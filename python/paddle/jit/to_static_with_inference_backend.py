# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import sys
import textwrap
from pathlib import Path

import paddle
from paddle.inference import Config, PrecisionType, create_predictor
from paddle.nn import Layer
from paddle.static import InputSpec


def get_inference_precision(precision_str):
    if precision_str == "float32":
        return PrecisionType.Float32
    elif precision_str == "float16":
        return PrecisionType.Half
    elif precision_str == "bfloat16":
        return PrecisionType.Bfloat16
    else:
        raise AssertionError(f"unsupported precision {precision_str}")


def register_triton_custom_ops(model_dir):
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith("_package.so"):
                so_full_path = os.path.join(root, file)
                paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                    so_full_path
                )


def paddle_inference_decorator(function=None, **kwargs):
    used_as_at_decorator = False
    if function is None:
        used_as_at_decorator = True

    def decorator(func=None):
        if isinstance(func, paddle.nn.Layer):
            func = func.forward

        predictors = [None]
        signature = inspect.signature(func)
        arg_names = [v.name for v in signature.parameters.values()]

        if "*" in str(signature):
            raise ValueError(
                f"your function named {func.__name__} definition has * or ** args, please modify your function definition, but when calling this function, you can still use positional arguments."
            )

        arg_defaults = [v.default for v in signature.parameters.values()]

        memory_pool_init_size_mb = kwargs.get("memory_pool_init_size_mb", 1000)
        cache_static_model = kwargs.get("cache_static_model", False)
        save_model_dir = kwargs.get(
            "save_model_dir", os.path.join(Path.home(), ".cache")
        )
        save_model_dir += "/" + func.__name__
        precision_mode = kwargs.get("precision_mode", "float32")
        switch_ir_optim = kwargs.get("switch_ir_optim", True)
        switch_ir_debug = kwargs.get("switch_ir_debug", False)
        enable_cinn = kwargs.get("enable_cinn", False)

        with_trt = kwargs.get("with_trt", False)
        trt_precision_mode = kwargs.get("trt_precision_mode", "float32")
        trt_use_static = kwargs.get("trt_use_static", False)
        collect_shape = kwargs.get("collect_shape", False)
        default_delete_pass_lists = [
            "trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass",
            "add_support_int8_pass",
        ]
        delete_pass_lists = kwargs.get(
            "delete_pass_lists", default_delete_pass_lists
        )
        enable_new_ir = kwargs.get("enable_new_ir", False)
        exp_enable_use_cutlass = kwargs.get("exp_enable_use_cutlass", False)

        py_script = textwrap.dedent(inspect.getsource(func))
        py_script = py_script[py_script.find("def") :]
        if used_as_at_decorator:
            assert arg_names[0] == "self"
        save_path = save_model_dir + "/infer"
        d2s_input_info_path = save_path + "_d2s_input_info.txt"
        d2s_input_shapes = []
        d2s_input_names = []

        # get old d2s shapes!
        if os.path.exists(d2s_input_info_path) and cache_static_model:
            with open(d2s_input_info_path, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    name_shape = line.split(":")
                    assert len(name_shape) == 2
                    name = name_shape[0]
                    shape = name_shape[1]
                    if len(shape) > 0:
                        # this is for None input
                        shape = [int(s) for s in shape.split(",")]
                    d2s_input_shapes.append(shape)
                    d2s_input_names.append(name)

        class WrappedLayer(paddle.nn.Layer):
            def __init__(self, layer):
                super().__init__()
                self.fn = func
                self.layer = layer

            def forward(self, args):
                return (
                    paddle.jit.dy2static.program_translator.convert_to_static(
                        self.fn
                    )(self.layer, *args)
                )

        # This is the inner_most decorator, ie. when user invoke the function decorated by @paddle.jit.to_static(backend='inference', )
        # he is actually invoke this internel function.
        def innermost_decorator(*args, **kwargs):
            def get_tensor(run_time_args, arg_name):
                if isinstance(run_time_args, paddle.Tensor):
                    return [run_time_args]
                elif isinstance(run_time_args, list):
                    this_input_tensor_lists = []
                    for ele in run_time_args:
                        assert isinstance(
                            ele, paddle.Tensor
                        ), f"the elements in {arg_name} must be paddle.Tensor"
                        this_input_tensor_lists.append(ele)
                    return this_input_tensor_lists
                elif run_time_args is None:
                    return [None]
                else:
                    raise AssertionError(
                        f'''we only support adding @paddle.jit.to_static(backend='inference', ) in functions whose arguments are paddle.Tensor or list[paddle.Tensor] or None,
                        but here we get {arg_name} in your function is {type(run_time_args)}, please modify your function to meet our requirement.'''
                    )

            input_tensor_lists = []
            collected_names = []
            for i in range(len(args)):
                collected_names.append(arg_names[i])
                if i == 0 and used_as_at_decorator:
                    continue
                input_tensor_lists += get_tensor(args[i], arg_names[i])

            position_arguments_num = len(args)
            # some are invoked from keyword arguments.
            for i in range(position_arguments_num, len(arg_names)):
                if arg_names[i] in kwargs.keys():
                    this_input = kwargs[arg_names[i]]
                    input_tensor_lists += get_tensor(this_input, arg_names[i])
                    collected_names.append(arg_names[i])
                else:
                    this_input = arg_defaults[i]
                    if this_input is not None:
                        raise ValueError(
                            f"{arg_names[i]}'s default value must be None."
                        )
                    input_tensor_lists += [this_input]
                    collected_names.append(arg_names[i])

            if collected_names != arg_names:
                unspecified_names = str(set(arg_names) - set(collected_names))
                raise ValueError(
                    f"some arguments are not specified when you invoke your function, you must specify your all arguments, below arguments are not specified: {unspecified_names}"
                )

            # initiate the d2s_input_shapes.
            if len(d2s_input_shapes) == 0:
                for tensor in input_tensor_lists:
                    if tensor is None:
                        d2s_input_shapes.append([])
                    else:
                        d2s_input_shapes.append(tensor.shape)

            re_do_d2s = False
            # check whether the shape is changed
            for i in range(len(d2s_input_shapes)):
                if input_tensor_lists[i] is None:
                    continue
                # The rank of this tensor has changed
                if len(d2s_input_shapes[i]) != len(input_tensor_lists[i].shape):
                    re_do_d2s = True
                    d2s_input_shapes[i] = input_tensor_lists[i]
                    print("rank is changed, we need re do d2s.")
                    continue
                for j in range(len(d2s_input_shapes[i])):
                    if (
                        d2s_input_shapes[i][j] != -1
                        and d2s_input_shapes[i][j]
                        != input_tensor_lists[i].shape[j]
                    ):
                        re_do_d2s = True
                        d2s_input_shapes[i][j] = -1
                        print("shape is changed, we need re do d2s.")
                sys.stdout.flush()

            remove_non_input_tensor_lists = [
                ele for ele in input_tensor_lists if ele is not None
            ]

            if predictors[0] is not None and not re_do_d2s:
                results = predictors[0].run(remove_non_input_tensor_lists)
                return results if len(results) > 1 else results[0]

            if (
                not os.path.exists(save_path + ".pdmodel")
                or not cache_static_model
                or re_do_d2s
            ):
                # we need do jit.to_static and jit.save.
                def get_d2s_spec(run_time_args, name):
                    if isinstance(run_time_args, paddle.Tensor):
                        return InputSpec.from_tensor(run_time_args, name=name)
                    elif isinstance(run_time_args, list):
                        this_input_spec = []
                        suffix = 0
                        for ele in run_time_args:
                            assert isinstance(ele, paddle.Tensor)
                            this_input_spec.append(
                                InputSpec.from_tensor(
                                    ele, name=name + "_" + str(suffix)
                                )
                            )
                            suffix += 1
                        return this_input_spec
                    elif run_time_args is None:
                        # we need to add a None input_spec!
                        return None

                # we need do ds2.
                input_specs = []
                # first we handle Positional Arguments
                for i in range(len(args)):
                    if i == 0 and used_as_at_decorator:
                        assert isinstance(args[i], paddle.nn.Layer)
                    else:
                        input_specs.append(
                            get_d2s_spec(args[i], name=arg_names[i])
                        )
                # second we handle Keyword Arguments
                for i in range(position_arguments_num, len(arg_names)):
                    if arg_names[i] in kwargs.keys():
                        this_input = kwargs[arg_names[i]]
                        input_specs.append(
                            get_d2s_spec(this_input, name=arg_names[i])
                        )
                    else:
                        this_input = arg_defaults[i]
                        if this_input is not None:
                            raise ValueError(
                                f"{arg_names[i]}'s default value must be None."
                            )
                        input_specs.append(None)

                # update the input_spec's shape for doing d2s
                d2s_shapes_id = 0
                # initial the d2s_input_names!
                if len(d2s_input_names) == 0:
                    d2s_input_names.extend([None] * len(input_tensor_lists))
                for i in range(len(input_specs)):
                    if type(input_specs[i]) == list:
                        for j in range(len(input_specs[i])):
                            input_specs[i][j].shape = d2s_input_shapes[
                                d2s_shapes_id
                            ]
                            d2s_input_names[d2s_shapes_id] = input_specs[i][
                                j
                            ].name
                            d2s_shapes_id += 1
                    elif type(input_specs[i]) == paddle.static.InputSpec:
                        input_specs[i].shape = d2s_input_shapes[d2s_shapes_id]
                        d2s_input_names[d2s_shapes_id] = input_specs[i].name
                        d2s_shapes_id += 1
                    elif input_specs[i] is None:
                        if used_as_at_decorator:
                            d2s_input_names[d2s_shapes_id] = arg_names[i + 1]
                        else:
                            d2s_input_names[d2s_shapes_id] = arg_names[i]
                        d2s_shapes_id += 1

                os.environ["TRITON_KERNEL_CACHE_DIR"] = save_model_dir

                print("we are doing d2s!!")
                print("input_specs: ", input_specs)
                sys.stdout.flush()

                to_d2s_thing = func
                if used_as_at_decorator:
                    to_d2s_thing = WrappedLayer(args[0])
                    input_specs = [input_specs]

                model = paddle.jit.to_static(
                    to_d2s_thing,
                    input_spec=input_specs,
                    full_graph=True,
                )
                paddle.jit.save(model, save_path, skip_prune_program=True)

                # save d2s_shapes
                assert len(d2s_input_names) == len(d2s_input_shapes)
                with open(d2s_input_info_path, "w") as f:
                    for i in range(len(d2s_input_names)):
                        line = d2s_input_names[i] + ":"
                        line += (
                            ",".join([str(s) for s in d2s_input_shapes[i]])
                            + "\n"
                        )
                        f.write(line)
                print("d2s are done!!")
                sys.stdout.flush()
            else:
                # we need register some triton ops.
                register_triton_custom_ops(save_model_dir)

            # create predictor
            model_file = save_model_dir + "/infer.pdmodel"
            params_file = save_model_dir + "/infer.pdiparams"

            config = Config(model_file, params_file)
            config.enable_memory_optim()
            config.switch_ir_debug(switch_ir_debug)
            config.switch_ir_optim(switch_ir_optim)
            if exp_enable_use_cutlass:
                config.exp_enable_use_cutlass()
            if enable_cinn:
                config.enable_cinn()
            config.enable_new_ir(enable_new_ir)

            device_num = paddle.device.get_device()
            if 'gpu' in device_num:
                gpu_id = int(device_num.split(':')[1])
                config.enable_use_gpu(
                    memory_pool_init_size_mb,
                    gpu_id,
                    get_inference_precision(precision_mode),
                )

            if with_trt:
                dynamic_names = []
                min_input_shape = {}
                max_input_shape = {}
                opt_input_shape = {}
                shape_range_file = save_model_dir + "/trt_shape.txt"
                if collect_shape:
                    config.collect_shape_range_info(shape_range_file)
                elif os.path.exists(shape_range_file):
                    config.enable_tuned_tensorrt_dynamic_shape(
                        shape_range_file, True
                    )
                else:
                    for i in range(len(input_tensor_lists)):
                        if input_tensor_lists[i] is not None:
                            min_input_shape[
                                d2s_input_names[i]
                            ] = input_tensor_lists[i].shape
                            max_input_shape[
                                d2s_input_names[i]
                            ] = input_tensor_lists[i].shape
                            opt_input_shape[
                                d2s_input_names[i]
                            ] = input_tensor_lists[i].shape

                    config.set_trt_dynamic_shape_info(
                        min_input_shape, max_input_shape, opt_input_shape
                    )
                    config.enable_tensorrt_engine(
                        workspace_size=1 << 30,
                        max_batch_size=1,
                        min_subgraph_size=3,
                        precision_mode=get_inference_precision(
                            trt_precision_mode
                        ),
                        use_static=trt_use_static,
                        use_calib_mode=False,
                    )

            if predictors[0] is not None:
                predictors[0] = None

            for pass_name in delete_pass_lists:
                config.delete_pass(pass_name)

            predictors[0] = create_predictor(config)

            results = predictors[0].run(remove_non_input_tensor_lists)
            return results if len(results) > 1 else results[0]

        return innermost_decorator

    if function is not None:
        if isinstance(function, Layer):
            function.forward = decorator(function)
            return function
        else:
            return decorator(function)

    return decorator
