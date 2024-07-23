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


# get paddle.Tensor for paddle inference use.
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


# get paddle.Tensor's input_spec for doing dynamic to static.
def get_d2s_spec(run_time_args, name):
    if isinstance(run_time_args, paddle.Tensor):
        return InputSpec.from_tensor(run_time_args, name=name)
    elif isinstance(run_time_args, list):
        this_input_spec = []
        suffix = 0
        for ele in run_time_args:
            assert isinstance(ele, paddle.Tensor)
            this_input_spec.append(
                InputSpec.from_tensor(ele, name=name + "_" + str(suffix))
            )
            suffix += 1
        return this_input_spec
    elif run_time_args is None:
        # we need to add a None input_spec!
        return None


class InferenceEngine:
    def __init__(self, func, used_as_at_decorator, **kwargs):
        super().__init__()
        self.used_as_at_decorator = used_as_at_decorator

        self.predictor = None
        signature = inspect.signature(func)
        self.arg_names = [v.name for v in signature.parameters.values()]

        if "*" in str(signature):
            raise ValueError(
                f"your function named {func.__name__} definition has * or ** args, please modify your function definition, but when calling this function, you can still use positional arguments."
            )

        self.arg_defaults = [v.default for v in signature.parameters.values()]

        self.memory_pool_init_size_mb = kwargs.get(
            "memory_pool_init_size_mb", 1000
        )
        self.cache_static_model = kwargs.get("cache_static_model", False)
        self.save_model_dir = kwargs.get(
            "save_model_dir",
            os.path.join(Path.home(), ".cache", "paddle", "inference_models"),
        )
        self.save_model_dir = os.path.join(self.save_model_dir, func.__name__)
        self.precision_mode = kwargs.get("precision_mode", "float32")
        self.switch_ir_optim = kwargs.get("switch_ir_optim", True)
        self.switch_ir_debug = kwargs.get("switch_ir_debug", False)
        self.enable_cinn = kwargs.get("enable_cinn", False)

        self.with_trt = kwargs.get("with_trt", False)
        self.trt_precision_mode = kwargs.get("trt_precision_mode", "float32")
        self.trt_use_static = kwargs.get("trt_use_static", False)
        self.collect_shape = kwargs.get("collect_shape", False)
        default_delete_pass_lists = [
            "trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass",
            "add_support_int8_pass",
        ]
        self.delete_pass_lists = kwargs.get(
            "delete_pass_lists", default_delete_pass_lists
        )
        self.enable_new_ir = kwargs.get("enable_new_ir", False)
        self.exp_enable_use_cutlass = kwargs.get(
            "exp_enable_use_cutlass", False
        )

        py_script = textwrap.dedent(inspect.getsource(func))
        py_script = py_script[py_script.find("def") :]
        if used_as_at_decorator:
            assert self.arg_names[0] == "self"
        self.save_path = os.path.join(self.save_model_dir, "infer")
        d2s_input_info_path = self.save_path + "_d2s_input_info.txt"
        d2s_input_shapes = []
        d2s_input_names = []

        # get old d2s shapes!
        if os.path.exists(d2s_input_info_path) and self.cache_static_model:
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

        self.d2s_input_info_path = d2s_input_info_path
        self.d2s_input_shapes = d2s_input_shapes
        self.d2s_input_names = d2s_input_names

    def check_and_update_d2s_input_shapes(self, input_tensor_lists):
        d2s_input_shapes = self.d2s_input_shapes
        # initiate the d2s_input_shapes.
        if len(d2s_input_shapes) == 0:
            for tensor in input_tensor_lists:
                if tensor is None:
                    d2s_input_shapes.append([])
                else:
                    d2s_input_shapes.append(tensor.shape)

        self.re_do_d2s = False
        # check whether the shape is changed
        for i in range(len(d2s_input_shapes)):
            if input_tensor_lists[i] is None:
                continue
            # The rank of this tensor has changed
            if len(d2s_input_shapes[i]) != len(input_tensor_lists[i].shape):
                self.re_do_d2s = True
                print(
                    f"{self.d2s_input_names[i]}'s rank is changed from {len(d2s_input_shapes[i])} to {len(input_tensor_lists[i].shape)}, need re do jit.save"
                )
                d2s_input_shapes[i] = input_tensor_lists[i].shape
                continue
            for j in range(len(d2s_input_shapes[i])):
                if (
                    d2s_input_shapes[i][j] != -1
                    and d2s_input_shapes[i][j] != input_tensor_lists[i].shape[j]
                ):
                    self.re_do_d2s = True
                    print(
                        f"{self.d2s_input_names[i]}'s shape is changed from {d2s_input_shapes[i]} to {input_tensor_lists[i].shape}, need re do jit.save"
                    )
                    d2s_input_shapes[i][j] = -1
            sys.stdout.flush()
        # update the d2s_input_shapes, because of dynamic shape
        self.d2s_input_shapes = d2s_input_shapes

    def to_static_model(self, func, input_tensor_lists, *args, **kwargs):
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

        arg_names = self.arg_names
        arg_defaults = self.arg_defaults

        # we need do ds2.
        input_specs = []
        # first we handle Positional Arguments
        for i in range(len(args)):
            if i == 0 and self.used_as_at_decorator:
                assert isinstance(args[i], paddle.nn.Layer)
            else:
                input_specs.append(get_d2s_spec(args[i], name=arg_names[i]))
        position_arguments_num = len(args)
        # second we handle Keyword Arguments
        for i in range(position_arguments_num, len(arg_names)):
            if arg_names[i] in kwargs.keys():
                this_input = kwargs[arg_names[i]]
                input_specs.append(get_d2s_spec(this_input, name=arg_names[i]))
            else:
                this_input = arg_defaults[i]
                if this_input is not None:
                    raise ValueError(
                        f"{arg_names[i]}'s default value must be None."
                    )
                input_specs.append(None)

        # update the input_spec's shape for doing d2s
        d2s_shapes_id = 0
        # initial the self.d2s_input_names!
        if len(self.d2s_input_names) == 0:
            self.d2s_input_names.extend([None] * len(input_tensor_lists))
        for i in range(len(input_specs)):
            if isinstance(input_specs[i], list):
                for j in range(len(input_specs[i])):
                    input_specs[i][j].shape = self.d2s_input_shapes[
                        d2s_shapes_id
                    ]
                    self.d2s_input_names[d2s_shapes_id] = input_specs[i][j].name
                    d2s_shapes_id += 1
            elif isinstance(input_specs[i], paddle.static.InputSpec):
                input_specs[i].shape = self.d2s_input_shapes[d2s_shapes_id]
                self.d2s_input_names[d2s_shapes_id] = input_specs[i].name
                d2s_shapes_id += 1
            elif input_specs[i] is None:
                if self.used_as_at_decorator:
                    self.d2s_input_names[d2s_shapes_id] = arg_names[i + 1]
                else:
                    self.d2s_input_names[d2s_shapes_id] = arg_names[i]
                d2s_shapes_id += 1

        print(
            f"now will use paddle.jit.save to save the {func.__name__} function to {self.save_path}.pdmodel"
        )
        print("input_specs: ", input_specs)
        sys.stdout.flush()

        to_d2s_thing = func
        if self.used_as_at_decorator:
            to_d2s_thing = WrappedLayer(args[0])
            input_specs = [input_specs]

        model = paddle.jit.to_static(
            to_d2s_thing,
            input_spec=input_specs,
            full_graph=True,
        )
        paddle.jit.save(model, self.save_path, skip_prune_program=True)

        # save d2s_shapes
        assert len(self.d2s_input_names) == len(self.d2s_input_shapes)
        with open(self.d2s_input_info_path, "w") as f:
            for i in range(len(self.d2s_input_names)):
                line = self.d2s_input_names[i] + ":"
                line += (
                    ",".join([str(s) for s in self.d2s_input_shapes[i]]) + "\n"
                )
                f.write(line)
        print(
            f"the {func.__name__} function is sucessfully saved to {self.save_path}.pdmodel"
        )
        sys.stdout.flush()

    def get_input_tensor_lists(self, *args, **kwargs):
        collected_names = []
        input_tensor_lists = []
        arg_names = self.arg_names
        arg_defaults = self.arg_defaults
        for i in range(len(args)):
            collected_names.append(arg_names[i])
            if i == 0 and self.used_as_at_decorator:
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
        return input_tensor_lists

    # why we need input_tensor_lists? this is for TensorRT max/min/opt shape.
    def create_predictor(self, input_tensor_lists):
        # create predictor
        model_file = os.path.join(self.save_model_dir, "infer.pdmodel")
        params_file = os.path.join(self.save_model_dir, "infer.pdiparams")

        config = Config(model_file, params_file)
        config.enable_memory_optim()
        config.switch_ir_debug(self.switch_ir_debug)
        config.switch_ir_optim(self.switch_ir_optim)
        if self.exp_enable_use_cutlass:
            config.exp_enable_use_cutlass()
        if self.enable_cinn:
            config.enable_cinn()
        config.enable_new_ir(self.enable_new_ir)

        device_num = paddle.device.get_device()
        if 'gpu' in device_num:
            gpu_id = int(device_num.split(':')[1])
            config.enable_use_gpu(
                self.memory_pool_init_size_mb,
                gpu_id,
                get_inference_precision(self.precision_mode),
            )

        if self.with_trt:
            dynamic_names = []
            min_input_shape = {}
            max_input_shape = {}
            opt_input_shape = {}
            shape_range_file = os.path.join(
                self.save_model_dir, "trt_shape.txt"
            )
            if self.collect_shape:
                config.collect_shape_range_info(shape_range_file)
            elif os.path.exists(shape_range_file):
                config.enable_tuned_tensorrt_dynamic_shape(
                    shape_range_file, True
                )
            else:
                for i in range(len(input_tensor_lists)):
                    if input_tensor_lists[i] is not None:
                        min_input_shape[
                            self.d2s_input_names[i]
                        ] = input_tensor_lists[i].shape
                        max_input_shape[
                            self.d2s_input_names[i]
                        ] = input_tensor_lists[i].shape
                        opt_input_shape[
                            self.d2s_input_names[i]
                        ] = input_tensor_lists[i].shape

                config.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape
                )
            config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=3,
                precision_mode=get_inference_precision(self.trt_precision_mode),
                use_static=self.trt_use_static,
                use_calib_mode=False,
            )

        if self.predictor is not None:
            self.predictor = None

        if self.enable_new_ir:
            config.delete_pass(self.delete_pass_lists)
        else:
            for pass_name in self.delete_pass_lists:
                config.delete_pass(pass_name)

        self.predictor = create_predictor(config)


def paddle_inference_decorator(function=None, **kwargs):
    """
    Converts dynamic graph APIs into static graph saved in disk. Then will use Paddle Inference to predictor based on
    the static model in the disk.
    This function return a callable function, user can use it to inference just like dynamic function.
    Args:
        function (callable): Callable dynamic graph function. It must be a member function of paddle.nn.Layer.
        If it used as a decorator, the decorated function will be parsed as this parameter.

        kwargs: Support keys including `property`, set `property` to True if the function
            is python property.

    Returns:
        Tensor(s): the decorated function can be sued for inference.

    Examples:
        .. code-block:: python
            >>> # doctest: +SKIP('`paddle.jit.paddle_inference_decorator` can not run in xdoctest')
            >>> import paddle
            >>> from paddle.jit import paddle_inference_decorator
            >>> class ExampleLayer(paddle.nn.Layer):
            ...     def __init__(self, hidd):
            ...         super().__init__()
            ...         self.fn = paddle.nn.Linear(hidd, hidd, bias_attr=False)
            ...     def forward(self, x):
            ...         for i in range(10):
            ...             x = paddle.nn.functional.softmax(x,-1)
            ...         x = x.cast("float32")
            ...         x = self.func(x)
            ...         return x
            ...     def func(self, x):
            ...         x = x + x
            ...         return self.fn(x)

            >>> batch = 4096
            >>> hidd = 1024
            >>> dtype = "bfloat16"
            >>> x = paddle.rand([batch, hidd], dtype=dtype)
            >>> mylayer = ExampleLayer(hidd)
            >>> dynamic_result = mylayer(x)
            >>> mylayer = paddle.jit.to_static(mylayer, backend='inference')
            >>> decorator_result = mylayer(x)

    """

    used_as_at_decorator = function is None

    def decorator(func=None):
        if isinstance(func, paddle.nn.Layer):
            func = func.forward

        infer_engine = InferenceEngine(func, used_as_at_decorator, **kwargs)

        # This is the inner_most decorator, ie. when user invoke the function decorated by @paddle.jit.to_static(backend='inference', )
        # he is actually invoke this internel function.
        def innermost_decorator(*args, **kwargs):
            input_tensor_lists = infer_engine.get_input_tensor_lists(
                *args, **kwargs
            )

            # this function will update infer_engine.re_do_d2s.
            infer_engine.check_and_update_d2s_input_shapes(input_tensor_lists)

            remove_non_input_tensor_lists = [
                ele for ele in input_tensor_lists if ele is not None
            ]

            if (
                infer_engine.predictor is not None
                and not infer_engine.re_do_d2s
            ):
                results = infer_engine.predictor.run(
                    remove_non_input_tensor_lists
                )
                return results if len(results) > 1 else results[0]

            # we need do jit.to_static and jit.save!
            if (
                not os.path.exists(infer_engine.save_path + ".pdmodel")
                or not infer_engine.cache_static_model
                or infer_engine.re_do_d2s
            ):
                infer_engine.to_static_model(
                    func, input_tensor_lists, *args, **kwargs
                )

            infer_engine.create_predictor(input_tensor_lists)

            results = infer_engine.predictor.run(remove_non_input_tensor_lists)
            return results if len(results) > 1 else results[0]

        return innermost_decorator

    if function is not None:
        if isinstance(function, Layer):
            function.forward = decorator(function)
            return function
        else:
            return decorator(function)

    return decorator
