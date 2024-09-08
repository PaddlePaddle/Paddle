# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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

import os

import numpy as np

import paddle
from paddle.base import core, dygraph
from paddle.base.framework import (
    EagerParamBase,
    Variable,
)
from paddle.framework import use_pir_api
from paddle.jit.api import (
    _get_function_names_from_layer,
    get_ast_static_function,
    to_static,
)
from paddle.jit.dy2static.program_translator import (
    StaticFunction,
)
from paddle.nn import Layer
from paddle.tensorrt.converter import PaddleToTensorRTConverter
from paddle.tensorrt.util import (
    forbid_op_lower_trt,
    run_pir_pass,
    warmup_shape_infer,
)


class Input:
    def __init__(
        self,
        min_input_shape,
        max_input_shape,
        optim_input_shape=None,
        input_data_type=None,
        input_range=None,
    ):

        self.min_input_shape = min_input_shape
        self.max_input_shape = max_input_shape
        self.optim_input_shape = optim_input_shape
        self.input_data_type = input_data_type
        self.input_range = input_range

    def generate_input_data(self):
        if self.input_data_type is None:
            self.input_data_type = 'float32'

        if self.input_range is None:
            self.input_range = (
                (0.0, 1.0) if 'float' in self.input_data_type else (1, 10)
            )

        if 'int' in self.input_data_type:
            low, high = self.input_range
            self.input_min_data = np.random.randint(
                low, high, size=self.min_input_shape
            )
            self.input_optim_data = np.random.randint(
                low, high, size=self.optim_input_shape
            )
            self.input_max_data = np.random.randint(
                low, high, size=self.max_input_shape
            )
        else:
            low, high = self.input_range if self.input_range else (0, 1)
            self.input_min_data = np.random.uniform(
                low, high, size=self.min_input_shape
            ).astype(self.input_data_type)
            self.input_optim_data = np.random.uniform(
                low, high, size=self.optim_input_shape
            ).astype(self.input_data_type)
            self.input_max_data = np.random.uniform(
                low, high, size=self.max_input_shape
            ).astype(self.input_data_type)

        return self.input_min_data, self.input_optim_data, self.input_max_data


class TensorRTConfig:
    def __init__(
        self,
        inputs,
        min_subgraph_size=3,
        save_model_dir=None,
        disable_ops=None,
    ):
        self.inputs = (inputs,)
        self.min_subgraph_size = min_subgraph_size
        self.save_model_dir = save_model_dir
        self.disable_ops = disable_ops
        paddle.framework.set_flags(
            {'FLAGS_trt_min_group_size': min_subgraph_size}
        )


# return an optimized program with pd_op.tensorrt_engine operations.
def convert_to_trt(program, trt_config, scope):
    if not isinstance(program, paddle.base.libpaddle.pir.Program):
        raise TypeError(
            f"program type must be paddle.base.libpaddle.pir.Program, but received {type(program)}"
        )

    output_var = []
    feed_name = []

    for op in program.global_block().ops:
        if op.name() == "pd_op.fetch":
            for operand in op.operands():
                source = operand.source()
                output_var.append(source)
        if op.name() == "pd_op.data" or op.name() == "pd_op.feed":
            param_name = op.attrs()["name"]
            feed_name.append(param_name)

    with paddle.pir_utils.IrGuard():
        for i, input_instance in enumerate(trt_config.inputs):
            # get fake inputs
            min_data, _, max_data = input_instance[i].generate_input_data()
            program_with_output = program.list_vars()[-1]

            # run warmup for collecting shape
            warmup_shape_infer(
                program,
                min_shape_feed={feed_name[0]: min_data},
                max_shape_feed={feed_name[0]: max_data},
                fetch_var_list=output_var,
                scope=scope,
            )

        # run pir pass (including trt_op_marker_pass)
        program_with_pir = run_pir_pass(program, partition_mode=False)

        # specify certain operators to be excluded from entering TensorRT
        if trt_config.disable_ops:
            forbid_op_lower_trt(program, trt_config.disable_ops)

        # run pir pass (including trt_sub_graph_extract_pass)
        program_with_pir = run_pir_pass(program, partition_mode=True)
        trt_output_var = []

        for op in program_with_pir.global_block().ops:
            if op.name() == "pd_op.fetch":
                for operand in op.operands():
                    source = operand.source()
                    trt_output_var.append(source)

        # Step4: run TRTConverter (would lower group_op into tensorrt_engine_op)
        converter = PaddleToTensorRTConverter(program_with_pir, scope)
        converter.convert_program_to_trt()

        # Save PIR program as JSON
        if trt_config.save_model_dir:
            input_values = []
            input_values.extend(
                result
                for op in program_with_pir.global_block().ops
                if op.name() == "pd_op.data" or op.name() == "pd_op.feed"
                for result in op.results()
            )
            place = paddle.CUDAPlace(0)
            exe = paddle.static.Executor(place)

            paddle.static.save_inference_model(
                trt_config.save_model_dir,
                input_values,
                trt_output_var,
                exe,
                program=program_with_pir,
            )
        return program_with_pir


# Obtain a program with tensorrt_op for dynamic-to-static scenarios.
def export(funtion=None, input_spec=None, config=None, **kwargs):
    # Converts dynamic graph APIs into static graph
    static_net = paddle.jit.to_static(
        funtion,
        input_spec=input_spec,
        **kwargs,
    )
    is_prim_infer = core._is_fwd_prim_enabled() and core._is_bwd_prim_enabled()
    if isinstance(static_net, paddle.DataParallel):
        inner_layer = static_net._layers
    else:
        inner_layer = static_net

    inner_input_spec = None
    if input_spec is not None:
        if isinstance(static_net, Layer):
            for member_name in _get_function_names_from_layer(inner_layer):
                static_func = getattr(inner_layer, member_name, None)
                if (
                    isinstance(static_func, StaticFunction)
                    and 'forward' != member_name
                ):
                    raise ValueError(
                        f"If there are static functions other than 'forward' that need to be saved, the input 'input_spec' should be None, but received the type of 'input_spec' is {type(input_spec)}."
                    )
        if not isinstance(input_spec, (list, tuple)):
            raise TypeError(
                f"The input input_spec should be 'list', but received input_spec's type is {type(input_spec)}."
            )
        inner_input_spec = []
        for var in paddle.utils.flatten(input_spec):
            if isinstance(var, paddle.static.InputSpec):
                inner_input_spec.append(var)
            elif isinstance(
                var, (core.eager.Tensor, Variable, paddle.pir.Value)
            ):
                inner_input_spec.append(
                    paddle.static.InputSpec.from_tensor(var)
                )
            else:
                inner_input_spec.append(var)

    with_hook = False
    scope = core.Scope()
    extra_var_info = {}
    if isinstance(static_net, Layer):
        functions = list(set(_get_function_names_from_layer(static_net)))
        functions = sorted(functions)
        if static_net._forward_pre_hooks or static_net._forward_post_hooks:
            with_hook = True
    else:
        functions = [static_net]

    combine_vars = {}
    combine_program = []
    property_vals = []  # (value, key)
    concrete_program = None

    for attr_func in functions:
        if isinstance(static_net, Layer):
            static_func = get_ast_static_function(
                getattr(inner_layer, attr_func, None)
            )
            if isinstance(static_func, StaticFunction):
                if static_func.is_property:
                    # property method to be exported
                    immediate_val = static_func()
                    property_vals.append(
                        (
                            immediate_val,
                            static_net.__class__.__name__ + '.' + attr_func,
                        )
                    )
                    continue
                concrete_program = (
                    static_func.concrete_program_specify_input_spec(
                        inner_input_spec,
                        with_hook=with_hook,
                        is_prim_infer=is_prim_infer,
                    )
                )
            elif 'forward' == attr_func:
                if inner_input_spec:
                    inner_input_spec = paddle.utils.pack_sequence_as(
                        input_spec, inner_input_spec
                    )
                static_forward = to_static(
                    inner_layer.forward,
                    input_spec=inner_input_spec,
                    full_graph=True,
                )
                concrete_program = (
                    static_forward.concrete_program_specify_input_spec(
                        with_hook=with_hook, is_prim_infer=is_prim_infer
                    )
                )
                # the input_spec has been used in declarative, which is equal to
                # @to_static with input_spec and jit.save without input_spec,
                # avoid needless warning
                inner_input_spec = None
            else:
                continue
        else:
            if isinstance(attr_func, StaticFunction):
                static_func = get_ast_static_function(attr_func)
                if static_func.is_property:
                    immediate_val = static_func()
                    property_vals.append((immediate_val, static_func))
                    continue

                concrete_program = (
                    static_func.concrete_program_specify_input_spec(
                        inner_input_spec, is_prim_infer=is_prim_infer
                    )
                )
            else:
                static_func = get_ast_static_function(attr_func)
                if inner_input_spec:
                    inner_input_spec = paddle.utils.pack_sequence_as(
                        input_spec, inner_input_spec
                    )
                static_function = to_static(
                    static_func,
                    input_spec=inner_input_spec,
                    full_graph=True,
                )
                concrete_program = static_function.concrete_program

        dygraph_state_dict = None
        if isinstance(inner_layer, Layer):
            dygraph_state_dict = inner_layer.to_static_state_dict()
        elif isinstance(attr_func, StaticFunction):
            if static_func._class_instance:
                dygraph_state_dict = (
                    static_func._class_instance.to_static_state_dict()
                )
        if dygraph_state_dict:
            state_names_dict = {}
            state_var_dict = {}
            for strcutured_name, var in dygraph_state_dict.items():
                state_names_dict[var.name] = strcutured_name
                state_var_dict[var.name] = var
        with dygraph.guard():
            if use_pir_api():
                for tensor, value in zip(*concrete_program.parameters):
                    if not value.persistable:
                        continue
                    param_or_buffer_tensor = scope.var(value.name).get_tensor()

                    src_tensor = (
                        state_var_dict[tensor.name].value().get_tensor()
                    )
                    param_or_buffer_tensor._share_data_with(src_tensor)
            else:
                for param_or_buffer in concrete_program.parameters:
                    if param_or_buffer.type == core.VarDesc.VarType.VOCAB:
                        scr_tensor = param_or_buffer.value().get_map_tensor()
                        tgt_var = scope.var(param_or_buffer.name)
                        tgt_var.set_vocab(scr_tensor)
                    else:
                        param_or_buffer_tensor = scope.var(
                            param_or_buffer.name
                        ).get_tensor()
                        # src_tensor = param_or_buffer.value().get_tensor()
                        src_tensor = (
                            state_var_dict[param_or_buffer.name]
                            .value()
                            .get_tensor()
                        )
                        param_or_buffer_tensor._share_data_with(src_tensor)
                    if param_or_buffer.name not in extra_var_info:
                        extra_info_dict = {}
                        if param_or_buffer.name in state_names_dict:
                            extra_info_dict['structured_name'] = (
                                state_names_dict[param_or_buffer.name]
                            )
                        extra_info_dict['stop_gradient'] = (
                            param_or_buffer.stop_gradient
                        )
                        if isinstance(param_or_buffer, EagerParamBase):
                            extra_info_dict['trainable'] = (
                                param_or_buffer.trainable
                            )
                        extra_var_info[param_or_buffer.name] = extra_info_dict

    with paddle.pir_utils.IrGuard():
        main_program = static_net.forward.main_program
        program_with_trt = convert_to_trt(main_program, config, scope)
        return program_with_trt, scope


# Obtain a program with tensorrt_op by directly loading the model.
def export_loaded_model(model_dir, trt_config):
    if os.path.abspath(trt_config.save_model_dir) == os.path.abspath(model_dir):
        raise ValueError(
            "The `trt_config.save_model_dir` and `model_dir` cannot be the same. Please specify a different directory for saving the model."
        )

    scope = paddle.static.global_scope()
    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)

    model_filename = model_dir + '.json'
    if os.path.exists(model_dir + '.json'):
        model_filename = model_dir + '.json'
    elif os.path.exists(model_dir + '.pdmodel'):
        model_filename = model_dir + '.pdmodel'
    else:
        raise ValueError(
            f"No valid model file found in the directory '{model_dir}'. Expected either 'json' or 'pdmodel'. Please ensure that the directory contains one of these files."
        )

    params_filename = model_dir + '.pdiparams'

    with paddle.pir_utils.IrGuard():
        [program, feed_target_names, fetch_targets] = (
            paddle.static.io.load_inference_model(
                model_dir,
                executor=exe,
                model_filename=model_filename,
                params_filename=params_filename,
            )
        )

    return convert_to_trt(program, trt_config, scope)
