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
from paddle.tensorrt.converter import PaddleToTensorRTConverter
from paddle.tensorrt.util import (
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

    def forbid_op_lower_trt(self, program, disabled_ops):
        if isinstance(disabled_ops, str):
            disabled_ops = [disabled_ops]
        for op in program.global_block().ops:
            if op.name() in disabled_ops:
                op.set_bool_attr("__l_trt__", False)


# return an optimized program with pd_op.tensorrt_engine operations.
def converter_to_trt(program, trt_config, scope):
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
            )

        # run pir pass (including trt_op_marker_pass)
        program_with_pir = run_pir_pass(program, partition_mode=False)

        # specify certain operators to be excluded from entering TensorRT
        if trt_config.disable_ops:
            trt_config.forbid_op_lower_trt(program, trt_config.disable_ops)

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
    main_program = static_net.main_program
    scope = paddle.static.global_scope()
    return converter_to_trt(main_program, config, scope)


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

    return converter_to_trt(program, trt_config, scope)
