# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from functools import partial

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertSetValue(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1():
            return np.random.random([1, 6, 20, 50, 10, 3]).astype(np.float32)

        def generate_input2():
            return np.random.random([1, 6, 20, 50, 10, 1]).astype(np.float32)

        ops_config = [
            {
                "op_type": "set_value",
                "op_inputs": {
                    "Input": ["input_data"],
                    "ValueTensor": ["update_data"],
                },
                "op_outputs": {"Out": ["set_output_data"]},
                "op_attrs": {
                    "axes": [5],
                    "starts": [0],
                    "ends": [1],
                    "steps": [1],
                },
            },
            {
                "op_type": "gelu",
                "op_inputs": {
                    "X": ["set_output_data"],
                },
                "op_outputs": {"Out": ["set_tmp_output_data"]},
                "op_attrs": {"approximate": True},
            },
            {
                "op_type": "slice",
                "op_inputs": {"Input": ["set_tmp_output_data"]},
                "op_outputs": {"Out": ["slice3_output_data"]},
                "op_attrs": {
                    "decrease_axis": [],
                    "axes": [5],
                    "starts": [1],
                    "ends": [2],
                },
            },
            {
                "op_type": "scale",
                "op_inputs": {"X": ["slice3_output_data"]},
                "op_outputs": {"Out": ["scale5_output_data"]},
                "op_attrs": {
                    "scale": 62.1,
                    "bias": 1,
                    "bias_after_scale": True,
                },
            },
            {
                "op_type": "scale",
                "op_inputs": {"X": ["scale5_output_data"]},
                "op_outputs": {"Out": ["scale6_output_data"]},
                "op_attrs": {
                    "scale": 0.1,
                    "bias": 0,
                    "bias_after_scale": True,
                },
            },
            {
                "op_type": "set_value",
                "op_inputs": {
                    "Input": ["set_tmp_output_data"],
                    "ValueTensor": ["scale6_output_data"],
                },
                "op_outputs": {"Out": ["output_data"]},
                "op_attrs": {
                    "axes": [5],
                    "starts": [1],
                    "ends": [2],
                    "steps": [1],
                },
            },
        ]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input1)),
                "update_data": TensorConfig(data_gen=partial(generate_input2)),
            },
            outputs=["output_data"],
        )

        yield program_config

    def sample_predictor_configs(self, program_config):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, 6, 20, 50, 10, 3],
                "update_data": [1, 6, 20, 50, 10, 1],
                "output_data": [1, 6, 20, 50, 10, 3],
                "set_output_data": [1, 6, 20, 50, 10, 3],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [1, 6, 20, 50, 10, 3],
                "update_data": [1, 6, 20, 50, 10, 1],
                "output_data": [1, 6, 20, 50, 10, 3],
                "set_output_data": [1, 6, 20, 50, 10, 3],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [1, 6, 20, 50, 10, 3],
                "update_data": [1, 6, 20, 50, 10, 1],
                "output_data": [1, 6, 20, 50, 10, 3],
                "set_output_data": [1, 6, 20, 50, 10, 3],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape:
                ver = paddle_infer.get_trt_compile_version()
                if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8200:
                    return 1, 5
                return 1, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        self.trt_param.workspace_size = 2013265920
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-5, 1e-4)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-3, 1e-3)

    def test(self):
        self.run_test()
