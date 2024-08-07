# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import unittest
from functools import partial
from typing import Any

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


# Special case
class TrtConvertConv3dTransposeTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8400:
            return False
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(batch, num_channels, attrs: list[dict[str, Any]]):
            return np.ones([batch, num_channels, 4, 20, 30]).astype(np.float32)

        def generate_weight1(num_channels, attrs: list[dict[str, Any]]):
            return np.random.random([num_channels, 64, 3, 3, 3]).astype(
                np.float32
            )

        num_channels = 128
        batch = 1
        # in_channels
        self.num_channels = num_channels
        dics = [
            {
                "data_format": 'NCHW',
                "dilations": [1, 1, 1],
                "padding_algorithm": 'EXPLICIT',
                "groups": 1,
                "paddings": [1, 1, 1],
                "strides": [2, 2, 2],
                "output_padding": [1, 1, 1],
                "output_size": [],
            }
        ]

        ops_config = [
            {
                "op_type": "conv3d_transpose",
                "op_inputs": {
                    "Input": ["input_data"],
                    "Filter": ["conv3d_weight"],
                },
                "op_outputs": {"Output": ["output_data"]},
                "op_attrs": dics[0],
            }
        ]
        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "conv3d_weight": TensorConfig(
                    data_gen=partial(generate_weight1, num_channels, dics)
                )
            },
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input1, batch, num_channels, dics)
                )
            },
            outputs=["output_data"],
        )

        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, 128, 4, 20, 30],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [1, 128, 4, 20, 30],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [1, 128, 4, 20, 30],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()

    def test_quant(self):
        self.add_skip_trt_case()
        self.run_test(quant=True)


if __name__ == "__main__":
    unittest.main()
