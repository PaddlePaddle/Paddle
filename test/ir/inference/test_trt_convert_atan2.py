# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from functools import partial
from typing import List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertAtan2TestOneInputSpecialCase1(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1 << 32

        def generate_input(shape):
            if self.dims == 1:
                return np.random.random(shape).astype(np.float32)
            elif self.dims == 2:
                return np.random.random(shape).astype(np.float32)
            elif self.dims == 3:
                return np.random.random(shape).astype(np.float32)

        for shape in [[1], [2], [32, 64]]:
            self.dims = len(shape)
            dics = [{}]
            ops_config = [
                {
                    "op_type": "atan2",
                    "op_inputs": {
                        "X1": ["input_data1"],
                        "X2": ["input_data2"],
                    },
                    "op_outputs": {"Out": ["output_data"]},
                    "op_attrs": dics[0],
                }
            ]
            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    "input_data1": TensorConfig(
                        data_gen=partial(generate_input, shape)
                    ),
                    "input_data2": TensorConfig(
                        data_gen=partial(generate_input, shape)
                    ),
                },
                outputs=["output_data"],
            )

            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1],
                    "input_data2": [1],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [2],
                    "input_data2": [2],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [1],
                    "input_data2": [1],
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 64],
                    "input_data2": [1, 64],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [128, 64],
                    "input_data2": [128, 64],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [32, 64],
                    "input_data2": [32, 64],
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 32, 16],
                    "input_data2": [1, 32, 16],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [128, 32, 16],
                    "input_data2": [128, 32, 16],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 32, 16],
                    "input_data2": [2, 32, 16],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            runtime_version = paddle_infer.get_trt_runtime_version()
            if not dynamic_shape or (
                runtime_version[0] * 1000
                + runtime_version[1] * 100
                + runtime_version[2] * 10
                < 8400
            ):
                return 0, 4
            return 1, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), (1e-3, 1e-3)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-3, 1e-3)

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
