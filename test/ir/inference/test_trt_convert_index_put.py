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

from __future__ import annotations

import unittest
from functools import partial

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertIndexPut(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8200:
            return False
        return True

    def sample_program_configs(self):
        def generate_input1():
            return np.random.random([1, 80, 2]).astype(np.float32)

        def generate_input2():
            return np.random.randint(0, 2, (1, 80)).astype(np.float32)

        def generate_input3():
            if self.value_num == 2:
                return np.random.random([2]).astype(np.float32)
            else:
                return np.random.random([1]).astype(np.float32)

        for v_num in [1, 2]:
            self.value_num = v_num
            ops_config = [
                {
                    "op_type": "cast",
                    "op_inputs": {
                        "X": ["input_data2"],
                    },
                    "op_outputs": {
                        "Out": [
                            "cast_output_data",
                        ]
                    },
                    "op_attrs": {'in_dtype': 5, 'out_dtype': 0},
                },
                {
                    "op_type": "index_put",
                    "op_inputs": {
                        "x": ["input_data1"],
                        "indices": ["cast_output_data"],
                        "value": ["input_data3"],
                    },
                    "op_outputs": {
                        "out": [
                            "output_data",
                        ]
                    },
                    "op_attrs": {'accumulate': False},
                },
            ]
            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    "input_data1": TensorConfig(
                        data_gen=partial(generate_input1)
                    ),
                    "input_data2": TensorConfig(
                        data_gen=partial(generate_input2)
                    ),
                    "input_data3": TensorConfig(
                        data_gen=partial(generate_input3)
                    ),
                },
                outputs=["output_data"],
            )

            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, list[int], float):
        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_dynamic_shape(attrs):
            if self.value_num == 2:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 80, 2],
                    "input_data2": [1, 80],
                    "input_data3": [2],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [1, 81, 2],
                    "input_data2": [1, 81],
                    "input_data3": [2],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [1, 80, 2],
                    "input_data2": [1, 80],
                    "input_data3": [2],
                }
            else:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 80, 2],
                    "input_data2": [1, 80],
                    "input_data3": [1],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [1, 81, 2],
                    "input_data2": [1, 81],
                    "input_data3": [2],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [1, 80, 2],
                    "input_data2": [1, 80],
                    "input_data3": [1],
                }

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if not dynamic_shape:
                return 0, 6
            if self.value_num == 2:
                return 1, 5
            return 1, 4

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        self.trt_param.workspace_size = 1073741824
        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-3

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
