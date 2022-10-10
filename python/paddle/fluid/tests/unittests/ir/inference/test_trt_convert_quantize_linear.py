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

import unittest
import itertools
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import numpy as np
import paddle.inference as paddle_infer
from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig


class TrtConvertConv2dTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(attrs: List[Dict[str, Any]]):
            return np.ones([1, 3, 64, 64]).astype(np.float32) / 4

        def generate_weight1(attrs: List[Dict[str, Any]]):
            return np.random.random([1]).astype(np.float32) - 0.5

        attrs = [
            {
                "quant_axis": -1,
            },
        ]

        ops_config = [
            {
                "op_type": "dequantize_linear",
                "op_inputs": {
                    "X": ["input_data"],
                    "Scale": ["scale_data"],
                    "ZeroPoint": ["zeropoint_data"],
                },
                "op_outputs": {
                    "Y": ["output_data"]
                },
                "op_attrs": attrs[0]
            },
        ]

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "scale_data":
                TensorConfig(data_gen=partial(generate_weight1, attrs)),
                "zeropoint_data":
                TensorConfig(data_gen=partial(generate_weight1, attrs))
            },
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input1, attrs))
            },
            outputs=["output_data"])

        yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):

        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, 3, 64, 64],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [1, 3, 64, 64],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [1, 3, 64, 64],
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

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), (1e-2, 1e-2)

    def test(self):
        self.run_test()

    def test_quant(self):
        self.run_test(quant=True)


if __name__ == "__main__":
    unittest.main()
