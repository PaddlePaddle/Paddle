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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import unittest
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set


class TrtConvertSplitTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        for shape in [[2, 3, 4]]:
            for dtype in [5]:
                for str_value in ["2", "23", "-1"]:
                    dics = [{
                        "str_value": str_value,
                        "shape": shape,
                        "dtype": dtype
                    }, {
                        "axis": -1
                    }]
                    ops_config = [
                        {
                            "op_type": "fill_constant",
                            "op_inputs": {},
                            "op_outputs": {
                                "Out": ["X_data"],
                            },
                            "op_attrs": dics[0]
                        },
                        {
                            "op_type": "elementwise_add",
                            "op_inputs": {
                                "X": ["X_data"],
                                "Y": ["Y_data"]
                            },
                            "op_outputs": {
                                "Out": ["out_data"],
                            },
                            "op_attrs": dics[1]
                        },
                    ]

                    def generate_input():
                        return np.random.random([1, 1]).astype(np.float32)

                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "Y_data":
                            TensorConfig(data_gen=partial(generate_input))
                        },
                        outputs=["out_data"])

                    yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):

        def generate_dynamic_shape(attrs):
            self.input_shape = [1, 1]
            max_shape = list(self.input_shape)
            min_shape = list(self.input_shape)
            opt_shape = list(self.input_shape)
            for i in range(len(self.input_shape)):
                max_shape[i] = max_shape[i] + 1
            self.dynamic_shape.min_input_shape = {"Y_data": min_shape}
            self.dynamic_shape.max_input_shape = {"Y_data": max_shape}
            self.dynamic_shape.opt_input_shape = {"Y_data": opt_shape}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        # Don't test static shape

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-5

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
