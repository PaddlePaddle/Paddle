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
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set


class TrtConvertFcTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(batch):
            return np.random.randn(batch, 32).astype(np.float32)

        def generate_weight1():
            return np.random.randn(32, 32).astype(np.float32)

        def generate_weight2():
            return np.random.randn(32).astype(np.float32)

        for batch in [1, 2, 4]:
            for axis in [1]:
                for x in [1]:
                    for y in [1]:
                        dics = [{
                            "x_num_col_dims": x,
                            "y_num_col_dims": y
                        }, {
                            "axis": axis
                        }]
                        ops_config = [{
                            "op_type": "mul",
                            "op_inputs": {
                                "X": ["input_data"],
                                "Y": ["mul_weight"]
                            },
                            "op_outputs": {
                                "Out": ["mul_output"]
                            },
                            "op_attrs": dics[0]
                        }, {
                            "op_type": "elementwise_add",
                            "op_inputs": {
                                "X": ["mul_output"],
                                "Y": ["elementwise_add_weight"]
                            },
                            "op_outputs": {
                                "Out": ["elementwise_add_output"]
                            },
                            "op_attrs": dics[1]
                        }]
                        ops = self.generate_op_config(ops_config)

                        program_config = ProgramConfig(
                            ops=ops,
                            weights={
                                "mul_weight": TensorConfig(
                                    data_gen=partial(generate_weight1)),
                                "elementwise_add_weight":
                                TensorConfig(data_gen=partial(generate_weight2))
                            },
                            inputs={
                                "input_data": TensorConfig(
                                    data_gen=partial(generate_input1, batch)),
                            },
                            outputs=["elementwise_add_output"])

                        yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 8]}
            self.dynamic_shape.max_input_shape = {"input_data": [64, 256]}
            self.dynamic_shape.opt_input_shape = {"input_data": [2, 16]}

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 2), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 2), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), 1e-5

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
