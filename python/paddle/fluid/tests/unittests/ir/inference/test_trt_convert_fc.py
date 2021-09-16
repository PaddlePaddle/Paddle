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
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        def generate_weight1(mul_weight_shape):
            return np.random.random(mul_weight_shape).astype(np.float32)

        def generate_weight2(shape):
            return np.random.random(shape).astype(np.float32)

        for batch in [1, 2, 4]:
            for x in [1, 2, 3]:
                if x == 1:
                    input_shape = [[batch, 64], [batch, 4, 16],
                                   [batch, 4, 4, 4]]
                elif x == 2:
                    input_shape = [[batch, 4, 64], [batch, 8, 16, 4]]
                elif x == 3:
                    input_shape = [[batch, 8, 16, 64]]
                for shape in input_shape:
                    self.input_dim = len(shape)
                    for with_elt in [False, True]:
                        for y in [1, 2, 3]:
                            if y == 1:
                                weight1_shape = [[64, 64], [64, 8, 8],
                                                 [64, 4, 4, 4]]
                            elif y == 2:
                                weight1_shape = [[8, 8, 64], [16, 4, 4, 16]]
                            elif y == 3:
                                weight1_shape = [[4, 4, 4, 64]]
                            for mul_weight_shape in weight1_shape:
                                dics = [{
                                    "x_num_col_dims": x,
                                    "y_num_col_dims": y
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
                                }]
                                if with_elt:
                                    ops_config.append({
                                        "op_type": "elementwise_add",
                                        "op_inputs": {
                                            "X": ["mul_output"],
                                            "Y": ["elementwise_add_weight"]
                                        },
                                        "op_outputs": {
                                            "Out": ["elementwise_add_output"]
                                        },
                                        "op_attrs": {
                                            "axis": -1
                                        }
                                    })
                                ops = self.generate_op_config(ops_config)

                                program_config = ProgramConfig(
                                    ops=ops,
                                    weights={
                                        "mul_weight": TensorConfig(
                                            data_gen=partial(generate_weight1,
                                                             mul_weight_shape)),
                                        "elementwise_add_weight":
                                        TensorConfig(data_gen=partial(
                                            generate_weight2,
                                            mul_weight_shape[-1]))
                                    } if with_elt else {
                                        "mul_weight": TensorConfig(
                                            data_gen=partial(generate_weight1,
                                                             mul_weight_shape))
                                    },
                                    inputs={
                                        "input_data": TensorConfig(
                                            data_gen=partial(generate_input,
                                                             shape)),
                                    },
                                    outputs=["elementwise_add_output"]
                                    if with_elt else ["mul_output"])

                                yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.input_dim == 2:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 8]}
                self.dynamic_shape.max_input_shape = {"input_data": [64, 256]}
                self.dynamic_shape.opt_input_shape = {"input_data": [2, 16]}
            elif self.input_dim == 3:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 4, 4]}
                self.dynamic_shape.max_input_shape = {
                    "input_data": [64, 256, 128]
                }
                self.dynamic_shape.opt_input_shape = {"input_data": [2, 16, 32]}
            elif self.input_dim == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 4, 4, 4]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [64, 256, 128, 256]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [2, 16, 32, 64]
                }

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
        yield self.create_inference_config(), (1, 2), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        yield self.create_inference_config(), (1, 2), (1e-5, 1e-5)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        yield self.create_inference_config(), (1, 3), 1e-5

    def test(self):
        self.run_test()

    # def test_quant(self):
    #     self.run_test(quant=True)


if __name__ == "__main__":
    unittest.main()
