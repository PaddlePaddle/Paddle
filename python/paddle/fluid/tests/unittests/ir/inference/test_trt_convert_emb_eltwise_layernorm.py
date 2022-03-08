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
import unittest


class TrtConvertEmbEltwiseLayernormTest1(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(batch, input_size):
            return np.random.randint(
                0, 7, size=(batch, input_size, 1)).astype(np.int64)

        def generate_weight1(size11, size2):
            return np.random.randn(size11, size2).astype(np.float32)

        def generate_weight2(size12, size2):
            return np.random.randn(size12, size2).astype(np.float32)

        def generate_weight3(size13, size2):
            return np.random.randn(size13, size2).astype(np.float32)

        def generate_weight4(size2):
            return np.random.randn(size2).astype(np.float32)

        for input_size in [16, 128]:
            for batch in [1, 2, 4]:
                for size1 in [[8, 513, 768], [513, 768, 8], [768, 8, 513]]:
                    size11 = size1[0]
                    size12 = size1[1]
                    size13 = size1[2]
                    for size2 in [32, 768]:
                        for norm_axis in [2]:
                            for epsilon in [0.0001, 0.0005]:
                                for axis1 in [0, -1]:
                                    for axis2 in [0, -1]:
                                        for type in [
                                                "lookup_table",
                                                "lookup_table_v2"
                                        ]:
                                            dics = [{
                                                "is_sparse": False,
                                                "is_distributed": False,
                                                "padding_idx": -1,
                                                "is_test": True
                                            }, {
                                                "is_sparse": False,
                                                "is_distributed": False,
                                                "padding_idx": -1,
                                            }, {
                                                "axis": axis1
                                            }, {
                                                "axis": axis2
                                            }, {
                                                "begin_norm_axis": norm_axis,
                                                "epsilon": epsilon
                                            }]
                                            ops_config = [{
                                                "op_type": type,
                                                "op_inputs": {
                                                    "Ids": ["input_data1"],
                                                    "W": ["embedding1_weight"]
                                                },
                                                "op_outputs": {
                                                    "Out":
                                                    ["embedding1_output"]
                                                },
                                                "op_attrs": dics[0]
                                                if type == "lookup_table" else
                                                dics[1]
                                            }, {
                                                "op_type": type,
                                                "op_inputs": {
                                                    "Ids": ["input_data2"],
                                                    "W": ["embedding2_weight"]
                                                },
                                                "op_outputs": {
                                                    "Out":
                                                    ["embedding2_output"]
                                                },
                                                "op_attrs": dics[0]
                                                if type == "lookup_table" else
                                                dics[1]
                                            }, {
                                                "op_type": type,
                                                "op_inputs": {
                                                    "Ids": ["input_data3"],
                                                    "W": ["embedding3_weight"]
                                                },
                                                "op_outputs": {
                                                    "Out":
                                                    ["embedding3_output"]
                                                },
                                                "op_attrs": dics[0]
                                                if type == "lookup_table" else
                                                dics[1]
                                            }, {
                                                "op_type": "elementwise_add",
                                                "op_inputs": {
                                                    "X": ["embedding2_output"],
                                                    "Y": ["embedding3_output"]
                                                },
                                                "op_outputs": {
                                                    "Out": [
                                                        "elementwise_add1_output"
                                                    ]
                                                },
                                                "op_attrs": dics[2]
                                            }, {
                                                "op_type": "elementwise_add",
                                                "op_inputs": {
                                                    "X": [
                                                        "elementwise_add1_output"
                                                    ],
                                                    "Y": ["embedding1_output"]
                                                },
                                                "op_outputs": {
                                                    "Out": [
                                                        "elementwise_add2_output"
                                                    ]
                                                },
                                                "op_attrs": dics[3]
                                            }, {
                                                "op_type": "layer_norm",
                                                "op_inputs": {
                                                    "X": [
                                                        "elementwise_add2_output"
                                                    ],
                                                    "Bias":
                                                    ["layer_norm_bias"],
                                                    "Scale":
                                                    ["layer_norm_scale"]
                                                },
                                                "op_outputs": {
                                                    "Y":
                                                    ["layer_norm_output1"],
                                                    "Mean":
                                                    ["layer_norm_output2"],
                                                    "Variance":
                                                    ["layer_norm_output3"]
                                                },
                                                "op_attrs": dics[4]
                                            }]
                                            ops = self.generate_op_config(
                                                ops_config)

                                            program_config = ProgramConfig(
                                                ops=ops,
                                                weights={
                                                    "embedding1_weight":
                                                    TensorConfig(
                                                        data_gen=partial(
                                                            generate_weight1,
                                                            size11, size2)),
                                                    "embedding2_weight":
                                                    TensorConfig(
                                                        data_gen=partial(
                                                            generate_weight2,
                                                            size12, size2)),
                                                    "embedding3_weight":
                                                    TensorConfig(
                                                        data_gen=partial(
                                                            generate_weight3,
                                                            size13, size2)),
                                                    "layer_norm_bias":
                                                    TensorConfig(
                                                        data_gen=partial(
                                                            generate_weight4,
                                                            size2)),
                                                    "layer_norm_scale":
                                                    TensorConfig(
                                                        data_gen=partial(
                                                            generate_weight4,
                                                            size2))
                                                },
                                                inputs={
                                                    "input_data1": TensorConfig(
                                                        data_gen=partial(
                                                            generate_input,
                                                            batch, input_size)),
                                                    "input_data2": TensorConfig(
                                                        data_gen=partial(
                                                            generate_input,
                                                            batch, input_size)),
                                                    "input_data3": TensorConfig(
                                                        data_gen=partial(
                                                            generate_input,
                                                            batch, input_size))
                                                },
                                                outputs=["layer_norm_output1"])

                                            yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data1": [1, 4, 1],
                "input_data2": [1, 4, 1],
                "input_data3": [1, 4, 1]
            }
            self.dynamic_shape.max_input_shape = {
                "input_data1": [4, 512, 1],
                "input_data2": [4, 512, 1],
                "input_data3": [4, 512, 1]
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data1": [2, 128, 1],
                "input_data2": [2, 128, 1],
                "input_data3": [2, 128, 1]
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
        yield self.create_inference_config(), (0, 5), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (0, 5), 2e-2

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 4), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 4), 2e-2

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
