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


class TrtConvertInstanceNormTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        if attrs[0]['epsilon'] < 0 or attrs[0]['epsilon'] > 0.001:
            return False

        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]], shape_input):
            return np.random.random(shape_input).astype(np.float32)

        def generate_input2(attrs: List[Dict[str, Any]], shape_input):
            return np.random.random(shape_input[1]).astype(np.float32)

        for batch in [1, 2, 4]:
            for shape_input in [[batch, 16], [batch, 32, 64],
                                [batch, 16, 32, 64]]:
                self.in_dim = len(shape_input)
                for epsilon in [0.0005, -1, 1]:
                    dics = [{"epsilon": epsilon}]
                    ops_config = [{
                        "op_type": "instance_norm",
                        "op_inputs": {
                            "X": ["input_data"],
                            "Scale": ["scale_data"],
                            "Bias": ["bias_data"]
                        },
                        "op_outputs": {
                            "Y": ["y_data"],
                            "SavedMean": ["saved_mean_data"],
                            "SavedVariance": ["saved_variance_data"]
                        },
                        "op_attrs": dics[0]
                    }]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(
                        ops=ops,
                        weights={
                            "bias_data": TensorConfig(data_gen=partial(
                                generate_input2, dics, shape_input)),
                            "scale_data": TensorConfig(data_gen=partial(
                                generate_input2, dics, shape_input))
                        },
                        inputs={
                            "input_data": TensorConfig(data_gen=partial(
                                generate_input1, dics, shape_input))
                        },
                        outputs=["y_data"])

                    yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.in_dim == 2:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 4]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 64]}
                self.dynamic_shape.opt_input_shape = {"input_data": [2, 16]}
            elif self.in_dim == 3:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 1, 4]}
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 32, 256]
                }
                self.dynamic_shape.opt_input_shape = {"input_data": [2, 3, 32]}
            elif self.in_dim == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 1, 4, 4]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 32, 128, 256]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [2, 3, 32, 32]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape or self.in_dim != 4:
                return 0, 3
            return 1, 2

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
