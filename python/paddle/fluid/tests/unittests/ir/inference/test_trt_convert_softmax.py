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


class TrtConvertSoftmaxTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        outputs = program_config.outputs

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        #The input dimension should be less than or equal to the set axis.
        if len(inputs['softmax_input'].shape) <= attrs[0]['axis']:
            return False

        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]], batch):
            if self.dims == 4:
                return np.ones([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 32]).astype(np.float32)

        for dims in [2, 3, 4]:
            for batch in [1, 2, 4]:
                for axis in [-1, 0, 1, 2, 3]:
                    self.dims = dims
                    dics = [{"axis": axis}, {}]
                    ops_config = [{
                        "op_type": "softmax",
                        "op_inputs": {
                            "X": ["softmax_input"]
                        },
                        "op_outputs": {
                            "Out": ["softmax_out"]
                        },
                        "op_attrs": dics[0]
                    }]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "softmax_input": TensorConfig(data_gen=partial(
                                generate_input1, dics, batch))
                        },
                        outputs=["softmax_out"])

                    yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "softmax_input": [1, 3, 24, 24]
                }
                self.dynamic_shape.max_input_shape = {
                    "softmax_input": [4, 3, 48, 48]
                }
                self.dynamic_shape.opt_input_shape = {
                    "softmax_input": [1, 3, 24, 48]
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "softmax_input": [1, 3, 24]
                }
                self.dynamic_shape.max_input_shape = {
                    "softmax_input": [4, 3, 48]
                }
                self.dynamic_shape.opt_input_shape = {
                    "softmax_input": [1, 3, 48]
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {"softmax_input": [1, 32]}
                self.dynamic_shape.max_input_shape = {"softmax_input": [4, 64]}
                self.dynamic_shape.opt_input_shape = {"softmax_input": [1, 32]}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 2

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]
        # for static_shape
        clear_dynamic_shape()
        if attrs[0]['axis'] == 0:
            pass
        else:
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
