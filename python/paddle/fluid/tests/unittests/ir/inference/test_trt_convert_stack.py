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


class TrtConvertStackTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        outputs = program_config.outputs

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]
        #The input dimension should be less than the set axis.
        if len(inputs['stack_input1'].shape) < attrs[0]['axis']:
            return False

        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]], batch):
            if self.dims == 4:
                return np.ones([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.ones([24]).astype(np.float32)

        def generate_input2(attrs: List[Dict[str, Any]], batch):
            if self.dims == 4:
                return np.ones([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.ones([24]).astype(np.float32)

        def generate_input3(attrs: List[Dict[str, Any]], batch):
            if self.dims == 4:
                return np.ones([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.ones([24]).astype(np.float32)

        for dims in [1, 2, 3, 4]:
            for batch in [1, 4]:
                for axis in [-2, -1, 0, 1, 2, 3]:
                    self.dims = dims
                    dics = [{"axis": axis}, {}]
                    ops_config = [{
                        "op_type": "stack",
                        "op_inputs": {
                            "X":
                            ["stack_input1", "stack_input2", "stack_input3"]
                        },
                        "op_outputs": {
                            "Y": ["stack_output"]
                        },
                        "op_attrs": dics[0]
                    }]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "stack_input1": TensorConfig(data_gen=partial(
                                generate_input1, dics, batch)),
                            "stack_input2": TensorConfig(data_gen=partial(
                                generate_input2, dics, batch)),
                            "stack_input3": TensorConfig(data_gen=partial(
                                generate_input3, dics, batch))
                        },
                        outputs=["stack_output"])

                    yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "stack_input1": [1, 3, 24, 24],
                    "stack_input2": [1, 3, 24, 24],
                    "stack_input3": [1, 3, 24, 24]
                }
                self.dynamic_shape.max_input_shape = {
                    "stack_input1": [4, 3, 48, 48],
                    "stack_input2": [4, 3, 48, 48],
                    "stack_input3": [4, 3, 48, 48]
                }
                self.dynamic_shape.opt_input_shape = {
                    "stack_input1": [1, 3, 24, 24],
                    "stack_input2": [1, 3, 24, 24],
                    "stack_input3": [1, 3, 24, 24]
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "stack_input1": [1, 3, 24],
                    "stack_input2": [1, 3, 24],
                    "stack_input3": [1, 3, 24]
                }
                self.dynamic_shape.max_input_shape = {
                    "stack_input1": [4, 3, 48],
                    "stack_input2": [4, 3, 48],
                    "stack_input3": [4, 3, 48]
                }
                self.dynamic_shape.opt_input_shape = {
                    "stack_input1": [1, 3, 24],
                    "stack_input2": [1, 3, 24],
                    "stack_input3": [1, 3, 24]
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "stack_input1": [1, 24],
                    "stack_input2": [1, 24],
                    "stack_input3": [1, 24]
                }
                self.dynamic_shape.max_input_shape = {
                    "stack_input1": [4, 48],
                    "stack_input2": [4, 48],
                    "stack_input3": [4, 48]
                }
                self.dynamic_shape.opt_input_shape = {
                    "stack_input1": [1, 24],
                    "stack_input2": [1, 24],
                    "stack_input3": [1, 24]
                }
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {
                    "stack_input1": [24],
                    "stack_input2": [24],
                    "stack_input3": [24]
                }
                self.dynamic_shape.max_input_shape = {
                    "stack_input1": [48],
                    "stack_input2": [48],
                    "stack_input3": [48]
                }
                self.dynamic_shape.opt_input_shape = {
                    "stack_input1": [24],
                    "stack_input2": [24],
                    "stack_input3": [24]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape == True:
                return 1, 4
            else:
                return 0, 5

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

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
