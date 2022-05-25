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
import unittest
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set


class TrtConvertConv2dTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        if inputs['input_data'].shape[1] != weights['conv2d_weight'].shape[
                1] * attrs[0]['groups']:
            return False

        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[0] * 10 < 7000:
            if attrs[0]['padding_algorithm'] == 'SAME' and (
                    attrs[0]['strides'][0] > 1 or attrs[0]['strides'][1] > 1):
                return False

        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(batch, attrs: List[Dict[str, Any]]):
            return np.ones(
                [batch, attrs[0]['groups'] * 3, 64, 64]).astype(np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            return np.random.random([24, 3, 3, 3]).astype(np.float32)

        for batch in [1, 4]:
            for strides in [[1, 1], [2, 2], [1, 2]]:
                for paddings in [[0, 3], [1, 2, 3, 4]]:
                    for groups in [1, 3]:
                        for padding_algorithm in ['EXPLICIT', 'SAME', 'VALID']:
                            for dilations in [[1, 1], [2, 2], [1, 2]]:
                                for data_format in ['NCHW']:

                                    dics = [{
                                        "data_fromat": data_format,
                                        "dilations": dilations,
                                        "padding_algorithm": padding_algorithm,
                                        "groups": groups,
                                        "paddings": paddings,
                                        "strides": strides,
                                        "data_format": data_format
                                    }, {}]

                                    ops_config = [{
                                        "op_type": "conv2d",
                                        "op_inputs": {
                                            "Input": ["input_data"],
                                            "Filter": ["conv2d_weight"]
                                        },
                                        "op_outputs": {
                                            "Output": ["conv_output_data"]
                                        },
                                        "op_attrs": dics[0]
                                    }, {
                                        "op_type": "relu",
                                        "op_inputs": {
                                            "X": ["conv_output_data"]
                                        },
                                        "op_outputs": {
                                            "Out": ["output_data"]
                                        },
                                        "op_attrs": dics[1]
                                    }]

                                    ops = self.generate_op_config(ops_config)

                                    program_config = ProgramConfig(
                                        ops=ops,
                                        weights={
                                            "conv2d_weight":
                                            TensorConfig(data_gen=partial(
                                                generate_weight1, dics))
                                        },
                                        inputs={
                                            "input_data":
                                            TensorConfig(data_gen=partial(
                                                generate_input1, batch, dics))
                                        },
                                        outputs=["output_data"])

                                    yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            input_groups = attrs[0]['groups'] * 3
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, input_groups, 32, 32],
                "output_data": [1, 24, 32, 32]
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [4, input_groups, 64, 64],
                "output_data": [4, 24, 64, 64]
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [1, input_groups, 64, 64],
                "output_data": [1, 24, 64, 64]
            }

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
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), (1e-5, 1e-5)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), (1e-5, 1e-5)

    def test(self):
        self.run_test()

    def test_quant(self):
        self.run_test(quant=True)


if __name__ == "__main__":
    unittest.main()
