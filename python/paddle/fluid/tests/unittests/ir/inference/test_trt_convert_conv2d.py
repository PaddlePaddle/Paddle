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


class TrtConvertConv2dTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        # TODO: This is just the example to remove the wrong attrs.
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        # groups restriction.
        if inputs['input_data'].shape[1] != weights['conv2d_weight'].shape[
                1] * attrs[0]['groups']:
            return False

        # others restriction, todo.

        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]]):
            # TODO: This is just the example to illustrate the releation between axis and input.
            # for each attr, can generate different datas
            if attrs[0]['groups'] == 1:
                return np.ones([2, 3, 64, 64]).astype(np.float32)
            else:
                return np.ones([1, 3, 64, 64]).astype(np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            return np.random.random([24, 3, 3, 3]).astype(np.float32)

        # for strides in [[1, 1], [2, 2], [1, 2], [2, 3]]:
        #     for paddings in [[0, 3], [3, 1], [1, 1, 1, 1]]:
        #         for groups in [1, 2]:
        #             for padding_algotithm in ['EXPLICIT', 'SAME', 'VALID']:
        #                 for dilations in [[1, 1], [1, 2]]:
        #                     for data_format in ['NCHW']:
        for strides in [[1, 1], [2, 2]]:
            for paddings in [[0, 3], [3, 1]]:
                for groups in [1]:
                    for padding_algotithm in ['EXPLICIT']:
                        for dilations in [[1, 1]]:
                            for data_format in ['NCHW']:

                                dics = [{
                                    "data_fromat": data_format,
                                    "dilations": dilations,
                                    "padding_algorithm": padding_algotithm,
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
                                        "Out": ["relu_output_data"]
                                    },
                                    "op_attrs": dics[1]
                                }]
                                ops = self.generate_op_config(ops_config)

                                program_config = ProgramConfig(
                                    ops=ops,
                                    weights={
                                        "conv2d_weight": TensorConfig(
                                            data_gen=partial(generate_weight1,
                                                             dics))
                                    },
                                    inputs={
                                        "input_data": TensorConfig(
                                            data_gen=partial(generate_input1,
                                                             dics))
                                    },
                                    outputs=["relu_output_data"])

                                yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if len(attrs[0]['paddings']) == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 3, 32, 32],
                    '': []
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 3, 64, 64],
                    '': []
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 3, 64, 64],
                    '': []
                }
            else:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 3, 32, 32]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 3, 64, 64]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 3, 64, 64]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            # TODO: This is just the example, need to be fixed.
            if len(attrs[0]['paddings']) == 4:
                return 1, 2
            else:
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

    def add_skip_trt_case(self):
        # TODO(wilber): This is just the example to illustrate the skip usage.
        def teller1(program_config, predictor_config):
            if len(program_config.ops[0].attrs['paddings']) == 4:
                return True
            return False

        self.add_skip_case(
            teller1, SkipReasons.TRT_NOT_IMPLEMENTED,
            "NOT Implemented: we need to add support in the future ....TODO, just for the example"
        )

        def teller2(program_config, predictor_config):
            if (
                    program_config.ops[0].attrs['dilations'][0] == 1 and
                    program_config.ops[0].attrs['dilations'][0] == 2
            ) or program_config.ops[0].attrs['padding_algorithm'] != 'EXPLICIT':
                return True
            return False

        self.add_skip_case(teller2, SkipReasons.TRT_NOT_SUPPORT,
                           "TODO, just for the example")
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()

    def test_quant(self):
        self.add_skip_trt_case()
        self.run_test(quant=True)


if __name__ == "__main__":
    unittest.main()
