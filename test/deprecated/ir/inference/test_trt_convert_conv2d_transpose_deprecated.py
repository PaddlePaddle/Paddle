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

from __future__ import annotations

import unittest
from functools import partial
from typing import Any

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import SkipReasons, TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertConv2dTransposeTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        if (
            inputs['input_data'].shape[1]
            != weights['conv2d_weight'].shape[1] * attrs[0]['groups']
        ):
            return False

        if inputs['input_data'].shape[1] != weights['conv2d_weight'].shape[0]:
            return False

        if attrs[0]['dilations'][0] != 1 or attrs[0]['dilations'][1] != 1:
            return False

        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 7000:
            return False

        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(batch, num_channels, attrs: list[dict[str, Any]]):
            return np.ones([batch, num_channels, 64, 64]).astype(np.float32)

        def generate_weight1(num_channels, attrs: list[dict[str, Any]]):
            if attrs[0]['groups'] == 1:
                return np.random.random(
                    [num_channels, num_channels, 3, 3]
                ).astype(np.float32)
            else:
                return np.random.random(
                    [num_channels, int(num_channels / 2), 3, 3]
                ).astype(np.float32)

        for num_channels in [2, 4, 6]:
            for batch in [1, 4]:
                for strides in [[2, 2], [1, 2]]:
                    for paddings in [[0, 3], [1, 2, 3, 4]]:
                        for groups in [2]:
                            for padding_algorithm in [
                                'EXPLICIT',
                                'SAME',
                                'VALID',
                            ]:
                                for dilations in [[2, 2], [1, 2]]:
                                    for data_format in ['NCHW']:
                                        self.num_channels = num_channels
                                        dics = [
                                            {
                                                "dilations": dilations,
                                                "padding_algorithm": padding_algorithm,
                                                "groups": groups,
                                                "paddings": paddings,
                                                "strides": strides,
                                                "data_format": data_format,
                                                "output_size": [],
                                                "output_padding": [],
                                            }
                                        ]

                                        ops_config = [
                                            {
                                                "op_type": "conv2d_transpose",
                                                "op_inputs": {
                                                    "Input": ["input_data"],
                                                    "Filter": ["conv2d_weight"],
                                                },
                                                "op_outputs": {
                                                    "Output": ["output_data"]
                                                },
                                                "op_attrs": dics[0],
                                            }
                                        ]
                                        ops = self.generate_op_config(
                                            ops_config
                                        )

                                        program_config = ProgramConfig(
                                            ops=ops,
                                            weights={
                                                "conv2d_weight": TensorConfig(
                                                    data_gen=partial(
                                                        generate_weight1,
                                                        num_channels,
                                                        dics,
                                                    )
                                                )
                                            },
                                            inputs={
                                                "input_data": TensorConfig(
                                                    data_gen=partial(
                                                        generate_input1,
                                                        batch,
                                                        num_channels,
                                                        dics,
                                                    )
                                                )
                                            },
                                            outputs=["output_data"],
                                        )

                                        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            if self.num_channels == 2:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 2, 32, 32],
                    "output_data": [1, 24, 32, 32],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 2, 64, 64],
                    "output_data": [4, 24, 64, 64],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 2, 64, 64],
                    "output_data": [1, 24, 64, 64],
                }
            elif self.num_channels == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 4, 32, 32],
                    "output_data": [1, 24, 32, 32],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 4, 64, 64],
                    "output_data": [4, 24, 64, 64],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 4, 64, 64],
                    "output_data": [1, 24, 64, 64],
                }
            else:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 6, 32, 32],
                    "output_data": [1, 24, 32, 32],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 6, 64, 64],
                    "output_data": [4, 24, 64, 64],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 6, 64, 64],
                    "output_data": [1, 24, 64, 64],
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

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), (1e-3, 1e-3)
        # self.trt_param.precision = paddle_infer.PrecisionType.Int8
        # yield self.create_inference_config(), generate_trt_nodes_num(
        #     attrs, False), (1e-5, 1e-5)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-3, 1e-3)
        # self.trt_param.precision = paddle_infer.PrecisionType.Int8
        # yield self.create_inference_config(), generate_trt_nodes_num(
        #     attrs, True), (1e-5, 1e-5)

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if self.trt_param.precision == paddle_infer.PrecisionType.Int8:
                return True
            return False

        self.add_skip_case(
            teller1,
            SkipReasons.TRT_NOT_IMPLEMENTED,
            "When precisionType is int8 without relu op, output is different between Trt and Paddle.",
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()

    def test_quant(self):
        self.add_skip_trt_case()
        self.run_test(quant=True)


# Special case
class TrtConvertConv2dTransposeTest2(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 7000:
            return False
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(batch, num_channels, attrs: list[dict[str, Any]]):
            return (
                np.ones([batch, num_channels, 20, 30]).astype(np.float32) / 100
            )

        def generate_weight1(num_channels, attrs: list[dict[str, Any]]):
            return (
                np.random.random([num_channels, 64, 3, 3]).astype(np.float32)
                / 100
            )

        num_channels = 128
        batch = 1

        self.num_channels = num_channels
        dics = [
            {
                "data_format": 'NCHW',
                "dilations": [1, 1],
                "padding_algorithm": 'EXPLICIT',
                "groups": 1,
                "paddings": [1, 1],
                "strides": [2, 2],
                "output_padding": [1, 1],
                "output_size": [],
            }
        ]

        ops_config = [
            {
                "op_type": "conv2d_transpose",
                "op_inputs": {
                    "Input": ["input_data"],
                    "Filter": ["conv2d_weight"],
                },
                "op_outputs": {"Output": ["output_data"]},
                "op_attrs": dics[0],
            }
        ]
        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "conv2d_weight": TensorConfig(
                    data_gen=partial(generate_weight1, num_channels, dics)
                )
            },
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input1, batch, num_channels, dics)
                )
            },
            outputs=["output_data"],
        )

        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, 128, 20, 30],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [1, 128, 20, 30],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [1, 128, 20, 30],
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

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-4
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), (1e0, 1e-3)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-4
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e0, 1e-3)

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()

    def test_quant(self):
        self.add_skip_trt_case()
        self.run_test(quant=True)


if __name__ == "__main__":
    unittest.main()
