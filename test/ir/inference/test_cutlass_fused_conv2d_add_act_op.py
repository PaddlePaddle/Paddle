# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from functools import partial
from itertools import product

import numpy as np
from auto_scan_test import CutlassAutoScanTest
from program_config import ProgramConfig, TensorConfig

import paddle.inference as paddle_infer


# cba pattern
class TestCutlassFusedConv2dAddActOp1(CutlassAutoScanTest):
    def sample_program_configs(self, *args, **kwargs):
        def generate_input1(input_shape):
            return (np.random.random(input_shape) - 0.5).astype(np.float32)

        def generate_weight(weight_shape):
            return np.random.random(weight_shape).astype(np.float32)

        def generate_bias(bias_shape):
            return np.random.random(bias_shape).astype(np.float32)

        input_shape_options = [[1, 16, 112, 112], [1, 8, 64, 64]]
        weight_shape_options = [[24, -1, 3, 3]]
        strides_options = [[1, 1], [2, 2]]
        paddings_options = [[1, 1], [1, 0, 1, 2]]
        groups_options = [1]
        padding_algorithm_options = ['EXPLICIT']
        dilations_options = [[2, 2], [1, 1]]
        data_format_options = ['NCHW']
        act_options = ['relu', 'leaky_relu', 'swish']

        configurations = [
            input_shape_options,
            weight_shape_options,
            strides_options,
            paddings_options,
            groups_options,
            padding_algorithm_options,
            dilations_options,
            data_format_options,
            act_options,
        ]

        for (
            input_shape,
            weight_shape,
            strides,
            paddings,
            groups,
            padding_algorithm,
            dilations,
            data_format,
            act,
        ) in product(*configurations):
            weight_shape[1] = input_shape[1]
            attrs = [
                {
                    "strides": strides,
                    "paddings": paddings,
                    "groups": groups,
                    "padding_algorithm": padding_algorithm,
                    "dilations": dilations,
                    "data_format": data_format,
                },
                {"axis": 1},
            ]

            ops_config = [
                {
                    "op_type": "conv2d",
                    "op_inputs": {
                        "Input": ["input_data"],
                        "Filter": ["conv2d_weight"],
                    },
                    "op_outputs": {"Output": ["conv_output_data"]},
                    "op_attrs": attrs[0],
                },
                {
                    "op_type": "elementwise_add",
                    "op_inputs": {
                        "X": ["conv_output_data"],
                        "Y": ["elementwise_weight"],
                    },
                    "op_outputs": {"Out": ["output_data0"]},
                    "op_attrs": attrs[1],
                },
                {
                    "op_type": act,
                    "op_inputs": {"X": ["output_data0"]},
                    "op_outputs": {"Out": ["output_data1"]},
                    "op_attrs": {
                        "alpha": 2.0,
                    },
                },
            ]

            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={
                    "conv2d_weight": TensorConfig(
                        data_gen=partial(generate_weight, weight_shape)
                    ),
                    "elementwise_weight": TensorConfig(
                        data_gen=partial(generate_bias, [weight_shape[0]])
                    ),
                },
                inputs={
                    "input_data": TensorConfig(
                        data_gen=partial(generate_input1, input_shape)
                    )
                },
                outputs=["output_data1"],
            )

            yield program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_gpu=True)
        config.enable_use_gpu(256, 0, paddle_infer.PrecisionType.Half)
        config.exp_enable_use_cutlass()
        yield config, (1e-2, 1e-2)

    def test(self, *args, **kwargs):
        self.run_test(quant=False, *args, **kwargs)


# cbaa pattern
class TestCutlassFusedConv2dAddActOp2(CutlassAutoScanTest):
    def sample_program_configs(self, *args, **kwargs):
        def generate_input(input_shape):
            return (np.random.random(input_shape) * 2 - 1).astype(np.float32)

        def generate_weight(weight_shape):
            return (np.random.random(weight_shape) * 2 - 1).astype(np.float32)

        def generate_bias(bias_shape):
            return np.random.random(bias_shape).astype(np.float32)

        input_shape_options = [[1, 16, 112, 112], [1, 24, 64, 64]]
        weight_shape_options = [[24, -1, 3, 3]]
        strides_options = [[2, 2], [1, 1]]
        paddings_options = [[1, 1]]
        groups_options = [1]
        padding_algorithm_options = ['EXPLICIT']
        dilations_options = [[1, 1]]
        data_format_options = ['NCHW']
        act_options = ['relu']

        configurations = [
            input_shape_options,
            weight_shape_options,
            strides_options,
            paddings_options,
            groups_options,
            padding_algorithm_options,
            dilations_options,
            data_format_options,
            act_options,
        ]

        for (
            input_shape,
            weight_shape,
            strides,
            paddings,
            groups,
            padding_algorithm,
            dilations,
            data_format,
            act,
        ) in product(*configurations):
            weight_shape[1] = input_shape[1]
            residual_shape = list(input_shape)
            residual_shape[1] = weight_shape[0]

            ih = input_shape[2]
            iw = input_shape[3]
            pad_h0 = 0
            pad_h1 = 0
            pad_w0 = 0
            pad_w1 = 0
            if len(paddings) == 2:
                pad_h0 = paddings[0]
                pad_h1 = paddings[0]
                pad_w0 = paddings[1]
                pad_w1 = paddings[1]
            elif len(paddings) == 4:
                pad_h0 = paddings[0]
                pad_h1 = paddings[1]
                pad_w0 = paddings[2]
                pad_w1 = paddings[3]
            dilation_h = dilations[0]
            dilation_w = dilations[1]
            kh = weight_shape[2]
            kw = weight_shape[3]
            stride_h = strides[0]
            stride_w = strides[1]
            residual_shape[2] = (int)(
                (ih + pad_h0 + pad_h1 - dilation_h * (kh - 1) - 1) / stride_h
            ) + 1
            residual_shape[3] = (int)(
                (iw + pad_w0 + pad_w1 - dilation_w * (kw - 1) - 1) / stride_w
            ) + 1

            attrs = [
                {
                    "strides": strides,
                    "paddings": paddings,
                    "groups": groups,
                    "padding_algorithm": padding_algorithm,
                    "dilations": dilations,
                    "data_format": data_format,
                },
                {"axis": 1},
            ]

            ops_config = [
                {
                    "op_type": "conv2d",
                    "op_inputs": {
                        "Input": ["input_data"],
                        "Filter": ["conv2d_weight"],
                    },
                    "op_outputs": {"Output": ["conv_output_data"]},
                    "op_attrs": attrs[0],
                },
                {
                    "op_type": "elementwise_add",
                    "op_inputs": {
                        "X": ["conv_output_data"],
                        "Y": ["elementwise_weight"],
                    },
                    "op_outputs": {"Out": ["output_data0"]},
                    "op_attrs": attrs[1],
                },
                {
                    "op_type": "elementwise_add",
                    "op_inputs": {
                        "X": ["residual_data"],
                        "Y": ["output_data0"],
                    },
                    "op_outputs": {"Out": ["output_data1"]},
                    "op_attrs": {},
                },
                {
                    "op_type": act,
                    "op_inputs": {"X": ["output_data1"]},
                    "op_outputs": {"Out": ["output_data2"]},
                    "op_attrs": {},
                },
            ]

            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={
                    "conv2d_weight": TensorConfig(
                        data_gen=partial(generate_weight, weight_shape)
                    ),
                    "elementwise_weight": TensorConfig(
                        data_gen=partial(generate_bias, [weight_shape[0]])
                    ),
                },
                inputs={
                    "input_data": TensorConfig(
                        data_gen=partial(generate_input, input_shape)
                    ),
                    "residual_data": TensorConfig(
                        data_gen=partial(generate_input, residual_shape)
                    ),
                },
                outputs=["output_data2"],
            )

            yield program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_gpu=True)
        config.enable_use_gpu(256, 0, paddle_infer.PrecisionType.Half)
        config.exp_enable_use_cutlass()
        yield config, (1e-2, 1e-2)

    def test(self, *args, **kwargs):
        self.run_test(quant=False, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()
