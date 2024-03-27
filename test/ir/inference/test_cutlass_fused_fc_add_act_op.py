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


# # fba pattern
# class TestCutlassFusedFcAddActOp1(CutlassAutoScanTest):
#     def sample_program_configs(self, *args, **kwargs):
#         def generate_input1(input_shape):
#             return (np.random.random(input_shape) - 0.5).astype(np.float32)

#         def generate_weight(weight_shape):
#             return np.random.random(weight_shape).astype(np.float32)

#         def generate_bias(bias_shape):
#             return np.random.random(bias_shape).astype(np.float32)

#         input_shape_options = [[3, 128], [8, 128]]
#         weight_shape_options = [[128, 104], [128, 64]]
#         data_format_options = ['RRR']
#         act_options = ['relu', 'swish', 'identity', 'sigmoid']      #, 'leaky_relu'

#         configurations = [
#             input_shape_options,
#             weight_shape_options,
#             data_format_options,
#             act_options,
#         ]

#         for (
#             input_shape,
#             weight_shape,
#             data_format,
#             act,
#         ) in product(*configurations):
#             attrs = [
#                 {
#                     "data_format": data_format,
#                 },
#                 {"axis": 1},
#             ]

#             ops_config = [
#                 {
#                     "op_type": "matmul",
#                     "op_inputs": {
#                         "Input": ["input_data"],
#                         "Weight": ["matmul_weight"],
#                     },
#                     "op_outputs": {"Output": ["matmul_output_data"]},
#                     "op_attrs": attrs[0],
#                 },
#                 {
#                     "op_type": "elementwise_add",
#                     "op_inputs": {
#                         "X": ["matmul_output_data"],
#                         "Y": ["elementwise_weight"],
#                     },
#                     "op_outputs": {"Out": ["output_data0"]},
#                     "op_attrs": attrs[1],
#                 },
#                 {
#                     "op_type": act,
#                     "op_inputs": {"X": ["output_data0"]},
#                     "op_outputs": {"Out": ["output_data1"]},
#                     "op_attrs": {
#                         "alpha": 1.0,
#                     },
#                 },
#             ]

#             ops = self.generate_op_config(ops_config)

#             program_config = ProgramConfig(
#                 ops=ops,
#                 weights={
#                     "matmul_weight": TensorConfig(
#                         data_gen=partial(generate_weight, weight_shape)
#                     ),
#                     "elementwise_weight": TensorConfig(
#                         data_gen=partial(generate_bias, [weight_shape[1]])
#                     ),
#                 },
#                 inputs={
#                     "input_data": TensorConfig(
#                         data_gen=partial(generate_input1, input_shape)
#                     )
#                 },
#                 outputs=["output_data1"],
#             )

#             yield program_config

#     def sample_predictor_configs(self, program_config):
#         config = self.create_inference_config(use_gpu=True)
#         config.enable_use_gpu(256, 0, paddle_infer.PrecisionType.Half)
#         config.exp_enable_use_cutlass()
#         yield config, (1e-2, 1e-2)

#     def test(self, *args, **kwargs):
#         self.run_test(quant=False, *args, **kwargs)


# fba pattern  cbaa是啥意思？
class TestCutlassFusedFcAddActOp2(CutlassAutoScanTest):
    def sample_program_configs(self, *args, **kwargs):
        def generate_input(input_shape):
            return (np.random.random(input_shape)).astype(np.float16)

        def generate_weight(weight_shape):
            return (np.random.random(weight_shape)).astype(np.float16)

        def generate_bias(bias_shape):
            return np.random.random(bias_shape).astype(np.float16)

        input_shape_options = [[1024, 104]]
        weight_shape_options = [[104, 256]]

        configurations = [
            input_shape_options,
            weight_shape_options,
        ]

        for (
            input_shape,
            weight_shape,
        ) in product(*configurations):
            attrs = [
                {
                    "x_num_col_dims": 1,
                    "y_num_col_dims": 1,
                },
                {
                    "axis": 1
                },
                {
                    "": ""
                }
            ]
            
            ops_config = [
                {
                    "op_type": "mul",
                    "op_inputs": {
                        "X": ["matmul_X"],
                        "Y": ["matmul_Y"],
                    },
                    "op_outputs": {"Out": ["matmul_output"]},
                    "op_attrs": attrs[0],
                },
                {
                    "op_type": "elementwise_add",
                    "op_inputs": {
                        "X": ["matmul_output"],
                        "Y": ["bias"],
                    },
                    "op_outputs": {"Out": ["elementwise_add_output"]},
                    "op_attrs": attrs[1],
                },
                {
                    "op_type": "relu",
                    "op_inputs": {"X": ["elementwise_add_output"]},
                    "op_outputs": {"Out": ["activation_output"]},
                    "op_attrs": attrs[2],
                },
            ]

            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={
                    "matmul_Y": TensorConfig(
                        data_gen=partial(generate_weight, weight_shape)
                    ),
                    "bias": TensorConfig(
                        data_gen=partial(generate_bias, [weight_shape[1]])
                    ),
                },
                inputs={
                    "matmul_X": TensorConfig(
                        data_gen=partial(generate_input, input_shape)
                    )
                },
                outputs=["activation_output"],
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
