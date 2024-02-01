# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from typing import List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer
from paddle.inference import InternalUtils


class TestTrtFp32MixPrecision(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_conv2d_input():
            return np.ones([1, 3, 64, 64]).astype(np.float32)

        def generate_conv2d_weight():
            return np.ones([9, 3, 3, 3]).astype(np.float32)

        def generate_elementwise_input(op_type):
            # elementwise_floordiv is integer only
            if op_type == "elementwise_mod":
                return np.random.uniform(
                    low=0.1, high=1.0, size=[33, 10]
                ).astype(np.float32)
            else:
                return np.random.random([33, 10]).astype(np.float32)

        def generate_elementwise_weight(op_type):
            if op_type == "elementwise_mod":
                return np.random.uniform(
                    low=0.1, high=1.0, size=[33, 1]
                ).astype(np.float32)
            else:
                return np.random.randn(33, 1).astype(np.float32)

        attrs = [
            {
                "data_fromat": 'NCHW',
                "dilations": [1, 2],
                "padding_algorithm": 'EXPLICIT',
                "groups": 1,
                "paddings": [0, 3],
                "strides": [2, 2],
            },
            {"axis": -1},
            {
                "trans_x": False,
                "trans_y": False,
            },
        ]
        for op_type in [
            "elementwise_add",
            "elementwise_mul",
            "elementwise_sub",
            "elementwise_div",
            "elementwise_pow",
            "elementwise_min",
            "elementwise_max",
            "elementwise_mod",
        ]:
            ops_config = [
                {
                    "op_type": "conv2d",
                    "op_inputs": {
                        "Input": ["conv2d_input"],
                        "Filter": ["conv2d_weight"],
                    },
                    "op_outputs": {"Output": ["conv_output_data"]},
                    "op_attrs": attrs[0],
                },
                {
                    "op_type": op_type,
                    "op_inputs": {
                        "X": ["elementwise_input"],
                        "Y": ["elementwise_weight"],
                    },
                    "op_outputs": {"Out": ["elementwise_output_data"]},
                    "op_attrs": attrs[1],
                    "outputs_dtype": {"output_data": np.float32},
                },
                {
                    "op_type": "matmul_v2",
                    "op_inputs": {
                        "X": ["conv_output_data"],
                        "Y": ["elementwise_output_data"],
                    },
                    "op_outputs": {"Out": ["matmul_v2_output_data"]},
                    "op_attrs": attrs[2],
                },
            ]

            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={
                    "conv2d_weight": TensorConfig(
                        data_gen=partial(generate_conv2d_weight)
                    ),
                    "elementwise_weight": TensorConfig(
                        data_gen=partial(generate_elementwise_weight, op_type)
                    ),
                },
                inputs={
                    "conv2d_input": TensorConfig(
                        data_gen=partial(generate_conv2d_input)
                    ),
                    "elementwise_input": TensorConfig(
                        data_gen=partial(generate_elementwise_input, op_type)
                    ),
                },
                outputs=["matmul_v2_output_data"],
            )

            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "conv2d_input": [1, 3, 64, 64],
                "elementwise_input": [33, 10],
            }
            self.dynamic_shape.max_input_shape = {
                "conv2d_input": [1, 3, 64, 64],
                "elementwise_input": [33, 10],
            }
            self.dynamic_shape.opt_input_shape = {
                "conv2d_input": [1, 3, 64, 64],
                "elementwise_input": [33, 10],
            }

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        config = self.create_inference_config()
        InternalUtils.disable_tensorrt_half_ops(
            config,
            {
                "conv_output_data",
                "elementwise_output_data",
                "matmul_v2_output_data",
            },
        )
        yield config, generate_trt_nodes_num(attrs, True), (1e-3, 1e-3)

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
