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

from __future__ import annotations

import unittest
from functools import partial

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtFloat64Test(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(shape, op_type):
            return np.random.randint(low=1, high=10000, size=shape).astype(
                np.float64
            )

        for op_type in [
            "elementwise_add",
            "elementwise_mul",
            "elementwise_sub",
        ]:
            for axis in [0, -1]:
                dics = [{"axis": axis}]
                ops_config = [
                    {
                        "op_type": op_type,
                        "op_inputs": {
                            "X": ["input_data1"],
                            "Y": ["input_data2"],
                        },
                        "op_outputs": {"Out": ["output_data"]},
                        "op_attrs": dics[0],
                        "outputs_dtype": {"slice_output_data": np.float64},
                    }
                ]
                ops = self.generate_op_config(ops_config)

                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "input_data1": TensorConfig(
                            data_gen=partial(
                                generate_input, [1, 8, 16, 32], op_type
                            )
                        ),
                        "input_data2": TensorConfig(
                            data_gen=partial(
                                generate_input, [1, 8, 16, 32], op_type
                            )
                        ),
                    },
                    outputs=["output_data"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data1": [1, 4, 4, 4],
                "input_data2": [1, 4, 4, 4],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data1": [8, 128, 64, 128],
                "input_data2": [8, 128, 64, 128],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data1": [2, 64, 32, 32],
                "input_data2": [2, 64, 32, 32],
            }

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 3), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), (1e-3, 1e-3)

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
