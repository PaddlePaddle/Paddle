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


class TrtConvertUnbind(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        # self.trt_param.workspace_size = 1073741824

        def generate_input1():
            self.input_shape = [3, 400, 196, 80]
            return np.random.random([3, 400, 196, 80]).astype(np.float32)

        for dims in [4]:
            for axis in [0]:
                # for type in ["int32", "int64", "float32", "float64"]:
                self.dims = dims
                ops_config = [
                    {
                        "op_type": "unbind",
                        "op_inputs": {
                            "X": ["input_data"],
                        },
                        "op_outputs": {
                            "Out": [
                                "output_data0",
                                "output_data1",
                                "output_data2",
                            ]
                        },
                        "op_attrs": {"axis": axis},
                    }
                ]
                ops = self.generate_op_config(ops_config)

                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "input_data": TensorConfig(
                            data_gen=partial(generate_input1)
                        ),
                    },
                    outputs=["output_data0", "output_data1", "output_data2"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 4

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [3, 100, 196, 80]
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [3, 400, 196, 80]
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [3, 400, 196, 80]
            }

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        # clear_dynamic_shape()

        # self.trt_param.precision = paddle_infer.PrecisionType.Float32
        # yield self.create_inference_config(), (0, 6), 1e-5
        # self.trt_param.precision = paddle_infer.PrecisionType.Half
        # yield self.create_inference_config(), (0, 6), 1e-3

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
        ), 1e-3

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
