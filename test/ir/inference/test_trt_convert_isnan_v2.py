# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer

if TYPE_CHECKING:
    from collections.abc import Generator


class TrtConvertIsnanV2Test(TrtLayerAutoScanTest):
    def sample_program_configs(self):
        def generate_input1(dims):
            if dims == 1:
                data = np.random.random([3]).astype(np.float32)
                mask = np.random.random([3]).astype(np.float32) < 0.3
                data[mask] = np.nan
                return data
            elif dims == 2:
                data = np.random.random([3, 64]).astype(np.float32)
                mask = np.random.random([3, 64]).astype(np.float32) < 0.3
                data[mask] = np.nan
                return data
            elif dims == 3:
                data = np.random.random([3, 64, 64]).astype(np.float32)
                mask = np.random.random([3, 64, 64]).astype(np.float32) < 0.3
                data[mask] = np.nan
                return data
            else:
                data = np.random.random([1, 3, 64, 64]).astype(np.float32)
                mask = np.random.random([1, 3, 64, 64]).astype(np.float32) < 0.3
                data[mask] = np.nan
                return data

        for dims in [1, 2, 3, 4]:
            self.dims = dims
            ops_config = [
                {
                    "op_type": "isnan_v2",
                    "op_inputs": {
                        "X": ["input_data"],
                    },
                    "op_outputs": {
                        "Out": ["isnan_v2_output_data"],
                    },
                    "op_attrs": {},
                },
                {
                    "op_type": "cast",
                    "op_inputs": {"X": ["isnan_v2_output_data"]},
                    "op_outputs": {"Out": ["output_data"]},
                    "op_attrs": {
                        "in_dtype": 0,
                        "out_dtype": 5,
                    },
                },
            ]
            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    "input_data": TensorConfig(
                        data_gen=partial(generate_input1, dims)
                    )
                },
                outputs=["output_data"],
            )

            yield program_config

    def sample_predictor_configs(self, program_config) -> Generator[
        tuple[
            paddle_infer.Config, tuple[int, int], tuple[float, float] | float
        ],
        None,
        None,
    ]:
        def generate_dynamic_shape(attrs):
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {"input_data": [1]}
                self.dynamic_shape.max_input_shape = {"input_data": [128]}
                self.dynamic_shape.opt_input_shape = {"input_data": [64]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 32]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 64]}
                self.dynamic_shape.opt_input_shape = {"input_data": [3, 64]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 32, 32]}
                self.dynamic_shape.max_input_shape = {
                    "input_data": [10, 64, 64]
                }
                self.dynamic_shape.opt_input_shape = {"input_data": [3, 64, 64]}
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

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if not dynamic_shape:
                return 0, 4
            return 1, 2

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

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

        # for dynamic_shape mode
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

    def test(self):
        if os.name != 'nt':
            self.run_test()


if __name__ == "__main__":
    unittest.main()
