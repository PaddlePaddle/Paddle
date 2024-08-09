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

from __future__ import annotations

import unittest
from functools import partial

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer
from paddle.framework import convert_np_dtype_to_dtype_


class TrtConvertCastTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if attrs[0]['in_dtype'] not in [0, 1, 2, 3, 4, 5] or attrs[0][
            'out_dtype'
        ] not in [0, 1, 2, 3, 4, 5]:
            return False
        compile_version = paddle_infer.get_trt_compile_version()
        runtime_version = paddle_infer.get_trt_runtime_version()
        if (
            compile_version[0] * 1000
            + compile_version[1] * 100
            + compile_version[2] * 10
            < 8400
        ):
            return False
        if (
            runtime_version[0] * 1000
            + runtime_version[1] * 100
            + runtime_version[2] * 10
            < 8400
        ):
            return False
        return True

    def sample_program_configs(self):
        def generate_input(type):
            if self.dims == 0:
                return np.ones([]).astype(type)
            else:
                return np.ones([1, 3, 64, 64]).astype(type)

        for dims in [0, 4]:
            self.dims = dims
            for in_dtype in [
                np.bool_,
                np.int32,
                np.float32,
                np.float64,
                np.int64,
            ]:
                self.has_bool_dtype = in_dtype == np.bool_
                dics = [
                    {
                        "in_dtype": convert_np_dtype_to_dtype_(in_dtype),
                        "out_dtype": convert_np_dtype_to_dtype_(np.float32),
                    }
                ]

                ops_config = [
                    {
                        "op_type": "cast",
                        "op_inputs": {"X": ["input_data"]},
                        "op_outputs": {"Out": ["cast_output_data"]},
                        "op_attrs": dics[0],
                        "outputs_dtype": {"cast_output_data": np.float32},
                    },
                ]

                ops = self.generate_op_config(ops_config)

                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "input_data": TensorConfig(
                            data_gen=partial(generate_input, in_dtype)
                        )
                    },
                    outputs=["cast_output_data"],
                    no_cast_list=["input_data"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            if self.dims == 0:
                self.dynamic_shape.min_input_shape = {"input_data": []}
                self.dynamic_shape.max_input_shape = {"input_data": []}
                self.dynamic_shape.opt_input_shape = {"input_data": []}
            else:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 3, 64, 64]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [1, 3, 64, 64]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 3, 64, 64]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if not dynamic_shape and (
                self.has_bool_dtype or self.dims == 1 or self.dims == 0
            ):
                return 0, 3
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-2

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-2

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
