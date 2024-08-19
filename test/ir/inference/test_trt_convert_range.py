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


class TrtConvertRangeDynamicTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input():
            return np.array([1]).astype(np.int32)

        for in_dtype in [2]:
            self.in_dtype = in_dtype
            dics = [{}]
            ops_config = [
                {
                    "op_type": "fill_constant",
                    "op_inputs": {},
                    "op_outputs": {"Out": ["start_data"]},
                    "op_attrs": {
                        "dtype": self.in_dtype,
                        "str_value": "7",
                        "shape": [1],
                    },
                },
                {
                    "op_type": "fill_constant",
                    "op_inputs": {},
                    "op_outputs": {"Out": ["end_data"]},
                    "op_attrs": {
                        "dtype": self.in_dtype,
                        "str_value": "256",
                        "shape": [1],
                    },
                },
                {
                    "op_type": "fill_constant",
                    "op_inputs": {},
                    "op_outputs": {"Out": ["step_data"]},
                    "op_attrs": {
                        "dtype": self.in_dtype,
                        "str_value": "1",
                        "shape": [1],
                    },
                },
                {
                    "op_type": "range",
                    "op_inputs": {
                        "Start": ["start_data"],
                        "End": ["end_data"],
                        "Step": ["step_data"],
                    },
                    "op_outputs": {"Out": ["range_output_data1"]},
                    "op_attrs": dics[0],
                },
                {
                    "op_type": "cast",
                    "op_inputs": {"X": ["range_output_data1"]},
                    "op_outputs": {"Out": ["range_output_data"]},
                    "op_attrs": {"in_dtype": self.in_dtype, "out_dtype": 5},
                },
            ]
            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    "step_data": TensorConfig(data_gen=partial(generate_input)),
                },
                outputs=["range_output_data"],
            )

            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "start_data": [1],
                "end_data": [1],
                "step_data": [1],
            }
            self.dynamic_shape.max_input_shape = {
                "start_data": [1],
                "end_data": [1],
                "step_data": [1],
            }
            self.dynamic_shape.opt_input_shape = {
                "start_data": [1],
                "end_data": [1],
                "step_data": [1],
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


class TrtConvertRangeStaticTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input():
            return np.array([0]).astype(np.int32)

        def generate_input1():
            return np.array([128]).astype(np.int32)

        def generate_input2():
            return np.array([1]).astype(np.int32)

        for in_dtype in [2]:
            self.in_dtype = in_dtype
            dics = [{}]
            ops_config = [
                {
                    "op_type": "range",
                    "op_inputs": {
                        "Start": ["start_data"],
                        "End": ["end_data"],
                        "Step": ["step_data"],
                    },
                    "op_outputs": {"Out": ["range_output_data1"]},
                    "op_attrs": dics[0],
                },
                {
                    "op_type": "cast",
                    "op_inputs": {"X": ["range_output_data1"]},
                    "op_outputs": {"Out": ["range_output_data"]},
                    "op_attrs": {"in_dtype": self.in_dtype, "out_dtype": 5},
                },
            ]
            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    "start_data": TensorConfig(
                        data_gen=partial(generate_input)
                    ),
                    "end_data": TensorConfig(data_gen=partial(generate_input1)),
                    "step_data": TensorConfig(
                        data_gen=partial(generate_input2)
                    ),
                },
                outputs=["range_output_data"],
            )

            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 0, 6

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

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
