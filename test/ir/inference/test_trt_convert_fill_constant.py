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

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertFillConstantTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_value_data():
            return np.array([1]).astype(np.int32)

        def generate_input_data():
            return np.random.random([2, 5, 7]).astype(np.float32)

        for dtype in [5, 2, 3]:
            for str_value in ["2", "23", "-1"]:
                value = float(str_value)
                if np.random.choice([False, True]):
                    str_value = str_value
                else:
                    str_value = ""
                dics = [
                    {
                        "str_value": str_value,
                        "value": value,
                        "dtype": dtype,
                    },
                    {"axis": -1},
                ]
                for mode in ["ValueTensor", "ShapeTensor", "ShapeTensorList"]:
                    self.mode = mode
                    if mode == "ValueTensor":
                        dics[0]["shape"] = [2, 3, 4]
                        ops_config = [
                            {
                                "op_type": "fill_constant",
                                "op_inputs": {"ValueTensor": ["value_data"]},
                                "op_outputs": {
                                    "Out": ["out_data"],
                                },
                                "op_attrs": dics[0],
                            },
                        ]
                    elif mode == "ShapeTensor":
                        ops_config = [
                            {
                                "op_type": "shape",
                                "op_inputs": {
                                    "Input": ["input_data"],
                                },
                                "op_outputs": {"Out": ["shape_data"]},
                                "op_attrs": {},
                            },
                            {
                                "op_type": "fill_constant",
                                "op_inputs": {
                                    "ShapeTensor": ["shape_data"],
                                },
                                "op_outputs": {
                                    "Out": ["out_data"],
                                },
                                "op_attrs": dics[0],
                            },
                        ]
                    else:
                        ops_config = [
                            {
                                "op_type": "shape",
                                "op_inputs": {
                                    "Input": ["input_data"],
                                },
                                "op_outputs": {"Out": ["shape_data"]},
                                "op_attrs": {},
                            },
                            {
                                "op_type": "split",
                                "op_inputs": {
                                    "X": ["shape_data"],
                                },
                                "op_outputs": {
                                    "Out": [
                                        "split_shape_data_0",
                                        "split_shape_data_1",
                                        "split_shape_data_2",
                                    ]
                                },
                                "op_attrs": {
                                    "axis": 0,
                                    "num": 3,
                                },
                            },
                            {
                                "op_type": "fill_constant",
                                "op_inputs": {
                                    "ShapeTensorList": [
                                        "split_shape_data_0",
                                        "split_shape_data_1",
                                        "split_shape_data_2",
                                    ],
                                },
                                "op_outputs": {
                                    "Out": ["out_data"],
                                },
                                "op_attrs": dics[0],
                            },
                        ]
                    ops = self.generate_op_config(ops_config)
                    if mode == "ValueTensor":
                        program_config = ProgramConfig(
                            ops=ops,
                            weights={},
                            inputs={
                                "value_data": TensorConfig(
                                    data_gen=partial(generate_value_data)
                                ),
                            },
                            outputs=["out_data"],
                        )
                    else:
                        program_config = ProgramConfig(
                            ops=ops,
                            weights={},
                            inputs={
                                "input_data": TensorConfig(
                                    data_gen=partial(generate_input_data)
                                ),
                            },
                            outputs=["out_data"],
                        )
                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            if self.mode == "ValueTensor":
                self.input_shape = [1, 1]
                max_shape = list(self.input_shape)
                min_shape = list(self.input_shape)
                opt_shape = list(self.input_shape)
                for i in range(len(self.input_shape)):
                    max_shape[i] = max_shape[i] + 1
                self.dynamic_shape.min_input_shape = {"Y_data": min_shape}
                self.dynamic_shape.max_input_shape = {"Y_data": max_shape}
                self.dynamic_shape.opt_input_shape = {"Y_data": opt_shape}
            else:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [2, 3, 7],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [2, 5, 7],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [2, 4, 7],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if self.mode == "ValueTensor":
                return 0, 3
            else:
                ver = paddle_infer.get_trt_compile_version()
                if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8500:
                    return 1, 3
                else:
                    return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        # Don't test static shape

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
