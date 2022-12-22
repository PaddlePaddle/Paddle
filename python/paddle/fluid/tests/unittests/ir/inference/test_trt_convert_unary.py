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
from typing import Any, Dict, List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertActivationTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[0] * 10 < 8200:
            if program_config.ops[0].type == "round":
                return False
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(dims, batch, attrs: List[Dict[str, Any]]):
            if dims == 2:
                return np.random.random([3, 32]).astype(np.float32)
            elif dims == 3:
                return np.random.random([3, 32, 32]).astype(np.float32)
            else:
                return np.random.random([batch, 3, 32, 32]).astype(np.float32)

        for dims in [2, 3, 4]:
            for batch in [1, 4]:
                for op_type in [
                    "exp",
                    "log",
                    "sqrt",
                    "abs",
                    "sin",
                    "cos",
                    "tan",
                    "sinh",
                    "cosh",
                    "asin",
                    "acos",
                    "atan",
                    "asinh",
                    "acosh",
                    "atanh",
                    "ceil",
                    "floor",
                    "rsqrt",
                    "reciprocal",
                    "round",
                    "sign",
                ]:
                    self.dims = dims
                    self.op_type = op_type
                    dics = [{}]

                    ops_config = [
                        {
                            "op_type": op_type,
                            "op_inputs": {"X": ["input_data"]},
                            "op_outputs": {"Out": ["output_data"]},
                            "op_attrs": dics[0],
                        }
                    ]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input_data": TensorConfig(
                                data_gen=partial(
                                    generate_input1, dims, batch, dics
                                )
                            )
                        },
                        outputs=["output_data"],
                    )

                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {"input_data": [1]}
                self.dynamic_shape.max_input_shape = {"input_data": [64]}
                self.dynamic_shape.opt_input_shape = {"input_data": [32]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 16]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 32]}
                self.dynamic_shape.opt_input_shape = {"input_data": [3, 32]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 16, 16]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 32, 32]}
                self.dynamic_shape.opt_input_shape = {"input_data": [3, 32, 32]}
            else:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 3, 16, 16]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 3, 32, 32]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 3, 32, 32]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            ver = paddle_infer.get_trt_compile_version()
            if self.dims == 1 or (
                self.op_type == "sign"
                and (
                    not dynamic_shape
                    or ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8200
                )
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
        ), 1e-4
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-4
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-3

    def test(self):
        self.run_test()


class TrtConvertLogicalNotTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        for shape in [[2, 16], [2, 16, 32], [1, 32, 16, 32]]:
            for op_type in ["logical_not"]:
                for axis in [-1]:
                    self.dims = len(shape)
                    dics = [
                        {"axis": axis},
                        {"in_dtype": 5, "out_dtype": 0},
                        {"in_dtype": 0, "out_dtype": 5},
                    ]
                    ops_config = [
                        {
                            "op_type": "cast",
                            "op_inputs": {"X": ["input_data"]},
                            "op_outputs": {"Out": ["cast_output_data1"]},
                            "op_attrs": dics[1],
                            "outputs_dtype": {"cast_output_data1": np.bool},
                        },
                        {
                            "op_type": op_type,
                            "op_inputs": {
                                "X": ["cast_output_data1"],
                            },
                            "op_outputs": {"Out": ["cast_output_data0"]},
                            "op_attrs": dics[0],
                            "outputs_dtype": {"cast_output_data0": np.bool},
                        },
                        {
                            "op_type": "cast",
                            "op_inputs": {"X": ["cast_output_data0"]},
                            "op_outputs": {"Out": ["output_data"]},
                            "op_attrs": dics[2],
                        },
                    ]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input_data": TensorConfig(
                                data_gen=partial(generate_input, shape)
                            ),
                        },
                        outputs=["output_data"],
                    )

                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [2, 16],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [2, 16],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [2, 16],
                }
            if self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [2, 16, 32],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [2, 16, 32],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [2, 16, 32],
                }
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 32, 16, 32],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [1, 32, 16, 32],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 32, 16, 32],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape:
                ver = paddle_infer.get_trt_compile_version()
                if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8400:
                    return 0, 5
                return 1, 2
            return 0, 5

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
        ), (1e-3, 1e-3)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-3, 1e-3)

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
