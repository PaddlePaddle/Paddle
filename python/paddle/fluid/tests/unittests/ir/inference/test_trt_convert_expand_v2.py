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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import List, Dict, Any
import unittest


class TrtConvertExpandV2Test(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if len(attrs[0]['shape']) < self.dims:
            return False
        if self.dims == 1:
            if len(attrs[0]['shape']) == 4:
                return False
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]]):
            if self.dims == 4:
                self.input_shape = [1, 1, 4, 6]
                return np.random.random([1, 1, 4, 6]).astype(np.float32)
            elif self.dims == 3:
                self.input_shape = [1, 8, 6]
                return np.random.random([1, 8, 6]).astype(np.float32)
            elif self.dims == 2:
                self.input_shape = [1, 48]
                return np.random.random([1, 48]).astype(np.float32)
            elif self.dims == 1:
                self.input_shape = [48]
                return np.random.random([48]).astype(np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            return np.array([1, 48]).astype(np.int32)

        def generate_shapeT1_data(attrs: List[Dict[str, Any]]):
            return np.array([2]).astype(np.int32)

        def generate_shapeT2_data(attrs: List[Dict[str, Any]]):
            return np.array([24]).astype(np.int32)

        for dims in [4, 3, 2, 1]:
            for shape in [[10, 12, -1, -1], [8, 64, -1, -1], [6, 8, -1]]:
                dics = [
                    {
                        "shape": shape,
                    },
                ]
                self.dims = dims
                dics_intput = [{"X": ["expand_v2_input"]}]

                ops_config = [
                    {
                        "op_type": "expand_v2",
                        "op_inputs": dics_intput[0],
                        "op_outputs": {"Out": ["expand_v2_out"]},
                        "op_attrs": dics[0],
                    }
                ]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "expand_v2_input": TensorConfig(
                            data_gen=partial(generate_input1, dics)
                        )
                    },
                    outputs=["expand_v2_out"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "expand_v2_input": [1, 1, 4, 6]
                }
                self.dynamic_shape.max_input_shape = {
                    "expand_v2_input": [10, 1, 4, 6]
                }
                self.dynamic_shape.opt_input_shape = {
                    "expand_v2_input": [1, 1, 4, 6]
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "expand_v2_input": [1, 8, 6]
                }
                self.dynamic_shape.max_input_shape = {
                    "expand_v2_input": [4, 8, 6]
                }
                self.dynamic_shape.opt_input_shape = {
                    "expand_v2_input": [1, 8, 6]
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "expand_v2_input": [1, 48]
                }
                self.dynamic_shape.max_input_shape = {
                    "expand_v2_input": [4, 48]
                }
                self.dynamic_shape.opt_input_shape = {
                    "expand_v2_input": [1, 48]
                }
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {"expand_v2_input": [48]}
                self.dynamic_shape.max_input_shape = {"expand_v2_input": [48]}
                self.dynamic_shape.opt_input_shape = {"expand_v2_input": [48]}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape:
                return 1, 2
            else:
                return 0, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-3

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


class TrtConvertExpandV2Test2(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]]):
            if self.dims == 1:
                self.input_shape = [1]
                return np.random.random([1]).astype(np.float32)

        for dims in [1]:
            for shape in [[10, 12, -1, -1], [8, 64, -1, -1]]:
                dics = [
                    {
                        "shape": shape,
                    },
                ]
                self.dims = dims
                dics_intput = [
                    {"X": ["expand_v2_input"], "Shape": ["shapeT1_data"]},
                ]
                ops_config = [
                    {
                        "op_type": "fill_constant",
                        "op_inputs": {},
                        "op_outputs": {"Out": ["shapeT1_data"]},
                        "op_attrs": {
                            "dtype": 2,
                            "str_value": "10",
                            "shape": [1],
                        },
                    },
                    {
                        "op_type": "expand_v2",
                        "op_inputs": dics_intput[0],
                        "op_outputs": {"Out": ["expand_v2_out"]},
                        "op_attrs": dics[0],
                    },
                ]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "expand_v2_input": TensorConfig(
                            data_gen=partial(generate_input1, dics)
                        )
                    },
                    outputs=["expand_v2_out"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape():
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {"expand_v2_input": [1]}
                self.dynamic_shape.max_input_shape = {"expand_v2_input": [1]}
                self.dynamic_shape.opt_input_shape = {"expand_v2_input": [1]}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        clear_dynamic_shape()
        # for dynamic_shape
        generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        # fill_constant will be folded by constnt folding pass!
        yield self.create_inference_config(), (0, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (0, 3), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertExpandV2Test3(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]]):
            if self.dims == 4:
                self.input_shape = [1, 1, 4, 6]
                return np.random.random([1, 1, 4, 6]).astype(np.float32)
            elif self.dims == 3:
                self.input_shape = [1, 4, 6]
                return np.random.random([1, 4, 6]).astype(np.float32)

        for dims in [4, 3]:
            for shape in [[10, 12, -1, -1], [8, 64, -1, -1]]:
                dics = [
                    {
                        "shape": shape,
                    },
                ]
                self.dims = dims
                dics_intput = [
                    {
                        "X": ["expand_v2_input"],
                        "expand_shapes_tensor": [
                            "shapeT1_data",
                            "shapeT2_data",
                            "shapeT3_data",
                            "shapeT4_data",
                        ],
                    },
                ]
                ops_config = [
                    {
                        "op_type": "fill_constant",
                        "op_inputs": {},
                        "op_outputs": {"Out": ["shapeT1_data"]},
                        "op_attrs": {
                            "dtype": 2,
                            "str_value": "10",
                            "shape": [1],
                        },
                    },
                    {
                        "op_type": "fill_constant",
                        "op_inputs": {},
                        "op_outputs": {"Out": ["shapeT2_data"]},
                        "op_attrs": {
                            "dtype": 2,
                            "str_value": "12",
                            "shape": [1],
                        },
                    },
                    {
                        "op_type": "fill_constant",
                        "op_inputs": {},
                        "op_outputs": {"Out": ["shapeT3_data"]},
                        "op_attrs": {
                            "dtype": 2,
                            "str_value": "4",
                            "shape": [1],
                        },
                    },
                    {
                        "op_type": "fill_constant",
                        "op_inputs": {},
                        "op_outputs": {"Out": ["shapeT4_data"]},
                        "op_attrs": {
                            "dtype": 2,
                            "str_value": "6",
                            "shape": [1],
                        },
                    },
                    {
                        "op_type": "expand_v2",
                        "op_inputs": dics_intput[0],
                        "op_outputs": {"Out": ["expand_v2_out"]},
                        "op_attrs": dics[0],
                    },
                ]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "expand_v2_input": TensorConfig(
                            data_gen=partial(generate_input1, dics)
                        )
                    },
                    outputs=["expand_v2_out"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape():
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "expand_v2_input": [1, 1, 4, 6]
                }
                self.dynamic_shape.max_input_shape = {
                    "expand_v2_input": [10, 1, 4, 6]
                }
                self.dynamic_shape.opt_input_shape = {
                    "expand_v2_input": [1, 1, 4, 6]
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "expand_v2_input": [1, 4, 6]
                }
                self.dynamic_shape.max_input_shape = {
                    "expand_v2_input": [4, 4, 6]
                }
                self.dynamic_shape.opt_input_shape = {
                    "expand_v2_input": [1, 4, 6]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        clear_dynamic_shape()
        # for dynamic_shape
        generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        # fill_constant will be folded by constnt folding pass!
        yield self.create_inference_config(), (0, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (0, 3), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
