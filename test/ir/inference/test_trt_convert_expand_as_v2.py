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
from typing import Any, Dict, List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertExpandASV2Test(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if len(attrs[0]['target_shape']) < self.dims:
            return False
        if self.dims == 1:
            if len(attrs[0]['target_shape']) == 4:
                return False
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]]):
            if self.dims == 4:
                self.input_shape = [1, 8, 1, 32]
                return np.random.random([1, 8, 1, 32]).astype(np.float32)
            elif self.dims == 3:
                self.input_shape = [1, 32, 32]
                return np.random.random([1, 32, 32]).astype(np.float32)
            elif self.dims == 2:
                self.input_shape = [1, 32]
                return np.random.random([1, 32]).astype(np.float32)
            elif self.dims == 1:
                self.input_shape = [32]
                return np.random.random([32]).astype(np.float32)
            elif self.dims == 0:
                self.input_shape = []
                return np.random.random([]).astype(np.float32)

        for dims in [0, 1, 2, 3, 4]:
            for shape in [
                [10, 8, 32, 32],
                [2, 8, 32, 32],
                [8, 32, 32],
                [2, 32],
                [32],
            ]:
                dics = [
                    {
                        "target_shape": shape,
                    },
                ]
                self.dims = dims

                ops_config = [
                    {
                        "op_type": "expand_as_v2",
                        "op_inputs": {"X": ["expand_v2_input"]},
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
                    "expand_v2_input": [1, 8, 1, 32]
                }
                self.dynamic_shape.max_input_shape = {
                    "expand_v2_input": [10, 8, 1, 32]
                }
                self.dynamic_shape.opt_input_shape = {
                    "expand_v2_input": [1, 8, 1, 32]
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "expand_v2_input": [1, 32, 32]
                }
                self.dynamic_shape.max_input_shape = {
                    "expand_v2_input": [8, 32, 32]
                }
                self.dynamic_shape.opt_input_shape = {
                    "expand_v2_input": [1, 32, 32]
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "expand_v2_input": [1, 32]
                }
                self.dynamic_shape.max_input_shape = {
                    "expand_v2_input": [4, 32]
                }
                self.dynamic_shape.opt_input_shape = {
                    "expand_v2_input": [1, 32]
                }
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {"expand_v2_input": [32]}
                self.dynamic_shape.max_input_shape = {"expand_v2_input": [64]}
                self.dynamic_shape.opt_input_shape = {"expand_v2_input": [32]}
            elif self.dims == 0:
                self.dynamic_shape.min_input_shape = {"expand_v2_input": []}
                self.dynamic_shape.max_input_shape = {"expand_v2_input": []}
                self.dynamic_shape.opt_input_shape = {"expand_v2_input": []}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            ver = paddle_infer.get_trt_compile_version()
            ver_num = ver[0] * 1000 + ver[1] * 100 + ver[2] * 10
            if dynamic_shape and (ver_num > 8000 or self.dims > 0):
                return 1, 2
            else:
                return 0, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        clear_dynamic_shape()
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
            for shape in [[10]]:
                dics = [
                    {
                        "target_shape": shape,
                    },
                ]
                self.dims = dims
                dics_intput = [
                    {"X": ["expand_v2_input"], "Y": ["shapeT1_data"]},
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
                        "op_type": "expand_as_v2",
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
        program_config.set_input_type(np.float32)
        # fill_constant will be folded by constnt folding pass!
        yield self.create_inference_config(), (1, 2), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (1, 2), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
