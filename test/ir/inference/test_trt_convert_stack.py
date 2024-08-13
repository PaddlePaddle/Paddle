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
from typing import Any

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertStackTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        outputs = program_config.outputs

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        # axis must be inside [-(rank+1), rank+1)
        if len(inputs['stack_input1'].shape) < attrs[0]['axis']:
            return False
        if -(len(inputs['stack_input1'].shape) + 1) > attrs[0]['axis']:
            return False

        return True

    def sample_program_configs(self):
        def generate_input1(attrs: list[dict[str, Any]], batch):
            if self.dims == 4:
                return np.random.random([batch, 3, 24, 24]).astype(np.float32)
            else:
                return np.random.random([]).astype(np.float32)

        def generate_input2(attrs: list[dict[str, Any]], batch):
            if self.dims == 4:
                return np.random.random([batch, 3, 24, 24]).astype(np.float32)
            else:
                return np.random.random([]).astype(np.float32)

        def generate_input3(attrs: list[dict[str, Any]], batch):
            if self.dims == 4:
                return np.random.random([batch, 3, 24, 24]).astype(np.float32)
            else:
                return np.random.random([]).astype(np.float32)

        for dims in [0, 4]:
            for batch in [1]:
                for axis in [-1, 0, 1]:
                    self.dims = dims
                    dics = [{"axis": axis}, {}]
                    ops_config = [
                        {
                            "op_type": "stack",
                            "op_inputs": {
                                "X": [
                                    "stack_input1",
                                    "stack_input2",
                                    "stack_input3",
                                ]
                            },
                            "op_outputs": {"Y": ["stack_output"]},
                            "op_attrs": dics[0],
                        }
                    ]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "stack_input1": TensorConfig(
                                data_gen=partial(generate_input1, dics, batch)
                            ),
                            "stack_input2": TensorConfig(
                                data_gen=partial(generate_input2, dics, batch)
                            ),
                            "stack_input3": TensorConfig(
                                data_gen=partial(generate_input3, dics, batch)
                            ),
                        },
                        outputs=["stack_output"],
                    )

                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "stack_input1": [1, 3, 24, 24],
                    "stack_input2": [1, 3, 24, 24],
                    "stack_input3": [1, 3, 24, 24],
                }
                self.dynamic_shape.max_input_shape = {
                    "stack_input1": [4, 3, 48, 48],
                    "stack_input2": [4, 3, 48, 48],
                    "stack_input3": [4, 3, 48, 48],
                }
                self.dynamic_shape.opt_input_shape = {
                    "stack_input1": [1, 3, 24, 24],
                    "stack_input2": [1, 3, 24, 24],
                    "stack_input3": [1, 3, 24, 24],
                }
            else:
                self.dynamic_shape.min_input_shape = {
                    "stack_input1": [],
                    "stack_input2": [],
                    "stack_input3": [],
                }
                self.dynamic_shape.max_input_shape = {
                    "stack_input1": [],
                    "stack_input2": [],
                    "stack_input3": [],
                }
                self.dynamic_shape.opt_input_shape = {
                    "stack_input1": [],
                    "stack_input2": [],
                    "stack_input3": [],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape:
                return 1, 4
            else:
                return 0, 5

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
        ), 1e-3

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


if __name__ == "__main__":
    unittest.main()
