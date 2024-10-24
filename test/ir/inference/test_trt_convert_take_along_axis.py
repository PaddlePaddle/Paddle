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


class TrtConvertTakeAlongAxisTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if len(inputs['input_data'].shape) <= attrs[0]['Axis']:
            return False
        if len(inputs['input_data'].shape) != len(inputs['index_data'].shape):
            return False

        return True

    def sample_program_configs(self):
        def generate_input1(shape):
            return np.random.random(shape).astype(np.float32)

        def generate_input2(index):
            return np.zeros(index).astype(np.int32)

        def generate_input3(axis):
            return np.array([axis]).astype(np.int32)

        for shape in [[32], [3, 64], [1, 64, 16], [1, 64, 16, 32]]:
            for index in [[1], [1, 1], [1, 1, 2], [1, 1, 1, 1]]:
                for axis in [0, 1, 2, 3]:
                    self.shape = shape
                    self.axis = axis
                    dics = [{"Axis": axis}]
                    ops_config = [
                        {
                            "op_type": "take_along_axis",
                            "op_inputs": {
                                "Input": ["input_data"],
                                "Index": ["index_data"],
                            },
                            "op_outputs": {"Result": ["output_data"]},
                            "op_attrs": dics[0],
                        }
                    ]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input_data": TensorConfig(
                                data_gen=partial(generate_input1, shape)
                            ),
                            "index_data": TensorConfig(
                                data_gen=partial(generate_input2, index)
                            ),
                        },
                        outputs=["output_data"],
                        no_cast_list=["index_data"],
                    )

                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            if len(self.shape) == 1:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [4],
                    "index_data": [1],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [128],
                    "index_data": [4],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [16],
                    "index_data": [2],
                }
            elif len(self.shape) == 2:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [3, 64],
                    "index_data": [1, 1],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [3, 64],
                    "index_data": [1, 1],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [3, 64],
                    "index_data": [1, 1],
                }
            elif len(self.shape) == 3:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 64, 16],
                    "index_data": [1, 1, 2],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [1, 64, 16],
                    "index_data": [1, 1, 2],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 64, 16],
                    "index_data": [1, 1, 2],
                }
            elif len(self.shape) == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 64, 16, 32],
                    "index_data": [1, 1, 1, 1],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [1, 64, 16, 32],
                    "index_data": [1, 1, 1, 1],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 64, 16, 32],
                    "index_data": [1, 1, 1, 1],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(dynamic_shape):
            ver = paddle_infer.get_trt_compile_version()
            if (
                ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 > 8200
                and dynamic_shape
            ):
                return 1, 3
            else:
                return 0, 4

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            False
        ), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(True), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
