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

import unittest
from functools import partial
from typing import Any, Dict, List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtInt64Test1(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        out_shape = list(inputs['input_data'].shape)
        for x in range(len(attrs[0]["axes"])):
            start = 0
            end = 0
            if attrs[0]["starts"][x] < 0:
                start = (
                    attrs[0]["starts"][x]
                    + inputs['input_data'].shape[attrs[0]["axes"][x]]
                )
            else:
                start = attrs[0]["starts"][x]
            if attrs[0]["ends"][x] < 0:
                end = (
                    attrs[0]["ends"][x]
                    + inputs['input_data'].shape[attrs[0]["axes"][x]]
                )
            else:
                end = attrs[0]["ends"][x]
            start = max(0, start)
            end = max(0, end)
            out_shape[attrs[0]["axes"][x]] = end - start
            if start >= end:
                return False
        for x in attrs[0]["decrease_axis"]:
            if x < 0:
                return False
            if out_shape[x] != 1:
                return False
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]]):
            return (10 * np.random.random([6, 6, 64, 64])).astype(np.int64)

        for axes in [[0, 1], [1, 3], [2, 3]]:
            for starts in [[0, 1]]:
                for ends in [[2, 2], [5, 5], [1, -1]]:
                    for decrease_axis in [[], [1], [2], [-1], [-100]]:
                        for infer_flags in [[-1]]:
                            dics = [
                                {
                                    "axes": axes,
                                    "starts": starts,
                                    "ends": ends,
                                    "decrease_axis": decrease_axis,
                                    "infer_flags": infer_flags,
                                }
                            ]

                            ops_config = [
                                {
                                    "op_type": "slice",
                                    "op_inputs": {"Input": ["input_data"]},
                                    "op_outputs": {
                                        "Out": ["slice_output_data"]
                                    },
                                    "op_attrs": dics[0],
                                }
                            ]
                            ops = self.generate_op_config(ops_config)

                            program_config = ProgramConfig(
                                ops=ops,
                                weights={},
                                inputs={
                                    "input_data": TensorConfig(
                                        data_gen=partial(generate_input1, dics)
                                    )
                                },
                                outputs=["slice_output_data"],
                            )

                            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 32, 32]}
            self.dynamic_shape.max_input_shape = {"input_data": [8, 8, 64, 64]}
            self.dynamic_shape.opt_input_shape = {"input_data": [6, 6, 64, 64]}

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
        ), 1e-3

    def test(self):
        self.run_test()


class TrtInt64Test2(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(shape, op_type):
            return np.random.randint(
                low=1, high=10000, size=shape, dtype=np.int64
            )

        for shape in [[2, 32, 16], [1, 8, 16, 32]]:
            for op_type in [
                "elementwise_add",
                "elementwise_mul",
                "elementwise_sub",
            ]:
                for axis in [0, -1]:
                    self.dims = len(shape)
                    dics = [{"axis": axis}]
                    ops_config = [
                        {
                            "op_type": op_type,
                            "op_inputs": {
                                "X": ["input_data1"],
                                "Y": ["input_data2"],
                            },
                            "op_outputs": {"Out": ["output_data"]},
                            "op_attrs": dics[0],
                        }
                    ]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input_data1": TensorConfig(
                                data_gen=partial(generate_input, shape, op_type)
                            ),
                            "input_data2": TensorConfig(
                                data_gen=partial(generate_input, shape, op_type)
                            ),
                        },
                        outputs=["output_data"],
                    )

                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 4, 4],
                    "input_data2": [1, 4, 4],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [128, 128, 256],
                    "input_data2": [128, 128, 256],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 32, 16],
                    "input_data2": [2, 32, 16],
                }
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 4, 4, 4],
                    "input_data2": [1, 4, 4, 4],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [8, 128, 64, 128],
                    "input_data2": [8, 128, 64, 128],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 64, 32, 32],
                    "input_data2": [2, 64, 32, 32],
                }

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 3), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), (1e-3, 1e-3)

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
