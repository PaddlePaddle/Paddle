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

from __future__ import annotations

import unittest
from functools import partial

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertCumsum(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 7220:
            return False
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1():
            if self.dims == 0:
                self.input_shape = []
                return np.random.random([]).astype(np.float32)
            elif self.dims == 2:
                self.input_shape = [2, 3]
                return np.random.random([2, 3]).astype(np.int32)
            elif self.dims == 3:
                self.input_shape = [2, 3, 4]
                return np.random.random([2, 3, 4]).astype(np.int64)
            elif self.dims == 4:
                self.input_shape = [4, 3, 32, 32]
                return np.random.random([4, 3, 32, 32]).astype(np.float32) - 0.5

        for dims in [0, 2, 3, 4]:
            test_dims = dims
            if dims == 0:
                test_dims = 1
            for axis in range(-1, test_dims):
                for type in ["int32", "int64", "float32", "float64"]:
                    self.dims = dims
                    ops_config = [
                        {
                            "op_type": "cumsum",
                            "op_inputs": {
                                "X": ["input_data"],
                            },
                            "op_outputs": {"Out": ["output_data"]},
                            "op_attrs": {"axis": axis, "dtype": type},
                        }
                    ]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input_data": TensorConfig(
                                data_gen=partial(generate_input1)
                            ),
                        },
                        outputs=["output_data"],
                    )

                    yield program_config

        # no op_attrs
        for dims in [0, 2, 3, 4]:
            self.dims = dims
            ops_config = [
                {
                    "op_type": "cumsum",
                    "op_inputs": {
                        "X": ["input_data"],
                    },
                    "op_outputs": {"Out": ["output_data"]},
                    "op_attrs": {},
                }
            ]
            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    "input_data": TensorConfig(
                        data_gen=partial(generate_input1)
                    ),
                },
                outputs=["output_data"],
            )

            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape():
            if self.dims == 0:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [],
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [2, 3],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [2, 3],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [2, 3],
                }

            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [2, 3, 4],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [2, 3, 4],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [2, 3, 4],
                }

            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [4, 3, 32, 32],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 3, 32, 32],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [4, 3, 32, 32],
                }

        def generate_trt_nodes_num(attrs, dynamic_shape):
            ver = paddle_infer.get_trt_compile_version()
            if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 7220:
                return 0, 3
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

        # for dynamic_shape
        generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-2

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
