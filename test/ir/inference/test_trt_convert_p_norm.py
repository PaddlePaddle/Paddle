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
from typing import Any

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertPNormTest(TrtLayerAutoScanTest):
    def sample_program_configs(self):
        def generate_input1(dims, attrs: list[dict[str, Any]]):
            if dims == 1:
                return np.ones([3]).astype(np.float32)
            elif dims == 2:
                return np.ones([3, 64]).astype(np.float32)
            elif dims == 3:
                return np.ones([3, 64, 64]).astype(np.float32)
            else:
                return np.ones([1, 3, 64, 64]).astype(np.float32)

        for dims in [2, 3, 4]:
            # TODO(liuyuanle): support asvector = True
            for asvector in [False]:
                for keepdim in [False, True]:
                    for porder in [0, 1, 2, 3]:
                        for axis in [-1]:
                            self.dims = dims

                            dics = [
                                {
                                    "asvector": asvector,
                                    "keepdim": keepdim,
                                    "axis": axis,
                                    "porder": porder,
                                }
                            ]

                            ops_config = [
                                {
                                    "op_type": "p_norm",
                                    "op_inputs": {
                                        "X": ["input_data"],
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
                                    "input_data": TensorConfig(
                                        data_gen=partial(
                                            generate_input1, dims, dics
                                        )
                                    )
                                },
                                outputs=["output_data"],
                            )

                            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {"input_data": [1]}
                self.dynamic_shape.max_input_shape = {"input_data": [128]}
                self.dynamic_shape.opt_input_shape = {"input_data": [64]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 32]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 64]}
                self.dynamic_shape.opt_input_shape = {"input_data": [3, 64]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 32, 32]}
                self.dynamic_shape.max_input_shape = {
                    "input_data": [10, 64, 64]
                }
                self.dynamic_shape.opt_input_shape = {"input_data": [3, 64, 64]}
            else:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 3, 32, 32]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 3, 64, 64]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 3, 64, 64]
                }

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for dynamic_shape mode
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
        ), (1e-3, 1e-3)

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
