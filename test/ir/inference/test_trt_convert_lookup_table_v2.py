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


class TrtConvertLookupTableV2Test(TrtLayerAutoScanTest):
    def sample_program_configs(self):
        self.trt_param.workspace_size = 102400

        def generate_input1(dims, attrs: list[dict[str, Any]]):
            if dims == 1:
                return np.array([32, 2, 19]).astype(np.int64)
            elif dims == 2:
                return np.array([[3, 16, 24], [6, 4, 47]]).astype(np.int64)
            else:
                return np.array(
                    [
                        [[3, 16, 24], [30, 16, 14], [2, 6, 24]],
                        [[3, 26, 34], [3, 16, 24], [3, 6, 4]],
                        [[3, 16, 24], [53, 16, 54], [30, 1, 24]],
                    ]
                ).astype(np.int64)

        def generate_input2(dims, attrs: list[dict[str, Any]]):
            return np.random.uniform(-1, 1, [64, 4]).astype('float32')

        for dims in [1, 2, 3]:
            self.dims = dims

            ops_config = [
                {
                    "op_type": "lookup_table_v2",
                    "op_inputs": {"Ids": ["indices"], "W": ["data"]},
                    "op_outputs": {"Out": ["out_data"]},
                    "op_attrs": {},
                }
            ]
            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={
                    "data": TensorConfig(
                        data_gen=partial(generate_input2, {}, {})
                    )
                },
                inputs={
                    "indices": TensorConfig(
                        data_gen=partial(generate_input1, dims, {})
                    )
                },
                outputs=["out_data"],
                no_cast_list=["indices"],
            )

            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {
                    "indices": [1],
                    "data": [64, 4],
                }
                self.dynamic_shape.max_input_shape = {
                    "indices": [16],
                    "data": [64, 4],
                }
                self.dynamic_shape.opt_input_shape = {
                    "indices": [8],
                    "data": [64, 4],
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "indices": [1, 1],
                    "data": [64, 4],
                }
                self.dynamic_shape.max_input_shape = {
                    "indices": [16, 32],
                    "data": [64, 4],
                }
                self.dynamic_shape.opt_input_shape = {
                    "indices": [2, 16],
                    "data": [64, 4],
                }
            else:
                self.dynamic_shape.min_input_shape = {
                    "indices": [1, 1, 1],
                    "data": [64, 4],
                }
                self.dynamic_shape.max_input_shape = {
                    "indices": [16, 16, 16],
                    "data": [64, 4],
                }
                self.dynamic_shape.opt_input_shape = {
                    "indices": [2, 8, 8],
                    "data": [64, 4],
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
