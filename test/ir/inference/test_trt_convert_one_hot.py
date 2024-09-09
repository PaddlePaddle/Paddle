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


class TrtConvertOneHotTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8510:
            return False
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_indices(dims, batch):
            if dims == 2:
                return np.random.randint(0, 10, (batch, 4), dtype=np.int32)
            elif dims == 3:
                return np.random.randint(0, 10, (batch, 4, 6), dtype=np.int32)
            else:
                return np.random.randint(
                    0, 10, (batch, 4, 6, 8), dtype=np.int32
                )

        def generate_depth(dims, batch):
            return np.ones((1,), dtype=np.int32) * 10

        for dims in [2, 3, 4]:
            for batch in [1, 2]:
                self.dims = dims
                dics = [{"dtype": 5, "depth": 10}, {}]
                ops_config = [
                    {
                        "op_type": "one_hot_v2",
                        "op_inputs": {
                            "X": ["indices_tensor"],
                            "depth_tensor": ["depth_tensor_data"],
                        },
                        "op_outputs": {"Out": ["output_data"]},
                        "op_attrs": dics[0],
                        "outputs_dtype": {"output_data": np.int_},
                    },
                ]
                ops = self.generate_op_config(ops_config)

                program_config = ProgramConfig(
                    ops=ops,
                    weights={
                        "depth_tensor_data": TensorConfig(
                            data_gen=partial(generate_depth, dims, batch)
                        ),
                    },
                    inputs={
                        "indices_tensor": TensorConfig(
                            data_gen=partial(generate_indices, dims, batch)
                        ),
                    },
                    outputs=["output_data"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {
                    "indices_tensor": [1],
                }
                self.dynamic_shape.max_input_shape = {
                    "indices_tensor": [2],
                }
                self.dynamic_shape.opt_input_shape = {
                    "indices_tensor": [1],
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "indices_tensor": [1, 4],
                }
                self.dynamic_shape.max_input_shape = {
                    "indices_tensor": [2, 4],
                }
                self.dynamic_shape.opt_input_shape = {
                    "indices_tensor": [1, 4],
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "indices_tensor": [1, 4, 6],
                }
                self.dynamic_shape.max_input_shape = {
                    "indices_tensor": [2, 4, 6],
                }
                self.dynamic_shape.opt_input_shape = {
                    "indices_tensor": [1, 4, 6],
                }
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "indices_tensor": [1, 4, 6, 8],
                }
                self.dynamic_shape.max_input_shape = {
                    "indices_tensor": [2, 4, 6, 8],
                }
                self.dynamic_shape.opt_input_shape = {
                    "indices_tensor": [1, 4, 6, 8],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if not dynamic_shape:
                return 0, 3
            return 1, 2

        attrs = [op.attrs for op in program_config.ops]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
