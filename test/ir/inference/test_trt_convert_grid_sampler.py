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

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertGridSampler(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        self.trt_param.workspace_size = 1073741824
        return True

    def sample_program_configs(self):
        def generate_input1():
            if self.dims == 4:
                self.input_shape = [1, 3, 32, 32]
                return np.random.random([1, 3, 32, 32]).astype(np.float32)
            elif self.dims == 5:
                self.input_shape = [1, 3, 32, 32, 64]
                return np.random.random([1, 3, 32, 32, 64]).astype(np.float32)

        def generate_input2():
            if self.dims == 4:
                self.input_shape = [1, 3, 3, 2]
                return np.random.random([1, 3, 3, 2]).astype(np.float32)
            elif self.dims == 5:
                self.input_shape = [1, 3, 3, 2, 3]
                return np.random.random([1, 3, 3, 2, 3]).astype(np.float32)

        mode = ["bilinear", "nearest"]
        padding_mode = ["zeros", "reflection"]
        align_corners = [True]
        descs = []
        for m in mode:
            for p in padding_mode:
                for a in align_corners:
                    descs.append(
                        {
                            "mode": m,
                            "padding_mode": p,
                            "align_corners": a,
                        }
                    )

        for dims in [4, 5]:
            for desc in descs:
                self.dims = dims
                ops_config = [
                    {
                        "op_type": "grid_sampler",
                        "op_inputs": {
                            "X": ["input_data"],
                            "Grid": ["grid_data"],
                        },
                        "op_outputs": {"Output": ["output_data"]},
                        "op_attrs": desc,
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
                        "grid_data": TensorConfig(
                            data_gen=partial(generate_input2)
                        ),
                    },
                    outputs=["output_data"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape():
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 3, 32, 32],
                    "grid_data": [1, 3, 3, 2],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [1, 3, 64, 64],
                    "grid_data": [1, 3, 6, 2],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 3, 32, 32],
                    "grid_data": [1, 3, 3, 2],
                }
            elif self.dims == 5:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 3, 32, 32, 64],
                    "grid_data": [1, 3, 3, 2, 3],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [1, 3, 64, 64, 128],
                    "grid_data": [1, 3, 3, 6, 3],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 3, 32, 32, 64],
                    "grid_data": [1, 3, 3, 2, 3],
                }

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
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), 1e-3

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
