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

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertArgsort(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1():
            if self.dims == 4:
                return np.random.random([1, 3, 3, 4000]).astype(np.float32)
            elif self.dims == 3:
                return np.random.random([1, 3, 24]).astype(np.int32)
            elif self.dims == 2:
                return np.random.random([1, 24]).astype(np.float32)
            else:
                return np.random.random([24]).astype(np.float32)

        for dims in [1, 2, 3, 4]:
            for axis in [-1, 0]:
                for descending in [False, True]:
                    self.dims = dims
                    ops_config = [
                        {
                            "op_type": "argsort",
                            "op_inputs": {"X": ["input_data"]},
                            "op_outputs": {
                                "Out": ["output_data"],
                                "Indices": ["indices_data"],
                            },
                            "op_attrs": {
                                "axis": axis,
                                "descending": descending,
                            },
                        }
                    ]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input_data": TensorConfig(
                                data_gen=partial(generate_input1)
                            )
                        },
                        outputs=["output_data", "indices_data"],
                    )
                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, list[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 3, 3, 4000],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [9, 3, 3, 4000],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [6, 3, 3, 4000],
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 3, 24],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [9, 3, 24],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [6, 3, 24],
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 24],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [9, 24],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [6, 24],
                }
            else:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [24],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [25],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [24],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        self.trt_param.workspace_size = 1073741824
        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), 1e-3

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
