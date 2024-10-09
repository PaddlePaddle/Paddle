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


class TrtConvertExpandAsV2Test(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        return True

    def sample_program_configs(self):
        def generate_input1():
            return np.random.random([1, 4, 1]).astype(np.float32)

        def generate_input2():
            return np.random.random([1, 4, 10]).astype(np.float32)

        ops_config = [
            {
                "op_type": "expand_as_v2",
                "op_inputs": {
                    "X": ["expand_as_v2_input1"],
                    "Y": ["expand_as_v2_input2"],
                },
                "op_outputs": {"Out": ["expand_as_v2_out"]},
                "op_attrs": {"target_shape": [1, 4, 10]},
            },
        ]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "expand_as_v2_input1": TensorConfig(
                    data_gen=partial(generate_input1)
                ),
                "expand_as_v2_input2": TensorConfig(
                    data_gen=partial(generate_input2)
                ),
            },
            outputs=["expand_as_v2_out"],
        )

        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape():
            self.dynamic_shape.min_input_shape = {
                "expand_as_v2_input1": [1, 4, 1],
                "expand_as_v2_input2": [1, 4, 10],
            }
            self.dynamic_shape.max_input_shape = {
                "expand_as_v2_input1": [1, 4, 1],
                "expand_as_v2_input2": [1, 4, 10],
            }
            self.dynamic_shape.opt_input_shape = {
                "expand_as_v2_input1": [1, 4, 1],
                "expand_as_v2_input2": [1, 4, 10],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        clear_dynamic_shape()
        # for dynamic_shape
        generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (1, 3), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertExpandAsV2Test2(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        return True

    def sample_program_configs(self):
        def generate_input():
            return np.random.random([1, 4, 1]).astype(np.float32)

        ops_config = [
            {
                "op_type": "expand_as_v2",
                "op_inputs": {
                    "X": ["expand_as_v2_input1"],
                },
                "op_outputs": {"Out": ["expand_as_v2_out"]},
                "op_attrs": {"target_shape": [1, 4, 10]},
            },
        ]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "expand_as_v2_input1": TensorConfig(
                    data_gen=partial(generate_input)
                ),
            },
            outputs=["expand_as_v2_out"],
        )

        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape():
            self.dynamic_shape.min_input_shape = {
                "expand_as_v2_input1": [1, 4, 1],
            }
            self.dynamic_shape.max_input_shape = {
                "expand_as_v2_input1": [1, 4, 1],
            }
            self.dynamic_shape.opt_input_shape = {
                "expand_as_v2_input1": [1, 4, 1],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        clear_dynamic_shape()
        # for dynamic_shape
        generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
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
