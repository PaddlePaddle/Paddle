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
from trt_layer_auto_scan_test import SkipReasons, TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertPadTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        if attrs[0]['pad_value'] != 0.0:
            return False
        for x in attrs[0]['paddings']:
            if x < 0:
                return False

        return True

    def sample_program_configs(self):
        def generate_input1(attrs: list[dict[str, Any]]):
            return np.ones([1, 3, 64, 64]).astype(np.float32)

        def generate_weight1(attrs: list[dict[str, Any]]):
            return np.random.random([24, 3, 3, 3]).astype(np.float32)

        for pad_value in [0.0, 1.0, 2.0, -100, 100.0]:
            for paddings in [
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 2, 3, 4],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, -1, -1, 1, 1],
            ]:
                dics = [{"pad_value": pad_value, "paddings": paddings}, {}]

                ops_config = [
                    {
                        "op_type": "pad",
                        "op_inputs": {"X": ["input_data"]},
                        "op_outputs": {"Out": ["pad_output_data"]},
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
                    outputs=["pad_output_data"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 32, 32]}
            self.dynamic_shape.max_input_shape = {"input_data": [4, 3, 64, 64]}
            self.dynamic_shape.opt_input_shape = {"input_data": [1, 3, 64, 64]}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            for x in range(len(program_config.ops[0].attrs['paddings']) - 4):
                if program_config.ops[0].attrs['paddings'][x] != 0:
                    return 0, 3
            return 1, 2

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
        ), 1e-2

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
        ), 1e-2

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            for x in range(len(program_config.ops[0].attrs['paddings']) - 4):
                if program_config.ops[0].attrs['paddings'][x] != 0:
                    return True
            return False

        self.add_skip_case(
            teller1,
            SkipReasons.TRT_NOT_IMPLEMENTED,
            "NOT Implemented: we need to add support pad not only implement on h or w, such as paddings = [0, 0, 1, 1, 1, 1, 1, 1]",
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
