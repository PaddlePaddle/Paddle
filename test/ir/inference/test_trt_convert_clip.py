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
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertClipTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(dims, batch, dtype, attrs: list[dict[str, Any]]):
            if dims == 0:
                return np.ones([]).astype(dtype)
            elif dims == 1:
                return np.ones([32]).astype(dtype)
            elif dims == 2:
                return np.ones([3, 32]).astype(dtype)
            elif dims == 3:
                return np.ones([3, 32, 32]).astype(dtype)
            else:
                return np.ones([batch, 3, 32, 32]).astype(dtype)

        def generate_weight1(attrs: list[dict[str, Any]]):
            return np.array([np.random.uniform(1, 10)]).astype("float32")

        def generate_weight2(attrs: list[dict[str, Any]]):
            return np.array([np.random.uniform(10, 20)]).astype("float32")

        for dims in [0, 1, 2, 3, 4]:
            for batch in [1, 4]:
                for dtype in [np.float32, np.int32]:
                    for op_inputs in [
                        {"X": ["input_data"]},
                        {"X": ["input_data"], "Min": ["Min_"], "Max": ["Max_"]},
                    ]:
                        self.input_num = len(op_inputs)
                        self.dims = dims
                        dics = [
                            {
                                "min": np.random.uniform(1, 10),
                                "max": np.random.uniform(10, 20),
                            },
                            {"op_inputs": op_inputs},
                        ]
                        ops_config = [
                            {
                                "op_type": "clip",
                                "op_inputs": op_inputs,
                                "op_outputs": {"Out": ["output_data"]},
                                "op_attrs": dics[0],
                            }
                        ]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(
                            ops=ops,
                            weights={
                                "Min_": TensorConfig(
                                    data_gen=partial(generate_weight1, dics)
                                ),
                                "Max_": TensorConfig(
                                    data_gen=partial(generate_weight2, dics)
                                ),
                            },
                            inputs={
                                "input_data": TensorConfig(
                                    data_gen=partial(
                                        generate_input1,
                                        dims,
                                        batch,
                                        dtype,
                                        dics,
                                    )
                                )
                            },
                            outputs=["output_data"],
                        )

                        yield program_config

    def generate_dynamic_shape(self):
        if self.dims == 0:
            self.dynamic_shape.min_input_shape = {"input_data": []}
            self.dynamic_shape.max_input_shape = {"input_data": []}
            self.dynamic_shape.opt_input_shape = {"input_data": []}
        elif self.dims == 1:
            self.dynamic_shape.min_input_shape = {"input_data": [1]}
            self.dynamic_shape.max_input_shape = {"input_data": [64]}
            self.dynamic_shape.opt_input_shape = {"input_data": [32]}
        elif self.dims == 2:
            self.dynamic_shape.min_input_shape = {"input_data": [1, 16]}
            self.dynamic_shape.max_input_shape = {"input_data": [4, 32]}
            self.dynamic_shape.opt_input_shape = {"input_data": [3, 32]}
        elif self.dims == 3:
            self.dynamic_shape.min_input_shape = {"input_data": [1, 16, 16]}
            self.dynamic_shape.max_input_shape = {"input_data": [4, 32, 32]}
            self.dynamic_shape.opt_input_shape = {"input_data": [3, 32, 32]}
        else:
            self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 16, 16]}
            self.dynamic_shape.max_input_shape = {"input_data": [4, 3, 32, 32]}
            self.dynamic_shape.opt_input_shape = {"input_data": [1, 3, 32, 32]}
        return self.dynamic_shape

    def sample_predictor_configs(self, program_config, run_pir=False):
        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape and self.dims != 0 and self.input_num != 3:
                return 1, 2
            else:
                return 0, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        if not run_pir:
            clear_dynamic_shape()
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            yield self.create_inference_config(), generate_trt_nodes_num(
                attrs, False
            ), 1e-5
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            yield self.create_inference_config(), generate_trt_nodes_num(
                attrs, False
            ), (1e-3, 1e-3)

        # for dynamic_shape
        self.generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-3, 1e-3)

    def test(self):
        # test for old ir
        self.run_test()
        # test for pir
        self.run_test(run_pir=True)


if __name__ == "__main__":
    unittest.main()
