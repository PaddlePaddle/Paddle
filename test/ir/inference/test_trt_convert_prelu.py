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


class TrtConvertPreluTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(attrs: list[dict[str, Any]], batch):
            if self.dims == 0:
                return np.random.random([]).astype(np.float32)
            elif self.dims == 1:
                return np.random.random([16]).astype(np.float32)
            elif self.dims == 2:
                return np.random.random([1, 3]).astype(np.float32)
            elif self.dims == 3:
                if attrs[0]["data_format"] == "NCHW":
                    return np.random.random([batch, 3, 16]).astype(np.float32)
                elif attrs[0]["data_format"] == "NHWC":
                    return np.random.random([batch, 16, 3]).astype(np.float32)
                else:
                    raise AssertionError
            else:
                if attrs[0]["data_format"] == "NCHW":
                    return np.random.random([batch, 3, 16, 32]).astype(
                        np.float32
                    )
                else:
                    return np.random.random([batch, 16, 32, 3]).astype(
                        np.float32
                    )

        def generate_alpha(attrs: list[dict[str, Any]]):
            if self.dims == 0:
                return np.random.random([]).astype(np.float32)
            if attrs[0]["mode"] == "all":
                return np.random.random([1]).astype(np.float32)
            elif attrs[0]["mode"] == "channel":
                return np.random.random([3]).astype(np.float32)
            elif attrs[0]["mode"] == "element":
                if self.dims == 1:
                    return np.random.random([16]).astype(np.float32)
                elif self.dims == 2:
                    return np.random.random([1, 3]).astype(np.float32)
                elif self.dims == 3:
                    if attrs[0]["data_format"] == "NCHW":
                        return np.random.random([1, 3, 16]).astype(np.float32)
                    elif attrs[0]["data_format"] == "NHWC":
                        return np.random.random([1, 16, 3]).astype(np.float32)
                    else:
                        raise AssertionError
                else:
                    if attrs[0]["data_format"] == "NCHW":
                        return np.random.random([1, 3, 16, 32]).astype(
                            np.float32
                        )
                    elif attrs[0]["data_format"] == "NHWC":
                        return np.random.random([1, 16, 32, 3]).astype(
                            np.float32
                        )
                    else:
                        raise AssertionError

        for batch in [1, 4]:
            for dims in [0, 1, 2, 3, 4]:
                for mode in ["all", "element", "channel"]:
                    for data_format in ["NCHW", "NHWC"]:
                        if (mode == "element" or mode == "all") and dims == 0:
                            continue
                        if mode == "channel" and dims != 4:
                            continue
                        self.dims = dims
                        dics = [{"mode": mode, "data_format": data_format}]
                        ops_config = [
                            {
                                "op_type": "prelu",
                                "op_inputs": {
                                    "X": ["input_data"],
                                    "Alpha": ["alpha_weight"],
                                },
                                "op_outputs": {"Out": ["output_data"]},
                                "op_attrs": dics[0],
                            }
                        ]
                        ops = self.generate_op_config(ops_config)

                        program_config = ProgramConfig(
                            ops=ops,
                            weights={
                                "alpha_weight": TensorConfig(
                                    data_gen=partial(generate_alpha, dics)
                                )
                            },
                            inputs={
                                "input_data": TensorConfig(
                                    data_gen=partial(
                                        generate_input, dics, batch
                                    )
                                ),
                            },
                            outputs=["output_data"],
                        )

                        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            if self.dims == 0:
                self.dynamic_shape.min_input_shape = {"input_data": []}
                self.dynamic_shape.max_input_shape = {"input_data": []}
                self.dynamic_shape.opt_input_shape = {"input_data": []}
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {"input_data": [16]}
                self.dynamic_shape.max_input_shape = {"input_data": [16]}
                self.dynamic_shape.opt_input_shape = {"input_data": [16]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 3]}
                self.dynamic_shape.max_input_shape = {"input_data": [1, 3]}
                self.dynamic_shape.opt_input_shape = {"input_data": [1, 3]}
            elif self.dims == 3:
                if attrs[0]["data_format"] == "NCHW":
                    self.dynamic_shape.min_input_shape = {
                        "input_data": [1, 3, 16]
                    }
                    self.dynamic_shape.max_input_shape = {
                        "input_data": [4, 3, 16]
                    }
                    self.dynamic_shape.opt_input_shape = {
                        "input_data": [1, 3, 16]
                    }
                elif attrs[0]["data_format"] == "NHWC":
                    self.dynamic_shape.min_input_shape = {
                        "input_data": [1, 16, 3]
                    }
                    self.dynamic_shape.max_input_shape = {
                        "input_data": [4, 16, 3]
                    }
                    self.dynamic_shape.opt_input_shape = {
                        "input_data": [1, 16, 3]
                    }
                else:
                    raise AssertionError
            else:
                if attrs[0]["data_format"] == "NCHW":
                    self.dynamic_shape.min_input_shape = {
                        "input_data": [1, 3, 16, 32]
                    }
                    self.dynamic_shape.max_input_shape = {
                        "input_data": [4, 3, 16, 32]
                    }
                    self.dynamic_shape.opt_input_shape = {
                        "input_data": [1, 3, 16, 32]
                    }
                elif attrs[0]["data_format"] == "NHWC":
                    self.dynamic_shape.min_input_shape = {
                        "input_data": [1, 16, 32, 3]
                    }
                    self.dynamic_shape.max_input_shape = {
                        "input_data": [4, 16, 32, 3]
                    }
                    self.dynamic_shape.opt_input_shape = {
                        "input_data": [1, 16, 32, 3]
                    }
                else:
                    raise AssertionError

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if not dynamic_shape and (self.dims == 1 or self.dims == 0):
                return 0, 3
            return 1, 2

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
        ), (1e-3, 1e-3)

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
        ), (1e-3, 1e-3)

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
