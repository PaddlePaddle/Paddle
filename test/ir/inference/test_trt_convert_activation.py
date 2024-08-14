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


class TrtConvertActivationTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[0] * 10 < 8200:
            if program_config.ops[0].type == "round":
                return False
        return True

    def sample_program_configs(self):
        def generate_input(attrs: list[dict[str, Any]]):
            if self.dims == 0:
                return np.random.random([]).astype(np.float32)
            else:
                return np.random.random([1, 3, 32, 32]).astype(np.float32)

        for dims in [0, 4]:
            self.dims = dims
            for op_type in [
                "relu",
                "sigmoid",
                "relu6",
                "elu",
                "selu",
                "silu",
                "softsign",
                "stanh",
                "thresholded_relu",
                "celu",
                "logsigmoid",
                "tanh_shrink",
                "softplus",
                "hard_swish",
                "hard_sigmoid",
                "leaky_relu",
            ]:
                # few samples to reduce time
                # for beta in [-0.2, 0.5, 0.67, 3]:
                #    for alpha in [-0.2, 0.5, 0.67, 3]:
                for beta in [0.67]:
                    for alpha in [0.67]:
                        dics = [{}]
                        if op_type == "celu":
                            dics = [{"alpha": 1.0}]
                        if op_type == "elu":
                            dics = [{"alpha": alpha}]
                        if op_type == "selu":
                            dics = [{"alpha": beta, "scale": alpha}]
                        if op_type == "stanh":
                            dics = [{"scale_a": beta, "scale_b": alpha}]
                        if op_type == "thresholded_relu":
                            dics = [{"threshold": alpha}]
                        if op_type == "softplus":
                            dics = [{"beta": beta}]
                        if op_type == "hard_swish":
                            dics = [
                                {
                                    "threshold": 6.0,
                                    "scale": 6.0,
                                    "offset": 3.0,
                                }
                            ]
                        if op_type == "hard_sigmoid":
                            dics = [{"slope": beta, "offset": alpha}]
                        if op_type == "leaky_relu":
                            dics = [{"alpha": alpha}]

                        ops_config = [
                            {
                                "op_type": op_type,
                                "op_inputs": {"X": ["input_data"]},
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
                                    data_gen=partial(generate_input, dics)
                                )
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
            else:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 3, 16, 16]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 3, 32, 32]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 3, 32, 32]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if not dynamic_shape and self.dims == 0:
                return 0, 3
            runtime_version = paddle_infer.get_trt_runtime_version()
            if (
                runtime_version[0] * 1000
                + runtime_version[1] * 100
                + runtime_version[2] * 10
                < 8600
                and self.dims == 0
            ) and program_config.ops[0].type in [
                "celu",
                "logsigmoid",
                "tanh_shrink",
            ]:
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
        ), 1e-3

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
        ), 1e-3

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
