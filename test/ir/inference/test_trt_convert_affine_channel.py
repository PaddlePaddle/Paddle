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


class TrtConvertAffineChannelTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(batch, dims, attrs: list[dict[str, Any]]):
            if dims == 2:
                return np.ones([batch, 64]).astype(np.float32)
            else:
                if attrs[0]['data_layout'] == "NCHW":
                    return np.ones([batch, 3, 64, 64]).astype(np.float32)
                else:
                    return np.ones([batch, 64, 64, 3]).astype(np.float32)

        def generate_weight1(dims, attrs: list[dict[str, Any]]):
            if dims == 2:
                return np.random.random([64]).astype(np.float32)
            else:
                return np.random.random([3]).astype(np.float32)

        for dims in [2, 4]:
            for batch in [1, 2, 4]:
                for data_layout in ["NCHW", "NHWC"]:
                    self.dims = dims
                    dics = [{"data_layout": data_layout}]

                    ops_config = [
                        {
                            "op_type": "affine_channel",
                            "op_inputs": {
                                "X": ["input_data"],
                                "Scale": ["scale"],
                                "Bias": ["bias"],
                            },
                            "op_outputs": {"Out": ["output_data"]},
                            "op_attrs": dics[0],
                        }
                    ]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={
                            "scale": TensorConfig(
                                data_gen=partial(generate_weight1, dims, dics)
                            ),
                            "bias": TensorConfig(
                                data_gen=partial(generate_weight1, dims, dics)
                            ),
                        },
                        inputs={
                            "input_data": TensorConfig(
                                data_gen=partial(
                                    generate_input1, batch, dims, dics
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
            if self.dims == 2:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 32]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 64]}
                self.dynamic_shape.opt_input_shape = {"input_data": [2, 64]}
            else:
                if attrs[0]['data_layout'] == "NCHW":
                    self.dynamic_shape.min_input_shape = {
                        "input_data": [1, 3, 32, 32]
                    }
                    self.dynamic_shape.max_input_shape = {
                        "input_data": [4, 3, 64, 64]
                    }
                    self.dynamic_shape.opt_input_shape = {
                        "input_data": [1, 3, 64, 64]
                    }
                else:
                    self.dynamic_shape.min_input_shape = {
                        "input_data": [1, 32, 32, 3]
                    }
                    self.dynamic_shape.max_input_shape = {
                        "input_data": [4, 64, 64, 3]
                    }
                    self.dynamic_shape.opt_input_shape = {
                        "input_data": [1, 64, 64, 3]
                    }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if self.dims == 4 and attrs[0]['data_layout'] == "NCHW":
                return 1, 2
            else:
                return 0, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
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
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-3, 1e-3)

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
