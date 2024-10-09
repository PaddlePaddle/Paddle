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
from typing import Any

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertFusedTokenPruneTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_attn_or_mask(attrs: list[dict[str, Any]]):
            return np.ones([4, 12, 64, 64]).astype(np.float32)

        def generate_x(attrs: list[dict[str, Any]]):
            return np.random.random([4, 64, 76]).astype(np.float32)

        def generate_new_mask(attrs: list[dict[str, Any]]):
            return np.random.random([4, 12, 32, 32]).astype(np.float32)

        for keep_first_token in [True, False]:
            for keep_order in [True, False]:
                dics = [
                    {
                        "keep_first_token": keep_first_token,
                        "keep_order": keep_order,
                    }
                ]
                ops_config = [
                    {
                        "op_type": "fused_token_prune",
                        "op_inputs": {
                            "Attn": ["attn"],
                            "X": ["x"],
                            "Mask": ["mask"],
                            "NewMask": ["new_mask"],
                        },
                        "op_outputs": {
                            "SlimmedX": ["slimmed_x"],
                            "CLSInds": ["cls_inds"],
                        },
                        "op_attrs": dics[0],
                    }
                ]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "attn": TensorConfig(
                            data_gen=partial(generate_attn_or_mask, dics)
                        ),
                        "x": TensorConfig(data_gen=partial(generate_x, dics)),
                        "mask": TensorConfig(
                            data_gen=partial(generate_attn_or_mask, dics)
                        ),
                        "new_mask": TensorConfig(
                            data_gen=partial(generate_new_mask, dics)
                        ),
                    },
                    outputs=["slimmed_x", "cls_inds"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "attn": [4, 12, 64, 64],
                "x": [4, 64, 76],
                "mask": [4, 12, 64, 64],
                "new_mask": [4, 12, 32, 32],
            }
            self.dynamic_shape.max_input_shape = {
                "attn": [4, 12, 64, 64],
                "x": [4, 64, 76],
                "mask": [4, 12, 64, 64],
                "new_mask": [4, 12, 32, 32],
            }
            self.dynamic_shape.opt_input_shape = {
                "attn": [4, 12, 64, 64],
                "x": [4, 64, 76],
                "mask": [4, 12, 64, 64],
                "new_mask": [4, 12, 32, 32],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 6

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-2, 1e-2)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-1, 1e-2)

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
