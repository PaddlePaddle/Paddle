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

from auto_scan_test import PassAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


class TestSimplifyWithBasicOpsPass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        scale = draw(st.floats(min_value=0.01, max_value=1.0))
        bias = draw(st.floats(min_value=0.01, max_value=2.0))
        bias_after_scale = draw(st.booleans())
        fix_seed = draw(st.booleans())
        dropout_implementation = "upscale_in_train"
        dropout_prob = draw(st.floats(min_value=0.0, max_value=1.0))
        seed = draw(st.integers(min_value=1, max_value=512))
        x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=2, max_size=4))
        is_test = True

        attrs = [{
            "fix_seed": fix_seed,
            "dropout_implementation": dropout_implementation,
            "dropout_prob": dropout_prob,
            "seed": seed,
            "is_test": is_test,
        }, {
            "scale": scale,
            "bias": bias,
            "bias_after_scale": bias_after_scale
        }]

        ops_config = [
            {
                #"op_type": "dropout",
                #"op_inputs": {
                #    "X": ["input_data"]
                #},
                #"op_outputs": {
                #    "Out": ["dropout_output"]
                #},
                #"op_attrs": {
                #    "fix_seed": attrs[0]['fix_seed'],
                #    "dropout_implementation": attrs[0]['dropout_implementation'],
                #    "dropout_prob": attrs[0]['dropout_prob'],
                #    "seed": attrs[0]['seed'],
                #    "is_test": attrs[0]['is_test']
                #},
            },
            {
                "op_type": "scale",
                "op_inputs": {
                    "X": ["input_data"],
                },
                "op_outputs": {
                    "Out": ["scale_output"]
                },
                "op_attrs": {
                    "scale": attrs[1]['scale'],
                    "bias": attrs[1]['bias'],
                    "bias_after_scale": attrs[1]['bias_after_scale']
                }
            }
        ]

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={"input_data": TensorConfig(shape=x_shape), },
            outputs=["scale_output"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_gpu=True)
        yield config, ['scale'], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=10,
            passes=["simplify_with_basic_ops_pass"],
            min_success_num=10)


if __name__ == "__main__":
    unittest.main()
