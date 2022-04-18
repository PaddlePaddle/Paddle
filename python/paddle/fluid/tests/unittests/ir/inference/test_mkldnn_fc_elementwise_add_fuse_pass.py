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

from auto_scan_test import PassAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


class TestFCElementwiseAddMkldnnFusePass(PassAutoScanTest):
    def sample_program_config(self, draw):
        axis = draw(st.sampled_from([-1, 0, 1]))
        fc_as_x = draw(st.sampled_from([True, False]))
        fc_in = draw(st.sampled_from([32, 64]))
        fc_wei = draw(st.sampled_from([32, 64]))

        def generate_input():
            return np.random.random([fc_in, fc_wei]).astype(np.float32)

        def generate_fc_weight():
            return np.random.random([fc_wei, fc_wei]).astype(np.float32)

        def generate_fc_bias():
            return np.random.random([fc_wei]).astype(np.float32)

        relu_op = OpConfig(
            type="relu",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["relu_out"]},
            attrs={})

        fc_op = OpConfig(
            type="fc",
            inputs={
                "Input": ["relu_out"],
                "W": ["fc_weight"],
                "Bias": ["fc_bias"]
            },
            outputs={"Out": ["fc_output"]},
            attrs={
                "use_mkldnn": True,
                "padding_weights": False,
                "activation_type": "",
                "in_num_col_dims": 1,
            })

        if fc_as_x:
            inputs = {"X": ["fc_output"], "Y": ["input_data"]}
        else:
            inputs = {"X": ["input_data"], "Y": ["fc_output"]}

        elt_add_op = OpConfig(
            type="elementwise_add",
            inputs=inputs,
            outputs={"Out": ["elementwise_output"]},
            attrs={'axis': axis})

        model_net = [relu_op, fc_op, elt_add_op]

        program_config = ProgramConfig(
            ops=model_net,
            weights={
                "fc_weight": TensorConfig(data_gen=partial(generate_fc_weight)),
                "fc_bias": TensorConfig(data_gen=partial(generate_fc_bias)),
            },
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["elementwise_output"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ["relu", "fc"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False, passes=["fc_elementwise_add_mkldnn_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
