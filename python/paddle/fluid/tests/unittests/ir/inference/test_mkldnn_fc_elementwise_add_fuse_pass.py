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
        fusing_mode = draw(st.sampled_from(["FC_as_X", "FC_as_Y"]))

        def generate_input():
            return np.random.random(
                [32, 64]).astype(np.float32)
        
        def generate_fc_weight():
            return np.random.random(
                [64, 64]).astype(np.float32)

        def generate_fc_bias():
            return np.random.random([64]).astype(np.float32)

        relu_op = OpConfig(
            type="relu",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["relu_out"]},
            attrs={}
        )

        fc_op = OpConfig(
            type="fc",
            inputs={"Input": ["relu_out"],
                    "W": ["fc_weight"],
                    "Bias": ["fc_bias"]},
            outputs={"Out": ["fc_output"]},
            attrs={
                "use_mkldnn": True,
                "padding_weights": False,
                "activation_type": "",
                "in_num_col_dims": 1,
                # 
            })

        if fusing_mode == "FC_as_X":
            elt_op = OpConfig(
                type="elementwise_add",
                inputs={"X": ["fc_output"],
                        "Y": ["relu_out"]},
                outputs={"Out": ["elementwise_output"]},
                attrs={'axis': axis})
        else:
            elt_op = OpConfig(
                type="elementwise_add",
                inputs={"X": ["relu_out"],
                        "Y": ["fc_output"]},
                outputs={"Out": ["elementwise_output"]},
                attrs={'axis': axis})

        relu2_op = OpConfig(
            type="relu",
            inputs={"X": ["elementwise_output"]},
            outputs={"Out": ["relu2_out"]},
            attrs={}
        )        

        model_net = [relu_op, fc_op, elt_op, relu2_op]

        program_config = ProgramConfig(
            ops=model_net,
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input))
            },
            weights={
                "fc_weight":
                TensorConfig(data_gen=partial(generate_fc_weight)),
                "fc_bias":
                TensorConfig(data_gen=partial(generate_fc_bias))
            },
            outputs=["relu2_out"])
        
        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True, passes=["fc_elementwise_add_mkldnn_fuse_pass"])
        yield config, ["relu", "fc", "relu"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False, passes=["fc_elementwise_add_mkldnn_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
