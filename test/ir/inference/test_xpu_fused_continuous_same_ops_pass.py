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

import unittest

import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestFusedSameUnSqueezePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ['scale', 'unsqueeze2'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        scale_x = draw(
            st.lists(
                st.integers(min_value=1, max_value=20), min_size=1, max_size=3
            )
        )
        first_unsqueeze_axis = 0
        second_unsqueeze_axis = 1
        third_unsqueeze_axis = 2
        scale_op0 = OpConfig(
            "scale",
            inputs={
                "X": ["scale_x"],
            },
            scale=2.0,
            bias=1.0,
            bias_after_scale=True,
            outputs={"Out": ["scale0_out"]},
        )
        unsqueeze_op0 = OpConfig(
            "unsqueeze2",
            inputs={
                "X": ["scale0_out"],
            },
            axes=[first_unsqueeze_axis],
            outputs={"Out": ["unsqueeze0_out"]},
        )
        unsqueeze_op1 = OpConfig(
            "unsqueeze2",
            inputs={
                "X": ["unsqueeze0_out"],
            },
            axes=[second_unsqueeze_axis],
            outputs={"Out": ["unsqueeze1_out"]},
        )
        unsqueeze_op2 = OpConfig(
            "unsqueeze2",
            inputs={
                "X": ["unsqueeze1_out"],
            },
            axes=[third_unsqueeze_axis],
            outputs={"Out": ["unsqueeze2_out"]},
        )
        ops = [scale_op0, unsqueeze_op0, unsqueeze_op1, unsqueeze_op2]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "scale_x": TensorConfig(shape=scale_x),
            },
            outputs=["unsqueeze2_out"],
        )
        return program_config


class TestFusedSameReshapePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ['scale', 'reshape2'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        scale_x = draw(
            st.sampled_from([[8, 16], [16, 32], [64, 16], [16, 8], [16, 16]])
        )
        first_reshape_shape = [-1, 16, 4]
        second_reshape_shape = [-1, 8]
        scale_op0 = OpConfig(
            "scale",
            inputs={
                "X": ["scale_x"],
            },
            scale=2.0,
            bias=1.0,
            bias_after_scale=True,
            outputs={"Out": ["scale0_out"]},
        )
        reshape_op0 = OpConfig(
            "reshape2",
            inputs={
                "X": ["scale0_out"],
            },
            shape=first_reshape_shape,
            outputs={"Out": ["reshape0_out"]},
        )
        reshape_op1 = OpConfig(
            "reshape2",
            inputs={
                "X": ["reshape0_out"],
            },
            shape=second_reshape_shape,
            outputs={"Out": ["reshape1_out"]},
        )
        ops = [scale_op0, reshape_op0, reshape_op1]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "scale_x": TensorConfig(shape=scale_x),
            },
            outputs=["reshape1_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            min_success_num=5,
            passes=["fused_continuous_same_ops_pass"],
        )


if __name__ == "__main__":
    unittest.main()
