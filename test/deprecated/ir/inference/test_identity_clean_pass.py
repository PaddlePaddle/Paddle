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

import unittest

import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestIdentityScaleCleanPass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_gpu=True)
        yield config, ['relu', 'relu', 'scale'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        bias_after_scale = draw(st.booleans())
        n = draw(st.integers(min_value=1, max_value=4))
        c = draw(st.integers(min_value=1, max_value=20))
        h = draw(st.integers(min_value=1, max_value=20))
        w = draw(st.integers(min_value=1, max_value=20))

        relu_op1 = OpConfig(
            "relu", inputs={"X": ["relu_x"]}, outputs={"Out": ["relu_op1_out"]}
        )
        scale_op1 = OpConfig(
            "scale",
            inputs={"X": ["relu_op1_out"]},
            outputs={"Out": ["scale_op1_out"]},
            bias=0.0,
            scale=1.0,
            bias_after_scale=True,
        )
        scale_op2 = OpConfig(
            "scale",
            inputs={"X": ["scale_op1_out"]},
            outputs={"Out": ["scale_op2_out"]},
            bias=0.0,
            scale=1.0,
            bias_after_scale=True,
        )
        relu_op2 = OpConfig(
            "relu",
            inputs={"X": ["relu_op1_out"]},
            outputs={"Out": ["relu_op2_out"]},
        )
        program_config = ProgramConfig(
            ops=[relu_op1, relu_op2, scale_op1, scale_op2],
            weights={},
            inputs={"relu_x": TensorConfig(shape=[n, c, h, w])},
            outputs=["scale_op2_out", "relu_op2_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(max_examples=25, passes=["identity_op_clean_pass"])


class TestIdentityScaleCleanPass_V1(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_gpu=True)
        yield config, ['relu'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        bias_after_scale = draw(st.booleans())
        n = draw(st.integers(min_value=1, max_value=4))
        c = draw(st.integers(min_value=1, max_value=20))
        h = draw(st.integers(min_value=1, max_value=20))
        w = draw(st.integers(min_value=1, max_value=20))

        relu_op1 = OpConfig(
            "relu", inputs={"X": ["relu_x"]}, outputs={"Out": ["relu_op1_out"]}
        )
        scale_op1 = OpConfig(
            "scale",
            inputs={"X": ["relu_op1_out"]},
            outputs={"Out": ["scale_op1_out"]},
            bias=0.0,
            scale=1.0,
            bias_after_scale=True,
        )
        scale_op2 = OpConfig(
            "scale",
            inputs={"X": ["scale_op1_out"]},
            outputs={"Out": ["scale_op2_out"]},
            bias=0.0,
            scale=1.0,
            bias_after_scale=True,
        )
        program_config = ProgramConfig(
            ops=[relu_op1, scale_op1, scale_op2],
            weights={},
            inputs={"relu_x": TensorConfig(shape=[n, c, h, w])},
            outputs=["scale_op2_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(max_examples=25, passes=["identity_op_clean_pass"])


class TestIdentityScaleCleanPass_V2(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_gpu=True)
        yield config, ['scale', 'relu'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        bias_after_scale = draw(st.booleans())
        n = draw(st.integers(min_value=1, max_value=4))
        c = draw(st.integers(min_value=1, max_value=20))
        h = draw(st.integers(min_value=1, max_value=20))
        w = draw(st.integers(min_value=1, max_value=20))

        scale_op1 = OpConfig(
            "scale",
            inputs={"X": ["scale_op1_in"]},
            outputs={"Out": ["scale_op1_out"]},
            bias=0.0,
            scale=1.0,
            bias_after_scale=True,
        )
        scale_op2 = OpConfig(
            "scale",
            inputs={"X": ["scale_op1_out"]},
            outputs={"Out": ["scale_op2_out"]},
            bias=0.0,
            scale=1.0,
            bias_after_scale=True,
        )
        relu_op1 = OpConfig(
            "relu",
            inputs={"X": ["scale_op2_out"]},
            outputs={"Out": ["relu_op1_out"]},
        )
        program_config = ProgramConfig(
            ops=[scale_op1, scale_op2, relu_op1],
            weights={},
            inputs={"scale_op1_in": TensorConfig(shape=[n, c, h, w])},
            outputs=["relu_op1_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(max_examples=25, passes=["identity_op_clean_pass"])


class TestIdentityCastCleanPass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_gpu=True)
        yield config, ['relu', 'relu'], (1e-2, 1e-2)

    def sample_program_config(self, draw):
        n = draw(st.integers(min_value=1, max_value=4))
        c = draw(st.integers(min_value=1, max_value=20))
        h = draw(st.integers(min_value=1, max_value=20))
        w = draw(st.integers(min_value=1, max_value=20))

        relu_op_1 = OpConfig(
            "relu",
            inputs={"X": ["relu_op_1_in"]},
            outputs={"Out": ["relu_op_1_out"]},
        )
        cast_op_1 = OpConfig(
            "cast",
            inputs={"X": ["relu_op_1_out"]},
            outputs={"Out": ["cast_op_1_out"]},
            in_dtype=5,
            out_dtype=5,
        )
        relu_op_2 = OpConfig(
            "relu",
            inputs={"X": ["cast_op_1_out"]},
            outputs={"Out": ["relu_op_2_out"]},
        )
        cast_op_2 = OpConfig(
            "cast",
            inputs={"X": ["relu_op_2_out"]},
            outputs={"Out": ["cast_op_2_out"]},
            in_dtype=5,
            out_dtype=4,
        )
        cast_op_3 = OpConfig(
            "cast",
            inputs={"X": ["cast_op_2_out"]},
            outputs={"Out": ["cast_op_3_out"]},
            in_dtype=4,
            out_dtype=5,
        )

        program_config = ProgramConfig(
            ops=[relu_op_1, cast_op_1, relu_op_2, cast_op_2, cast_op_3],
            weights={},
            inputs={"relu_op_1_in": TensorConfig(shape=[n, c, h, w])},
            outputs=["cast_op_3_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(max_examples=25, passes=["identity_op_clean_pass"])


if __name__ == "__main__":
    unittest.main()
