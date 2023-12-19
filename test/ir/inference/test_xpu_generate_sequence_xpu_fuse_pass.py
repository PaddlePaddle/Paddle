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


class TestGenerateSequenceXPUFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["generate_sequence_xpu"], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        fill_any_like_x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=32), min_size=2, max_size=2
            )
        )
        fill_any_like_dtype = draw(st.sampled_from([2, 3, 5]))

        fill_any_like_op = OpConfig(
            "fill_any_like",
            inputs={"X": ["fill_any_like_x"]},
            outputs={"Out": ["fill_any_like_out"]},
            dtype=fill_any_like_dtype,
            value=1.0,
        )
        cumsum_op = OpConfig(
            "cumsum",
            inputs={"X": ["fill_any_like_out"]},
            outputs={"Out": ["cumsum_out"]},
            axis=1,
            exclusive=False,
            flatten=False,
            reverse=False,
        )
        elementwise_sub_op = OpConfig(
            "elementwise_sub",
            inputs={"X": ["cumsum_out"], "Y": ["fill_any_like_out"]},
            outputs={"Out": ["elementwise_sub_out"]},
            axis=-1,
        )

        ops = [fill_any_like_op, cumsum_op, elementwise_sub_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "fill_any_like_x": TensorConfig(shape=fill_any_like_x_shape),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["generate_sequence_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
