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


class TestDeleteConcatOpPass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["relu"], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )

        relu_op = OpConfig(
            "relu",
            inputs={
                "X": ["relu_x"],
            },
            outputs={"Out": ["relu_out"]},
        )
        concat_op = OpConfig(
            "concat",
            inputs={
                "X": ["relu_out"],
            },
            axis=0,
            outputs={"Out": ["concat_out"]},
        )
        ops = [relu_op, concat_op]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "relu_x": TensorConfig(shape=x_shape),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["delete_concat_op_pass"],
        )


if __name__ == "__main__":
    unittest.main()
