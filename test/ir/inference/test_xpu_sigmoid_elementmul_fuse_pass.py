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
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestSigmoidElementmulFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["swish"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        # 1. sigmoid
        sigmoid_x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )

        sigmoid_op = OpConfig(
            "sigmoid",
            inputs={"X": ["sigmoid_x"]},
            outputs={"Out": ["sigmoid_out"]},
            trans_x=False,
            trans_y=False,
        )
        mul_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["sigmoid_x"], "Y": ["sigmoid_out"]},
            outputs={"Out": ["out"]},
            axis=-1,
        )
        ops = [sigmoid_op, mul_op]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "sigmoid_x": TensorConfig(shape=sigmoid_x_shape),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["sigmoid_elementmul_fuse_pass"],
        )


if __name__ == "__main__":
    np.random.seed(200)
    unittest.main()
