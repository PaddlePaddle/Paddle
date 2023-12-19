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


class TestTransposeFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["transpose2"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        batch_size = draw(st.integers(min_value=1, max_value=4))
        C = draw(st.integers(min_value=1, max_value=64))
        H = draw(st.integers(min_value=1, max_value=64))
        W = draw(st.integers(min_value=1, max_value=64))
        in_shape = [batch_size, C, H, W]
        transpose_op1 = OpConfig(
            type='transpose2',
            inputs={
                "X": ["transpose_in"],
            },
            outputs={"Out": ["transpose_out1"]},
            attrs={"axis": [0, 2, 1, 3]},
        )

        transpose_op2 = OpConfig(
            type='transpose2',
            inputs={
                "X": ["transpose_out1"],
            },
            outputs={"Out": ["transpose_out2"]},
            attrs={"axis": [0, 3, 2, 1]},
        )

        transpose_op3 = OpConfig(
            type='transpose2',
            inputs={
                "X": ["transpose_out2"],
            },
            outputs={"Out": ["transpose_out3"]},
            attrs={"axis": [0, 1, 3, 2]},
        )

        ops = [
            transpose_op1,
            transpose_op2,
            transpose_op3,
        ]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "transpose_in": TensorConfig(shape=in_shape),
            },
            outputs=["transpose_out3"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["duplicated_transpose_fuse_pass"],
        )


if __name__ == "__main__":
    np.random.seed(200)
    unittest.main()
