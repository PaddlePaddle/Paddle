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


class TestRelu6FusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["relu6"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        # 1. clip
        clip_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )
        min_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=1), min_size=1, max_size=1
            )
        )

        def min_data():
            return np.full(min_shape, 0).astype(np.float32)

        def max_data():
            return np.full(min_shape, 6).astype(np.float32)

        clip_op = OpConfig(
            "clip",
            inputs={"X": ["clip_x"], "Min": ["min"], "Max": ["max"]},
            outputs={"Out": ["clip_out"]},
            min=0.0,
            max=6.0,
        )

        ops = [clip_op]

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "min": TensorConfig(data_gen=min_data),
                "max": TensorConfig(data_gen=max_data),
            },
            inputs={
                "clip_x": TensorConfig(shape=clip_shape),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["relu6_fuse_pass"],
        )


if __name__ == "__main__":
    np.random.seed(200)
    unittest.main()
