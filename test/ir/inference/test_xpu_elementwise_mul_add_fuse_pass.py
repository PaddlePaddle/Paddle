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
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestGatherAddTransposePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["addcmul_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=3, max_size=4
            )
        )

        def generate_data(shape):
            return np.random.random(shape).astype(np.float32)

        mul_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["mul_x"], "Y": ["mul_y"]},
            outputs={"Out": ["mul_out"]},
        )

        add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["mul_out"], "Y": ["add_w"]},
            outputs={"Out": ["add_out"]},
        )

        ops = [mul_op, add_op]

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "mul_x": TensorConfig(data_gen=partial(generate_data, x_shape)),
                "mul_y": TensorConfig(data_gen=partial(generate_data, x_shape)),
                "add_w": TensorConfig(data_gen=partial(generate_data, x_shape)),
            },
            weights={},
            outputs=["add_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["elementwise_mul_add_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
