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


class TestFcXPUFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["fc_xpu", "fc_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        # 1. matmul0
        matmul0_x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )
        matmul0_y_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=8), min_size=2, max_size=2
            )
        )
        matmul0_y_shape[0] = matmul0_x_shape[-1]
        # 2. add0
        add0_bias_shape = [matmul0_y_shape[1]]
        # 3. matmul1
        matmul1_y_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=8), min_size=2, max_size=2
            )
        )
        matmul1_y_shape[0] = matmul0_y_shape[-1]
        # 4. add1
        add1_bias_shape = [matmul1_y_shape[1]]

        matmul0_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["matmul0_x"], "Y": ["matmul0_y"]},
            outputs={"Out": ["matmul0_out"]},
            trans_x=False,
            trans_y=False,
        )
        add0_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["matmul0_out"], "Y": ["add0_bias"]},
            outputs={"Out": ["add0_out"]},
            axis=-1,
        )
        matmul1_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["add0_out"], "Y": ["matmul1_y"]},
            outputs={"Out": ["matmul1_out"]},
            trans_x=False,
            trans_y=False,
        )
        add1_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["matmul1_out"], "Y": ["add1_bias"]},
            outputs={"Out": ["add1_out"]},
            axis=-1,
        )
        ops = [matmul0_op, add0_op, matmul1_op, add1_op]

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "matmul0_y": TensorConfig(shape=matmul0_y_shape),
                "add0_bias": TensorConfig(shape=add0_bias_shape),
                "matmul1_y": TensorConfig(shape=matmul1_y_shape),
                "add1_bias": TensorConfig(shape=add1_bias_shape),
            },
            inputs={
                "matmul0_x": TensorConfig(shape=matmul0_x_shape),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["fc_xpu_fuse_pass", "link_xpu_op_max_pass"],
        )


if __name__ == "__main__":
    unittest.main()
