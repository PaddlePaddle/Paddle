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


class TestXpuRedundantSqueezeUnsqueezeEliminationPass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["leaky_relu"], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        x_shape = draw(st.sampled_from([[1, 32, 1, 4]]))
        alpha = 0.009999999776482582
        axes = [2]
        squeeze_op = OpConfig(
            "squeeze2",
            inputs={
                "X": ["squeeze_input"],
            },
            outputs={"Out": ["squeeze_out"]},
            axes=axes,
        )
        leaky_relu_op = OpConfig(
            "leaky_relu",
            inputs={
                "X": ["squeeze_out"],
            },
            outputs={"Out": ["leaky_relu_out"]},
            alpha=alpha,
        )
        unsqueeze_op = OpConfig(
            "unsqueeze2",
            inputs={
                "X": ["leaky_relu_out"],
            },
            outputs={"Out": ["unsqueeze_out"]},
            axes=axes,
        )

        ops = [squeeze_op, leaky_relu_op, unsqueeze_op]

        def generate_data(shape):
            return np.random.random(shape).astype(np.float32)

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "squeeze_input": TensorConfig(
                    data_gen=partial(generate_data, x_shape)
                ),
            },
            weights={},
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            min_success_num=1,
            passes=["redundant_squeeze_unsqueeze_elimination_pass"],
        )


class TestXpuRedundantSqueezeUnsqueezeEliminationPass2(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["leaky_relu", "elementwise_add", "leaky_relu"], (
            1e-5,
            1e-5,
        )

    def sample_program_config(self, draw):
        x_shape = draw(st.sampled_from([[1, 32, 1, 4]]))
        alpha = 0.009999999776482582
        axes = [2]
        squeeze_op_1 = OpConfig(
            "squeeze2",
            inputs={
                "X": ["squeeze_1_input"],
            },
            outputs={"Out": ["squeeze_1_out"]},
            axes=axes,
        )
        leaky_relu_op_1 = OpConfig(
            "leaky_relu",
            inputs={
                "X": ["squeeze_1_out"],
            },
            outputs={"Out": ["leaky_relu_1_out"]},
            alpha=alpha,
        )
        squeeze_op_2 = OpConfig(
            "squeeze2",
            inputs={
                "X": ["squeeze_2_input"],
            },
            outputs={"Out": ["squeeze_2_out"]},
            axes=axes,
        )
        elementwise_add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["leaky_relu_1_out"], "Y": ["squeeze_2_out"]},
            outputs={"Out": ["elementwise_add_out"]},
        )
        leaky_relu_op_2 = OpConfig(
            "leaky_relu",
            inputs={
                "X": ["elementwise_add_out"],
            },
            outputs={"Out": ["leaky_relu_2_out"]},
            alpha=alpha,
        )
        unsqueeze_op_1 = OpConfig(
            "unsqueeze2",
            inputs={
                "X": ["leaky_relu_2_out"],
            },
            outputs={"Out": ["unsqueeze_1_out"]},
            axes=axes,
        )
        unsqueeze_op_2 = OpConfig(
            "unsqueeze2",
            inputs={
                "X": ["leaky_relu_2_out"],
            },
            outputs={"Out": ["unsqueeze_2_out"]},
            axes=axes,
        )

        ops = [
            squeeze_op_1,
            leaky_relu_op_1,
            squeeze_op_2,
            elementwise_add_op,
            leaky_relu_op_2,
            unsqueeze_op_1,
            unsqueeze_op_2,
        ]

        def generate_data(shape):
            return np.random.random(shape).astype(np.float32)

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "squeeze_1_input": TensorConfig(
                    data_gen=partial(generate_data, x_shape)
                ),
                "squeeze_2_input": TensorConfig(
                    data_gen=partial(generate_data, x_shape)
                ),
            },
            weights={},
            outputs=["unsqueeze_1_out", "unsqueeze_2_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            min_success_num=1,
            passes=["redundant_squeeze_unsqueeze_elimination_pass"],
        )


if __name__ == "__main__":
    unittest.main()
