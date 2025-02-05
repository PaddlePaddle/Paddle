# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest
from functools import partial

import hypothesis.strategies as st
import numpy as np

sys.path.append("../../../ir/inference")
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestMulGruFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        x_col = draw(st.sampled_from([1]))
        y_col = draw(st.sampled_from([1]))
        activation = draw(st.sampled_from(['sigmoid', 'tanh']))
        is_reverse = draw(st.booleans())
        has_origin_mode = draw(st.booleans())
        origin_mode = False
        gate_activation = draw(st.sampled_from(['sigmoid', 'tanh']))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input():
            shape = [batch_size, 128, 6, 120]
            return np.full(shape, 0.001).astype(np.float32)

        def generate_weight(shape):
            return np.full(shape, 0.0001).astype(np.float32)

        im2sequence_op = OpConfig(
            type="im2sequence",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["seq_out"]},
            attrs={
                "kernels": [6, 1],
                "out_stride": [1, 1],
                "paddings": [0, 0, 0, 0],
                "strides": [1, 1],
            },
        )

        mul_op = OpConfig(
            type="mul",
            inputs={"X": ["seq_out"], "Y": ["mul_weight"]},
            outputs={"Out": ["mul_out"]},
            attrs={"x_num_col_dims": x_col, "y_num_col_dims": y_col},
        )

        if has_origin_mode:
            gru_op = OpConfig(
                type="gru",
                inputs={
                    "Input": ["mul_out"],
                    "Weight": ["gru_weight"],
                    "Bias": ["gru_bias"],
                },
                outputs={
                    "BatchGate": ["batch_gate"],
                    "BatchHidden": ["batch_hidden"],
                    "BatchResetHiddenPrev": ["batch_reset"],
                    "Hidden": ["hidden"],
                },
                attrs={
                    'activation': activation,
                    'is_reverse': is_reverse,
                    'gate_activation': gate_activation,
                    'is_test': True,
                    'origin_mode': origin_mode,
                },
            )
        else:
            gru_op = OpConfig(
                type="gru",
                inputs={
                    "Input": ["mul_out"],
                    "Weight": ["gru_weight"],
                    "Bias": ["gru_bias"],
                },
                outputs={
                    "BatchGate": ["batch_gate"],
                    "BatchHidden": ["batch_hidden"],
                    "BatchResetHiddenPrev": ["batch_reset"],
                    "Hidden": ["hidden"],
                },
                attrs={
                    'activation': activation,
                    'is_reverse': is_reverse,
                    'gate_activation': gate_activation,
                    'is_test': True,
                },
            )

        model_net = [im2sequence_op, mul_op, gru_op]

        program_config = ProgramConfig(
            ops=model_net,
            weights={
                "mul_weight": TensorConfig(
                    data_gen=partial(generate_weight, [768, 600])
                ),
                "gru_weight": TensorConfig(
                    data_gen=partial(generate_weight, [200, 600])
                ),
                "gru_bias": TensorConfig(
                    data_gen=partial(generate_weight, [1, 600])
                ),
            },
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["hidden"],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config()
        yield config, ["im2sequence", "fusion_gru"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False, max_duration=600, passes=["mul_gru_fuse_pass"]
        )


if __name__ == "__main__":
    unittest.main()
