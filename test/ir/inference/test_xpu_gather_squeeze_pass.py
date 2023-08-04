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
        yield config, [
            "transpose2",
            "gather",
            "transpose2",
            "gather",
            "squeeze2",
            "squeeze2",
        ], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=3, max_size=3
            )
        )

        def generate_data(shape):
            return np.random.random(shape).astype(np.float32)

        def generate_index(*args, **kwargs):
            return np.array([0]).astype(np.int64)

        axis = 2
        axes = [2]
        gather_op0 = OpConfig(
            "gather",
            inputs={"X": ["gather_in"], "Index": ["gather_index0"]},
            outputs={"Out": ["gather_out0"]},
            axis=axis,
        )

        gather_op1 = OpConfig(
            "gather",
            inputs={"X": ["gather_in"], "Index": ["gather_index1"]},
            outputs={"Out": ["gather_out1"]},
            axis=axis,
        )

        squeeze_op0 = OpConfig(
            "squeeze2",
            inputs={
                "X": ["gather_out0"],
            },
            outputs={"Out": ["squeeze_out0"]},
            axes=axes,
        )

        squeeze_op1 = OpConfig(
            "squeeze2",
            inputs={
                "X": ["gather_out1"],
            },
            outputs={"Out": ["squeeze_out1"]},
            axes=axes,
        )

        ops = [gather_op0, gather_op1, squeeze_op0, squeeze_op1]

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "gather_in": TensorConfig(
                    data_gen=partial(generate_data, x_shape)
                ),
                "gather_index0": TensorConfig(data_gen=partial(generate_index)),
                "gather_index1": TensorConfig(data_gen=partial(generate_index)),
            },
            weights={},
            outputs=["squeeze_out0", "squeeze_out1"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False, max_examples=25, passes=["gather_squeeze_pass"]
        )


if __name__ == "__main__":
    unittest.main()
