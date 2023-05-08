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


class TestDeleteRepeatedShapePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ['shape', 'cast', 'cast', 'cast'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=20), min_size=2, max_size=4
            )
        )

        shape_op0 = OpConfig(
            "shape",
            inputs={
                "Input": ["shape_x"],
            },
            outputs={"Out": ["shape0_out"]},
        )
        cast_op0 = OpConfig(
            "cast",
            inputs={
                "X": ["shape0_out"],
            },
            in_dtype=2,
            out_dtype=5,
            outputs={"Out": ["cast0_out"]},
        )
        shape_op1 = OpConfig(
            "shape",
            inputs={
                "Input": ["shape_x"],
            },
            outputs={"Out": ["shape1_out"]},
        )
        cast_op1 = OpConfig(
            "cast",
            inputs={
                "X": ["shape1_out"],
            },
            in_dtype=2,
            out_dtype=5,
            outputs={"Out": ["cast1_out"]},
        )
        shape_op2 = OpConfig(
            "shape",
            inputs={
                "Input": ["shape_x"],
            },
            outputs={"Out": ["shape2_out"]},
        )
        cast_op2 = OpConfig(
            "cast",
            inputs={
                "X": ["shape2_out"],
            },
            in_dtype=2,
            out_dtype=5,
            outputs={"Out": ["cast2_out"]},
        )
        ops = [shape_op0, cast_op0, shape_op1, cast_op1, shape_op2, cast_op2]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "shape_x": TensorConfig(shape=x_shape),
            },
            outputs=["cast0_out", "cast1_out", "cast2_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["delete_repeated_ops_pass"],
        )


class TestDeleteRepeatedSlicePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ['slice', 'cast', 'cast', 'cast'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        slice_x = draw(
            st.lists(
                st.integers(min_value=1, max_value=20), min_size=2, max_size=4
            )
        )

        slice_op0 = OpConfig(
            "slice",
            inputs={
                "Input": ["slice_x"],
            },
            starts=[0],
            ends=[1],
            axes=[0],
            decrease_axis=[0],
            outputs={"Out": ["slice0_out"]},
        )
        cast_op0 = OpConfig(
            "cast",
            inputs={
                "X": ["slice0_out"],
            },
            in_dtype=5,
            out_dtype=5,
            outputs={"Out": ["cast0_out"]},
        )
        slice_op1 = OpConfig(
            "slice",
            inputs={
                "Input": ["slice_x"],
            },
            starts=[0],
            ends=[1],
            axes=[0],
            decrease_axis=[0],
            outputs={"Out": ["slice1_out"]},
        )
        cast_op1 = OpConfig(
            "cast",
            inputs={
                "X": ["slice1_out"],
            },
            in_dtype=5,
            out_dtype=5,
            outputs={"Out": ["cast1_out"]},
        )
        slice_op2 = OpConfig(
            "slice",
            inputs={
                "Input": ["slice_x"],
            },
            starts=[0],
            ends=[1],
            axes=[0],
            decrease_axis=[0],
            outputs={"Out": ["slice2_out"]},
        )
        cast_op2 = OpConfig(
            "cast",
            inputs={
                "X": ["slice2_out"],
            },
            in_dtype=5,
            out_dtype=5,
            outputs={"Out": ["cast2_out"]},
        )
        ops = [slice_op0, cast_op0, slice_op1, cast_op1, slice_op2, cast_op2]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "slice_x": TensorConfig(shape=slice_x),
            },
            outputs=["cast0_out", "cast1_out", "cast2_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["delete_repeated_ops_pass"],
        )


if __name__ == "__main__":
    unittest.main()
