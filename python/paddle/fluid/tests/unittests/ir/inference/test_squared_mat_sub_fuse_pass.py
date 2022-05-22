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

from auto_scan_test import PassAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


class TestSquaredMatSubFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        transpose_X = False
        transpose_Y = False
        alpha1 = 1.0
        alpha2 = 1.0
        axis1 = draw(st.sampled_from([-1, 0]))
        place_type = draw(st.sampled_from([-1, 0]))
        has_str_value = draw(st.booleans())
        str_value = ''
        value = draw(st.floats(min_value=-10, max_value=10))
        shape = draw(st.sampled_from([[1]]))
        axis2 = draw(st.sampled_from([-1, 0]))
        input_dim = draw(st.sampled_from([32, 64]))

        def generate_input(type):
            shape_x = [32, input_dim]
            shape_y = [input_dim, 16]

            if type == "x":
                return np.random.random(shape_x).astype(np.float32)
            else:
                return np.random.random(shape_y).astype(np.float32)

        matmul_op1 = OpConfig(
            type="matmul",
            inputs={"X": ["input_data1"],
                    "Y": ["input_data2"]},
            outputs={"Out": ["matmul1_output"]},
            attrs={
                "transpose_X": transpose_X,
                "transpose_Y": transpose_Y,
                "alpha": alpha1,
                "fused_reshape_X": [],
                "fused_reshape_Y": [],
                "fused_transpose_X": [],
                "fused_transpose_Y": [],
                "fused_reshape_Out": [],
                "fused_transpose_Out": []
            })

        square_op1 = OpConfig(
            type="square",
            inputs={"X": ["matmul1_output"]},
            outputs={"Out": ["square1_output"]},
            attrs={})

        square_op2 = OpConfig(
            type="square",
            inputs={"X": ["input_data1"]},
            outputs={"Out": ["square2_output"]},
            attrs={})

        square_op3 = OpConfig(
            type="square",
            inputs={"X": ["input_data2"]},
            outputs={"Out": ["square3_output"]},
            attrs={})

        matmul_op2 = OpConfig(
            type="matmul",
            inputs={"X": ["square2_output"],
                    "Y": ["square3_output"]},
            outputs={"Out": ["matmul2_output"]},
            attrs={
                "transpose_X": transpose_X,
                "transpose_Y": transpose_Y,
                "alpha": alpha2,
                "fused_reshape_X": [],
                "fused_reshape_Y": [],
                "fused_transpose_X": [],
                "fused_transpose_Y": [],
                "fused_reshape_Out": [],
                "fused_transpose_Out": []
            })

        elt_sub_op = OpConfig(
            type="elementwise_sub",
            inputs={"X": ["square1_output"],
                    "Y": ["matmul2_output"]},
            outputs={"Out": ["sub_out"]},
            attrs={"axis": axis1})

        if has_str_value:
            fill_constant_op = OpConfig(
                type="fill_constant",
                inputs={},
                outputs={"Out": ["constant_out"]},
                attrs={
                    "dtype": 5,
                    "place_type": place_type,
                    "str_value": str_value,
                    "value": value,
                    "shape": shape
                })
        else:
            fill_constant_op = OpConfig(
                type="fill_constant",
                inputs={},
                outputs={"Out": ["constant_out"]},
                attrs={
                    "dtype": 5,
                    "place_type": place_type,
                    "value": value,
                    "shape": shape
                })

        elt_mul_op = OpConfig(
            type="elementwise_mul",
            inputs={"X": ["sub_out"],
                    "Y": ["constant_out"]},
            outputs={"Out": ["mul_out"]},
            attrs={"axis": axis2})

        model_net = [
            matmul_op1, square_op1, square_op2, square_op3, matmul_op2,
            elt_sub_op, fill_constant_op, elt_mul_op
        ]

        program_config = ProgramConfig(
            ops=model_net,
            weights={},
            inputs={
                "input_data1":
                TensorConfig(data_gen=partial(generate_input, "x")),
                "input_data2":
                TensorConfig(data_gen=partial(generate_input, "y"))
            },
            outputs=["mul_out"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config()
        yield config, ["fusion_squared_mat_sub"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(quant=False, passes=["squared_mat_sub_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
