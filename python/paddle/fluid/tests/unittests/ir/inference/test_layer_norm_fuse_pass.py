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

from auto_scan_test import PassAutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestFcFusePass(PassAutoScanTest):
    """
             x_var
              / \
             /   reduce_mean "u(x)"
             \   /
         elementwise_sub     "x - u(x)"
         /           \   sqr_pow_var(persistable) = 2
         |            \  /
         |      elementwise_pow  "(x - u(x))^2"
         |             |
         |       reduce_mean     "sigma^2 = 1/C*Sum{(x - u(x))^2}"
         |             |     eps_var(persistable)
         |             |     /
         |       elementwise_add "sigma^2 + epsilon"
         \             |
          \           sqrt       "sqrt(sigma^2 + epsilon)"
           \          /
            \        /
          elementwise_div        "lnorm = {x-u(x)}/{sqrt(sigma^2 + epsilon)}"
                 |
                 |  gamma_var(persistable)
                 |   /
          elementwise_mul        "scale: gamma(C) * lnorm"
                 |
                 |  beta_var(persistable)
                 |  /
          elementwise_add        "shift: gamma(C) * lnorm + beta(C)"
    """

    def sample_predictor_configs(self, program_config):
        # cpu
        config = self.create_inference_config(use_gpu=False)
        yield config, ["layer_norm"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        # Here we put some skip rules to avoid known bugs
        def teller1(program_config, predictor_config):
            x_shape = list(program_config.inputs["x"].shape)
            reduce_mean_dim = program_config.ops[0].attrs["dim"]
            if reduce_mean_dim[-1] != len(x_shape) - 1:
                return True
            for i in range(1, len(reduce_mean_dim)):
                if reduce_mean_dim[i] - reduce_mean_dim[i - 1] != 1:
                    return True
            return False

        self.add_ignore_check_case(
            teller1,
            IgnoreReasons.PASS_ACCURACY_ERROR,
            "Use bad case to test pass.", )

    def sample_program_config(self, draw):
        # 1. Generate shape of input:X 
        x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=4, max_size=5))
        x_shape_rank = len(x_shape)
        # 2. Generate attrs of reduce_mean
        keep_dim = draw(st.booleans())
        reduce_all = False
        begin_norm_axis = draw(
            st.integers(
                min_value=1, max_value=x_shape_rank - 1))
        if begin_norm_axis == x_shape_rank - 1 and draw(st.booleans()):
            reduce_mean_dim = [-1]
        else:
            reduce_mean_dim = [i for i in range(x_shape_rank)]
            reduce_mean_dim = reduce_mean_dim[begin_norm_axis:]
        error_test_ratio = draw(st.integers(min_value=1, max_value=10))
        if error_test_ratio > 9:
            keep_dim = True
            reduce_mean_dim = [1, ]
        elif error_test_ratio > 8:
            keep_dim = True
            begin_norm_axis = 1
            reduce_mean_dim = [1, x_shape_rank - 1]
        # 3. Generate attrs of elementwise_sub
        sub_axis = 0
        if keep_dim and draw(st.booleans()):
            sub_axis = -1
        # 4. Generate data of pow
        pow_axis = -1

        def generate_pow_data():
            return np.array([2, ], dtype="float32")

        # 5. Generate attrs of elementwise_add
        if keep_dim:
            add_axis = draw(
                st.integers(
                    min_value=-1, max_value=x_shape_rank - 1))
        else:
            add_axis = draw(
                st.integers(
                    min_value=-1, max_value=begin_norm_axis - 1))

        def generate_epsilon_data():
            return np.array([1e-5, ], dtype="float32")

        # 6. Generate attrs of elementwise_div
        div_axis = 0
        if keep_dim and draw(st.booleans()):
            sub_axis = -1
        # 6. Generate attrs gamma、beta
        mul_axis = -1
        if draw(st.booleans()):
            mul_axis = begin_norm_axis
        add_axis2 = -1
        if draw(st.booleans()):
            add_axis2 = begin_norm_axis
        gamma_shape = x_shape[begin_norm_axis:]
        beta_shape = gamma_shape[:]

        mean_op1 = OpConfig(
            "reduce_mean",
            inputs={"X": ["x"], },
            outputs={"Out": ["mean_out"]},
            dim=reduce_mean_dim,
            keep_dim=keep_dim,
            reduce_all=reduce_all, )
        sub_op = OpConfig(
            "elementwise_sub",
            inputs={"X": ["x"],
                    "Y": ["mean_out"]},
            outputs={"Out": ["sub_out"]},
            axis=sub_axis, )
        pow_op = OpConfig(
            "elementwise_pow",
            inputs={"X": ["sub_out"],
                    "Y": ["pow_y"]},
            outputs={"Out": ["pow_out"]},
            axis=pow_axis, )
        mean_op2 = OpConfig(
            "reduce_mean",
            inputs={"X": ["pow_out"], },
            outputs={"Out": ["mean_out2"]},
            dim=reduce_mean_dim,
            keep_dim=keep_dim,
            reduce_all=reduce_all, )
        add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["mean_out2"],
                    "Y": ["epsilon_var"]},
            outputs={"Out": ["add_out"]},
            axis=add_axis, )
        sqrt_op = OpConfig(
            "sqrt",
            inputs={"X": ["add_out"], },
            outputs={"Out": ["sqrt_out"]}, )
        div_op = OpConfig(
            "elementwise_div",
            inputs={"X": ["sub_out"],
                    "Y": ["sqrt_out"]},
            outputs={"Out": ["div_out"]},
            axis=div_axis, )
        mul_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["div_out"],
                    "Y": ["gamma_var"]},
            outputs={"Out": ["mul_out"]},
            axis=mul_axis, )
        add_op2 = OpConfig(
            "elementwise_add",
            inputs={"X": ["mul_out"],
                    "Y": ["beta_var"]},
            outputs={"Out": ["add_out2"]},
            axis=add_axis2, )

        ops = [
            mean_op1, sub_op, pow_op, mean_op2, add_op, sqrt_op, div_op, mul_op,
            add_op2
        ]

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "pow_y": TensorConfig(data_gen=generate_pow_data),
                "epsilon_var": TensorConfig(data_gen=generate_epsilon_data),
                "gamma_var": TensorConfig(shape=gamma_shape),
                "beta_var": TensorConfig(shape=beta_shape),
            },
            inputs={"x": TensorConfig(shape=x_shape), },
            outputs=ops[-1].outputs["Out"], )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=300,
            passes=["layer_norm_fuse_pass"], )


if __name__ == "__main__":
    unittest.main()
