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
"""Test for fusion of subgraph expressing layer normalization."""

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from inference_pass_test import InferencePassTest
from paddle import enable_static
from paddle.fluid.core import PassVersionChecker


class LayerNormFusePassTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[3, 64, 120], dtype="float32")
            sqr_pow = fluid.layers.fill_constant(
                shape=[1], value=2, dtype="float32")
            eps = fluid.layers.fill_constant(
                shape=[1], value=1e-5, dtype="float32")
            gamma = fluid.layers.create_parameter(
                shape=[120], dtype="float32", is_bias=True)
            beta = fluid.layers.create_parameter(
                shape=[120], dtype="float32", is_bias=True)

            x_mean_out = fluid.layers.reduce_mean(data, dim=-1, keep_dim=True)
            x_sub_mean_out = fluid.layers.elementwise_sub(data, x_mean_out)
            x_sub_mean_sqr_out = fluid.layers.elementwise_pow(x_sub_mean_out,
                                                              sqr_pow)
            std_dev_out = fluid.layers.reduce_mean(
                x_sub_mean_sqr_out, dim=-1, keep_dim=True)
            std_dev_eps_out = fluid.layers.elementwise_add(std_dev_out, eps)
            std_dev_eps_sqrt_out = fluid.layers.sqrt(std_dev_eps_out)
            division_out = fluid.layers.elementwise_div(x_sub_mean_out,
                                                        std_dev_eps_sqrt_out)
            scale_out = fluid.layers.elementwise_mul(division_out, gamma)
            shift_out = fluid.layers.elementwise_add(scale_out, beta)

        self.feeds = {
            "data": np.random.random((3, 64, 120)).astype("float32"),
        }
        self.fetch_list = [shift_out]

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)
        self.assertTrue(PassVersionChecker.IsCompatible("layer_norm_fuse_pass"))


if __name__ == "__main__":
    enable_static()
    unittest.main()
