# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from utils import dygraph_guard, static_guard

import paddle
from paddle import base


class TestGroupNormAPIV2_ParamInit(unittest.TestCase):
    def setUp(self):
        self.num_groups = 8
        self.num_channels = 16
        self.x_shape = [2, self.num_channels, 4, 4]
        self.param_shape = [self.num_channels]

    def _check_params(self, weight_np, bias_np, expected_weight, expected_bias):
        assert tuple(weight_np.shape) == tuple(self.param_shape)
        assert tuple(bias_np.shape) == tuple(self.param_shape)
        np.testing.assert_allclose(weight_np, expected_weight)
        np.testing.assert_allclose(bias_np, expected_bias)

    def test_dygraph(self):
        """test that weight_attr and bias_attr can override the default initialization when affine=True."""
        paddle.disable_static()
        with dygraph_guard():
            weight_val = 2.0
            bias_val = 3.0
            weight_attr = paddle.nn.initializer.Constant(value=weight_val)
            bias_attr = paddle.nn.initializer.Constant(value=bias_val)

            layer = paddle.nn.GroupNorm(
                num_groups=self.num_groups,
                num_channels=self.num_channels,
                affine=True,
                weight_attr=weight_attr,
                bias_attr=bias_attr,
            )

            expected_weight = np.full(self.param_shape, weight_val)
            expected_bias = np.full(self.param_shape, bias_val)
            self._check_params(
                layer.weight.numpy(),
                layer.bias.numpy(),
                expected_weight,
                expected_bias,
            )

    def test_static(self):
        """test that weight_attr and bias_attr can override the default initialization when affine=True."""
        paddle.enable_static()
        with static_guard():
            main = base.Program()
            start = base.Program()
            with (
                base.unique_name.guard(),
                base.program_guard(main, start),
            ):
                weight_val = 2.0
                bias_val = 3.0
                weight_attr = paddle.nn.initializer.Constant(value=weight_val)
                bias_attr = paddle.nn.initializer.Constant(value=bias_val)
                layer = paddle.nn.GroupNorm(
                    num_groups=self.num_groups,
                    num_channels=self.num_channels,
                    affine=True,
                    weight_attr=weight_attr,
                    bias_attr=bias_attr,
                )
            exe = base.Executor()
            exe.run(start)
            weight_np, bias_np = exe.run(
                main, fetch_list=[layer.weight, layer.bias]
            )

            expected_weight = np.full(self.param_shape, weight_val)
            expected_bias = np.full(self.param_shape, bias_val)
            self._check_params(
                weight_np, bias_np, expected_weight, expected_bias
            )


if __name__ == '__main__':
    unittest.main()
