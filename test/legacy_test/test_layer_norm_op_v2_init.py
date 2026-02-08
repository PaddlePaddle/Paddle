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


class TestLayerNormParamInit(unittest.TestCase):
    def setUp(self):
        self.normalized_shape = [6]
        self.x_shape = [2, 4, 4, 6]

    def test_dygraph(self):
        """test custom initialization using weight_attr and bias_attr."""
        paddle.disable_static()
        with dygraph_guard():
            weight_val = 2.5
            bias_val = -1.0
            weight_initializer = paddle.nn.initializer.Constant(
                value=weight_val
            )
            bias_initializer = paddle.nn.initializer.Constant(value=bias_val)

            layer = paddle.nn.LayerNorm(
                normalized_shape=self.normalized_shape,
                elementwise_affine=True,
                weight_attr=weight_initializer,
                bias_attr=bias_initializer,
            )

            expected_weight = np.full(self.normalized_shape, weight_val)
            expected_bias = np.full(self.normalized_shape, bias_val)

            np.testing.assert_allclose(layer.weight.numpy(), expected_weight)
            np.testing.assert_allclose(layer.bias.numpy(), expected_bias)

    def test_static(self):
        """test custom initialization in static graph mode."""
        paddle.enable_static()
        with static_guard():
            main = base.Program()
            start = base.Program()
            with (
                base.unique_name.guard(),
                base.program_guard(main, start),
            ):
                weight_val = 2.5
                bias_val = -1.0
                weight_initializer = paddle.nn.initializer.Constant(
                    value=weight_val
                )
                bias_initializer = paddle.nn.initializer.Constant(
                    value=bias_val
                )

                layer = paddle.nn.LayerNorm(
                    normalized_shape=self.normalized_shape,
                    elementwise_affine=True,
                    weight_attr=weight_initializer,
                    bias_attr=bias_initializer,
                )

            exe = base.Executor()
            exe.run(start)
            weight_np, bias_np = exe.run(
                main, fetch_list=[layer.weight, layer.bias]
            )

            expected_weight = np.full(self.normalized_shape, weight_val)
            expected_bias = np.full(self.normalized_shape, bias_val)

            np.testing.assert_allclose(weight_np, expected_weight)
            np.testing.assert_allclose(bias_np, expected_bias)


if __name__ == '__main__':
    unittest.main()
