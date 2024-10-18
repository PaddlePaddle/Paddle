# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import utils

import paddle
from paddle.base import core


class GroupNormSubGraph(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias, data_format):
        return paddle.nn.functional.group_norm(
            x,
            num_groups=32,
            epsilon=1e-6,
            weight=weight,
            bias=bias,
            data_format=data_format,
        )


class TestGroupNormSubGraph(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.shape = [80, 128, 256, 128]
        self.dtype = "bfloat16"
        self.data_format = "NHWC"
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randn(self.shape, dtype=self.dtype)
        self.x.stop_gradient = False
        self.weight = paddle.randn([128], dtype=self.dtype)
        self.weight.stop_gradient = False
        self.bias = paddle.randn([128], dtype=self.dtype)
        self.bias.stop_gradient = False

    def eval(self, use_cinn, use_prim=False):
        if self.x.grad is not None:
            self.x.clear_grad()
            self.weight.clear_grad()
            self.bias.clear_grad()

        if use_prim:
            core._set_prim_all_enabled(True)
        net = GroupNormSubGraph()
        net = utils.apply_to_static(net, use_cinn=use_cinn)
        out = net(self.x, self.weight, self.bias, self.data_format)
        loss = out.sum()
        loss.backward()

        core._set_prim_all_enabled(False)
        return (
            out,
            self.x.gradient(),
            self.weight.gradient(),
            self.bias.gradient(),
        )

    def _test_cinn(self):
        cinn_out, cinn_x_grad, cinn_weight_grad, cinn_bias_grad = self.eval(
            use_cinn=True, use_prim=True
        )
        dy_out, dy_x_grad, dy_weight_grad, dy_bias_grad = self.eval(
            use_cinn=False
        )
        # np.testing.assert_allclose(
        #     cinn_x_grad, dy_x_grad, atol=1e-2
        # )
        # np.testing.assert_allclose(
        #     cinn_weight_grad, dy_weight_grad, atol=1e-6
        # )
        # np.testing.assert_allclose(
        #     cinn_bias_grad, dy_bias_grad, atol=1e-6
        # )


# Todo: Origin group_norm does not support float16 or bfloat16 in NHWC and rank3
class _TestGroupNormSubGraphF16NHWCRank3(TestGroupNormSubGraph):
    def setUp(self):
        paddle.seed(2024)
        self.shape = [80, 128, 128]
        self.dtype = "float16"
        self.data_format = "NHWC"
        self.prepare_data()

    def test_prim(self):
        cinn_out, cinn_x_grad, cinn_weight_grad, cinn_bias_grad = self.eval(
            use_cinn=False, use_prim=True
        )
        dy_out, dy_x_grad, dy_weight_grad, dy_bias_grad = self.eval(
            use_cinn=False
        )
        np.testing.assert_allclose(cinn_x_grad, dy_x_grad, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cinn_weight_grad, dy_weight_grad, atol=5e-4)
        np.testing.assert_allclose(cinn_bias_grad, dy_bias_grad, rtol=1e-5)


class TestGroupNormSubGraphRank3(TestGroupNormSubGraph):
    def setUp(self):
        paddle.seed(2024)
        self.shape = [80, 128, 128]
        self.dtype = "float32"
        self.data_format = "NHWC"
        self.prepare_data()

    def test_prim(self):
        cinn_out, cinn_x_grad, cinn_weight_grad, cinn_bias_grad = self.eval(
            use_cinn=False, use_prim=True
        )
        dy_out, dy_x_grad, dy_weight_grad, dy_bias_grad = self.eval(
            use_cinn=False
        )
        np.testing.assert_allclose(cinn_x_grad, dy_x_grad, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cinn_weight_grad, dy_weight_grad, atol=5e-4)
        np.testing.assert_allclose(cinn_bias_grad, dy_bias_grad, rtol=1e-5)


class TestGroupNormSubGraphNCHW(TestGroupNormSubGraphRank3):
    def setUp(self):
        paddle.seed(2024)
        self.shape = [80, 128, 128]
        self.dtype = "bfloat16"
        self.data_format = "NCHW"
        self.prepare_data()

    def test_prim(self):
        cinn_out, cinn_x_grad, cinn_weight_grad, cinn_bias_grad = self.eval(
            use_cinn=False, use_prim=True
        )
        dy_out, dy_x_grad, dy_weight_grad, dy_bias_grad = self.eval(
            use_cinn=False
        )
        np.testing.assert_allclose(cinn_x_grad, dy_x_grad, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(cinn_weight_grad, dy_weight_grad, rtol=1e-2)
        np.testing.assert_allclose(cinn_bias_grad, dy_bias_grad, rtol=1e-5)


class TestGroupNormSubGraphRank6NHWC(TestGroupNormSubGraph):
    def setUp(self):
        paddle.seed(2024)
        self.shape = [80, 4, 4, 4, 4, 128]
        self.dtype = "float32"
        self.data_format = "NHWC"
        self.prepare_data()

    def test_prim(self):
        cinn_out, cinn_x_grad, cinn_weight_grad, cinn_bias_grad = self.eval(
            use_cinn=False, use_prim=True
        )
        dy_out, dy_x_grad, dy_weight_grad, dy_bias_grad = self.eval(
            use_cinn=False
        )
        np.testing.assert_allclose(cinn_x_grad, dy_x_grad, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cinn_weight_grad, dy_weight_grad, atol=5e-4)
        np.testing.assert_allclose(cinn_bias_grad, dy_bias_grad, rtol=1e-5)


class TestGroupNormSubGraphRank6NCHW(TestGroupNormSubGraph):
    def setUp(self):
        paddle.seed(2024)
        self.shape = [80, 128, 8, 4, 4, 4]
        self.dtype = "float32"
        self.data_format = "NCHW"
        self.prepare_data()

    def test_prim(self):
        cinn_out, cinn_x_grad, cinn_weight_grad, cinn_bias_grad = self.eval(
            use_cinn=False, use_prim=True
        )
        dy_out, dy_x_grad, dy_weight_grad, dy_bias_grad = self.eval(
            use_cinn=False
        )
        np.testing.assert_allclose(cinn_x_grad, dy_x_grad, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cinn_weight_grad, dy_weight_grad, atol=5e-4)
        np.testing.assert_allclose(cinn_bias_grad, dy_bias_grad, rtol=1e-5)


class TestGroupNormSubGraphRank7NHWC(TestGroupNormSubGraph):
    def setUp(self):
        paddle.seed(2024)
        self.shape = [80, 4, 4, 4, 4, 4, 128]
        self.dtype = "float32"
        self.data_format = "NHWC"
        self.prepare_data()

    def test_prim(self):
        cinn_out, cinn_x_grad, cinn_weight_grad, cinn_bias_grad = self.eval(
            use_cinn=False, use_prim=True
        )
        dy_out, dy_x_grad, dy_weight_grad, dy_bias_grad = self.eval(
            use_cinn=False
        )
        np.testing.assert_allclose(cinn_x_grad, dy_x_grad, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cinn_weight_grad, dy_weight_grad, atol=5e-4)
        np.testing.assert_allclose(cinn_bias_grad, dy_bias_grad, rtol=1e-5)


class TestGroupNormSubGraphRank7NCHW(TestGroupNormSubGraph):
    def setUp(self):
        paddle.seed(2024)
        self.shape = [80, 128, 8, 4, 4, 4, 4]
        self.dtype = "float32"
        self.data_format = "NCHW"
        self.prepare_data()

    def test_prim(self):
        cinn_out, cinn_x_grad, cinn_weight_grad, cinn_bias_grad = self.eval(
            use_cinn=False, use_prim=True
        )
        dy_out, dy_x_grad, dy_weight_grad, dy_bias_grad = self.eval(
            use_cinn=False
        )
        np.testing.assert_allclose(cinn_x_grad, dy_x_grad, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cinn_weight_grad, dy_weight_grad, atol=5e-4)
        np.testing.assert_allclose(cinn_bias_grad, dy_bias_grad, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
