# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net, build_strategy=build_strategy, full_graph=True
    )


def softmax(x, axis):
    """define composite rule of op softmax"""
    is_amp = False
    from paddle.base.data_feeder import convert_dtype

    # Softmax need fp32 compute since it has sum op in
    dtype = convert_dtype(x.dtype)
    if dtype in ["float16", "uint16"]:
        is_amp = True
        x = paddle.cast(x, "float32")
    if not x.shape:
        # do not return 1, to ensure gradients
        res = paddle.exp(x - x)
        if is_amp:
            res = paddle.cast(res, "float16")
        return res
    max_temp = paddle.max(x, axis, keepdim=True)
    max_temp.stop_gradient = True
    molecular = paddle.exp(x - max_temp)
    denominator = paddle.sum(molecular, axis=axis, keepdim=True)
    res = paddle.divide(molecular, denominator)
    if is_amp:
        res = paddle.cast(res, dtype)
    return res


def exp_sub(x):
    y = paddle.exp(x)
    z = y - x
    return z


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = exp_sub

    def forward(self, x):
        out = self.fn(x)
        return out


class CINNSoftmaxSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = softmax

    def forward(self, x, axis=-1):
        out = self.fn(x, axis=axis)
        return out


class TestCinnSubGraphBase(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.shape = [64, 128]
        self.axis = -1
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False

    def train(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet()
        net = apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x, self.axis)
        return out

    def test_forward(self):
        cinn_out = self.train(use_cinn=True)
        dy_out = self.train(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnSoftmax(TestCinnSubGraphBase):
    def train(self, use_cinn):
        paddle.seed(2022)
        net = CINNSoftmaxSubGraphNet()
        net = apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x, self.axis)
        return out


if __name__ == '__main__':
    unittest.main()
