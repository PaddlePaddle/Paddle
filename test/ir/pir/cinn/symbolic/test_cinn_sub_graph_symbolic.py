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

import numpy as np

import paddle
from paddle.static import InputSpec


def apply_to_static(net, use_cinn, input_spec=None):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )


def exp_sub(x):
    y = paddle.exp(x)
    z = y - x
    return z


def reshape(x):
    y = paddle.exp(x)
    z = y - x
    i = paddle.shape(x)[0]
    j = paddle.shape(y)[1]
    out = paddle.reshape(z, shape=[i, j])
    # out = paddle.reshape(z, shape=[128, 4, 2])
    return out


def broadcast(x, y):
    z = x + y
    return z


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = exp_sub

    def forward(self, x):
        out = self.fn(x)
        return out


class CINNReshapeSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = reshape

    def forward(self, x):
        out = self.fn(x)
        return out


class CINNBroadcastSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = broadcast

    def forward(self, x, y):
        out = self.fn(x, y)
        return out


class TestCinnSubGraphBase(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [64, 128]
        self.axis = -1
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet()
        input_spec = [InputSpec(shape=[None, 128], dtype='float32')]
        net = apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnDyShapeBase(TestCinnSubGraphBase):
    def prepare_data(self):
        self.shape = [4, 256]
        self.axis = -1
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNReshapeSubGraphNet()
        input_spec = [InputSpec(shape=[None, 256], dtype='float32')]
        net = apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        return out

    def test_eval_symbolic(self):
        import os

        is_debug = os.getenv('IS_DEBUG_DY_SHAPE')
        if is_debug:
            cinn_out = self.eval_symbolic(use_cinn=True)

        dy_out = self.eval_symbolic(use_cinn=False)
        # np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnDyShapeBC(TestCinnDyShapeBase):
    def prepare_data(self):
        self.x_shape = [2, 4, 1]
        self.x = paddle.randn(self.x_shape, dtype="float32")
        self.x.stop_gradient = False

        self.y_shape = [4, 5]
        self.y = paddle.randn(self.y_shape, dtype="float32")
        self.y.stop_gradient = False

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNBroadcastSubGraphNet()
        input_spec = [
            InputSpec(shape=[None, None, None], dtype='float32'),
            InputSpec(shape=[None, None], dtype='float32'),
        ]
        net = apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x, self.y)
        return out

    def test_eval_symbolic(self):
        # cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        # np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class LlamaRMSNorm(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4096
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        # 1. variance = hidden_states.pow(2).mean(-1, keepdim=True)

        axis_rst = -1
        # 1.1 decomp pow -> elementwise_pow
        pow_tensor = paddle.full([1], axis_rst, hidden_states.dtype)
        pow_rst = paddle.pow(hidden_states, pow_tensor)

        # 1.2 decomp mean -> sum & div
        sum_rst = paddle.sum(pow_rst, [axis_rst], keepdim=True)
        shape_rst = paddle.shape(sum_rst)
        div_by = paddle.full(shape_rst, hidden_states.shape[axis_rst])
        variance = paddle.divide(sum_rst, div_by)

        # 2. hidden_states = (paddle.rsqrt(variance + self.variance_epsilon) * hidden_states)

        # 2.1 decomp variance + self.variance_epsilon -> full + scale
        scale_tensor = paddle.full([1], 1.0)
        scale_rst = paddle.scale(variance, scale_tensor, self.variance_epsilon)

        # 2.2 decomp rsqrt -> pow(-0.5)
        rsqrt_tensor = paddle.full([1], -0.5)
        rsqrt_rst = paddle.pow(scale_rst, rsqrt_tensor)

        hidden_states = rsqrt_rst * hidden_states

        return hidden_states * self.weight


class TestCinnDyShapeRMSNorm(TestCinnDyShapeBase):
    def prepare_data(self):
        self.hidden_states_shape = [1, 300, 4096]
        self.hidden_states = paddle.randn(
            self.hidden_states_shape, dtype="float32"
        )
        self.hidden_states.stop_gradient = False

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = LlamaRMSNorm()
        input_spec = [
            InputSpec(shape=[None, None, 4096], dtype='float32'),
        ]
        net = apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.hidden_states)
        return out

    def test_eval_symbolic(self):
        # cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        # np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
