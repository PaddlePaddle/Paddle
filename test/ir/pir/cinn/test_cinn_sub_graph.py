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


def exp_sub(x):
    y = paddle.exp(x)
    z = y - x
    return z


def layer_norm(x, weight, bias):
    num = paddle.full([1], x.shape[-1])
    eps = paddle.full([1], 1e-5)
    sum1 = paddle.sum(x, axis=-1, keepdim=True)
    mean = sum1 / num
    t1 = x - mean
    t2 = t1 * t1
    t3 = paddle.sum(t2, axis=-1, keepdim=True)
    t3 = t3 / num
    t4 = t3 + eps
    t5 = paddle.sqrt(t4)
    t7 = t1 / t5

    return t7 * weight + bias


def dropout(x):
    rand = paddle.uniform(x.shape, min=0.0, max=1.0, dtype="float32")
    zero = paddle.full([1], 0.0)

    mask = paddle.greater_equal(rand, zero)

    out = x * paddle.cast(mask, x.dtype)
    return out


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
        self.fn = paddle.nn.functional.softmax

    def forward(self, x, axis=-1):
        out = self.fn(x, axis=axis)
        return out


class CINNLayerNormSubGraphNet(paddle.nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.fn = layer_norm
        self.weight = self.create_parameter(
            shape=[hidden_size], dtype="float32"
        )
        self.bias = self.create_parameter(shape=[hidden_size], dtype="float32")

    def forward(self, x, weight, bias):
        out = paddle.nn.functional.layer_norm(
            x, x.shape[-1], self.weight, self.bias
        )
        return out


class CINNAddDropoutLayerNormSubGraphNet(paddle.nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.add = paddle.add
        self.dropout = dropout
        self.layer_norm = layer_norm

        self.weight = self.create_parameter(
            shape=[hidden_size], dtype="float32"
        )
        self.bias = self.create_parameter(shape=[hidden_size], dtype="float32")

    def forward(self, x, y, weight, bias):
        t1 = self.add(x, y)
        t2 = self.dropout(t1)
        out = self.layer_norm(t2, self.weight, self.bias)
        return out


class CINNDropoutSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = paddle.nn.functional.dropout

    def forward(self, x):
        out = self.fn(x)
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

    def eval(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet()
        net = apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x)
        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True)
        dy_out = self.eval(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnSoftmax(TestCinnSubGraphBase):
    def train(self, use_cinn):
        paddle.seed(2022)
        net = CINNSoftmaxSubGraphNet()
        net = apply_to_static(net, use_cinn)
        out = net(self.x, self.axis)

        loss = out.mean()
        loss.backward()
        return out

    def test_forward(self):
        cinn_out = self.train(use_cinn=True)
        dy_out = self.train(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


# class TestCinnLayerNorm(TestCinnSubGraphBase):
#     def train(self, use_cinn):
#         paddle.seed(2022)
#         net = CINNLayerNormSubGraphNet(self.shape[-1])
#         net = apply_to_static(net, use_cinn)
#         # net.eval()
#         weight = paddle.ones(shape=[self.shape[-1]], dtype="float32")
#         bias = paddle.ones(shape=[self.shape[-1]], dtype="float32")
#         out = net(self.x, weight, bias)
#         return out

#     def test_forward(self):
#         cinn_out = self.train(use_cinn=True)
#         print(cinn_out)
#         # dy_out = self.train(use_cinn=False)
#         # np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestAddDropoutLayerNorm(TestCinnSubGraphBase):
    def train(self, use_cinn):
        paddle.seed(2022)
        net = CINNAddDropoutLayerNormSubGraphNet(self.shape[-1])
        net = apply_to_static(net, use_cinn)
        net.eval()
        weight = paddle.ones(shape=[self.shape[-1]], dtype="float32")
        bias = paddle.ones(shape=[self.shape[-1]], dtype="float32")
        out = net(self.x, self.x, weight, bias)
        return out

    def test_forward(self):
        cinn_out = self.train(use_cinn=True)
        dy_out = self.train(use_cinn=False)

        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-8, rtol=1e-4
        )


class TestCinnDropout(TestCinnSubGraphBase):
    def train(self, use_cinn):
        paddle.seed(2022)
        net = CINNDropoutSubGraphNet()
        net = apply_to_static(net, use_cinn)
        out = net(self.x)

        loss = out.mean()
        loss.backward()

        return out

    def test_forward(self):
        cinn_out = self.train(use_cinn=True)
        # dy_out = self.train(use_cinn=False)
        # np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnEvalPrim(TestCinnSubGraphBase):
    def prepare_data(self):
        self.shape = [1, 2048, 768]
        self.hidden_states = paddle.randn(self.shape, dtype="float32")
        self.hidden_states.stop_gradient = False

    def eval(self, use_cinn):
        paddle.seed(2022)
        net = CINNSoftmaxSubGraphNet()
        if use_cinn:
            net = apply_to_static(net, True)
        net.eval()
        out = net(self.hidden_states)

        if use_cinn:
            ops = [
                op.name()
                for op in net.forward.program_cache.last()[-1][-1]
                .train_program.program.global_block()
                .ops
            ]
            assert (
                "pd_op.softmax" not in ops
            ), f"after prim, pd_op.softmax should not exist, but got {ops}"
            assert (
                "pd_op.exp" in ops
            ), f"after prim, pd_op.softmax should not exist, but got {ops}"

        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True)
        dy_out = self.eval(use_cinn=False)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
