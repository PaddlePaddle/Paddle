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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_legacy_and_pt,
    test_legacy_and_pt_and_pir,
)

import paddle
from paddle import nn
from paddle.jit import to_static


class TestTensorHook(Dy2StTestBase):
    @test_legacy_and_pt
    def test_hook_for_different_parameter(self):
        def f(x):
            def h(g):
                return 2 * g

            y = x + 4
            f = y + x
            z = f**2
            y.register_hook(h)
            f.register_hook(h)
            x.register_hook(h)
            return z

        x = paddle.to_tensor([2.0])
        x.stop_gradient = False
        loss = f(x)
        loss.backward()

        x_jit = paddle.to_tensor([2.0])
        x_jit.stop_gradient = False
        jit_f = to_static(f)
        loss = jit_f(x_jit)
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), x_jit.grad.numpy())

    @test_legacy_and_pt
    def test_hook_in_sub_block(self):
        def f(x):
            def hook1(grad):
                return 2 * grad

            def hook2(grad):
                return 3 * grad

            if x > 1:
                y = x + 4
                z = y**2
                y.register_hook(hook1)
            else:
                y = x - 4
                z = y**3
                y.register_hook(hook2)
            return z

        x = paddle.to_tensor([2.0])
        x.stop_gradient = False
        loss = f(x)
        loss.backward()

        x_jit = paddle.to_tensor([2.0])
        x_jit.stop_gradient = False
        jit_f = to_static(f)
        loss = jit_f(x_jit)
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), x_jit.grad.numpy())

    @test_legacy_and_pt
    def test_hook_sub_attr(self):
        IMAGE_SIZE = 784
        CLASS_NUM = 10

        def hook(grad):
            return grad * 2

        class Layer(nn.Layer):
            def __init__(self):
                super().__init__()
                self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

            def forward(self, x):
                # breakpoint()
                self._linear.weight.register_hook(hook)
                y = self._linear(x)
                return y

        paddle.seed(0)
        data = np.random.random([IMAGE_SIZE]).astype('float32')
        x = paddle.to_tensor(data)
        x.stop_gradient = False
        layer = Layer()
        loss = layer(x)
        loss.backward()

        paddle.seed(0)
        x_jit = paddle.to_tensor(data)
        x_jit.stop_gradient = False
        jit_layer = to_static(Layer())
        loss = jit_layer(x_jit)
        loss.backward()
        np.testing.assert_allclose(
            layer._linear.weight.grad.numpy(),
            jit_layer._linear.weight.grad.numpy(),
        )

    @test_legacy_and_pt
    def test_hook_for_reassignment_parameter(self):
        def f(x):
            def h(g):
                return 2 * g

            y = x + 4
            x = y * 5
            z = x**2
            x.register_hook(h)
            return z

        x = paddle.to_tensor([2.0])
        x.stop_gradient = False
        loss = f(x)
        loss.backward()

        x_jit = paddle.to_tensor([2.0])
        x_jit.stop_gradient = False
        jit_f = to_static(f)
        loss = jit_f(x_jit)
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), x_jit.grad.numpy())

    @test_legacy_and_pt
    def test_hook_for_repeat_register(self):
        def f(x):
            def h(g):
                return 2 * g

            y = x + 4
            z = y**2
            x.register_hook(h)
            x.register_hook(h)
            return z

        x = paddle.to_tensor([2.0])
        x.stop_gradient = False
        loss = f(x)
        loss.backward()

        x_jit = paddle.to_tensor([2.0])
        x_jit.stop_gradient = False
        jit_f = to_static(f)
        loss = jit_f(x_jit)
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), x_jit.grad.numpy())

    @test_legacy_and_pt_and_pir
    def test_hook_in_init_for_layer(self):
        def hook(grad):
            return grad * 2

        IMAGE_SIZE = 784
        CLASS_NUM = 10

        class LinearNet(nn.Layer):
            def __init__(self):
                super().__init__()
                self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
                # register_hook in init
                self._linear.parameters()[0].register_hook(hook)

            def forward(self, x):
                return self._linear(x)

        # create network
        layer = LinearNet()
        jit_layer = to_static(LinearNet())
        data = np.random.random([IMAGE_SIZE]).astype('float32')
        image = paddle.to_tensor(data)
        image_jit = paddle.to_tensor(data)
        loss = layer(image)
        loss_jit = jit_layer(image_jit)
        loss_jit.backward()
        loss.backward()
        np.testing.assert_allclose(
            layer.parameters()[0].grad.numpy(),
            jit_layer.parameters()[0].grad.numpy(),
        )


if __name__ == '__main__':
    unittest.main()
