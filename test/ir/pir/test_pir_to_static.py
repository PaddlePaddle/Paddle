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

import os
import unittest

import numpy as np

import paddle


class TestDy2staticPir(unittest.TestCase):
    def test_basic_network(self):
        def func(x):
            out = paddle.mean(x)
            return out

        static_func = paddle.jit.to_static(func, full_graph=True)
        x = paddle.randn((3, 3))
        y = paddle.randn((3, 3))
        x.stop_gradient = False
        y.stop_gradient = False
        ans = func(x)
        out = static_func(x)

        np.testing.assert_allclose(
            out.numpy(), ans.numpy(), rtol=1e-05, atol=1e-8
        )

    def test_basic_network_backward(self):
        def func(x):
            out = paddle.mean(x)
            return out

        # ==== dygraph computation ====
        static_func = paddle.jit.to_static(func, full_graph=True)
        x = paddle.randn((3, 3))
        y = paddle.randn((3, 3))
        x.stop_gradient = False
        y.stop_gradient = False
        loss = func(x) * 2
        loss.backward()
        x_grad_ans = x.grad.numpy()
        x.clear_gradient()

        # ==== to static computation ====
        out = static_func(x)
        out = out * 2
        out.backward()
        st_grad = x.grad

        np.testing.assert_allclose(
            x_grad_ans, st_grad.numpy(), rtol=1e-05, atol=1e-8
        )


class TestDy2staticPir2(unittest.TestCase):
    def test_basic_layer(self):
        class SimpleNet(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.linear = paddle.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        net = SimpleNet()
        x = paddle.randn((10, 10))
        x.stop_gradient = False
        ans = net(x)
        net = paddle.jit.to_static(net, full_graph=True)
        out = net(x)
        np.testing.assert_allclose(
            out.numpy(), ans.numpy(), rtol=1e-05, atol=1e-8
        )


class TestDy2staticPir3(unittest.TestCase):
    def test_complex_layer(self):
        def output_pure_func(x, y):
            outx = paddle.mean(x)
            outy = paddle.mean(y)
            outy.stop_gradient = True
            return paddle.add(outx, outy), outy

        def run_function(to_static=True):
            paddle.seed(2023)
            x = paddle.randn((10, 10))
            y = paddle.randn((10, 10))
            x.stop_gradient = False
            y.stop_gradient = True
            func = output_pure_func
            if to_static:
                func = paddle.jit.to_static(func, full_graph=True)
            y, y_mean = func(x, y)
            loss = y.mean()
            loss.backward()
            return (y, x.grad)

        for dy, st in zip(run_function(False), run_function(True)):
            np.testing.assert_allclose(
                dy.numpy(), st.numpy(), rtol=1e-05, atol=1e-8
            )


class TestLossFor10Steps(unittest.TestCase):
    def test_loss_for_10_steps(self):
        # Dy2static RunProgramOp support nn.Layer's forward and backward training.
        class SimpleNet(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.linear = paddle.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        def train_step(to_static=True):
            paddle.seed(2023)
            x = paddle.randn((10, 10), dtype='float32')
            y = paddle.randn((10, 10), dtype='float32')
            loss_fn = paddle.nn.loss.MSELoss()
            net = SimpleNet()
            optimizer = paddle.optimizer.SGD(
                learning_rate=0.1, parameters=net.parameters()
            )
            if to_static:
                net = paddle.jit.to_static(net, full_graph=True)
            losses = []
            for step in range(100):
                y_pred = net(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                losses.append(loss.numpy())
            return losses

        expected_losses = train_step(True)
        losses = train_step(False)
        np.testing.assert_allclose(
            losses, expected_losses, rtol=1e-05, atol=1e-8
        )


class TestDy2staticPir5(unittest.TestCase):
    def test_run(self):
        # Dy2static RunProgramOp support nn.Layer's forward and backward training.
        class SimpleNet(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.linear = paddle.nn.Linear(10, 10)

            def forward(self, x, y):
                if y is True:
                    return self.linear(x)
                else:
                    m = self.linear(x)
                    return m * m

        def train_step(to_static=True, full_graph=True):
            paddle.seed(2023)
            x = paddle.randn((10, 10), dtype='float32')
            y = paddle.randn((10, 10), dtype='float32')
            loss_fn = paddle.nn.loss.MSELoss()
            net = SimpleNet()
            optimizer = paddle.optimizer.SGD(
                learning_rate=0.1, parameters=net.parameters()
            )
            if to_static:
                net = paddle.jit.to_static(net, full_graph=full_graph)
            losses = []
            for step in range(100):
                y_pred = net(x, step % 2 == 1)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                losses.append(loss.numpy())
            return losses

        expected_losses = train_step(True)
        losses = train_step(False)
        np.testing.assert_allclose(
            losses, expected_losses, rtol=1e-05, atol=1e-8
        )
        os.environ['MIN_GRAPH_SIZE'] = '0'
        sot_losses = train_step(True, False)
        np.testing.assert_allclose(losses, sot_losses, rtol=1e-05, atol=1e-8)


class TestDy2staticPir6(unittest.TestCase):
    # test basic-indexing __getitem__ for Value
    def test_basic_network(self):
        def func(x):
            shape = paddle.shape(x)
            out = shape[1:]
            return out

        static_func = paddle.jit.to_static(func, full_graph=True)
        x = paddle.randn((2, 3, 4))
        x.stop_gradient = False
        ans = func(x)
        out = static_func(x)

        np.testing.assert_allclose(out.numpy(), ans.numpy())


class TestDy2staticPir7(unittest.TestCase):
    # test basic-indexing __getitem__ for Value
    def test_basic_network(self):
        def func(x):
            x = x * 2
            x = x + 1
            return 1

        static_func = paddle.jit.to_static(func, full_graph=True)
        x = paddle.randn((2, 3, 4))
        x.stop_gradient = False
        ans = func(x)
        out = static_func(x)
        np.testing.assert_allclose(out, ans)


if __name__ == "__main__":
    unittest.main()
