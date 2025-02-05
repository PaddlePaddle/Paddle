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
from paddle.incubate.optimizer import (
    lbfgs as incubate_lbfgs,
    line_search_dygraph,
)
from paddle.optimizer import lbfgs

np.random.seed(123)

# func()should be func(w, x)where w is parameter to be optimize ,x is input of optimizer func
# np_w is the init parameter of w


class Net(paddle.nn.Layer):
    def __init__(self, np_w, func):
        super().__init__()
        self.func = func
        w = paddle.to_tensor(np_w)
        self.w = paddle.create_parameter(
            shape=w.shape,
            dtype=w.dtype,
            default_initializer=paddle.nn.initializer.Assign(w),
        )

    def forward(self, x):
        return self.func(self.w, x)


def train_step(inputs, targets, net, opt):
    def closure():
        outputs = net(inputs)
        loss = paddle.nn.functional.mse_loss(outputs, targets)
        opt.clear_grad()
        loss.backward()
        return loss

    loss = opt.step(closure)
    return loss


class TestLbfgs(unittest.TestCase):
    def test_function_fix_incubate(self):
        paddle.disable_static()
        np_w = np.random.rand(1).astype(np.float32)

        input = np.random.rand(1).astype(np.float32)
        weights = [np.random.rand(1).astype(np.float32) for i in range(5)]
        targets = [weights[i] * input for i in range(5)]

        def func(w, x):
            return w * x

        net = Net(np_w, func)
        opt = incubate_lbfgs.LBFGS(
            learning_rate=1,
            max_iter=10,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=5,
            line_search_fn='strong_wolfe',
            parameters=net.parameters(),
        )

        for weight, target in zip(weights, targets):
            input = paddle.to_tensor(input)
            target = paddle.to_tensor(target)
            loss = 1
            while loss > 1e-4:
                loss = train_step(input, target, net, opt)
            np.testing.assert_allclose(net.w, weight, rtol=1e-05)

    def test_inf_minima_incubate(self):
        # not converge
        input = np.random.rand(1).astype(np.float32)

        def outputs1(x):
            # weight[0] = 1.01 weight[1] = 0.99
            return x * x * x - 3 * x * x + 3 * 1.01 * 0.99 * x

        def outputs2(x):
            # weight[0] = 4 weight[1] = 2
            return pow(x, 4) + 5 * pow(x, 2)

        targets = [outputs1(input), outputs2(input)]
        input = paddle.to_tensor(input)

        def func1(extreme_point, x):
            return (
                x * x * x
                - 3 * x * x
                + 3 * extreme_point[0] * extreme_point[1] * x
            )

        def func2(extreme_point, x):
            return pow(x, extreme_point[0]) + 5 * pow(x, extreme_point[1])

        extreme_point = np.array([-2.34, 1.45]).astype('float32')
        net1 = Net(extreme_point, func1)
        # converge of old_sk.pop()
        opt1 = incubate_lbfgs.LBFGS(
            learning_rate=1,
            max_iter=10,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=1,
            line_search_fn='strong_wolfe',
            parameters=net1.parameters(),
        )

        net2 = Net(extreme_point, func2)
        # converge of line_search = None
        opt2 = incubate_lbfgs.LBFGS(
            learning_rate=1,
            max_iter=50,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=10,
            line_search_fn=None,
            parameters=net2.parameters(),
        )

        n_iter = 0
        while n_iter < 20:
            loss = train_step(input, paddle.to_tensor(targets[0]), net1, opt1)
            n_iter = opt1.state_dict()["state"]["func_evals"]

        n_iter = 0
        while n_iter < 10:
            loss = train_step(input, paddle.to_tensor(targets[1]), net2, opt2)
            n_iter = opt1.state_dict()["state"]["func_evals"]

    def test_error_incubate(self):
        # test parameter is not Paddle Tensor
        def error_func1():
            extreme_point = np.array([-1, 2]).astype('float32')
            extreme_point = paddle.to_tensor(extreme_point)
            return incubate_lbfgs.LBFGS(
                learning_rate=1,
                max_iter=10,
                max_eval=None,
                tolerance_grad=1e-07,
                tolerance_change=1e-09,
                history_size=3,
                line_search_fn='strong_wolfe',
                parameters=extreme_point,
            )

        self.assertRaises(TypeError, error_func1)

    def test_error2_incubate(self):
        # not converge
        input = np.random.rand(1).astype(np.float32)

        def outputs2(x):
            # weight[0] = 4 weight[1] = 2
            return pow(x, 4) + 5 * pow(x, 2)

        targets = [outputs2(input)]
        input = paddle.to_tensor(input)

        def func2(extreme_point, x):
            return pow(x, extreme_point[0]) + 5 * pow(x, extreme_point[1])

        extreme_point = np.array([-2.34, 1.45]).astype('float32')
        net2 = Net(extreme_point, func2)
        # converge of line_search = None
        opt2 = incubate_lbfgs.LBFGS(
            learning_rate=1,
            max_iter=50,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=10,
            line_search_fn='None',
            parameters=net2.parameters(),
        )

        def error_func():
            n_iter = 0
            while n_iter < 10:
                loss = train_step(
                    input, paddle.to_tensor(targets[0]), net2, opt2
                )
                n_iter = opt2.state_dict()["state"]["func_evals"]

        self.assertRaises(RuntimeError, error_func)

    def test_line_search_incubate(self):
        def func1(x, alpha, d):
            return paddle.to_tensor(x + alpha * d), paddle.to_tensor([0.0])

        def func2(x, alpha, d):
            return paddle.to_tensor(x + alpha * d), paddle.to_tensor([1.0])

        def func3(x, alpha, d):
            return paddle.to_tensor(x + alpha * d), paddle.to_tensor([-1.0])

        line_search_dygraph._strong_wolfe(
            func1,
            paddle.to_tensor([1.0]),
            paddle.to_tensor([0.001]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([0.0]),
            paddle.to_tensor([0.0]),
            max_ls=1,
        )

        line_search_dygraph._strong_wolfe(
            func1,
            paddle.to_tensor([1.0]),
            paddle.to_tensor([0.001]),
            paddle.to_tensor([0.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([0.0]),
            paddle.to_tensor([0.0]),
            max_ls=0,
        )

        line_search_dygraph._strong_wolfe(
            func2,
            paddle.to_tensor([1.0]),
            paddle.to_tensor([-0.001]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            max_ls=1,
        )

        line_search_dygraph._strong_wolfe(
            func3,
            paddle.to_tensor([1.0]),
            paddle.to_tensor([-0.001]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            max_ls=1,
        )

        line_search_dygraph._cubic_interpolate(
            paddle.to_tensor([2.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([0.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([2.0]),
            paddle.to_tensor([0.0]),
            [0.1, 0.5],
        )

        line_search_dygraph._cubic_interpolate(
            paddle.to_tensor([2.0]),
            paddle.to_tensor([0.0]),
            paddle.to_tensor([-3.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([-0.1]),
            [0.1, 0.5],
        )

    def test_error3_incubate(self):
        # test parameter shape size <= 0
        def error_func3():
            extreme_point = np.array([-1, 2]).astype('float32')
            extreme_point = paddle.to_tensor(extreme_point)

            def func(w, x):
                return w * x

            net = Net(extreme_point, func)
            net.w = paddle.create_parameter(
                shape=[-1, 2],
                dtype=net.w.dtype,
            )
            opt = incubate_lbfgs.LBFGS(
                learning_rate=1,
                max_iter=10,
                max_eval=None,
                tolerance_grad=1e-07,
                tolerance_change=1e-09,
                history_size=5,
                line_search_fn='strong_wolfe',
                parameters=net.parameters(),
            )

        self.assertRaises(AssertionError, error_func3)

    def test_function_fix(self):
        paddle.disable_static()
        np_w = np.random.rand(1).astype(np.float32)

        input = np.random.rand(1).astype(np.float32)
        weights = [np.random.rand(1).astype(np.float32) for i in range(5)]
        targets = [weights[i] * input for i in range(5)]

        def func(w, x):
            return w * x

        net = Net(np_w, func)
        opt = lbfgs.LBFGS(
            learning_rate=1,
            max_iter=10,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=5,
            line_search_fn='strong_wolfe',
            parameters=net.parameters(),
        )

        for weight, target in zip(weights, targets):
            input = paddle.to_tensor(input)
            target = paddle.to_tensor(target)
            loss = 1
            while loss > 1e-4:
                loss = train_step(input, target, net, opt)
            np.testing.assert_allclose(net.w, weight, rtol=1e-05)

    def test_inf_minima(self):
        # not converge
        input = np.random.rand(1).astype(np.float32)

        def outputs1(x):
            # weight[0] = 1.01 weight[1] = 0.99
            return x * x * x - 3 * x * x + 3 * 1.01 * 0.99 * x

        def outputs2(x):
            # weight[0] = 4 weight[1] = 2
            return pow(x, 4) + 5 * pow(x, 2)

        targets = [outputs1(input), outputs2(input)]
        input = paddle.to_tensor(input)

        def func1(extreme_point, x):
            return (
                x * x * x
                - 3 * x * x
                + 3 * extreme_point[0] * extreme_point[1] * x
            )

        def func2(extreme_point, x):
            return pow(x, extreme_point[0]) + 5 * pow(x, extreme_point[1])

        extreme_point = np.array([-2.34, 1.45]).astype('float32')
        net1 = Net(extreme_point, func1)
        # converge of old_sk.pop()
        opt1 = lbfgs.LBFGS(
            learning_rate=1,
            max_iter=10,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=1,
            line_search_fn='strong_wolfe',
            parameters=net1.parameters(),
        )

        net2 = Net(extreme_point, func2)
        # converge of line_search = None
        opt2 = lbfgs.LBFGS(
            learning_rate=1,
            max_iter=50,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=10,
            line_search_fn=None,
            parameters=net2.parameters(),
        )

        n_iter = 0
        while n_iter < 20:
            loss = train_step(input, paddle.to_tensor(targets[0]), net1, opt1)
            n_iter = opt1.state_dict()["state"]["func_evals"]

        n_iter = 0
        while n_iter < 10:
            loss = train_step(input, paddle.to_tensor(targets[1]), net2, opt2)
            n_iter = opt1.state_dict()["state"]["func_evals"]

    def test_error(self):
        # test parameter is not Paddle Tensor
        def error_func1():
            extreme_point = np.array([-1, 2]).astype('float32')
            extreme_point = paddle.to_tensor(extreme_point)
            return lbfgs.LBFGS(
                learning_rate=1,
                max_iter=10,
                max_eval=None,
                tolerance_grad=1e-07,
                tolerance_change=1e-09,
                history_size=3,
                line_search_fn='strong_wolfe',
                parameters=extreme_point,
            )

        self.assertRaises(TypeError, error_func1)

    def test_error2(self):
        # not converge
        input = np.random.rand(1).astype(np.float32)

        def outputs2(x):
            # weight[0] = 4 weight[1] = 2
            return pow(x, 4) + 5 * pow(x, 2)

        targets = [outputs2(input)]
        input = paddle.to_tensor(input)

        def func2(extreme_point, x):
            return pow(x, extreme_point[0]) + 5 * pow(x, extreme_point[1])

        extreme_point = np.array([-2.34, 1.45]).astype('float32')
        net2 = Net(extreme_point, func2)
        # converge of line_search = None
        opt2 = lbfgs.LBFGS(
            learning_rate=1,
            max_iter=50,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=10,
            line_search_fn='None',
            parameters=net2.parameters(),
        )

        def error_func():
            n_iter = 0
            while n_iter < 10:
                loss = train_step(
                    input, paddle.to_tensor(targets[0]), net2, opt2
                )
                n_iter = opt2.state_dict()["state"]["func_evals"]

        self.assertRaises(RuntimeError, error_func)

    def test_line_search(self):
        def func1(x, alpha, d):
            return paddle.to_tensor(x + alpha * d), paddle.to_tensor([0.0])

        def func2(x, alpha, d):
            return paddle.to_tensor(x + alpha * d), paddle.to_tensor([1.0])

        def func3(x, alpha, d):
            return paddle.to_tensor(x + alpha * d), paddle.to_tensor([-1.0])

        lbfgs._strong_wolfe(
            func1,
            paddle.to_tensor([1.0]),
            paddle.to_tensor([0.001]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([0.0]),
            paddle.to_tensor([0.0]),
            max_ls=1,
        )

        lbfgs._strong_wolfe(
            func1,
            paddle.to_tensor([1.0]),
            paddle.to_tensor([0.001]),
            paddle.to_tensor([0.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([0.0]),
            paddle.to_tensor([0.0]),
            max_ls=0,
        )

        lbfgs._strong_wolfe(
            func2,
            paddle.to_tensor([1.0]),
            paddle.to_tensor([-0.001]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            max_ls=1,
        )
        lbfgs._strong_wolfe(
            func2,
            paddle.to_tensor([1.0]),
            -0.001,
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            max_ls=1,
        )

        lbfgs._strong_wolfe(
            func3,
            paddle.to_tensor([1.0]),
            paddle.to_tensor([-0.001]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            max_ls=1,
        )

        lbfgs._cubic_interpolate(
            paddle.to_tensor([2.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([0.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([2.0]),
            paddle.to_tensor([0.0]),
            [0.1, 0.5],
        )

        lbfgs._cubic_interpolate(
            paddle.to_tensor([2.0]),
            paddle.to_tensor([0.0]),
            paddle.to_tensor([-3.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            paddle.to_tensor([-0.1]),
            [0.1, 0.5],
        )

    def test_error3(self):
        # test parameter shape size <= 0
        def error_func3():
            extreme_point = np.array([-1, 2]).astype('float32')
            extreme_point = paddle.to_tensor(extreme_point)

            def func(w, x):
                return w * x

            net = Net(extreme_point, func)
            net.w = paddle.create_parameter(
                shape=[-1, 2],
                dtype=net.w.dtype,
            )
            opt = lbfgs.LBFGS(
                learning_rate=1,
                max_iter=10,
                max_eval=None,
                tolerance_grad=1e-07,
                tolerance_change=1e-09,
                history_size=5,
                line_search_fn='strong_wolfe',
                parameters=net.parameters(),
            )

        self.assertRaises(AssertionError, error_func3)

    def test_error4(self):
        # test call minimize(loss)
        paddle.disable_static()

        def error_func4():
            inputs = np.random.rand(1).astype(np.float32)
            targets = paddle.to_tensor([inputs * 2])
            inputs = paddle.to_tensor(inputs)

            extreme_point = np.array([-1, 1]).astype('float32')

            def func(extreme_point, x):
                return x * extreme_point[0] + 5 * x * extreme_point[1]

            net = Net(extreme_point, func)
            opt = lbfgs.LBFGS(
                learning_rate=1,
                max_iter=10,
                max_eval=None,
                tolerance_grad=1e-07,
                tolerance_change=1e-09,
                history_size=5,
                line_search_fn='strong_wolfe',
                parameters=net.parameters(),
            )
            loss = train_step(inputs, targets, net, opt)
            opt.minimize(loss)

        self.assertRaises(NotImplementedError, error_func4)


if __name__ == '__main__':
    unittest.main()
