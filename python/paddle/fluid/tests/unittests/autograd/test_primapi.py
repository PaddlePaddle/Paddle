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

import typing
import unittest

import numpy as np
import paddle
from paddle.incubate.autograd import primapi

import config
import utils


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'xs', 'v', 'dtype'),
    (('matmul', paddle.matmul,
      (np.random.rand(2, 3), np.random.rand(3, 2)), None, 'float32'), ))
class TestWithoutProgramGuard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.xs = tuple(x.astype(cls.dtype) for x in cls.xs)
        cls._rtol = config.TOLERANCE.get(str(
            cls.dtype)).get("first_order_grad").get("rtol")
        cls._atol = config.TOLERANCE.get(str(
            cls.dtype)).get("first_order_grad").get("atol")

    def setUp(self):
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()

    def tearDown(self):
        paddle.incubate.autograd.disable_prim()
        paddle.disable_static()

    def test_forward_grad_without_program_guard(self):

        def with_program_guard():
            paddle.incubate.autograd.enable_prim()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False)
                ys = self.fun(*static_xs) if isinstance(
                    static_xs, typing.Sequence) else self.fun(static_xs)
                ys_grad = paddle.incubate.autograd.forward_grad(
                    ys, static_xs, static_v)
                paddle.incubate.autograd.prim2orig(mp.block(0))
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=ys_grad)
            paddle.incubate.autograd.disable_prim()
            return out

        def without_program_guard():
            paddle.incubate.autograd.enable_prim()
            feed, static_xs, static_v = utils.gen_static_data_and_feed(
                self.xs, self.v, stop_gradient=False)
            ys = self.fun(*static_xs) if isinstance(
                static_xs, typing.Sequence) else self.fun(static_xs)
            ys_grad = paddle.incubate.autograd.forward_grad(
                ys, static_xs, static_v)
            sp = paddle.fluid.framework.default_startup_program()
            mp = paddle.fluid.framework.default_main_program()
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=ys_grad)
            paddle.incubate.autograd.disable_prim()
            return out

        expected = with_program_guard()
        actual = without_program_guard()
        self.assertEqual(type(actual), type(expected))
        np.testing.assert_allclose(np.concatenate(actual),
                                   np.concatenate(expected),
                                   rtol=self._rtol,
                                   atol=self._atol)

    def test_grad_without_program_guard(self):

        def with_program_guard():
            paddle.incubate.autograd.enable_prim()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False)
                ys = self.fun(*static_xs) if isinstance(
                    static_xs, typing.Sequence) else self.fun(static_xs)
                xs_grad = paddle.incubate.autograd.grad(ys, static_xs, static_v)
                paddle.incubate.autograd.prim2orig(mp.block(0))
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=xs_grad)
            paddle.incubate.autograd.disable_prim()
            return out

        def without_program_guard():
            paddle.incubate.autograd.enable_prim()
            feed, static_xs, static_v = utils.gen_static_data_and_feed(
                self.xs, self.v, stop_gradient=False)
            ys = self.fun(*static_xs) if isinstance(
                static_xs, typing.Sequence) else self.fun(static_xs)
            xs_grad = paddle.incubate.autograd.grad(ys, static_xs, static_v)
            sp = paddle.fluid.framework.default_startup_program()
            mp = paddle.fluid.framework.default_main_program()
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=xs_grad)
            paddle.incubate.autograd.disable_prim()
            return out

        expected = with_program_guard()
        actual = without_program_guard()
        for i, j in zip(actual, expected):
            self.assertEqual(type(i), type(j))
            np.testing.assert_allclose(np.concatenate(i),
                                       np.concatenate(j),
                                       rtol=self._rtol,
                                       atol=self._atol)


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'xs', 'v', 'dtype'),
    (('matmul', paddle.matmul,
      (np.random.rand(2, 3), np.random.rand(3, 2)), None, 'float32'),
     ('multiply', paddle.multiply,
      (np.random.rand(2, 3), np.random.rand(2, 3)), None, 'float64'),
     ('add', paddle.add,
      (np.random.rand(2, 3), np.random.rand(2, 3)), None, 'float32'),
     ('input_not_sequence', paddle.tanh,
      (np.random.rand(5, 5), ), None, 'float64'),
     ('input_gradients_not_none', paddle.matmul,
      (np.random.rand(3, 3), np.random.rand(3, 3)),
      (np.random.rand(3, 3), np.random.rand(3, 3)), 'float64')))
class TestForwardGrad(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.xs = tuple(x.astype(cls.dtype) for x in cls.xs)
        cls._rtol = config.TOLERANCE.get(str(
            cls.dtype)).get("first_order_grad").get("rtol")
        cls._atol = config.TOLERANCE.get(str(
            cls.dtype)).get("first_order_grad").get("atol")

    def setUp(self):
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()

    def tearDown(self):
        paddle.incubate.autograd.disable_prim()
        paddle.disable_static()

    def test_forward_grad(self):

        def expected():
            paddle.incubate.autograd.disable_prim()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False)
                _, ys_grad = paddle.incubate.autograd.jvp(
                    self.fun, static_xs, static_v)
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=ys_grad)
            paddle.incubate.autograd.enable_prim()
            return out

        def actual():
            paddle.incubate.autograd.enable_prim()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False)
                ys = self.fun(*static_xs) if isinstance(
                    static_xs, typing.Sequence) else self.fun(static_xs)
                ys_grad = paddle.incubate.autograd.forward_grad(
                    ys, static_xs, static_v)
                paddle.incubate.autograd.prim2orig(mp.block(0))
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=ys_grad)
            paddle.incubate.autograd.disable_prim()
            return out

        actual = actual()
        expected = expected()
        self.assertEqual(type(actual), type(expected))
        np.testing.assert_allclose(np.concatenate(actual),
                                   np.concatenate(expected),
                                   rtol=self._rtol,
                                   atol=self._atol)

    def test_prim_disabled(self):
        paddle.incubate.autograd.disable_prim()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with self.assertRaises(RuntimeError):
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False)
                ys = self.fun(*static_xs) if isinstance(
                    static_xs, typing.Sequence) else self.fun(static_xs)
                ys_grad = primapi.forward_grad(ys, static_xs, static_v)
                paddle.incubate.autograd.prim2orig(mp.block(0))
            exe = paddle.static.Executor()
            exe.run(sp)
            exe.run(mp, feed=feed, fetch_list=ys_grad)
        paddle.incubate.autograd.enable_prim()

    def test_illegal_param(self):
        paddle.incubate.autograd.enable_prim()
        with self.assertRaises(TypeError):
            primapi.forward_grad(1, paddle.static.data('inputs', shape=[1]))

        with self.assertRaises(TypeError):
            primapi.forward_grad(paddle.static.data('targets', shape=[1]), 1)
        paddle.incubate.autograd.disable_prim()


class TestGrad(unittest.TestCase):

    def setUp(self):
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()

    def tearDown(self):
        paddle.incubate.autograd.disable_prim()
        paddle.disable_static()

    def test_third_order(self):
        paddle.incubate.autograd.enable_prim()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name='x', shape=[1], dtype='float32')
            x2 = paddle.multiply(x, x)
            x3 = paddle.multiply(x2, x)
            x4 = paddle.multiply(x3, x)

            grad1, = paddle.incubate.autograd.grad([x4], [x])
            grad2, = paddle.incubate.autograd.grad([grad1], [x])
            grad3, = paddle.incubate.autograd.grad([grad2], [x])

            paddle.incubate.autograd.prim2orig(main.block(0))

        feed = {x.name: np.array([2.]).astype('float32')}
        fetch_list = [grad3.name]
        result = [np.array([48.])]

        place = paddle.CPUPlace()
        if paddle.device.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        exe.run(startup)
        outs = exe.run(main, feed=feed, fetch_list=fetch_list)
        np.allclose(outs, result)
        paddle.incubate.autograd.disable_prim()

    def test_fourth_order(self):
        paddle.incubate.autograd.enable_prim()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name='x', shape=[1], dtype='float32')
            x2 = paddle.multiply(x, x)
            x3 = paddle.multiply(x2, x)
            x4 = paddle.multiply(x3, x)
            x5 = paddle.multiply(x4, x)
            out = paddle.sqrt(x5 + x4)

            grad1, = paddle.incubate.autograd.grad([out], [x])
            grad2, = paddle.incubate.autograd.grad([grad1], [x])
            grad3, = paddle.incubate.autograd.grad([grad2], [x])
            grad4, = paddle.incubate.autograd.grad([grad3], [x])

            paddle.incubate.autograd.prim2orig(main.block(0))

        feed = {
            x.name: np.array([2.]).astype('float32'),
        }
        fetch_list = [grad4.name]
        # (3*(-5*x^2-16*x-16))/(16*(x+1)^3.5)
        result = [np.array([-0.27263762711])]

        place = paddle.CPUPlace()
        if paddle.device.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        exe.run(startup)
        outs = exe.run(main, feed=feed, fetch_list=fetch_list)
        np.allclose(outs, result)
        paddle.incubate.autograd.disable_prim()

    def test_disable_prim(self):

        def actual(x: np.array):
            paddle.incubate.autograd.disable_prim()
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                var_x = paddle.static.data('x', shape=x.shape, dtype=x.dtype)
                var_x.stop_gradient = False
                y = paddle.tanh(var_x)
                y_grad = paddle.incubate.autograd.grad(y, var_x)
                y_second_grad = paddle.incubate.autograd.grad(y_grad, var_x)
            exe = paddle.static.Executor()
            exe.run(startup)
            return exe.run(main,
                           feed={'x': x},
                           fetch_list=[y_grad, y_second_grad])

        def expect(x: np.array):
            paddle.incubate.autograd.disable_prim()
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                var_x = paddle.static.data('x', shape=x.shape, dtype=x.dtype)
                var_x.stop_gradient = False
                y = paddle.tanh(var_x)
                y_grad = paddle.static.gradients(y, var_x)
                y_second_grad = paddle.static.gradients(y_grad, var_x)
            exe = paddle.static.Executor()
            exe.run(startup)
            return exe.run(main,
                           feed={'x': x},
                           fetch_list=[y_grad, y_second_grad])

        x = np.random.randn(100, 200)
        for i, j in zip(actual(x), expect(x)):
            np.testing.assert_allclose(i, j)


if __name__ == '__main__':
    unittest.main()
