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
@utils.parameterize((utils.TEST_CASE_NAME, 'fun', 'xs', 'v', 'dtype'), (
    ('matmul', paddle.matmul,
     (np.random.rand(2, 3), np.random.rand(3, 2)), None, 'float32'),
    ('multiply', paddle.multiply,
     (np.random.rand(2, 3), np.random.rand(2, 3)), None, 'float64'),
    ('add', paddle.add,
     (np.random.rand(2, 3), np.random.rand(2, 3)), None, 'float32'),
    ('input_not_sequence', paddle.tanh,
     (np.random.rand(5, 5), ), None, 'float64'),
    ('input_gradients_not_none', paddle.matmul,
     (np.random.rand(3, 3), np.random.rand(3, 3)),
     (np.random.rand(3, 3), np.random.rand(3, 3)), 'float64'),
    ('log', paddle.log, (np.random.rand(3, 4), ), None, 'float32'),
))
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
                ys_grad = paddle.incubate.autograd.forward_grad(
                    ys, static_xs, static_v)
                paddle.incubate.autograd.prim2orig(mp.block(0))
            exe = paddle.static.Executor()
            exe.run(sp)
            exe.run(mp, feed=feed, fetch_list=ys_grad)
        paddle.incubate.autograd.enable_prim()

    def test_illegal_param(self):
        paddle.incubate.autograd.enable_prim()
        with self.assertRaises(TypeError):
            paddle.incubate.autograd.forward_grad(
                1, paddle.static.data('inputs', shape=[1]))

        with self.assertRaises(TypeError):
            paddle.incubate.autograd.forward_grad(
                paddle.static.data('targets', shape=[1]), 1)
        paddle.incubate.autograd.disable_prim()


@utils.place(config.DEVICES)
@utils.parameterize((utils.TEST_CASE_NAME, 'fun', 'xs', 'v', 'dtype'), (
    ('matmul', paddle.matmul,
     (np.random.rand(2, 3), np.random.rand(3, 2)), None, 'float32'),
    ('multiply', paddle.multiply,
     (np.random.rand(2, 3), np.random.rand(2, 3)), None, 'float64'),
    ('add', paddle.add,
     (np.random.rand(2, 3), np.random.rand(2, 3)), None, 'float32'),
    ('input_not_sequence', paddle.tanh,
     (np.random.rand(5, 5), ), None, 'float64'),
    ('input_gradients_not_none', paddle.matmul,
     (np.random.rand(3, 3), np.random.rand(3, 3)),
     (np.random.rand(3, 3), ), 'float64'),
    ('sin', paddle.sin, (np.random.rand(100, 200), ), None, 'float32'),
    ('cos', paddle.cos, (np.random.rand(200, 90), ), None, 'float32'),
    ('exp', paddle.exp, (np.random.rand(299, 320), ), None, 'float32'),
    ('log', paddle.log, (np.random.rand(3, 4), ), None, 'float32'),
))
class TestGrad(unittest.TestCase):

    def setUp(self):
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()

    def tearDown(self):
        paddle.incubate.autograd.disable_prim()
        paddle.disable_static()

    @classmethod
    def setUpClass(cls):
        cls.xs = tuple(x.astype(cls.dtype) for x in cls.xs)
        cls._rtol = config.TOLERANCE.get(str(
            cls.dtype)).get("first_order_grad").get("rtol")
        cls._atol = config.TOLERANCE.get(str(
            cls.dtype)).get("first_order_grad").get("atol")

    def test_grad(self):

        def expected():
            paddle.incubate.autograd.disable_prim()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False)
                _, ys_grad = paddle.incubate.autograd.vjp(
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
                ys_grad = paddle.incubate.autograd.grad(ys, static_xs, static_v)
                paddle.incubate.autograd.prim2orig(mp.block(0))
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=ys_grad)
            paddle.incubate.autograd.disable_prim()
            return out

        actual = actual()
        expected = expected()
        self.assertEqual(type(actual), type(expected))
        for i, j in zip(actual, expected):
            np.testing.assert_allclose(i, j, rtol=self._rtol, atol=self._atol)

    def test_illegal_param(self):
        paddle.incubate.autograd.enable_prim()
        with self.assertRaises(TypeError):
            paddle.incubate.autograd.grad(
                1, paddle.static.data('inputs', shape=[1]))

        with self.assertRaises(TypeError):
            paddle.incubate.autograd.grad(
                paddle.static.data('targets', shape=[1]), 1)
        paddle.incubate.autograd.disable_prim()

    def test_disable_prim(self):

        def expected():
            paddle.incubate.autograd.disable_prim()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False)
                ys = self.fun(*static_xs) if isinstance(
                    static_xs, typing.Sequence) else self.fun(static_xs)
                ys_grad = paddle.incubate.autograd.grad(ys, static_xs, static_v)
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=ys_grad)
            paddle.incubate.autograd.enable_prim()
            return out

        def actual():
            paddle.incubate.autograd.disable_prim()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False)
                ys = self.fun(*static_xs) if isinstance(
                    static_xs, typing.Sequence) else self.fun(static_xs)
                ys_grad = paddle.static.gradients(ys, static_xs, static_v)
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=ys_grad)
            paddle.incubate.autograd.enable_prim()
            return out

        actual = actual()
        expected = expected()
        self.assertEqual(type(actual), type(expected))
        for i, j in zip(actual, expected):
            np.testing.assert_allclose(i, j, rtol=self._rtol, atol=self._atol)


class TestGradWithHigherOrder(unittest.TestCase):

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
        np.testing.assert_allclose(outs, result, rtol=1e-5, atol=1e-5)
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
        np.testing.assert_allclose(outs, result, rtol=1e-5, atol=1e-5)
        paddle.incubate.autograd.disable_prim()

    def test_fifth_order(self):
        paddle.incubate.autograd.enable_prim()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name='x', shape=[1], dtype='float32')
            x2 = paddle.multiply(x, x)
            x3 = paddle.multiply(x2, x)
            x4 = paddle.multiply(x3, x)
            x5 = paddle.multiply(x4, x)
            x6 = paddle.multiply(x5, x)
            out = x6 + x5

            grad1, = paddle.incubate.autograd.grad([out], [x])
            grad2, = paddle.incubate.autograd.grad([grad1], [x])
            grad3, = paddle.incubate.autograd.grad([grad2], [x])
            grad4, = paddle.incubate.autograd.grad([grad3], [x])
            grad5, = paddle.incubate.autograd.grad([grad4], [x])

            paddle.incubate.autograd.prim2orig()

        feed = {
            x.name: np.array([2.]).astype('float32'),
        }
        fetch_list = [grad5.name]
        result = [np.array([1560.0])]

        place = paddle.CPUPlace()
        if paddle.device.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        exe.run(startup)
        outs = exe.run(main, feed=feed, fetch_list=fetch_list)
        np.testing.assert_allclose(outs, result, rtol=1e-5, atol=1e-5)
        paddle.incubate.autograd.disable_prim()


if __name__ == '__main__':
    unittest.main()
