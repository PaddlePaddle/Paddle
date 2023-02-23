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

import autograd
import autograd.numpy as anp
import autograd.scipy as ascipy
import config
import numpy as np
import utils

import paddle
from paddle.incubate.autograd import primx


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'xs', 'dtype'),
    (
        (
            'uniform_random',
            lambda: paddle.uniform(
                [1, 2, 3], dtype='float32', min=0, max=1.0, seed=1
            ),
            (),
            'int32',
        ),
        (
            'sigmoid',
            paddle.nn.functional.sigmoid,
            (
                np.random.rand(
                    5,
                ),
            ),
            'float32',
        ),
    ),
)
class TestFowardApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xs = tuple(x.astype(cls.dtype) for x in cls.xs)

    def setUp(self):
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()

    def tearDown(self):
        paddle.incubate.autograd.disable_prim()
        paddle.disable_static()

    def test_grad(self):
        def expected():
            paddle.incubate.autograd.disable_prim()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs = utils.gen_static_inputs_and_feed(
                    self.xs, stop_gradient=False
                )
                out = self.fun(*static_xs)
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=out)
            paddle.incubate.autograd.enable_prim()
            return out

        def actual():
            paddle.incubate.autograd.enable_prim()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs = utils.gen_static_inputs_and_feed(
                    self.xs, stop_gradient=False
                )
                out = self.fun(*static_xs)
                primx.orig2prim(mp.block(0))
                primx.prim2orig(mp.block(0))
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=out)
            paddle.incubate.autograd.disable_prim()
            return out

        expected = expected()
        actual = actual()
        self.assertEqual(type(actual), type(expected))
        for i, j in zip(actual, expected):
            np.testing.assert_allclose(i, j, rtol=1e-6)


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'xs', 'v', 'dtype'),
    (
        (
            'dropout',
            paddle.nn.functional.dropout,
            (np.random.rand(5000, 5000),),
            None,
            'float32',
        ),
    ),
)
class TestDropoutGrad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xs = tuple(x.astype(cls.dtype) for x in cls.xs)
        cls._rtol = (
            config.TOLERANCE.get(str(cls.dtype))
            .get("first_order_grad")
            .get("rtol")
        )
        cls._atol = (
            config.TOLERANCE.get(str(cls.dtype))
            .get("first_order_grad")
            .get("atol")
        )

    def setUp(self):
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()

    def tearDown(self):
        paddle.incubate.autograd.disable_prim()
        paddle.disable_static()

    def test_grad(self):
        def expected():
            paddle.incubate.autograd.disable_prim()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False
                )
                _, ys_grad = paddle.incubate.autograd.vjp(
                    self.fun, static_xs, static_v
                )
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
                    self.xs, self.v, stop_gradient=False
                )
                ys = (
                    self.fun(*static_xs)
                    if isinstance(static_xs, typing.Sequence)
                    else self.fun(static_xs)
                )
                ys_grad = paddle.incubate.autograd.grad(ys, static_xs, static_v)
                paddle.incubate.autograd.prim2orig(mp.block(0))
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=ys_grad)
            paddle.incubate.autograd.disable_prim()
            return out

        expected = expected()
        actual = actual()
        self.assertEqual(type(actual), type(expected))
        for i, j in zip(actual, expected):
            np.testing.assert_allclose(np.sum(i), np.sum(j), rtol=1e-1)


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'xs', 'v', 'dtype'),
    (
        (
            'matmul',
            paddle.matmul,
            (np.random.rand(2, 3), np.random.rand(3, 2)),
            None,
            'float32',
        ),
    ),
)
class TestWithoutProgramGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xs = tuple(x.astype(cls.dtype) for x in cls.xs)
        cls._rtol = (
            config.TOLERANCE.get(str(cls.dtype))
            .get("first_order_grad")
            .get("rtol")
        )
        cls._atol = (
            config.TOLERANCE.get(str(cls.dtype))
            .get("first_order_grad")
            .get("atol")
        )

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
                    self.xs, self.v, stop_gradient=False
                )
                ys = (
                    self.fun(*static_xs)
                    if isinstance(static_xs, typing.Sequence)
                    else self.fun(static_xs)
                )
                ys_grad = paddle.incubate.autograd.forward_grad(
                    ys, static_xs, static_v
                )
                paddle.incubate.autograd.prim2orig(mp.block(0))
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=ys_grad)
            paddle.incubate.autograd.disable_prim()
            return out

        def without_program_guard():
            paddle.incubate.autograd.enable_prim()
            feed, static_xs, static_v = utils.gen_static_data_and_feed(
                self.xs, self.v, stop_gradient=False
            )
            ys = (
                self.fun(*static_xs)
                if isinstance(static_xs, typing.Sequence)
                else self.fun(static_xs)
            )
            ys_grad = paddle.incubate.autograd.forward_grad(
                ys, static_xs, static_v
            )
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
        np.testing.assert_allclose(
            np.concatenate(actual),
            np.concatenate(expected),
            rtol=self._rtol,
            atol=self._atol,
        )

    def test_grad_without_program_guard(self):
        def with_program_guard():
            paddle.incubate.autograd.enable_prim()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False
                )
                ys = (
                    self.fun(*static_xs)
                    if isinstance(static_xs, typing.Sequence)
                    else self.fun(static_xs)
                )
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
                self.xs, self.v, stop_gradient=False
            )
            ys = (
                self.fun(*static_xs)
                if isinstance(static_xs, typing.Sequence)
                else self.fun(static_xs)
            )
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
            np.testing.assert_allclose(
                np.concatenate(i),
                np.concatenate(j),
                rtol=self._rtol,
                atol=self._atol,
            )


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'xs', 'v', 'dtype'),
    (
        (
            'matmul',
            paddle.matmul,
            (np.random.rand(2, 3), np.random.rand(3, 2)),
            None,
            'float32',
        ),
        (
            'multiply',
            paddle.multiply,
            (np.random.rand(2, 3), np.random.rand(2, 3)),
            None,
            'float64',
        ),
        (
            'add',
            paddle.add,
            (np.random.rand(2, 3), np.random.rand(2, 3)),
            None,
            'float32',
        ),
        (
            'input_not_sequence',
            paddle.tanh,
            (np.random.rand(5, 5),),
            None,
            'float64',
        ),
        (
            'input_gradients_not_none',
            paddle.matmul,
            (np.random.rand(3, 3), np.random.rand(3, 3)),
            (np.random.rand(3, 3), np.random.rand(3, 3)),
            'float64',
        ),
        ('log', paddle.log, (np.random.rand(3, 4),), None, 'float32'),
        (
            'abs',
            paddle.abs,
            (np.random.uniform(-10, 10, (10, 10)),),
            None,
            'float32',
        ),
        ('rsqrt', paddle.rsqrt, (np.random.rand(100, 200),), None, 'float32'),
        (
            'sigmoid',
            paddle.nn.functional.sigmoid,
            (
                np.random.rand(
                    5,
                ),
            ),
            None,
            'float32',
        ),
    ),
)
# paddle.where, paddle.pow, paddle.maximum has no double grad definition,
# can not compute forward grad use double trick
class TestForwardGrad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xs = tuple(x.astype(cls.dtype) for x in cls.xs)
        cls._rtol = (
            config.TOLERANCE.get(str(cls.dtype))
            .get("first_order_grad")
            .get("rtol")
        )
        cls._atol = (
            config.TOLERANCE.get(str(cls.dtype))
            .get("first_order_grad")
            .get("atol")
        )

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
                    self.xs, self.v, stop_gradient=False
                )
                _, ys_grad = paddle.incubate.autograd.jvp(
                    self.fun, static_xs, static_v
                )
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
                    self.xs, self.v, stop_gradient=False
                )
                ys = (
                    self.fun(*static_xs)
                    if isinstance(static_xs, typing.Sequence)
                    else self.fun(static_xs)
                )
                ys_grad = paddle.incubate.autograd.forward_grad(
                    ys, static_xs, static_v
                )
                paddle.incubate.autograd.prim2orig(mp.block(0))
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=ys_grad)
            paddle.incubate.autograd.disable_prim()
            return out

        actual = actual()
        expected = expected()
        self.assertEqual(type(actual), type(expected))
        np.testing.assert_allclose(
            np.concatenate(actual),
            np.concatenate(expected),
            rtol=self._rtol,
            atol=self._atol,
        )

    def test_prim_disabled(self):
        paddle.incubate.autograd.disable_prim()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with self.assertRaises(RuntimeError):
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False
                )
                ys = (
                    self.fun(*static_xs)
                    if isinstance(static_xs, typing.Sequence)
                    else self.fun(static_xs)
                )
                ys_grad = paddle.incubate.autograd.forward_grad(
                    ys, static_xs, static_v
                )
                paddle.incubate.autograd.prim2orig(mp.block(0))
            exe = paddle.static.Executor()
            exe.run(sp)
            exe.run(mp, feed=feed, fetch_list=ys_grad)
        paddle.incubate.autograd.enable_prim()

    def test_illegal_param(self):
        paddle.incubate.autograd.enable_prim()
        with self.assertRaises(TypeError):
            paddle.incubate.autograd.forward_grad(
                1, paddle.static.data('inputs', shape=[1])
            )

        with self.assertRaises(TypeError):
            paddle.incubate.autograd.forward_grad(
                paddle.static.data('targets', shape=[1]), 1
            )
        paddle.incubate.autograd.disable_prim()


where_wrap = lambda x, y: paddle.where(paddle.eye(3, 4) == 1, x, y)


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'xs', 'v', 'dtype'),
    (
        (
            'matmul',
            paddle.matmul,
            (np.random.rand(2, 3), np.random.rand(3, 2)),
            None,
            'float32',
        ),
        (
            'multiply',
            paddle.multiply,
            (np.random.rand(2, 3), np.random.rand(2, 3)),
            None,
            'float64',
        ),
        (
            'div',
            paddle.divide,
            (np.random.rand(2, 3), np.random.rand(2, 3)),
            None,
            'float64',
        ),
        (
            'add',
            paddle.add,
            (np.random.rand(2, 3), np.random.rand(2, 3)),
            None,
            'float32',
        ),
        (
            'input_not_sequence',
            paddle.tanh,
            (np.random.rand(5, 5),),
            None,
            'float64',
        ),
        (
            'input_gradients_not_none',
            paddle.matmul,
            (np.random.rand(3, 3), np.random.rand(3, 3)),
            (np.random.rand(3, 3),),
            'float64',
        ),
        ('sin', paddle.sin, (np.random.rand(100, 200),), None, 'float32'),
        ('rsqrt', paddle.rsqrt, (np.random.rand(100, 200),), None, 'float32'),
        ('cos', paddle.cos, (np.random.rand(200, 90),), None, 'float32'),
        ('exp', paddle.exp, (np.random.rand(299, 320),), None, 'float32'),
        # In where op, grad of condition computed by paddle.static.gradients is None,
        # and paddle.incubate.autograd.grad will replace None with zeros while transpose
        # will just return None because cond_dot is unused, that is a diff.
        (
            'select',
            where_wrap,
            (np.random.rand(3, 4), np.random.rand(3, 4)),
            None,
            'float32',
        ),
        # pow_p and pow has diff when compute z_dot of 0^0
        (
            'pow',
            paddle.pow,
            (np.array([1, 2, 3]), np.array([0, 2, 7])),
            None,
            'float32',
        ),
        # To make max_p consistent with paddle.maximum, be sure x.grad = 0 and y.grad = 1 when x==y.
        (
            'max',
            paddle.maximum,
            (
                np.array([1, 2, 3]),
                np.array([2, 2, 2]),
            ),
            None,
            'float32',
        ),
        ('erf', paddle.erf, (np.random.rand(300, 288),), None, 'float32'),
        (
            'gelu',
            paddle.nn.functional.gelu,
            (np.random.rand(200, 189),),
            None,
            'float32',
        ),
        (
            'gelu_approximate',
            lambda x: paddle.nn.functional.gelu(x, True),
            (np.random.rand(200, 189),),
            None,
            'float32',
        ),
        ('sum', paddle.sum, (np.random.rand(200, 345),), None, 'float32'),
        (
            'sigmoid',
            paddle.nn.functional.sigmoid,
            (
                np.random.rand(
                    5,
                ),
            ),
            None,
            'float32',
        ),
        (
            'sum_with_axis',
            lambda x: paddle.sum(x, axis=1),
            (np.random.rand(200, 345),),
            None,
            'float32',
        ),
        (
            'sum_with_keepdim',
            lambda x: paddle.sum(x, keepdim=True),
            (np.random.rand(200, 345),),
            None,
            'float32',
        ),
        ('mean', paddle.mean, (np.random.rand(200, 345),), None, 'float32'),
        (
            'mean_with_axis',
            lambda x: paddle.mean(x, axis=1),
            (np.random.rand(200, 345),),
            None,
            'float32',
        ),
        (
            'mean_with_keepdim',
            lambda x: paddle.mean(x, keepdim=True),
            (np.random.rand(200, 345),),
            None,
            'float32',
        ),
        (
            'mean_with_axis_keepdim',
            lambda x: paddle.mean(x, axis=0, keepdim=True),
            (np.random.rand(200, 345),),
            None,
            'float32',
        ),
        (
            'abs',
            paddle.abs,
            (np.random.uniform(-10, 10, (200, 345)),),
            None,
            'float32',
        ),
        (
            'cast_float',
            lambda x: paddle.cast(x, paddle.float64),
            (np.random.rand(10, 20),),
            None,
            'float32',
        ),
        (
            'cast_int',
            lambda x: paddle.cast(x, paddle.int32),
            (np.random.rand(10, 20),),
            None,
            'float32',
        ),
        ('square', paddle.square, (np.random.rand(100),), None, 'float32'),
        (
            'pow_scalar',
            lambda x: paddle.pow(x, 2),
            (np.random.rand(20, 30),),
            None,
            'float32',
        ),
        (
            'var',
            lambda x: paddle.var(x, unbiased=False),
            (np.random.rand(200, 324),),
            None,
            'float32',
        ),
        (
            'var_with_axis',
            lambda x: paddle.var(x, axis=1, unbiased=False),
            (np.random.rand(10, 20, 30),),
            None,
            'float32',
        ),
        (
            'var_with_keepdim',
            lambda x: paddle.var(x, axis=1, keepdim=True, unbiased=False),
            (np.random.rand(10, 20, 30),),
            None,
            'float32',
        ),
        (
            'bn',
            lambda x, w, b: paddle.nn.functional.batch_norm(
                x, paddle.ones((10,)), paddle.ones((10,)), w, b
            ),
            (np.random.rand(10, 10), np.random.rand(10), np.random.rand(10)),
            None,
            'float32',
        ),
        (
            'bn_train',
            lambda x, w, b: paddle.nn.functional.batch_norm(
                x, paddle.ones((10,)), paddle.ones((10,)), w, b, training=True
            ),
            (np.random.rand(10, 10), np.random.rand(10), np.random.rand(10)),
            None,
            'float32',
        ),
        (
            'bn_nhwc',
            lambda x, w, b: paddle.nn.functional.batch_norm(
                x,
                paddle.ones((10,)) + 1,
                paddle.ones((10,)),
                w,
                b,
                training=True,
                data_format='NHWC',
            ),
            (np.random.rand(10, 10), np.random.rand(10), np.random.rand(10)),
            None,
            'float32',
        ),
        (
            'bn_global_stat',
            lambda x, w, b: paddle.nn.functional.batch_norm(
                x,
                paddle.ones((10,)) + 3.2,
                paddle.ones((10,)) + 6.7,
                w,
                b,
                training=True,
                data_format='NHWC',
                use_global_stats=True,
            ),
            (np.random.rand(10, 10), np.random.rand(10), np.random.rand(10)),
            None,
            'float32',
        ),
    ),
)
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
        cls._rtol = (
            config.TOLERANCE.get(str(cls.dtype))
            .get("first_order_grad")
            .get("rtol")
        )
        cls._atol = (
            config.TOLERANCE.get(str(cls.dtype))
            .get("first_order_grad")
            .get("atol")
        )

    def test_grad(self):
        def expected():
            paddle.incubate.autograd.disable_prim()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False
                )
                _, ys_grad = paddle.incubate.autograd.vjp(
                    self.fun, static_xs, static_v
                )
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
                    self.xs, self.v, stop_gradient=False
                )
                ys = (
                    self.fun(*static_xs)
                    if isinstance(static_xs, typing.Sequence)
                    else self.fun(static_xs)
                )
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
                1, paddle.static.data('inputs', shape=[1])
            )

        with self.assertRaises(TypeError):
            paddle.incubate.autograd.grad(
                paddle.static.data('targets', shape=[1]), 1
            )
        paddle.incubate.autograd.disable_prim()

    def test_disable_prim(self):
        def expected():
            paddle.incubate.autograd.disable_prim()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False
                )
                ys = (
                    self.fun(*static_xs)
                    if isinstance(static_xs, typing.Sequence)
                    else self.fun(static_xs)
                )
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
                    self.xs, self.v, stop_gradient=False
                )
                ys = (
                    self.fun(*static_xs)
                    if isinstance(static_xs, typing.Sequence)
                    else self.fun(static_xs)
                )
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


def multiply_pd(x):
    x2 = paddle.multiply(x, x)
    x3 = paddle.multiply(x2, x2)
    x4 = paddle.multiply(x3, x)
    return x4


multiply_ag = lambda xs: xs[0] * xs[0] * xs[0] * xs[0] * xs[0]
sin_ag = lambda xs: anp.sin(xs[0])
cos_ag = lambda xs: anp.cos(xs[0])
exp_ag = lambda xs: anp.exp(xs[0])
pow_ag = lambda xs: xs[0] ** xs[1]
log_ag = lambda xs: anp.log(xs[0])
erf_ag = lambda xs: ascipy.special.erf(xs[0])
sigmoid_ag = lambda xs: 1.0 / (1 + anp.exp(-xs[0]))


def gelu_ag(x, approximate=False):
    if approximate:
        sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x.dtype)
        cdf = 0.5 * (1.0 + anp.tanh(sqrt_2_over_pi * (x + 0.044715 * (x**3))))
        return x * cdf
    else:
        return x * (ascipy.special.erf(x / np.sqrt(2)) + 1) / 2


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun_pd', 'fun_ag', 'xs', 'v', 'dtype'),
    (
        (
            'multiply',
            multiply_pd,
            multiply_ag,
            (np.random.rand(3, 5),),
            None,
            'float32',
        ),
        ('sin', paddle.sin, sin_ag, (np.random.rand(2, 3),), None, 'float32'),
        ('cos', paddle.cos, cos_ag, (np.random.rand(3, 4),), None, 'float32'),
        ('exp', paddle.exp, exp_ag, (np.random.rand(2, 3),), None, 'float32'),
        (
            'pow',
            paddle.pow,
            pow_ag,
            (np.random.rand(2, 3), np.random.rand(2, 3)),
            None,
            'float32',
        ),
        ('log', paddle.log, log_ag, (np.random.rand(3, 8),), None, 'float32'),
        (
            'erf',
            paddle.erf,
            erf_ag,
            (np.random.rand(100, 200),),
            None,
            'float32',
        ),
        (
            'gelu',
            paddle.nn.functional.gelu,
            lambda xs: gelu_ag(xs[0]),
            (np.random.rand(10, 20, 30),),
            None,
            'float32',
        ),
        (
            'gelu_approximate',
            lambda x: paddle.nn.functional.gelu(x, approximate=True),
            lambda xs: gelu_ag(xs[0], approximate=True),
            (np.random.rand(10, 20, 30),),
            None,
            'float32',
        ),
        (
            'sigmoid',
            paddle.nn.functional.sigmoid,
            sigmoid_ag,
            (np.random.rand(10, 20),),
            None,
            'float32',
        ),
    ),
)
class TestGradWithHigherOrder(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()

    def tearDown(self):
        paddle.incubate.autograd.disable_prim()
        paddle.disable_static()

    @classmethod
    def setUpClass(cls):
        cls.xs = tuple(x.astype(cls.dtype) for x in cls.xs)
        cls._rtol = (
            config.TOLERANCE.get(str(cls.dtype))
            .get("first_order_grad")
            .get("rtol")
        )
        cls._atol = (
            config.TOLERANCE.get(str(cls.dtype))
            .get("first_order_grad")
            .get("atol")
        )

    def test_grad(self):
        def expected():
            egrad = autograd.elementwise_grad
            grad_3 = egrad(egrad(egrad(self.fun_ag)))(self.xs)
            grad_4 = egrad(egrad(egrad(egrad(self.fun_ag))))(self.xs)
            grad_5 = egrad(egrad(egrad(egrad(egrad(self.fun_ag)))))(self.xs)
            # the output of egrad is tuple
            return list(grad_3 + grad_4 + grad_5)

        def actual():
            paddle_grad = paddle.incubate.autograd.grad
            paddle.incubate.autograd.enable_prim()
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False
                )
                ys = (
                    self.fun_pd(*static_xs)
                    if isinstance(static_xs, typing.Sequence)
                    else self.fun_pd(static_xs)
                )

                grad1 = paddle_grad(ys, static_xs, static_v)
                grad2 = paddle_grad(grad1, static_xs, static_v)
                grad3 = paddle_grad(grad2, static_xs, static_v)
                grad4 = paddle_grad(grad3, static_xs, static_v)
                grad5 = paddle_grad(grad4, static_xs, static_v)
                paddle.incubate.autograd.prim2orig()

            fetch_list = [grad3, grad4, grad5]

            place = paddle.CPUPlace()
            if paddle.device.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
            exe = paddle.static.Executor(place)
            exe.run(startup)
            outs = exe.run(main, feed=feed, fetch_list=fetch_list)
            paddle.incubate.autograd.disable_prim()
            return outs

        actual = actual()
        expected = expected()
        self.assertEqual(type(actual), type(expected))
        for i, j in zip(actual, expected):
            np.testing.assert_allclose(i, j, rtol=self._rtol, atol=self._atol)


if __name__ == '__main__':
    unittest.main()
