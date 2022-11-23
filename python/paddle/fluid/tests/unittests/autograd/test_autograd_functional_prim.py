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

<<<<<<< HEAD
import typing
=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
import unittest

import numpy as np
import paddle

import config
import utils


@utils.place(config.DEVICES)
<<<<<<< HEAD
@utils.parameterize((utils.TEST_CASE_NAME, 'fun', 'args', 'dtype'), (
    ('unary_float32', paddle.tanh, (np.random.rand(2, 3), ), 'float32'),
    ('binary_float32', paddle.matmul,
     (np.random.rand(2, 3), np.random.rand(3, 2)), 'float32'),
    ('unary_float64', paddle.tanh, (np.random.rand(2, 3), ), 'float64'),
    ('binary_float64', paddle.matmul,
     (np.random.rand(2, 3), np.random.rand(3, 2)), 'float64'),
))
class TestJacobianPrim(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.args = [arg.astype(cls.dtype) for arg in cls.args]
        cls._rtol = config.TOLERANCE.get(
            cls.dtype).get('first_order_grad').get('rtol')
        cls._atol = config.TOLERANCE.get(
            cls.dtype).get('first_order_grad').get('atol')
=======
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'args', 'dtype'),
    (
        ('unary_float32', paddle.tanh, (np.random.rand(2, 3),), 'float32'),
        (
            'binary_float32',
            paddle.matmul,
            (np.random.rand(2, 3), np.random.rand(3, 2)),
            'float32',
        ),
        ('unary_float64', paddle.tanh, (np.random.rand(2, 3),), 'float64'),
        (
            'binary_float64',
            paddle.matmul,
            (np.random.rand(2, 3), np.random.rand(3, 2)),
            'float64',
        ),
    ),
)
class TestJacobianPrim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = [arg.astype(cls.dtype) for arg in cls.args]
        cls._rtol = (
            config.TOLERANCE.get(cls.dtype).get('first_order_grad').get('rtol')
        )
        cls._atol = (
            config.TOLERANCE.get(cls.dtype).get('first_order_grad').get('atol')
        )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

    def setUp(self):
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()

    def tearDown(self):
        paddle.incubate.autograd.disable_prim()
        paddle.disable_static()

    def test_jacobian_prim(self):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        def wrapper(fun, args):
            mp = paddle.static.Program()
            sp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                static_args = [
                    paddle.static.data(f'arg{i}', arg.shape, self.dtype)
                    for i, arg in enumerate(args)
                ]
                for arg in static_args:
                    arg.stop_gradient = False
                jac = paddle.incubate.autograd.Jacobian(fun, static_args)[:]
                if paddle.incubate.autograd.prim_enabled():
                    paddle.incubate.autograd.prim2orig()
            exe = paddle.static.Executor()
            exe.run(sp)
<<<<<<< HEAD
            [jac] = exe.run(mp,
                            feed={f'arg{i}': arg
                                  for i, arg in enumerate(args)},
                            fetch_list=[jac])
=======
            [jac] = exe.run(
                mp,
                feed={f'arg{i}': arg for i, arg in enumerate(args)},
                fetch_list=[jac],
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            return jac

        paddle.incubate.autograd.enable_prim()
        prim_jac = wrapper(self.fun, self.args)
        paddle.incubate.autograd.disable_prim()
        orig_jac = wrapper(self.fun, self.args)

<<<<<<< HEAD
        np.testing.assert_allclose(orig_jac,
                                   prim_jac,
                                   rtol=self._rtol,
                                   atol=self._atol)


@utils.place(config.DEVICES)
@utils.parameterize((utils.TEST_CASE_NAME, 'fun', 'args', 'dtype'), (
    ('unary_float32', paddle.tanh, (np.random.rand(1), ), 'float32'),
    ('binary_float32', paddle.multiply,
     (np.random.rand(1), np.random.rand(1)), 'float32'),
    ('unary_float64', paddle.tanh, (np.random.rand(1), ), 'float64'),
    ('binary_float64', paddle.multiply,
     (np.random.rand(1), np.random.rand(1)), 'float64'),
))
class TestHessianPrim(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.args = [arg.astype(cls.dtype) for arg in cls.args]
        cls._rtol = config.TOLERANCE.get(
            cls.dtype).get('second_order_grad').get('rtol')
        cls._atol = config.TOLERANCE.get(
            cls.dtype).get('second_order_grad').get('atol')
=======
        np.testing.assert_allclose(
            orig_jac, prim_jac, rtol=self._rtol, atol=self._atol
        )


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'args', 'dtype'),
    (
        ('unary_float32', paddle.tanh, (np.random.rand(1),), 'float32'),
        (
            'binary_float32',
            paddle.multiply,
            (np.random.rand(1), np.random.rand(1)),
            'float32',
        ),
        ('unary_float64', paddle.tanh, (np.random.rand(1),), 'float64'),
        (
            'binary_float64',
            paddle.multiply,
            (np.random.rand(1), np.random.rand(1)),
            'float64',
        ),
    ),
)
class TestHessianPrim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = [arg.astype(cls.dtype) for arg in cls.args]
        cls._rtol = (
            config.TOLERANCE.get(cls.dtype).get('second_order_grad').get('rtol')
        )
        cls._atol = (
            config.TOLERANCE.get(cls.dtype).get('second_order_grad').get('atol')
        )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

    def setUp(self):
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()

    def tearDown(self):
        paddle.incubate.autograd.disable_prim()
        paddle.disable_static()

    def test_jacobian_prim(self):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        def wrapper(fun, args):
            mp = paddle.static.Program()
            sp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                static_args = [
                    paddle.static.data(f'arg{i}', arg.shape, self.dtype)
                    for i, arg in enumerate(args)
                ]
                for arg in static_args:
                    arg.stop_gradient = False
                hessian = paddle.incubate.autograd.Hessian(fun, static_args)[:]
                if paddle.incubate.autograd.prim_enabled():
                    paddle.incubate.autograd.prim2orig()
            exe = paddle.static.Executor()
            exe.run(sp)
<<<<<<< HEAD
            [hessian
             ] = exe.run(mp,
                         feed={f'arg{i}': arg
                               for i, arg in enumerate(args)},
                         fetch_list=[hessian])
=======
            [hessian] = exe.run(
                mp,
                feed={f'arg{i}': arg for i, arg in enumerate(args)},
                fetch_list=[hessian],
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            return hessian

        paddle.incubate.autograd.enable_prim()
        prim_jac = wrapper(self.fun, self.args)
        paddle.incubate.autograd.disable_prim()
        orig_jac = wrapper(self.fun, self.args)

<<<<<<< HEAD
        np.testing.assert_allclose(orig_jac,
                                   prim_jac,
                                   rtol=self._rtol,
                                   atol=self._atol)


@utils.place(config.DEVICES)
@utils.parameterize((utils.TEST_CASE_NAME, 'fun', 'args', 'dtype'), (
    ('unary_float32', paddle.tanh, (np.random.rand(2, 3), ), 'float32'),
    ('binary_float32', paddle.matmul,
     (np.random.rand(2, 3), np.random.rand(3, 2)), 'float32'),
    ('unary_float64', paddle.tanh, (np.random.rand(2, 3), ), 'float64'),
    ('binary_float64', paddle.matmul,
     (np.random.rand(2, 3), np.random.rand(3, 2)), 'float64'),
))
class TestJvpPrim(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.args = [arg.astype(cls.dtype) for arg in cls.args]
        cls._rtol = config.TOLERANCE.get(
            cls.dtype).get('first_order_grad').get('rtol')
        cls._atol = config.TOLERANCE.get(
            cls.dtype).get('first_order_grad').get('atol')
=======
        np.testing.assert_allclose(
            orig_jac, prim_jac, rtol=self._rtol, atol=self._atol
        )


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'args', 'dtype'),
    (
        ('unary_float32', paddle.tanh, (np.random.rand(2, 3),), 'float32'),
        (
            'binary_float32',
            paddle.matmul,
            (np.random.rand(2, 3), np.random.rand(3, 2)),
            'float32',
        ),
        ('unary_float64', paddle.tanh, (np.random.rand(2, 3),), 'float64'),
        (
            'binary_float64',
            paddle.matmul,
            (np.random.rand(2, 3), np.random.rand(3, 2)),
            'float64',
        ),
    ),
)
class TestJvpPrim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = [arg.astype(cls.dtype) for arg in cls.args]
        cls._rtol = (
            config.TOLERANCE.get(cls.dtype).get('first_order_grad').get('rtol')
        )
        cls._atol = (
            config.TOLERANCE.get(cls.dtype).get('first_order_grad').get('atol')
        )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

    def setUp(self):
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()

    def tearDown(self):
        paddle.incubate.autograd.disable_prim()
        paddle.disable_static()

    def test_jacobian_prim(self):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        def wrapper(fun, args):
            mp = paddle.static.Program()
            sp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                static_args = [
                    paddle.static.data(f'arg{i}', arg.shape, self.dtype)
                    for i, arg in enumerate(args)
                ]
                for arg in static_args:
                    arg.stop_gradient = False
                _, jvp_res = paddle.incubate.autograd.jvp(fun, static_args)
                if paddle.incubate.autograd.prim_enabled():
                    paddle.incubate.autograd.prim2orig()
            exe = paddle.static.Executor()
            exe.run(sp)
            jvp_res = exe.run(
                mp,
<<<<<<< HEAD
                feed={f'arg{i}': arg
                      for i, arg in enumerate(args)},
                fetch_list=[jvp_res])
=======
                feed={f'arg{i}': arg for i, arg in enumerate(args)},
                fetch_list=[jvp_res],
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            return jvp_res

        paddle.incubate.autograd.enable_prim()
        prim_jvp = wrapper(self.fun, self.args)
        paddle.incubate.autograd.disable_prim()
        orig_jvp = wrapper(self.fun, self.args)

<<<<<<< HEAD
        np.testing.assert_allclose(orig_jvp,
                                   prim_jvp,
                                   rtol=self._rtol,
                                   atol=self._atol)


@utils.place(config.DEVICES)
@utils.parameterize((utils.TEST_CASE_NAME, 'fun', 'args', 'dtype'), (
    ('unary_float32', paddle.tanh, (np.random.rand(2, 3), ), 'float32'),
    ('binary_float32', paddle.matmul,
     (np.random.rand(2, 3), np.random.rand(3, 2)), 'float32'),
    ('unary_float64', paddle.tanh, (np.random.rand(2, 3), ), 'float64'),
    ('binary_float64', paddle.matmul,
     (np.random.rand(2, 3), np.random.rand(3, 2)), 'float64'),
))
class TestVjpPrim(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.args = [arg.astype(cls.dtype) for arg in cls.args]
        cls._rtol = config.TOLERANCE.get(
            cls.dtype).get('first_order_grad').get('rtol')
        cls._atol = config.TOLERANCE.get(
            cls.dtype).get('first_order_grad').get('atol')
=======
        np.testing.assert_allclose(
            orig_jvp, prim_jvp, rtol=self._rtol, atol=self._atol
        )


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'args', 'dtype'),
    (
        ('unary_float32', paddle.tanh, (np.random.rand(2, 3),), 'float32'),
        (
            'binary_float32',
            paddle.matmul,
            (np.random.rand(2, 3), np.random.rand(3, 2)),
            'float32',
        ),
        ('unary_float64', paddle.tanh, (np.random.rand(2, 3),), 'float64'),
        (
            'binary_float64',
            paddle.matmul,
            (np.random.rand(2, 3), np.random.rand(3, 2)),
            'float64',
        ),
    ),
)
class TestVjpPrim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = [arg.astype(cls.dtype) for arg in cls.args]
        cls._rtol = (
            config.TOLERANCE.get(cls.dtype).get('first_order_grad').get('rtol')
        )
        cls._atol = (
            config.TOLERANCE.get(cls.dtype).get('first_order_grad').get('atol')
        )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

    def setUp(self):
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()

    def tearDown(self):
        paddle.incubate.autograd.disable_prim()
        paddle.disable_static()

    def test_jacobian_prim(self):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        def wrapper(fun, args):
            mp = paddle.static.Program()
            sp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                static_args = [
                    paddle.static.data(f'arg{i}', arg.shape, self.dtype)
                    for i, arg in enumerate(args)
                ]
                for arg in static_args:
                    arg.stop_gradient = False
                _, vjp_res = paddle.incubate.autograd.vjp(fun, static_args)
                if paddle.incubate.autograd.prim_enabled():
                    paddle.incubate.autograd.prim2orig()
            exe = paddle.static.Executor()
            exe.run(sp)
            vjp_res = exe.run(
                mp,
<<<<<<< HEAD
                feed={f'arg{i}': arg
                      for i, arg in enumerate(args)},
                fetch_list=[vjp_res])
=======
                feed={f'arg{i}': arg for i, arg in enumerate(args)},
                fetch_list=[vjp_res],
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            return vjp_res

        paddle.incubate.autograd.enable_prim()
        prim_vjp = wrapper(self.fun, self.args)
        paddle.incubate.autograd.disable_prim()
        orig_vjp = wrapper(self.fun, self.args)

        for orig, prim in zip(orig_vjp, prim_vjp):
<<<<<<< HEAD
            np.testing.assert_allclose(orig,
                                       prim,
                                       rtol=self._rtol,
                                       atol=self._atol)
=======
            np.testing.assert_allclose(
                orig, prim, rtol=self._rtol, atol=self._atol
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91


if __name__ == "__main__":
    unittest.main()
