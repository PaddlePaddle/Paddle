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
import parameterize as param
from distribution import config

import paddle
from paddle.distribution import transform, variable

np.random.seed(2022)
paddle.seed(2022)
paddle.enable_static()


@param.place(config.DEVICES)
class TestTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.Transform()

    @param.param_func(
        [
            (transform.Type.BIJECTION, True),
            (transform.Type.INJECTION, True),
            (transform.Type.SURJECTION, False),
            (transform.Type.OTHER, False),
        ]
    )
    def test_is_injective(self, type, expected):
        transform.Transform._type = type
        self.assertEqual(self._t._is_injective(), expected)

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Real))

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Real))

    @param.param_func(
        [
            (np.array(0), NotImplementedError),
            (np.random.random((2, 3)), NotImplementedError),
        ]
    )
    def test_forward(self, input, expected):
        with self.assertRaises(expected):
            exe = paddle.static.Executor()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                t = transform.Transform()
                static_input = paddle.static.data(
                    'input', input.shape, input.dtype
                )
                output = t.forward(static_input)
            exe.run(sp)
            exe.run(mp, feed={'input': input}, fetch_list=[output])

    @param.param_func(
        [
            (np.array(0), NotImplementedError),
            (np.random.random((2, 3)), NotImplementedError),
        ]
    )
    def test_inverse(self, input, expected):
        with self.assertRaises(expected):
            exe = paddle.static.Executor()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                t = transform.Transform()
                static_input = paddle.static.data(
                    'input', input.shape, input.dtype
                )
                output = t.inverse(static_input)
            exe.run(sp)
            exe.run(mp, feed={'input': input}, fetch_list=[output])

    @param.param_func(
        [
            (np.array(0), NotImplementedError),
            (paddle.rand((2, 3)), NotImplementedError),
        ]
    )
    def test_forward_log_det_jacobian(self, input, expected):
        with self.assertRaises(expected):
            exe = paddle.static.Executor()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                t = transform.Transform()
                static_input = paddle.static.data(
                    'input', input.shape, input.dtype
                )
                output = t.forward_log_det_jacobian(static_input)
            exe.run(sp)
            exe.run(mp, feed={'input': input}, fetch_list=[output])

    @param.param_func(
        [
            (np.array(0), NotImplementedError),
            (paddle.rand((2, 3)), NotImplementedError),
        ]
    )
    def test_inverse_log_det_jacobian(self, input, expected):
        with self.assertRaises(expected):
            exe = paddle.static.Executor()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                t = transform.Transform()
                static_input = paddle.static.data(
                    'input', input.shape, input.dtype
                )
                output = t.inverse_log_det_jacobian(static_input)
            exe.run(sp)
            exe.run(mp, feed={'input': input}, fetch_list=[output])

    @param.param_func([(0, TypeError)])
    def test_forward_shape(self, shape, expected):
        with self.assertRaises(expected):
            self._t.forward_shape(shape)

    @param.param_func([(0, TypeError)])
    def test_inverse_shape(self, shape, expected):
        with self.assertRaises(expected):
            self._t.forward_shape(shape)


@param.place(config.DEVICES)
class TestAbsTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.AbsTransform()

    def test_is_injective(self):
        self.assertFalse(self._t._is_injective())

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Real))
        self.assertEqual(self._t._domain.event_rank, 0)
        self.assertEqual(self._t._domain.is_discrete, False)

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Positive))
        self.assertEqual(self._t._codomain.event_rank, 0)
        self.assertEqual(self._t._codomain.is_discrete, False)

    @param.param_func(
        [
            (np.array([-1.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0])),
            (
                np.array([[1.0, -1.0, -0.1], [-3.0, -0.1, 0]]),
                np.array([[1.0, 1.0, 0.1], [3.0, 0.1, 0]]),
            ),
        ]
    )
    def test_forward(self, input, expected):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.AbsTransform()
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.forward(static_input)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': input}, fetch_list=[output])
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func([(np.array([1.0]), (-np.array([1.0]), np.array([1.0])))])
    def test_inverse(self, input, expected):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.AbsTransform()
            static_input = paddle.static.data('input', input.shape, input.dtype)
            actual0, actual1 = t.inverse(static_input)
        exe.run(sp)
        [actual0, actual1] = exe.run(
            mp, feed={'input': input}, fetch_list=[actual0, actual1]
        )
        expected0, expected1 = expected
        np.testing.assert_allclose(
            actual0,
            expected0,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )
        np.testing.assert_allclose(
            actual1,
            expected1,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    def test_forward_log_det_jacobian(self):
        input = np.random.random((10,))
        with self.assertRaises(NotImplementedError):
            exe = paddle.static.Executor()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                t = transform.AbsTransform()
                static_input = paddle.static.data(
                    'input', input.shape, input.dtype
                )
                output = t.forward_log_det_jacobian(static_input)
            exe.run(sp)
            [output] = exe.run(mp, feed={'input': input}, fetch_list=[output])

    @param.param_func(
        [
            (np.array([1.0]), (np.array([0.0]), np.array([0.0]))),
        ]
    )
    def test_inverse_log_det_jacobian(self, input, expected):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.AbsTransform()
            static_input = paddle.static.data('input', input.shape, input.dtype)
            actual0, actual1 = t.inverse_log_det_jacobian(static_input)
        exe.run(sp)
        [actual0, actual1] = exe.run(
            mp, feed={'input': input}, fetch_list=[actual0, actual1]
        )
        expected0, expected1 = expected
        np.testing.assert_allclose(
            actual0,
            expected0,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )
        np.testing.assert_allclose(
            actual1,
            expected1,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)


@param.place(config.DEVICES)
@param.param_cls(
    (param.TEST_CASE_NAME, 'loc', 'scale'),
    [
        ('normal', np.random.rand(8, 10), np.random.rand(8, 10)),
        ('broadcast', np.random.rand(2, 10), np.random.rand(10)),
    ],
)
class TestAffineTransform(unittest.TestCase):
    def setUp(self):
        self.sp = paddle.static.Program()
        self.mp = paddle.static.Program()
        with paddle.static.program_guard(self.mp, self.sp):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            scale = paddle.static.data(
                'scale', self.scale.shape, self.scale.dtype
            )
            self._t = transform.AffineTransform(loc, scale)

    def test_is_injective(self):
        self.assertTrue(self._t._is_injective())

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Real))
        self.assertEqual(self._t._domain.event_rank, 0)
        self.assertEqual(self._t._domain.is_discrete, False)

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Real))
        self.assertEqual(self._t._codomain.event_rank, 0)
        self.assertEqual(self._t._codomain.is_discrete, False)

    def test_forward(self):
        input = np.random.random(self.loc.shape)
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            scale = paddle.static.data(
                'scale', self.scale.shape, self.scale.dtype
            )
            t = transform.AffineTransform(loc, scale)
            static_input = paddle.static.data(
                'input', self.loc.shape, self.loc.dtype
            )
            output = t.forward(static_input)
        exe.run(sp)
        [output] = exe.run(
            mp,
            feed={'input': input, 'loc': self.loc, 'scale': self.scale},
            fetch_list=[output],
        )
        np.testing.assert_allclose(
            output,
            self._np_forward(input),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)),
        )

    def test_inverse(self):
        input = np.random.random(self.loc.shape)
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            scale = paddle.static.data(
                'scale', self.scale.shape, self.scale.dtype
            )
            t = transform.AffineTransform(loc, scale)
            static_input = paddle.static.data(
                'input', self.loc.shape, self.loc.dtype
            )
            output = t.inverse(static_input)
        exe.run(sp)
        [output] = exe.run(
            mp,
            feed={'input': input, 'loc': self.loc, 'scale': self.scale},
            fetch_list=[output],
        )
        np.testing.assert_allclose(
            output,
            self._np_inverse(input),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)),
        )

    def _np_forward(self, x):
        return self.loc + self.scale * x

    def _np_inverse(self, y):
        return (y - self.loc) / self.scale

    def _np_forward_jacobian(self, x):
        return np.log(np.abs(self.scale))

    def _np_inverse_jacobian(self, y):
        return -self._np_forward_jacobian(self._np_inverse(y))

    def test_inverse_log_det_jacobian(self):
        input = np.random.random(self.scale.shape)
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            scale = paddle.static.data(
                'scale', self.scale.shape, self.scale.dtype
            )
            t = transform.AffineTransform(loc, scale)
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.inverse_log_det_jacobian(static_input)
        exe.run(sp)
        [output] = exe.run(
            mp,
            feed={'input': input, 'loc': self.loc, 'scale': self.scale},
            fetch_list=[output],
        )
        np.testing.assert_allclose(
            output,
            self._np_inverse_jacobian(input),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)),
        )

    def test_forward_log_det_jacobian(self):
        input = np.random.random(self.scale.shape)
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            scale = paddle.static.data(
                'scale', self.scale.shape, self.scale.dtype
            )
            t = transform.AffineTransform(loc, scale)
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.forward_log_det_jacobian(static_input)
        exe.run(sp)
        [output] = exe.run(
            mp,
            feed={'input': input, 'loc': self.loc, 'scale': self.scale},
            fetch_list=[output],
        )
        np.testing.assert_allclose(
            output,
            self._np_forward_jacobian(input),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)),
        )

    def test_forward_shape(self):
        shape = self.loc.shape
        self.assertEqual(
            tuple(self._t.forward_shape(shape)),
            np.broadcast(np.random.random(shape), self.loc, self.scale).shape,
        )

    def test_inverse_shape(self):
        shape = self.scale.shape
        self.assertEqual(
            tuple(self._t.forward_shape(shape)),
            np.broadcast(np.random.random(shape), self.loc, self.scale).shape,
        )


@param.place(config.DEVICES)
class TestExpTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.ExpTransform()

    def test_is_injective(self):
        self.assertTrue(self._t._is_injective())

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Real))
        self.assertEqual(self._t._domain.event_rank, 0)
        self.assertEqual(self._t._domain.is_discrete, False)

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Positive))
        self.assertEqual(self._t._codomain.event_rank, 0)
        self.assertEqual(self._t._codomain.is_discrete, False)

    @param.param_func(
        [
            (
                np.array([0.0, 1.0, 2.0, 3.0]),
                np.exp(np.array([0.0, 1.0, 2.0, 3.0])),
            ),
            (
                np.array([[0.0, 1.0, 2.0, 3.0], [-5.0, 6.0, 7.0, 8.0]]),
                np.exp(np.array([[0.0, 1.0, 2.0, 3.0], [-5.0, 6.0, 7.0, 8.0]])),
            ),
        ]
    )
    def test_forward(self, input, expected):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.ExpTransform()
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.forward(static_input)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': input}, fetch_list=[output])
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func(
        [
            (np.array([1.0, 2.0, 3.0]), np.log(np.array([1.0, 2.0, 3.0]))),
            (
                np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]]),
                np.log(np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]])),
            ),
        ]
    )
    def test_inverse(self, input, expected):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.ExpTransform()
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.inverse(static_input)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': input}, fetch_list=[output])
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func(
        [
            (np.array([1.0, 2.0, 3.0]),),
            (np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]]),),
        ]
    )
    def test_forward_log_det_jacobian(self, input):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.ExpTransform()
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.forward_log_det_jacobian(static_input)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': input}, fetch_list=[output])
        np.testing.assert_allclose(
            output,
            self._np_forward_jacobian(input),
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    def _np_forward_jacobian(self, x):
        return x

    @param.param_func(
        [
            (np.array([1.0, 2.0, 3.0]),),
            (np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]]),),
        ]
    )
    def test_inverse_log_det_jacobian(self, input):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.ExpTransform()
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.inverse_log_det_jacobian(static_input)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': input}, fetch_list=[output])
        np.testing.assert_allclose(
            output,
            self._np_inverse_jacobian(input),
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    def _np_inverse_jacobian(self, y):
        return -self._np_forward_jacobian(np.log(y))

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)


@param.place(config.DEVICES)
class TestChainTransform(unittest.TestCase):
    @param.param_func(
        (
            (
                transform.ChainTransform(
                    (
                        transform.AbsTransform(),
                        transform.AffineTransform(
                            paddle.rand([1]), paddle.rand([1])
                        ),
                    )
                ),
                False,
            ),
            (
                transform.ChainTransform(
                    (
                        transform.AffineTransform(
                            paddle.rand([1]), paddle.rand([1])
                        ),
                        transform.ExpTransform(),
                    )
                ),
                True,
            ),
        )
    )
    def test_is_injective(self, chain, expected):
        self.assertEqual(chain._is_injective(), expected)

    @param.param_func(
        (
            (
                transform.ChainTransform(
                    (
                        transform.IndependentTransform(
                            transform.ExpTransform(), 1
                        ),
                        transform.IndependentTransform(
                            transform.ExpTransform(), 10
                        ),
                        transform.IndependentTransform(
                            transform.ExpTransform(), 8
                        ),
                    )
                ),
                variable.Independent(variable.real, 10),
            ),
        )
    )
    def test_domain(self, input, expected):
        self.assertIsInstance(input._domain, type(expected))
        self.assertEqual(input._domain.event_rank, expected.event_rank)
        self.assertEqual(input._domain.is_discrete, expected.is_discrete)

    @param.param_func(
        (
            (
                transform.ChainTransform(
                    (
                        transform.IndependentTransform(
                            transform.ExpTransform(), 9
                        ),
                        transform.IndependentTransform(
                            transform.ExpTransform(), 4
                        ),
                        transform.IndependentTransform(
                            transform.ExpTransform(), 5
                        ),
                    )
                ),
                variable.Independent(variable.real, 9),
            ),
        )
    )
    def test_codomain(self, input, expected):
        self.assertIsInstance(input._codomain, variable.Independent)
        self.assertEqual(input._codomain.event_rank, expected.event_rank)
        self.assertEqual(input._codomain.is_discrete, expected.is_discrete)

    @param.param_func(
        [
            (
                transform.ChainTransform(
                    (transform.ExpTransform(), transform.TanhTransform())
                ),
                np.array([[0.0, -1.0, 2.0, -3.0], [-5.0, 6.0, 7.0, -8.0]]),
                np.tanh(
                    np.exp(
                        np.array(
                            [[0.0, -1.0, 2.0, -3.0], [-5.0, 6.0, 7.0, -8.0]]
                        )
                    )
                ),
            )
        ]
    )
    def test_forward(self, chain, input, expected):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = chain
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.forward(static_input)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': input}, fetch_list=[output])
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func(
        [
            (
                transform.ChainTransform(
                    (transform.ExpTransform(), transform.TanhTransform())
                ),
                np.array([[0.0, 1.0, 2.0, 3.0], [5.0, 6.0, 7.0, 8.0]]),
                np.log(
                    np.arctanh(
                        np.array([[0.0, 1.0, 2.0, 3.0], [5.0, 6.0, 7.0, 8.0]])
                    )
                ),
            )
        ]
    )
    def test_inverse(self, chain, input, expected):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = chain
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.inverse(static_input)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': input}, fetch_list=[output])
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func(
        [
            (
                transform.ChainTransform(
                    (
                        transform.AffineTransform(
                            paddle.full([1], 0.0), paddle.full([1], -1.0)
                        ),
                        transform.ExpTransform(),
                    )
                ),
                (2, 3, 5),
                (2, 3, 5),
            ),
        ]
    )
    def test_forward_shape(self, chain, shape, expected_shape):
        self.assertEqual(chain.forward_shape(shape), expected_shape)

    @param.param_func(
        [
            (
                transform.ChainTransform(
                    (
                        transform.AffineTransform(
                            paddle.full([1], 0.0), paddle.full([1], -1.0)
                        ),
                        transform.ExpTransform(),
                    )
                ),
                (2, 3, 5),
                (2, 3, 5),
            ),
        ]
    )
    def test_inverse_shape(self, chain, shape, expected_shape):
        self.assertEqual(chain.forward_shape(shape), expected_shape)


@param.place(config.DEVICES)
@param.param_cls(
    (param.TEST_CASE_NAME, 'base', 'reinterpreted_batch_rank', 'x'),
    [
        (
            'rank-over-zero',
            transform.ExpTransform(),
            2,
            np.random.rand(2, 3, 3),
        ),
    ],
)
class TestIndependentTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.IndependentTransform(
            self.base, self.reinterpreted_batch_rank
        )

    def test_is_injective(self):
        self.assertEqual(self._t._is_injective(), self.base._is_injective())

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Independent))
        self.assertEqual(
            self._t._domain.event_rank,
            self.base._domain.event_rank + self.reinterpreted_batch_rank,
        )
        self.assertEqual(
            self._t._domain.is_discrete, self.base._domain.is_discrete
        )

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Independent))
        self.assertEqual(
            self._t._codomain.event_rank,
            self.base._codomain.event_rank + self.reinterpreted_batch_rank,
        )
        self.assertEqual(
            self._t._codomain.is_discrete, self.base._codomain.is_discrete
        )

    def test_forward(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.IndependentTransform(
                self.base, self.reinterpreted_batch_rank
            )
            static_input = paddle.static.data(
                'input', self.x.shape, self.x.dtype
            )
            output = t.forward(static_input)
            expected = self.base.forward(static_input)
        exe.run(sp)
        [output, expected] = exe.run(
            mp, feed={'input': self.x}, fetch_list=[output, expected]
        )
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(self.x.dtype)),
            atol=config.ATOL.get(str(self.x.dtype)),
        )

    def test_inverse(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.IndependentTransform(
                self.base, self.reinterpreted_batch_rank
            )
            static_input = paddle.static.data(
                'input', self.x.shape, self.x.dtype
            )
            output = t.inverse(static_input)
            expected = self.base.inverse(static_input)
        exe.run(sp)
        [output, expected] = exe.run(
            mp, feed={'input': self.x}, fetch_list=[output, expected]
        )
        np.testing.assert_allclose(
            expected,
            output,
            rtol=config.RTOL.get(str(self.x.dtype)),
            atol=config.ATOL.get(str(self.x.dtype)),
        )

    def test_forward_log_det_jacobian(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.IndependentTransform(
                self.base, self.reinterpreted_batch_rank
            )
            static_input = paddle.static.data(
                'input', self.x.shape, self.x.dtype
            )
            output = t.forward_log_det_jacobian(static_input)
            expected = self.base.forward_log_det_jacobian(
                static_input.sum(list(range(-self.reinterpreted_batch_rank, 0)))
            )
        exe.run(sp)
        [actual, expected] = exe.run(
            mp, feed={'input': self.x}, fetch_list=[output, expected]
        )
        self.assertEqual(
            tuple(actual.shape), self.x.shape[: -self.reinterpreted_batch_rank]
        )
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=config.RTOL.get(str(self.x.dtype)),
            atol=config.ATOL.get(str(self.x.dtype)),
        )

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)


@param.place(config.DEVICES)
class TestPowerTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.PowerTransform(paddle.full([1], 2.0))

    def test_init(self):
        with self.assertRaises(TypeError):
            transform.PowerTransform(1.0)

    def test_is_injective(self):
        self.assertTrue(self._t._is_injective())

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Real))
        self.assertEqual(self._t._domain.event_rank, 0)
        self.assertEqual(self._t._domain.is_discrete, False)

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Positive))
        self.assertEqual(self._t._codomain.event_rank, 0)
        self.assertEqual(self._t._codomain.is_discrete, False)

    @param.param_func(
        [
            (
                np.array([2.0]),
                np.array([0.0, -1.0, 2.0]),
                np.power(np.array([0.0, -1.0, 2.0]), 2.0),
            ),
            (
                np.array([[0.0], [3.0]]),
                np.array([[1.0, 0.0], [5.0, 6.0]]),
                np.power(
                    np.array([[1.0, 0.0], [5.0, 6.0]]), np.array([[0.0], [3.0]])
                ),
            ),
        ]
    )
    def test_forward(self, power, input, expected):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            static_power = paddle.static.data('power', power.shape, power.dtype)
            t = transform.PowerTransform(static_power)
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.forward(static_input)
        exe.run(sp)
        [output] = exe.run(
            mp, feed={'input': input, 'power': power}, fetch_list=[output]
        )
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func([(np.array([2.0]), np.array([4.0]), np.array([2.0]))])
    def test_inverse(self, power, input, expected):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            static_power = paddle.static.data('power', power.shape, power.dtype)
            t = transform.PowerTransform(static_power)
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.inverse(static_input)
        exe.run(sp)
        [output] = exe.run(
            mp, feed={'input': input, 'power': power}, fetch_list=[output]
        )
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func(((np.array([2.0]), np.array([3.0, 1.4, 0.8])),))
    def test_forward_log_det_jacobian(self, power, input):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            static_power = paddle.static.data('power', power.shape, power.dtype)
            t = transform.PowerTransform(static_power)
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.forward_log_det_jacobian(static_input)
        exe.run(sp)
        [output] = exe.run(
            mp, feed={'input': input, 'power': power}, fetch_list=[output]
        )
        np.testing.assert_allclose(
            output,
            self._np_forward_jacobian(power, input),
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    def _np_forward_jacobian(self, alpha, x):
        return np.abs(np.log(alpha * np.power(x, alpha - 1)))

    @param.param_func([((2, 3, 5), (2, 3, 5))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((2, 3, 5), (2, 3, 5))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)


@param.place(config.DEVICES)
class TestTanhTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.TanhTransform()

    def test_is_injective(self):
        self.assertTrue(self._t._is_injective())

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Real))
        self.assertEqual(self._t._domain.event_rank, 0)
        self.assertEqual(self._t._domain.is_discrete, False)

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Variable))
        self.assertEqual(self._t._codomain.event_rank, 0)
        self.assertEqual(self._t._codomain.is_discrete, False)
        self.assertEqual(self._t._codomain._constraint._lower, -1)
        self.assertEqual(self._t._codomain._constraint._upper, 1)

    @param.param_func(
        [
            (
                np.array([0.0, 1.0, 2.0, 3.0]),
                np.tanh(np.array([0.0, 1.0, 2.0, 3.0])),
            ),
            (
                np.array([[0.0, 1.0, 2.0, 3.0], [-5.0, 6.0, 7.0, 8.0]]),
                np.tanh(
                    np.array([[0.0, 1.0, 2.0, 3.0], [-5.0, 6.0, 7.0, 8.0]])
                ),
            ),
        ]
    )
    def test_forward(self, input, expected):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.TanhTransform()
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.forward(static_input)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': input}, fetch_list=[output])
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func(
        [
            (np.array([1.0, 2.0, 3.0]), np.arctanh(np.array([1.0, 2.0, 3.0]))),
            (
                np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]]),
                np.arctanh(np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]])),
            ),
        ]
    )
    def test_inverse(self, input, expected):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.TanhTransform()
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.inverse(static_input)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': input}, fetch_list=[output])
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func(
        [
            (np.array([1.0, 2.0, 3.0]),),
            (np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]]),),
        ]
    )
    def test_forward_log_det_jacobian(self, input):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.TanhTransform()
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.forward_log_det_jacobian(static_input)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': input}, fetch_list=[output])
        np.testing.assert_allclose(
            output,
            self._np_forward_jacobian(input),
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    def _np_forward_jacobian(self, x):
        return 2.0 * (np.log(2.0) - x - self._np_softplus(-2.0 * x))

    def _np_softplus(self, x, beta=1.0, threshold=20.0):
        if np.any(beta * x > threshold):
            return x
        return 1.0 / beta * np.log1p(np.exp(beta * x))

    def _np_inverse_jacobian(self, y):
        return -self._np_forward_jacobian(np.arctanh(y))

    @param.param_func(
        [
            (np.array([1.0, 2.0, 3.0]),),
            (np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]]),),
        ]
    )
    def test_inverse_log_det_jacobian(self, input):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.TanhTransform()
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.inverse_log_det_jacobian(static_input)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': input}, fetch_list=[output])
        np.testing.assert_allclose(
            output,
            self._np_inverse_jacobian(input),
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)


@param.place(config.DEVICES)
@param.param_cls(
    (param.TEST_CASE_NAME, 'in_event_shape', 'out_event_shape'),
    [
        ('regular_shape', (2, 3), (3, 2)),
    ],
)
class TestReshapeTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.ReshapeTransform(
            self.in_event_shape, self.out_event_shape
        )

    def test_is_injective(self):
        self.assertTrue(self._t._is_injective())

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Independent))

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Independent))

    def test_forward(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = paddle.ones(self.in_event_shape)
            t = transform.ReshapeTransform(
                self.in_event_shape, self.out_event_shape
            )
            output = self._t.forward(x)
        exe.run(sp)
        [output] = exe.run(mp, feed={}, fetch_list=[output])
        expected = np.ones(self.out_event_shape)
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(expected.dtype)),
            atol=config.ATOL.get(str(expected.dtype)),
        )

    def test_inverse(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = paddle.ones(self.out_event_shape)
            t = transform.ReshapeTransform(
                self.in_event_shape, self.out_event_shape
            )
            output = self._t.inverse(x)
        exe.run(sp)
        [output] = exe.run(mp, feed={}, fetch_list=[output])
        expected = np.ones(self.in_event_shape)

        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(expected.dtype)),
            atol=config.ATOL.get(str(expected.dtype)),
        )

    def test_forward_log_det_jacobian(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = paddle.ones(self.in_event_shape)
            t = transform.ReshapeTransform(
                self.in_event_shape, self.out_event_shape
            )
            output = self._t.forward_log_det_jacobian(x)
        exe.run(sp)
        [output] = exe.run(mp, feed={}, fetch_list=[output])
        expected = np.zeros([1])
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(expected.dtype)),
            atol=config.ATOL.get(str(expected.dtype)),
        )


def _np_softplus(x, beta=1.0, threshold=20.0):
    if np.any(beta * x > threshold):
        return x
    return 1.0 / beta * np.log1p(np.exp(beta * x))


class TestSigmoidTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.SigmoidTransform()

    def test_is_injective(self):
        self.assertTrue(self._t._is_injective())

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Real))

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Variable))

    @param.param_func(
        ((np.ones((5, 10)), 1 / (1 + np.exp(-np.ones((5, 10))))),)
    )
    def test_forward(self, input, expected):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', input.shape, input.dtype)
            model = transform.SigmoidTransform()
            out = model.forward(x)
            place = (
                paddle.CUDAPlace(0)
                if paddle.core.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            (result,) = exe.run(feed={'X': input}, fetch_list=[out])
        np.testing.assert_allclose(
            result,
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func(
        ((np.ones(10), np.log(np.ones(10)) - np.log1p(-np.ones(10))),)
    )
    def test_inverse(self, input, expected):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', input.shape, input.dtype)
            model = transform.SigmoidTransform()
            out = model.inverse(x)
            place = (
                paddle.CUDAPlace(0)
                if paddle.core.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            (result,) = exe.run(feed={'X': input}, fetch_list=[out])
        np.testing.assert_allclose(
            result,
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func(
        (
            (
                np.ones(10),
                -_np_softplus(-np.ones(10)) - _np_softplus(np.ones(10)),
            ),
        )
    )
    def test_forward_log_det_jacobian(self, input, expected):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', input.shape, input.dtype)
            model = transform.SigmoidTransform()
            out = model.forward_log_det_jacobian(x)
            place = (
                paddle.CUDAPlace(0)
                if paddle.core.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            (result,) = exe.run(feed={'X': input}, fetch_list=[out])
        np.testing.assert_allclose(
            result,
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([(np.array(1.0), np.array(1.0))])
    def test_zerodim(self, input, expected):
        shape = ()
        if paddle.framework.in_pir_mode():
            shape = []
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', input.shape, 'float32')
            model = transform.SigmoidTransform()
            self.assertEqual(model.forward(x).shape, shape)
            self.assertEqual(model.inverse(x).shape, shape)
            self.assertEqual(model.forward_log_det_jacobian(x).shape, shape)
            self.assertEqual(model.inverse_log_det_jacobian(x).shape, shape)
            self.assertEqual(model.forward_shape(x.shape), shape)
            self.assertEqual(model.inverse_shape(x.shape), shape)


class TestStickBreakingTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.StickBreakingTransform()

    def test_is_injective(self):
        self.assertTrue(self._t._is_injective())

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Independent))

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Variable))

    @param.param_func(((np.random.random(10),),))
    def test_forward(self, input):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', input.shape, input.dtype)
            model = transform.StickBreakingTransform()
            fwd = model.forward(x)
            out = model.inverse(fwd)
            place = (
                paddle.CUDAPlace(0)
                if paddle.core.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            (result,) = exe.run(feed={'X': input}, fetch_list=[out])
        np.testing.assert_allclose(
            result,
            input,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func([((2, 3, 5), (2, 3, 6))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((2, 3, 5), (2, 3, 4))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.inverse_shape(shape), expected_shape)

    @param.param_func(((np.random.random(10),),))
    def test_forward_log_det_jacobian(self, input):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', input.shape, input.dtype)
            model = transform.StickBreakingTransform()
            out = model.forward_log_det_jacobian(x)
            place = (
                paddle.CUDAPlace(0)
                if paddle.core.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            (result,) = exe.run(feed={'X': input}, fetch_list=[out])
        self.assertEqual(result.shape, ())


@param.place(config.DEVICES)
@param.param_cls(
    (param.TEST_CASE_NAME, 'transforms', 'axis'),
    [
        ('simple_one_transform', [transform.ExpTransform()], 0),
    ],
)
class TestStackTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.StackTransform(self.transforms, self.axis)

    def test_is_injective(self):
        self.assertTrue(self._t._is_injective())

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Stack))

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Stack))

    @param.param_func(
        [
            (np.array([[0.0, 1.0, 2.0, 3.0]]),),
            (np.array([[-5.0, 6.0, 7.0, 8.0]]),),
        ]
    )
    def test_forward(self, input):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', input.shape, input.dtype)
            model = transform.StackTransform(self.transforms, self.axis)
            out = model.forward(x)
            place = (
                paddle.CUDAPlace(0)
                if paddle.core.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            (result,) = exe.run(feed={'X': input}, fetch_list=[out])
        self.assertEqual(tuple(result.shape), input.shape)

    @param.param_func(
        [
            (np.array([[1.0, 2.0, 3.0]]),),
            (
                np.array(
                    [[6.0, 7.0, 8.0]],
                ),
            ),
        ]
    )
    def test_inverse(self, input):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', input.shape, input.dtype)
            model = transform.StackTransform(self.transforms, self.axis)
            out = model.inverse(x)
            place = (
                paddle.CUDAPlace(0)
                if paddle.core.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            (result,) = exe.run(feed={'X': input}, fetch_list=[out])
        self.assertEqual(tuple(result.shape), input.shape)

    @param.param_func(
        [(np.array([[1.0, 2.0, 3.0]]),), (np.array([[6.0, 7.0, 8.0]]),)]
    )
    def test_forward_log_det_jacobian(self, input):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', input.shape, input.dtype)
            model = transform.StackTransform(self.transforms, self.axis)
            out = model.forward_log_det_jacobian(x)
            place = (
                paddle.CUDAPlace(0)
                if paddle.core.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            (result,) = exe.run(feed={'X': input}, fetch_list=[out])
        self.assertEqual(tuple(result.shape), input.shape)

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    def test_axis(self):
        self.assertEqual(self._t.axis, self.axis)

    @param.param_func(
        [
            (0, 0, TypeError),
            ([0], 0, TypeError),
            ([paddle.distribution.ExpTransform()], 'axis', TypeError),
        ]
    )
    def test_init_exception(self, transforms, axis, exc):
        with self.assertRaises(exc):
            paddle.distribution.StackTransform(transforms, axis)

    def test_transforms(self):
        self.assertIsInstance((self._t.transforms), typing.Sequence)


if __name__ == '__main__':
    unittest.main()
