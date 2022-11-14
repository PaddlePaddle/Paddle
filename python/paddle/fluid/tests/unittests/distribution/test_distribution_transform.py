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
from paddle.distribution import transform, variable

import config
import parameterize as param

np.random.seed(2022)
paddle.seed(2022)


@param.place(config.DEVICES)
class TestTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.Transform()

    @param.param_func(
        [
            (
                paddle.distribution.Distribution(),
                paddle.distribution.TransformedDistribution,
            ),
            (
                paddle.distribution.ExpTransform(),
                paddle.distribution.ChainTransform,
            ),
        ]
    )
    def test_call(self, input, expected_type):
        t = transform.Transform()
        self.assertIsInstance(t(input), expected_type)

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
        [(0, TypeError), (paddle.rand((2, 3)), NotImplementedError)]
    )
    def test_forward(self, input, expected):
        with self.assertRaises(expected):
            self._t.forward(input)

    @param.param_func(
        [(0, TypeError), (paddle.rand((2, 3)), NotImplementedError)]
    )
    def test_inverse(self, input, expected):
        with self.assertRaises(expected):
            self._t.inverse(input)

    @param.param_func(
        [(0, TypeError), (paddle.rand((2, 3)), NotImplementedError)]
    )
    def test_forward_log_det_jacobian(self, input, expected):
        with self.assertRaises(expected):
            self._t.forward_log_det_jacobian(input)

    @param.param_func(
        [(0, TypeError), (paddle.rand((2, 3)), NotImplementedError)]
    )
    def test_inverse_log_det_jacobian(self, input, expected):
        with self.assertRaises(expected):
            self._t.inverse_log_det_jacobian(input)

    @param.param_func([(0, TypeError)])
    def test_forward_shape(self, shape, expected):
        with self.assertRaises(expected):
            self._t.forward_shape(shape)

    @param.param_func([(0, TypeError)])
    def test_inverse_shape(self, shape, expected):
        with self.assertRaises(expected):
            self._t.inverse_shape(shape)


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
        np.testing.assert_allclose(
            self._t.forward(paddle.to_tensor(input)).numpy(),
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func([(np.array(1.0), (-np.array(1.0), np.array(1.0)))])
    def test_inverse(self, input, expected):
        actual0, actual1 = self._t.inverse(paddle.to_tensor(input))
        expected0, expected1 = expected
        np.testing.assert_allclose(
            actual0.numpy(),
            expected0,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )
        np.testing.assert_allclose(
            actual1.numpy(),
            expected1,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    def test_forward_log_det_jacobian(self):
        with self.assertRaises(NotImplementedError):
            self._t.forward_log_det_jacobian(paddle.rand((10,)))

    @param.param_func(
        [
            (np.array(1.0), (np.array(0.0), np.array(0.0))),
        ]
    )
    def test_inverse_log_det_jacobian(self, input, expected):
        actual0, actual1 = self._t.inverse_log_det_jacobian(
            paddle.to_tensor(input)
        )
        expected0, expected1 = expected
        np.testing.assert_allclose(
            actual0.numpy(),
            expected0,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )
        np.testing.assert_allclose(
            actual1.numpy(),
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

    @param.param_func([(np.array(1.0), np.array(1.0))])
    def test_zerodim(self, input, expected):
        x = paddle.to_tensor(input).astype('float32')
        self.assertEqual(self._t.forward(x).shape, [])
        self.assertEqual(self._t.inverse(x)[0].shape, [])
        self.assertEqual(self._t.inverse(x)[1].shape, [])
        self.assertEqual(self._t.inverse_log_det_jacobian(x)[0].shape, [])
        self.assertEqual(self._t.inverse_log_det_jacobian(x)[1].shape, [])
        self.assertEqual(self._t.forward_shape(x.shape), [])
        self.assertEqual(self._t.inverse_shape(x.shape), [])


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
        self._t = transform.AffineTransform(
            paddle.to_tensor(self.loc), paddle.to_tensor(self.scale)
        )

    @param.param_func(
        [
            (paddle.rand([1]), 0, TypeError),
            (0, paddle.rand([1]), TypeError),
        ]
    )
    def test_init_exception(self, loc, scale, exc):
        with self.assertRaises(exc):
            paddle.distribution.AffineTransform(loc, scale)

    def test_scale(self):
        np.testing.assert_allclose(self._t.scale, self.scale)

    def test_loc(self):
        np.testing.assert_allclose(self._t.loc, self.loc)

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
        x = np.random.random(self.loc.shape)
        np.testing.assert_allclose(
            self._t.forward(paddle.to_tensor(x)).numpy(),
            self._np_forward(x),
            rtol=config.RTOL.get(str(self._t.loc.numpy().dtype)),
            atol=config.ATOL.get(str(self._t.loc.numpy().dtype)),
        )

    def test_inverse(self):
        y = np.random.random(self.loc.shape)
        np.testing.assert_allclose(
            self._t.inverse(paddle.to_tensor(y)).numpy(),
            self._np_inverse(y),
            rtol=config.RTOL.get(str(self._t.loc.numpy().dtype)),
            atol=config.ATOL.get(str(self._t.loc.numpy().dtype)),
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
        y = np.random.random(self.scale.shape)
        np.testing.assert_allclose(
            self._t.inverse_log_det_jacobian(paddle.to_tensor(y)).numpy(),
            self._np_inverse_jacobian(y),
            rtol=config.RTOL.get(str(self._t.loc.numpy().dtype)),
            atol=config.ATOL.get(str(self._t.loc.numpy().dtype)),
        )

    def test_forward_log_det_jacobian(self):
        x = np.random.random(self.scale.shape)
        np.testing.assert_allclose(
            self._t.forward_log_det_jacobian(paddle.to_tensor(x)).numpy(),
            self._np_forward_jacobian(x),
            rtol=config.RTOL.get(str(self._t.loc.numpy().dtype)),
            atol=config.ATOL.get(str(self._t.loc.numpy().dtype)),
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

    @param.param_func([(np.array(1.0), np.array(1.0))])
    def test_zerodim(self, input, expected):
        affine = transform.AffineTransform(paddle.zeros([]), paddle.ones([]))

        x = paddle.to_tensor(input).astype('float32')
        self.assertEqual(affine.forward(x).shape, [])
        self.assertEqual(affine.inverse(x).shape, [])
        self.assertEqual(affine.forward_log_det_jacobian(x).shape, [])
        self.assertEqual(affine.inverse_log_det_jacobian(x).shape, [])
        self.assertEqual(affine.forward_shape(x.shape), ())
        self.assertEqual(affine.inverse_shape(x.shape), ())


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
        np.testing.assert_allclose(
            self._t.forward(paddle.to_tensor(input)).numpy(),
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
        np.testing.assert_allclose(
            self._t.inverse(paddle.to_tensor(input)).numpy(),
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
        np.testing.assert_allclose(
            self._t.forward_log_det_jacobian(paddle.to_tensor(input)).numpy(),
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
        np.testing.assert_allclose(
            self._t.inverse_log_det_jacobian(paddle.to_tensor(input)).numpy(),
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

    @param.param_func([(np.array(1.0), np.array(1.0))])
    def test_zerodim(self, input, expected):
        x = paddle.to_tensor(input).astype('float32')
        self.assertEqual(self._t.forward(x).shape, [])
        self.assertEqual(self._t.inverse(x).shape, [])
        self.assertEqual(self._t.forward_log_det_jacobian(x).shape, [])
        self.assertEqual(self._t.inverse_log_det_jacobian(x).shape, [])
        self.assertEqual(self._t.forward_shape(x.shape), [])
        self.assertEqual(self._t.inverse_shape(x.shape), [])


@param.place(config.DEVICES)
class TestChainTransform(unittest.TestCase):
    @param.param_func(
        [(paddle.distribution.Transform, TypeError), ([0], TypeError)]
    )
    def test_init_exception(self, transforms, exception):
        with self.assertRaises(exception):
            paddle.distribution.ChainTransform(transforms)

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
                    (
                        transform.AffineTransform(
                            paddle.to_tensor(0.0), paddle.to_tensor(1.0)
                        ),
                        transform.ExpTransform(),
                    )
                ),
                np.array([0.0, 1.0, 2.0, 3.0]),
                np.exp(np.array([0.0, 1.0, 2.0, 3.0]) * 1.0),
            ),
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
            ),
        ]
    )
    def test_forward(self, chain, input, expected):
        np.testing.assert_allclose(
            chain.forward(paddle.to_tensor(input)).numpy(),
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
                            paddle.to_tensor(0.0), paddle.to_tensor(-1.0)
                        ),
                        transform.ExpTransform(),
                    )
                ),
                np.array([0.0, 1.0, 2.0, 3.0]),
                np.log(np.array([0.0, 1.0, 2.0, 3.0])) / (-1.0),
            ),
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
            ),
        ]
    )
    def test_inverse(self, chain, input, expected):
        np.testing.assert_allclose(
            chain.inverse(paddle.to_tensor(input)).numpy(),
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
                            paddle.to_tensor(0.0), paddle.to_tensor(-1.0)
                        ),
                        transform.PowerTransform(paddle.to_tensor(2.0)),
                    )
                ),
                np.array([1.0, 2.0, 3.0]),
                np.log(2.0 * np.array([1.0, 2.0, 3.0])),
            ),
        ]
    )
    def test_forward_log_det_jacobian(self, chain, input, expected):
        np.testing.assert_allclose(
            chain.forward_log_det_jacobian(paddle.to_tensor(input)).numpy(),
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
                            paddle.to_tensor(0.0), paddle.to_tensor(-1.0)
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
                            paddle.to_tensor(0.0), paddle.to_tensor(-1.0)
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
        self.assertEqual(chain.inverse_shape(shape), expected_shape)


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

    @param.param_func(
        [(0, 0, TypeError), (paddle.distribution.Transform(), -1, ValueError)]
    )
    def test_init_exception(self, base, rank, exc):
        with self.assertRaises(exc):
            paddle.distribution.IndependentTransform(base, rank)

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
        np.testing.assert_allclose(
            self._t.forward(paddle.to_tensor(self.x)).numpy(),
            self.base.forward(paddle.to_tensor(self.x)).numpy(),
            rtol=config.RTOL.get(str(self.x.dtype)),
            atol=config.ATOL.get(str(self.x.dtype)),
        )

    def test_inverse(self):
        np.testing.assert_allclose(
            self._t.inverse(paddle.to_tensor(self.x)).numpy(),
            self.base.inverse(paddle.to_tensor(self.x)).numpy(),
            rtol=config.RTOL.get(str(self.x.dtype)),
            atol=config.ATOL.get(str(self.x.dtype)),
        )

    def test_forward_log_det_jacobian(self):
        actual = self._t.forward_log_det_jacobian(paddle.to_tensor(self.x))
        self.assertEqual(
            tuple(actual.shape), self.x.shape[: -self.reinterpreted_batch_rank]
        )
        expected = self.base.forward_log_det_jacobian(
            paddle.to_tensor(self.x)
        ).sum(list(range(-self.reinterpreted_batch_rank, 0)))
        np.testing.assert_allclose(
            actual.numpy(),
            expected.numpy(),
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
        self._t = transform.PowerTransform(paddle.to_tensor(2.0))

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
    def test_forward(self, power, x, y):
        t = transform.PowerTransform(paddle.to_tensor(power))
        np.testing.assert_allclose(
            t.forward(paddle.to_tensor(x)).numpy(),
            y,
            rtol=config.RTOL.get(str(x.dtype)),
            atol=config.ATOL.get(str(x.dtype)),
        )

    @param.param_func([(np.array([2.0]), np.array([4.0]), np.array([2.0]))])
    def test_inverse(self, power, y, x):
        t = transform.PowerTransform(paddle.to_tensor(power))
        np.testing.assert_allclose(
            t.inverse(paddle.to_tensor(y)).numpy(),
            x,
            rtol=config.RTOL.get(str(x.dtype)),
            atol=config.ATOL.get(str(x.dtype)),
        )

    @param.param_func(((np.array([2.0]), np.array([3.0, 1.4, 0.8])),))
    def test_forward_log_det_jacobian(self, power, x):
        t = transform.PowerTransform(paddle.to_tensor(power))
        np.testing.assert_allclose(
            t.forward_log_det_jacobian(paddle.to_tensor(x)).numpy(),
            self._np_forward_jacobian(power, x),
            rtol=config.RTOL.get(str(x.dtype)),
            atol=config.ATOL.get(str(x.dtype)),
        )

    def _np_forward_jacobian(self, alpha, x):
        return np.abs(np.log(alpha * np.power(x, alpha - 1)))

    @param.param_func([((2, 3, 5), (2, 3, 5))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((2, 3, 5), (2, 3, 5))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([(np.array(2.0), np.array(1.0))])
    def test_zerodim(self, input, expected):
        power = transform.PowerTransform(paddle.full([], 2.0))

        x = paddle.to_tensor(input).astype('float32')
        self.assertEqual(power.forward(x).shape, [])
        self.assertEqual(power.inverse(x).shape, [])
        self.assertEqual(power.forward_log_det_jacobian(x).shape, [])
        self.assertEqual(power.inverse_log_det_jacobian(x).shape, [])
        self.assertEqual(power.forward_shape(x.shape), ())
        self.assertEqual(power.inverse_shape(x.shape), ())


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
        np.testing.assert_allclose(
            self._t.forward(paddle.to_tensor(input)).numpy(),
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
        np.testing.assert_allclose(
            self._t.inverse(paddle.to_tensor(input)).numpy(),
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
        np.testing.assert_allclose(
            self._t.forward_log_det_jacobian(paddle.to_tensor(input)).numpy(),
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
        np.testing.assert_allclose(
            self._t.inverse_log_det_jacobian(paddle.to_tensor(input)).numpy(),
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

    @param.param_func([(np.array(1.0), np.array(1.0))])
    def test_zerodim(self, input, expected):
        x = paddle.to_tensor(input).astype('float32')
        self.assertEqual(self._t.forward(x).shape, [])
        self.assertEqual(self._t.inverse(x).shape, [])
        self.assertEqual(self._t.forward_log_det_jacobian(x).shape, [])
        self.assertEqual(self._t.inverse_log_det_jacobian(x).shape, [])
        self.assertEqual(self._t.forward_shape(x.shape), [])
        self.assertEqual(self._t.inverse_shape(x.shape), [])


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

    @param.param_func([(0, 0, TypeError), ((1, 2), (1, 3), ValueError)])
    def test_init_exception(self, in_event_shape, out_event_shape, exc):
        with self.assertRaises(exc):
            paddle.distribution.ReshapeTransform(
                in_event_shape, out_event_shape
            )

    def test_is_injective(self):
        self.assertTrue(self._t._is_injective())

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Independent))

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Independent))

    def test_forward(self):
        x = paddle.ones(self.in_event_shape)
        np.testing.assert_allclose(
            self._t.forward(x),
            paddle.ones(self.out_event_shape),
            rtol=config.RTOL.get(str(x.numpy().dtype)),
            atol=config.ATOL.get(str(x.numpy().dtype)),
        )

    def test_inverse(self):
        x = paddle.ones(self.out_event_shape)
        np.testing.assert_allclose(
            self._t.inverse(x).numpy(),
            paddle.ones(self.in_event_shape).numpy(),
            rtol=config.RTOL.get(str(x.numpy().dtype)),
            atol=config.ATOL.get(str(x.numpy().dtype)),
        )

    def test_forward_log_det_jacobian(self):
        x = paddle.ones(self.in_event_shape)
        np.testing.assert_allclose(
            self._t.forward_log_det_jacobian(x).numpy(),
            paddle.zeros([1]).numpy(),
            rtol=config.RTOL.get(str(x.numpy().dtype)),
            atol=config.ATOL.get(str(x.numpy().dtype)),
        )

    def test_in_event_shape(self):
        self.assertEqual(self._t.in_event_shape, self.in_event_shape)

    def test_out_event_shape(self):
        self.assertEqual(self._t.out_event_shape, self.out_event_shape)

    @param.param_func([((), ValueError), ((1, 2), ValueError)])
    def test_forward_shape_exception(self, shape, exc):
        with self.assertRaises(exc):
            self._t.forward_shape(shape)

    @param.param_func([((), ValueError), ((1, 2), ValueError)])
    def test_inverse_shape_exception(self, shape, exc):
        with self.assertRaises(exc):
            self._t.inverse_shape(shape)

    @param.param_func([(np.array(2.0), np.array(1.0))])
    def test_zerodim(self, input, expected):
        reshape = transform.ReshapeTransform((), (1, 1))

        x = paddle.to_tensor(input).astype('float32')
        out = reshape.forward(x)

        self.assertEqual(out.shape, [1, 1])
        self.assertEqual(reshape.inverse(out).shape, [])
        # self.assertEqual(reshape.forward_log_det_jacobian(x).shape, [])
        # self.assertEqual(reshape.inverse_log_det_jacobian(out).shape, [])
        self.assertEqual(reshape.forward_shape(x.shape), (1, 1))
        self.assertEqual(reshape.inverse_shape(out.shape), ())


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
        np.testing.assert_allclose(
            self._t.forward(paddle.to_tensor(input)),
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func(
        ((np.ones(10), np.log(np.ones(10)) - np.log1p(-np.ones(10))),)
    )
    def test_inverse(self, input, expected):
        np.testing.assert_allclose(
            self._t.inverse(paddle.to_tensor(input)).numpy(),
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
        np.testing.assert_allclose(
            self._t.forward_log_det_jacobian(paddle.to_tensor(input)).numpy(),
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
        x = paddle.to_tensor(input).astype('float32')
        self.assertEqual(self._t.forward(x).shape, [])
        self.assertEqual(self._t.inverse(x).shape, [])
        self.assertEqual(self._t.forward_log_det_jacobian(x).shape, [])
        self.assertEqual(self._t.inverse_log_det_jacobian(x).shape, [])
        self.assertEqual(self._t.forward_shape(x.shape), [])
        self.assertEqual(self._t.inverse_shape(x.shape), [])


class TestSoftmaxTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.SoftmaxTransform()

    def test_is_injective(self):
        self.assertFalse(self._t._is_injective())

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Independent))

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Variable))

    @param.param_func(((np.random.random((5, 10)),),))
    def test_forward(self, input):
        np.testing.assert_allclose(
            self._t.forward(paddle.to_tensor(input)),
            self._np_forward(input),
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func(((np.random.random(10),),))
    def test_inverse(self, input):
        np.testing.assert_allclose(
            self._t.inverse(paddle.to_tensor(input)),
            self._np_inverse(input),
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    def _np_forward(self, x):
        x = np.exp(x - np.max(x, -1, keepdims=True)[0])
        return x / np.sum(x, -1, keepdims=True)

    def _np_inverse(self, y):
        return np.log(y)

    def test_forward_log_det_jacobian(self):
        with self.assertRaises(NotImplementedError):
            self._t.forward_log_det_jacobian(paddle.rand((2, 3)))

    @param.param_func([((2, 3, 5), (2, 3, 5))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((), ValueError)])
    def test_forward_shape_exception(self, shape, exc):
        with self.assertRaises(exc):
            self._t.forward_shape(shape)

    @param.param_func([((), ValueError)])
    def test_inverse_shape_exception(self, shape, exc):
        with self.assertRaises(exc):
            self._t.inverse_shape(shape)

    @param.param_func([((2, 3, 5), (2, 3, 5))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.inverse_shape(shape), expected_shape)


class TestStickBreakingTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.StickBreakingTransform()

    def test_is_injective(self):
        self.assertTrue(self._t._is_injective())

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Independent))

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Variable))

    @param.param_func(((np.random.random((10)),),))
    def test_forward(self, input):
        np.testing.assert_allclose(
            self._t.inverse(self._t.forward(paddle.to_tensor(input))),
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

    @param.param_func(((np.random.random((10)),),))
    def test_forward_log_det_jacobian(self, x):
        self.assertEqual(
            self._t.forward_log_det_jacobian(paddle.to_tensor(x)).shape, [1]
        )


# Todo
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
        self.assertEqual(
            tuple(self._t.forward(paddle.to_tensor(input)).shape), input.shape
        )

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
        self.assertEqual(
            tuple(self._t.inverse(paddle.to_tensor(input)).shape), input.shape
        )

    @param.param_func(
        [(np.array([[1.0, 2.0, 3.0]]),), (np.array([[6.0, 7.0, 8.0]]),)]
    )
    def test_forward_log_det_jacobian(self, input):
        self.assertEqual(
            tuple(
                self._t.forward_log_det_jacobian(paddle.to_tensor(input)).shape
            ),
            input.shape,
        )

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
