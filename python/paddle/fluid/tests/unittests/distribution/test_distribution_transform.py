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
from paddle.distribution import transform, variable

import config
import parameterize as param


@param.place(config.DEVICES)
class TestTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.Transform()

    @param.param_func(
        [(transform.Type.BIJECTION, True), (transform.Type.INJECTION, True),
         (transform.Type.SURJECTION, False), (transform.Type.OTHER, False)])
    def test_is_injective(self, type, expected):
        transform.Transform._type = type
        self.assertEqual(self._t._is_injective(), expected)

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Real))

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Real))

    @param.param_func([(0, TypeError), (paddle.rand((2, 3)),
                                        NotImplementedError)])
    def test_forward(self, input, expected):
        with self.assertRaises(expected):
            self._t.forward(input)

    @param.param_func([(0, TypeError), (paddle.rand((2, 3)),
                                        NotImplementedError)])
    def test_inverse(self, input, expected):
        with self.assertRaises(expected):
            self._t.inverse(input)

    @param.param_func([(0, TypeError), (paddle.rand((2, 3)),
                                        NotImplementedError)])
    def test_forward_log_det_jacobian(self, input, expected):
        with self.assertRaises(expected):
            self._t.forward_log_det_jacobian(input)

    @param.param_func([(0, TypeError), (paddle.rand((2, 3)),
                                        NotImplementedError)])
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

    @param.param_func([(np.array([-1., 1., 0.]), np.array([1., 1., 0.])),
                       (np.array([[1., -1., -0.1], [-3., -0.1, 0]]),
                        np.array([[1., 1., 0.1], [3., 0.1, 0]]))])
    def test_forward(self, input, expected):
        np.testing.assert_allclose(
            self._t.forward(paddle.to_tensor(input)).numpy(),
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)))

    @param.param_func([(np.array(1.), (-np.array(1.), np.array(1.)))])
    def test_inverse(self, input, expected):
        actual0, actual1 = self._t.inverse(paddle.to_tensor(input))
        expected0, expected1 = expected
        np.testing.assert_allclose(
            actual0.numpy(),
            expected0,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)))
        np.testing.assert_allclose(
            actual1.numpy(),
            expected1,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)))

    def test_forward_log_det_jacobian(self):
        with self.assertRaises(NotImplementedError):
            self._t.forward_log_det_jacobian(paddle.rand((10, )))

    @param.param_func([(np.array(1.), (np.array(0.), np.array(0.))), ])
    def test_inverse_log_det_jacobian(self, input, expected):
        actual0, actual1 = self._t.inverse_log_det_jacobian(
            paddle.to_tensor(input))
        expected0, expected1 = expected
        np.testing.assert_allclose(
            actual0.numpy(),
            expected0,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)))
        np.testing.assert_allclose(
            actual1.numpy(),
            expected1,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)))

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)


@param.place(config.DEVICES)
@param.param_cls((param.TEST_CASE_NAME, 'loc', 'scale'), [
    ('normal', np.random.rand(8, 10), np.random.rand(8, 10)),
    ('broadcast', np.random.rand(2, 10), np.random.rand(10)),
])
class TestAffineTransform(unittest.TestCase):
    def setUp(self):
        self._t = transform.AffineTransform(
            paddle.to_tensor(self.loc), paddle.to_tensor(self.scale))

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
            atol=config.ATOL.get(str(self._t.loc.numpy().dtype)))

    def test_inverse(self):
        y = np.random.random(self.loc.shape)
        np.testing.assert_allclose(
            self._t.inverse(paddle.to_tensor(y)).numpy(),
            self._np_inverse(y),
            rtol=config.RTOL.get(str(self._t.loc.numpy().dtype)),
            atol=config.ATOL.get(str(self._t.loc.numpy().dtype)))

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
            atol=config.ATOL.get(str(self._t.loc.numpy().dtype)))

    def test_forward_log_det_jacobian(self):
        x = np.random.random(self.scale.shape)
        np.testing.assert_allclose(
            self._t.forward_log_det_jacobian(paddle.to_tensor(x)).numpy(),
            self._np_forward_jacobian(x),
            rtol=config.RTOL.get(str(self._t.loc.numpy().dtype)),
            atol=config.ATOL.get(str(self._t.loc.numpy().dtype)))

    def test_forward_shape(self):
        shape = self.loc.shape
        self.assertEqual(
            tuple(self._t.forward_shape(shape)),
            np.broadcast(np.random.random(shape), self.loc, self.scale).shape)

    def test_inverse_shape(self):
        shape = self.scale.shape
        self.assertEqual(
            tuple(self._t.forward_shape(shape)),
            np.broadcast(np.random.random(shape), self.loc, self.scale).shape)


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
        [(np.array([0., 1., 2., 3.]), np.exp(np.array([0., 1., 2., 3.]))),
         (np.array([[0., 1., 2., 3.], [-5., 6., 7., 8.]]),
          np.exp(np.array([[0., 1., 2., 3.], [-5., 6., 7., 8.]])))])
    def test_forward(self, input, expected):
        np.testing.assert_allclose(
            self._t.forward(paddle.to_tensor(input)).numpy(),
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)))

    @param.param_func([(np.array([1., 2., 3.]), np.log(np.array([1., 2., 3.]))),
                       (np.array([[1., 2., 3.], [6., 7., 8.]]),
                        np.log(np.array([[1., 2., 3.], [6., 7., 8.]])))])
    def test_inverse(self, input, expected):
        np.testing.assert_allclose(
            self._t.inverse(paddle.to_tensor(input)).numpy(),
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)))

    @param.param_func([(np.array([1., 2., 3.]), ),
                       (np.array([[1., 2., 3.], [6., 7., 8.]]), )])
    def test_forward_log_det_jacobian(self, input):
        np.testing.assert_allclose(
            self._t.forward_log_det_jacobian(paddle.to_tensor(input)).numpy(),
            self._np_forward_jacobian(input),
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)))

    def _np_forward_jacobian(self, x):
        return x

    @param.param_func([(np.array([1., 2., 3.]), ),
                       (np.array([[1., 2., 3.], [6., 7., 8.]]), )])
    def test_inverse_log_det_jacobian(self, input):
        np.testing.assert_allclose(
            self._t.inverse_log_det_jacobian(paddle.to_tensor(input)).numpy(),
            self._np_inverse_jacobian(input),
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)))

    def _np_inverse_jacobian(self, y):
        return -self._np_forward_jacobian(np.log(y))

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)


if __name__ == '__main__':
    unittest.main()
