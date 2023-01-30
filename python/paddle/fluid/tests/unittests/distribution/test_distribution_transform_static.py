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

<<<<<<< HEAD
import config
import numpy as np
import parameterize as param

import paddle
from paddle.distribution import transform, variable
=======
import numpy as np
import paddle
from paddle.distribution import transform, variable, constraint

import config
import parameterize as param
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

np.random.seed(2022)
paddle.seed(2022)
paddle.enable_static()


@param.place(config.DEVICES)
class TestTransform(unittest.TestCase):
<<<<<<< HEAD
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
=======

    def setUp(self):
        self._t = transform.Transform()

    @param.param_func([(transform.Type.BIJECTION, True),
                       (transform.Type.INJECTION, True),
                       (transform.Type.SURJECTION, False),
                       (transform.Type.OTHER, False)])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_is_injective(self, type, expected):
        transform.Transform._type = type
        self.assertEqual(self._t._is_injective(), expected)

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Real))

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Real))

<<<<<<< HEAD
    @param.param_func(
        [
            (np.array(0), NotImplementedError),
            (np.random.random((2, 3)), NotImplementedError),
        ]
    )
=======
    @param.param_func([(np.array(0), NotImplementedError),
                       (np.random.random((2, 3)), NotImplementedError)])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_forward(self, input, expected):
        with self.assertRaises(expected):
            exe = paddle.static.Executor()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                t = transform.Transform()
<<<<<<< HEAD
                static_input = paddle.static.data(
                    'input', input.shape, input.dtype
                )
=======
                static_input = paddle.static.data('input', input.shape,
                                                  input.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                output = t.forward(static_input)
            exe.run(sp)
            exe.run(mp, feed={'input': input}, fetch_list=[output])

<<<<<<< HEAD
    @param.param_func(
        [
            (np.array(0), NotImplementedError),
            (np.random.random((2, 3)), NotImplementedError),
        ]
    )
=======
    @param.param_func([(np.array(0), NotImplementedError),
                       (np.random.random((2, 3)), NotImplementedError)])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_inverse(self, input, expected):
        with self.assertRaises(expected):
            exe = paddle.static.Executor()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                t = transform.Transform()
<<<<<<< HEAD
                static_input = paddle.static.data(
                    'input', input.shape, input.dtype
                )
=======
                static_input = paddle.static.data('input', input.shape,
                                                  input.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                output = t.inverse(static_input)
            exe.run(sp)
            exe.run(mp, feed={'input': input}, fetch_list=[output])

<<<<<<< HEAD
    @param.param_func(
        [
            (np.array(0), NotImplementedError),
            (paddle.rand((2, 3)), NotImplementedError),
        ]
    )
=======
    @param.param_func([(np.array(0), NotImplementedError),
                       (paddle.rand((2, 3)), NotImplementedError)])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_forward_log_det_jacobian(self, input, expected):
        with self.assertRaises(expected):
            exe = paddle.static.Executor()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                t = transform.Transform()
<<<<<<< HEAD
                static_input = paddle.static.data(
                    'input', input.shape, input.dtype
                )
=======
                static_input = paddle.static.data('input', input.shape,
                                                  input.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                output = t.forward_log_det_jacobian(static_input)
            exe.run(sp)
            exe.run(mp, feed={'input': input}, fetch_list=[output])

<<<<<<< HEAD
    @param.param_func(
        [
            (np.array(0), NotImplementedError),
            (paddle.rand((2, 3)), NotImplementedError),
        ]
    )
=======
    @param.param_func([(np.array(0), NotImplementedError),
                       (paddle.rand((2, 3)), NotImplementedError)])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_inverse_log_det_jacobian(self, input, expected):
        with self.assertRaises(expected):
            exe = paddle.static.Executor()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                t = transform.Transform()
<<<<<<< HEAD
                static_input = paddle.static.data(
                    'input', input.shape, input.dtype
                )
=======
                static_input = paddle.static.data('input', input.shape,
                                                  input.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
    @param.param_func(
        [
            (np.array([-1.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0])),
            (
                np.array([[1.0, -1.0, -0.1], [-3.0, -0.1, 0]]),
                np.array([[1.0, 1.0, 0.1], [3.0, 0.1, 0]]),
            ),
        ]
    )
=======
    @param.param_func([(np.array([-1., 1., 0.]), np.array([1., 1., 0.])),
                       (np.array([[1., -1., -0.1], [-3., -0.1, 0]]),
                        np.array([[1., 1., 0.1], [3., 0.1, 0]]))])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )

    @param.param_func([(np.array([1.0]), (-np.array([1.0]), np.array([1.0])))])
=======
        np.testing.assert_allclose(output,
                                   expected,
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))

    @param.param_func([(np.array([1.]), (-np.array([1.]), np.array([1.])))])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_inverse(self, input, expected):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.AbsTransform()
            static_input = paddle.static.data('input', input.shape, input.dtype)
            actual0, actual1 = t.inverse(static_input)
        exe.run(sp)
<<<<<<< HEAD
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
=======
        [actual0, actual1] = exe.run(mp,
                                     feed={'input': input},
                                     fetch_list=[actual0, actual1])
        expected0, expected1 = expected
        np.testing.assert_allclose(actual0,
                                   expected0,
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))
        np.testing.assert_allclose(actual1,
                                   expected1,
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))

    def test_forward_log_det_jacobian(self):
        input = np.random.random((10, ))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        with self.assertRaises(NotImplementedError):
            exe = paddle.static.Executor()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                t = transform.AbsTransform()
<<<<<<< HEAD
                static_input = paddle.static.data(
                    'input', input.shape, input.dtype
                )
=======
                static_input = paddle.static.data('input', input.shape,
                                                  input.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                output = t.forward_log_det_jacobian(static_input)
            exe.run(sp)
            [output] = exe.run(mp, feed={'input': input}, fetch_list=[output])

<<<<<<< HEAD
    @param.param_func(
        [
            (np.array([1.0]), (np.array([0.0]), np.array([0.0]))),
        ]
    )
=======
    @param.param_func([
        (np.array([1.]), (np.array([0.]), np.array([0.]))),
    ])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_inverse_log_det_jacobian(self, input, expected):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            t = transform.AbsTransform()
            static_input = paddle.static.data('input', input.shape, input.dtype)
            actual0, actual1 = t.inverse_log_det_jacobian(static_input)
        exe.run(sp)
<<<<<<< HEAD
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
=======
        [actual0, actual1] = exe.run(mp,
                                     feed={'input': input},
                                     fetch_list=[actual0, actual1])
        expected0, expected1 = expected
        np.testing.assert_allclose(actual0,
                                   expected0,
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))
        np.testing.assert_allclose(actual1,
                                   expected1,
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)


@param.place(config.DEVICES)
<<<<<<< HEAD
@param.param_cls(
    (param.TEST_CASE_NAME, 'loc', 'scale'),
    [
        ('normal', np.random.rand(8, 10), np.random.rand(8, 10)),
        ('broadcast', np.random.rand(2, 10), np.random.rand(10)),
    ],
)
class TestAffineTransform(unittest.TestCase):
=======
@param.param_cls((param.TEST_CASE_NAME, 'loc', 'scale'), [
    ('normal', np.random.rand(8, 10), np.random.rand(8, 10)),
    ('broadcast', np.random.rand(2, 10), np.random.rand(10)),
])
class TestAffineTransform(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
<<<<<<< HEAD
            scale = paddle.static.data(
                'scale', self.scale.shape, self.scale.dtype
            )
=======
            scale = paddle.static.data('scale', self.scale.shape,
                                       self.scale.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
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
=======
            scale = paddle.static.data('scale', self.scale.shape,
                                       self.scale.dtype)
            t = transform.AffineTransform(loc, scale)
            static_input = paddle.static.data('input', self.loc.shape,
                                              self.loc.dtype)
            output = t.forward(static_input)
        exe.run(sp)
        [output] = exe.run(mp,
                           feed={
                               'input': input,
                               'loc': self.loc,
                               'scale': self.scale
                           },
                           fetch_list=[output])
        np.testing.assert_allclose(output,
                                   self._np_forward(input),
                                   rtol=config.RTOL.get(str(self.loc.dtype)),
                                   atol=config.ATOL.get(str(self.loc.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_inverse(self):
        input = np.random.random(self.loc.shape)
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
<<<<<<< HEAD
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
=======
            scale = paddle.static.data('scale', self.scale.shape,
                                       self.scale.dtype)
            t = transform.AffineTransform(loc, scale)
            static_input = paddle.static.data('input', self.loc.shape,
                                              self.loc.dtype)
            output = t.inverse(static_input)
        exe.run(sp)
        [output] = exe.run(mp,
                           feed={
                               'input': input,
                               'loc': self.loc,
                               'scale': self.scale
                           },
                           fetch_list=[output])
        np.testing.assert_allclose(output,
                                   self._np_inverse(input),
                                   rtol=config.RTOL.get(str(self.loc.dtype)),
                                   atol=config.ATOL.get(str(self.loc.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
<<<<<<< HEAD
            scale = paddle.static.data(
                'scale', self.scale.shape, self.scale.dtype
            )
=======
            scale = paddle.static.data('scale', self.scale.shape,
                                       self.scale.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            t = transform.AffineTransform(loc, scale)
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.inverse_log_det_jacobian(static_input)
        exe.run(sp)
<<<<<<< HEAD
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
=======
        [output] = exe.run(mp,
                           feed={
                               'input': input,
                               'loc': self.loc,
                               'scale': self.scale
                           },
                           fetch_list=[output])
        np.testing.assert_allclose(output,
                                   self._np_inverse_jacobian(input),
                                   rtol=config.RTOL.get(str(self.loc.dtype)),
                                   atol=config.ATOL.get(str(self.loc.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_forward_log_det_jacobian(self):
        input = np.random.random(self.scale.shape)
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
<<<<<<< HEAD
            scale = paddle.static.data(
                'scale', self.scale.shape, self.scale.dtype
            )
=======
            scale = paddle.static.data('scale', self.scale.shape,
                                       self.scale.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            t = transform.AffineTransform(loc, scale)
            static_input = paddle.static.data('input', input.shape, input.dtype)
            output = t.forward_log_det_jacobian(static_input)
        exe.run(sp)
<<<<<<< HEAD
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
=======
        [output] = exe.run(mp,
                           feed={
                               'input': input,
                               'loc': self.loc,
                               'scale': self.scale
                           },
                           fetch_list=[output])
        np.testing.assert_allclose(output,
                                   self._np_forward_jacobian(input),
                                   rtol=config.RTOL.get(str(self.loc.dtype)),
                                   atol=config.ATOL.get(str(self.loc.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_forward_shape(self):
        shape = self.loc.shape
        self.assertEqual(
            tuple(self._t.forward_shape(shape)),
<<<<<<< HEAD
            np.broadcast(np.random.random(shape), self.loc, self.scale).shape,
        )
=======
            np.broadcast(np.random.random(shape), self.loc, self.scale).shape)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_inverse_shape(self):
        shape = self.scale.shape
        self.assertEqual(
            tuple(self._t.forward_shape(shape)),
<<<<<<< HEAD
            np.broadcast(np.random.random(shape), self.loc, self.scale).shape,
        )
=======
            np.broadcast(np.random.random(shape), self.loc, self.scale).shape)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


@param.place(config.DEVICES)
class TestExpTransform(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
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
=======
    @param.param_func([(np.array([0., 1., 2.,
                                  3.]), np.exp(np.array([0., 1., 2., 3.]))),
                       (np.array([[0., 1., 2., 3.], [-5., 6., 7., 8.]]),
                        np.exp(np.array([[0., 1., 2., 3.], [-5., 6., 7.,
                                                            8.]])))])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
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
=======
        np.testing.assert_allclose(output,
                                   expected,
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))

    @param.param_func([(np.array([1., 2., 3.]), np.log(np.array([1., 2., 3.]))),
                       (np.array([[1., 2., 3.], [6., 7., 8.]]),
                        np.log(np.array([[1., 2., 3.], [6., 7., 8.]])))])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
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
=======
        np.testing.assert_allclose(output,
                                   expected,
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))

    @param.param_func([(np.array([1., 2., 3.]), ),
                       (np.array([[1., 2., 3.], [6., 7., 8.]]), )])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        np.testing.assert_allclose(
            output,
            self._np_forward_jacobian(input),
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )
=======
        np.testing.assert_allclose(output,
                                   self._np_forward_jacobian(input),
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _np_forward_jacobian(self, x):
        return x

<<<<<<< HEAD
    @param.param_func(
        [
            (np.array([1.0, 2.0, 3.0]),),
            (np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]]),),
        ]
    )
=======
    @param.param_func([(np.array([1., 2., 3.]), ),
                       (np.array([[1., 2., 3.], [6., 7., 8.]]), )])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        np.testing.assert_allclose(
            output,
            self._np_inverse_jacobian(input),
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )
=======
        np.testing.assert_allclose(output,
                                   self._np_inverse_jacobian(input),
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
<<<<<<< HEAD
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
=======

    @param.param_func(((transform.ChainTransform(
        (transform.AbsTransform(),
         transform.AffineTransform(paddle.rand([1]), paddle.rand([1])))),
                        False), (transform.ChainTransform((
                            transform.AffineTransform(paddle.rand([1]),
                                                      paddle.rand([1])),
                            transform.ExpTransform(),
                        )), True)))
    def test_is_injective(self, chain, expected):
        self.assertEqual(chain._is_injective(), expected)

    @param.param_func(((transform.ChainTransform(
        (transform.IndependentTransform(transform.ExpTransform(), 1),
         transform.IndependentTransform(transform.ExpTransform(), 10),
         transform.IndependentTransform(transform.ExpTransform(), 8))),
                        variable.Independent(variable.real, 10)), ))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_domain(self, input, expected):
        self.assertIsInstance(input._domain, type(expected))
        self.assertEqual(input._domain.event_rank, expected.event_rank)
        self.assertEqual(input._domain.is_discrete, expected.is_discrete)

<<<<<<< HEAD
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
=======
    @param.param_func(((transform.ChainTransform(
        (transform.IndependentTransform(transform.ExpTransform(), 9),
         transform.IndependentTransform(transform.ExpTransform(), 4),
         transform.IndependentTransform(transform.ExpTransform(), 5))),
                        variable.Independent(variable.real, 9)), ))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_codomain(self, input, expected):
        self.assertIsInstance(input._codomain, variable.Independent)
        self.assertEqual(input._codomain.event_rank, expected.event_rank)
        self.assertEqual(input._codomain.is_discrete, expected.is_discrete)

<<<<<<< HEAD
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
=======
    @param.param_func([
        (transform.ChainTransform(
            (transform.ExpTransform(), transform.TanhTransform())),
         np.array([[0., -1., 2., -3.], [-5., 6., 7., -8.]]),
         np.tanh(np.exp(np.array([[0., -1., 2., -3.], [-5., 6., 7., -8.]]))))
    ])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
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
=======
        np.testing.assert_allclose(output,
                                   expected,
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))

    @param.param_func([
        (transform.ChainTransform(
            (transform.ExpTransform(), transform.TanhTransform())),
         np.array([[0., 1., 2., 3.], [5., 6., 7., 8.]]),
         np.log(np.arctanh(np.array([[0., 1., 2., 3.], [5., 6., 7., 8.]]))))
    ])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
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
=======
        np.testing.assert_allclose(output,
                                   expected,
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))

    @param.param_func([
        (transform.ChainTransform(
            (transform.AffineTransform(paddle.full([1], 0.0),
                                       paddle.full([1], -1.0)),
             transform.ExpTransform())), (2, 3, 5), (2, 3, 5)),
    ])
    def test_forward_shape(self, chain, shape, expected_shape):
        self.assertEqual(chain.forward_shape(shape), expected_shape)

    @param.param_func([
        (transform.ChainTransform(
            (transform.AffineTransform(paddle.full([1], 0.0),
                                       paddle.full([1], -1.0)),
             transform.ExpTransform())), (2, 3, 5), (2, 3, 5)),
    ])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_inverse_shape(self, chain, shape, expected_shape):
        self.assertEqual(chain.forward_shape(shape), expected_shape)


@param.place(config.DEVICES)
@param.param_cls(
<<<<<<< HEAD
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
=======
    (param.TEST_CASE_NAME, 'base', 'reinterpreted_batch_rank', 'x'), [
        ('rank-over-zero', transform.ExpTransform(), 2, np.random.rand(2, 3,
                                                                       3)),
    ])
class TestIndependentTransform(unittest.TestCase):

    def setUp(self):
        self._t = transform.IndependentTransform(self.base,
                                                 self.reinterpreted_batch_rank)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_is_injective(self):
        self.assertEqual(self._t._is_injective(), self.base._is_injective())

    def test_domain(self):
        self.assertTrue(isinstance(self._t._domain, variable.Independent))
        self.assertEqual(
            self._t._domain.event_rank,
<<<<<<< HEAD
            self.base._domain.event_rank + self.reinterpreted_batch_rank,
        )
        self.assertEqual(
            self._t._domain.is_discrete, self.base._domain.is_discrete
        )
=======
            self.base._domain.event_rank + self.reinterpreted_batch_rank)
        self.assertEqual(self._t._domain.is_discrete,
                         self.base._domain.is_discrete)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_codomain(self):
        self.assertTrue(isinstance(self._t._codomain, variable.Independent))
        self.assertEqual(
            self._t._codomain.event_rank,
<<<<<<< HEAD
            self.base._codomain.event_rank + self.reinterpreted_batch_rank,
        )
        self.assertEqual(
            self._t._codomain.is_discrete, self.base._codomain.is_discrete
        )
=======
            self.base._codomain.event_rank + self.reinterpreted_batch_rank)
        self.assertEqual(self._t._codomain.is_discrete,
                         self.base._codomain.is_discrete)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_forward(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
<<<<<<< HEAD
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
=======
            t = transform.IndependentTransform(self.base,
                                               self.reinterpreted_batch_rank)
            static_input = paddle.static.data('input', self.x.shape,
                                              self.x.dtype)
            output = t.forward(static_input)
            expected = self.base.forward(static_input)
        exe.run(sp)
        [output, expected] = exe.run(mp,
                                     feed={'input': self.x},
                                     fetch_list=[output, expected])
        np.testing.assert_allclose(output,
                                   expected,
                                   rtol=config.RTOL.get(str(self.x.dtype)),
                                   atol=config.ATOL.get(str(self.x.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_inverse(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
<<<<<<< HEAD
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
=======
            t = transform.IndependentTransform(self.base,
                                               self.reinterpreted_batch_rank)
            static_input = paddle.static.data('input', self.x.shape,
                                              self.x.dtype)
            output = t.inverse(static_input)
            expected = self.base.inverse(static_input)
        exe.run(sp)
        [output, expected] = exe.run(mp,
                                     feed={'input': self.x},
                                     fetch_list=[output, expected])
        np.testing.assert_allclose(expected,
                                   output,
                                   rtol=config.RTOL.get(str(self.x.dtype)),
                                   atol=config.ATOL.get(str(self.x.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_forward_log_det_jacobian(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
<<<<<<< HEAD
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
=======
            t = transform.IndependentTransform(self.base,
                                               self.reinterpreted_batch_rank)
            static_input = paddle.static.data('input', self.x.shape,
                                              self.x.dtype)
            output = t.forward_log_det_jacobian(static_input)
            expected = self.base.forward_log_det_jacobian(
                static_input.sum(list(range(-self.reinterpreted_batch_rank,
                                            0))))
        exe.run(sp)
        [actual, expected] = exe.run(mp,
                                     feed={'input': self.x},
                                     fetch_list=[output, expected])
        self.assertEqual(tuple(actual.shape),
                         self.x.shape[:-self.reinterpreted_batch_rank])
        np.testing.assert_allclose(actual,
                                   expected,
                                   rtol=config.RTOL.get(str(self.x.dtype)),
                                   atol=config.ATOL.get(str(self.x.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)


@param.place(config.DEVICES)
class TestPowerTransform(unittest.TestCase):
<<<<<<< HEAD
    def setUp(self):
        self._t = transform.PowerTransform(paddle.full([1], 2.0))

    def test_init(self):
        with self.assertRaises(TypeError):
            transform.PowerTransform(1.0)
=======

    def setUp(self):
        self._t = transform.PowerTransform(paddle.full([1], 2.))

    def test_init(self):
        with self.assertRaises(TypeError):
            transform.PowerTransform(1.)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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

<<<<<<< HEAD
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
=======
    @param.param_func([(np.array([2.]), np.array([0., -1., 2.]),
                        np.power(np.array([0., -1., 2.]), 2.)),
                       (np.array([[0.], [3.]]), np.array([[1., 0.], [5., 6.]]),
                        np.power(np.array([[1., 0.], [5., 6.]]),
                                 np.array([[0.], [3.]])))])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
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
=======
        [output] = exe.run(mp,
                           feed={
                               'input': input,
                               'power': power
                           },
                           fetch_list=[output])
        np.testing.assert_allclose(output,
                                   expected,
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))

    @param.param_func([(np.array([2.]), np.array([4.]), np.array([2.]))])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
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
=======
        [output] = exe.run(mp,
                           feed={
                               'input': input,
                               'power': power
                           },
                           fetch_list=[output])
        np.testing.assert_allclose(output,
                                   expected,
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))

    @param.param_func(((np.array([2.]), np.array([3., 1.4, 0.8])), ))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        [output] = exe.run(
            mp, feed={'input': input, 'power': power}, fetch_list=[output]
        )
        np.testing.assert_allclose(
            output,
            self._np_forward_jacobian(power, input),
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )
=======
        [output] = exe.run(mp,
                           feed={
                               'input': input,
                               'power': power
                           },
                           fetch_list=[output])
        np.testing.assert_allclose(output,
                                   self._np_forward_jacobian(power, input),
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
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
=======
    @param.param_func([(np.array([0., 1., 2.,
                                  3.]), np.tanh(np.array([0., 1., 2., 3.]))),
                       (np.array([[0., 1., 2., 3.], [-5., 6., 7., 8.]]),
                        np.tanh(np.array([[0., 1., 2., 3.], [-5., 6., 7.,
                                                             8.]])))])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
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
=======
        np.testing.assert_allclose(output,
                                   expected,
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))

    @param.param_func([(np.array([1., 2.,
                                  3.]), np.arctanh(np.array([1., 2., 3.]))),
                       (np.array([[1., 2., 3.], [6., 7., 8.]]),
                        np.arctanh(np.array([[1., 2., 3.], [6., 7., 8.]])))])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
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
=======
        np.testing.assert_allclose(output,
                                   expected,
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))

    @param.param_func([(np.array([1., 2., 3.]), ),
                       (np.array([[1., 2., 3.], [6., 7., 8.]]), )])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
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
=======
        np.testing.assert_allclose(output,
                                   self._np_forward_jacobian(input),
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))

    def _np_forward_jacobian(self, x):
        return 2. * (np.log(2.) - x - self._np_softplus(-2. * x))

    def _np_softplus(self, x, beta=1., threshold=20.):
        if np.any(beta * x > threshold):
            return x
        return 1. / beta * np.log1p(np.exp(beta * x))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _np_inverse_jacobian(self, y):
        return -self._np_forward_jacobian(np.arctanh(y))

<<<<<<< HEAD
    @param.param_func(
        [
            (np.array([1.0, 2.0, 3.0]),),
            (np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]]),),
        ]
    )
=======
    @param.param_func([(np.array([1., 2., 3.]), ),
                       (np.array([[1., 2., 3.], [6., 7., 8.]]), )])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        np.testing.assert_allclose(
            output,
            self._np_inverse_jacobian(input),
            rtol=config.RTOL.get(str(input.dtype)),
            atol=config.ATOL.get(str(input.dtype)),
        )
=======
        np.testing.assert_allclose(output,
                                   self._np_inverse_jacobian(input),
                                   rtol=config.RTOL.get(str(input.dtype)),
                                   atol=config.ATOL.get(str(input.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_forward_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)

    @param.param_func([((), ()), ((2, 3, 5), (2, 3, 5))])
    def test_inverse_shape(self, shape, expected_shape):
        self.assertEqual(self._t.forward_shape(shape), expected_shape)


@param.place(config.DEVICES)
<<<<<<< HEAD
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
=======
@param.param_cls((param.TEST_CASE_NAME, 'in_event_shape', 'out_event_shape'), [
    ('regular_shape', (2, 3), (3, 2)),
])
class TestReshapeTransform(unittest.TestCase):

    def setUp(self):
        self._t = transform.ReshapeTransform(self.in_event_shape,
                                             self.out_event_shape)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
<<<<<<< HEAD
            t = transform.ReshapeTransform(
                self.in_event_shape, self.out_event_shape
            )
=======
            t = transform.ReshapeTransform(self.in_event_shape,
                                           self.out_event_shape)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            output = self._t.forward(x)
        exe.run(sp)
        [output] = exe.run(mp, feed={}, fetch_list=[output])
        expected = np.ones(self.out_event_shape)
<<<<<<< HEAD
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(expected.dtype)),
            atol=config.ATOL.get(str(expected.dtype)),
        )
=======
        np.testing.assert_allclose(output,
                                   expected,
                                   rtol=config.RTOL.get(str(expected.dtype)),
                                   atol=config.ATOL.get(str(expected.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_inverse(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = paddle.ones(self.out_event_shape)
<<<<<<< HEAD
            t = transform.ReshapeTransform(
                self.in_event_shape, self.out_event_shape
            )
=======
            t = transform.ReshapeTransform(self.in_event_shape,
                                           self.out_event_shape)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            output = self._t.inverse(x)
        exe.run(sp)
        [output] = exe.run(mp, feed={}, fetch_list=[output])
        expected = np.ones(self.in_event_shape)

<<<<<<< HEAD
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(expected.dtype)),
            atol=config.ATOL.get(str(expected.dtype)),
        )
=======
        np.testing.assert_allclose(output,
                                   expected,
                                   rtol=config.RTOL.get(str(expected.dtype)),
                                   atol=config.ATOL.get(str(expected.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_forward_log_det_jacobian(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = paddle.ones(self.in_event_shape)
<<<<<<< HEAD
            t = transform.ReshapeTransform(
                self.in_event_shape, self.out_event_shape
            )
=======
            t = transform.ReshapeTransform(self.in_event_shape,
                                           self.out_event_shape)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            output = self._t.forward_log_det_jacobian(x)
        exe.run(sp)
        [output] = exe.run(mp, feed={}, fetch_list=[output])
        expected = np.zeros([1])
<<<<<<< HEAD
        np.testing.assert_allclose(
            output,
            expected,
            rtol=config.RTOL.get(str(expected.dtype)),
            atol=config.ATOL.get(str(expected.dtype)),
        )
=======
        np.testing.assert_allclose(output,
                                   expected,
                                   rtol=config.RTOL.get(str(expected.dtype)),
                                   atol=config.ATOL.get(str(expected.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()
