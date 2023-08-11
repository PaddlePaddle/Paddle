# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import sys
import typing
import unittest

sys.path.insert(0, '.')

import config
import numpy as np
import utils

import paddle
import paddle.nn.functional as F
from paddle.incubate.autograd.utils import as_tensors


def make_v(f, inputs):
    outputs = as_tensors(f(*inputs))
    return [paddle.ones_like(x) for x in outputs]


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'func', 'xs'),
    (
        ('1d_in_1d_out', utils.square, np.array([2.0, 3.0])),
        (
            'single_in_single_out',
            utils.square,
            np.random.rand(
                6,
            ),
        ),
        (
            'multi_in_single_out',
            paddle.matmul,
            (
                np.random.rand(
                    4,
                ),
                np.random.rand(
                    4,
                ),
            ),
        ),
    ),
)
class TestJacobianNoBatch(unittest.TestCase):
    def setUp(self):
        self._dtype = (
            self.xs[0].dtype
            if isinstance(self.xs, typing.Sequence)
            else self.xs.dtype
        )
        self._eps = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("eps")
        )
        self._rtol = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("rtol")
        )
        self._atol = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("atol")
        )

    def test_jacobian(self):
        xs = (
            [paddle.to_tensor(x, stop_gradient=False) for x in self.xs]
            if isinstance(self.xs, typing.Sequence)
            else paddle.to_tensor(self.xs, stop_gradient=False)
        )
        ys = (
            self.func(*xs) if isinstance(xs, typing.Sequence) else self.func(xs)
        )
        self._actual = paddle.autograd.jacobian(ys, xs, batch_axis=None)
        if isinstance(self._actual, (tuple, list)):
            self._actual = paddle.concat([x[:] for x in self._actual], axis=0)
        self._expected = self._get_expected()

        self.assertEqual(self._actual.numpy().dtype, self._expected.dtype)
        np.testing.assert_allclose(
            self._actual.flatten(),
            self._expected.flatten(),
            rtol=self._rtol,
            atol=self._atol,
        )

    def test_jacobian_attribute_operator(self):
        xs = (
            [paddle.to_tensor(x, stop_gradient=False) for x in self.xs]
            if isinstance(self.xs, typing.Sequence)
            else paddle.to_tensor(self.xs, stop_gradient=False)
        )
        ys = (
            self.func(*xs) if isinstance(xs, typing.Sequence) else self.func(xs)
        )
        self._actual = paddle.autograd.jacobian(ys, xs, batch_axis=None)
        if isinstance(self._actual, (tuple, list)):
            self._actual = paddle.concat([x[:] for x in self._actual], axis=0)
        self._expected = self._get_expected()

        self.assertEqual(self._actual.numpy().dtype, self._expected.dtype)
        np.testing.assert_allclose(
            self._actual.flatten(),
            self._expected.flatten(),
            rtol=self._rtol,
            atol=self._atol,
        )

    def _get_expected(self):
        xs = (
            [paddle.to_tensor(x, stop_gradient=False) for x in self.xs]
            if isinstance(self.xs, typing.Sequence)
            else paddle.to_tensor(self.xs, stop_gradient=False)
        )
        jac = utils._compute_numerical_jacobian(
            self.func, xs, self._eps, self._dtype
        )
        return utils._np_concat_matrix_sequence(jac, utils.MatrixFormat.NM)


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'func', 'xs'),
    (
        (
            '1d_in_1d_out',
            utils.square,
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 3.0]]),
        ),
        ('multi_in_single_out', utils.square, np.random.rand(2, 3)),
    ),
)
class TestJacobianBatchFirst(unittest.TestCase):
    def setUp(self):
        self._dtype = (
            self.xs[0].dtype
            if isinstance(self.xs, typing.Sequence)
            else self.xs.dtype
        )
        self._eps = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("eps")
        )
        self._rtol = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("rtol")
        )
        self._atol = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("atol")
        )

    def test_jacobian(self):
        xs = (
            [paddle.to_tensor(x, stop_gradient=False) for x in self.xs]
            if isinstance(self.xs, typing.Sequence)
            else paddle.to_tensor(self.xs, stop_gradient=False)
        )
        ys = (
            self.func(*xs) if isinstance(xs, typing.Sequence) else self.func(xs)
        )
        self._actual = paddle.autograd.jacobian(ys, xs, batch_axis=0)
        self._expected = self._get_expected()

        Index = collections.namedtuple('Index', ('type', 'value'))
        indexes = (
            Index(
                'all',
                (
                    slice(0, None, None),
                    slice(0, None, None),
                    slice(0, None, None),
                ),
            ),
            Index('row', (slice(0, None, None), 0, slice(0, None, None))),
            Index('col', (slice(0, None, None), slice(0, None, None), 0)),
            Index(
                'batch',
                (slice(0, 2, None), slice(0, None, None), slice(0, None, None)),
            ),
            Index(
                'multi_row',
                (slice(0, 1, None), slice(0, 2, 1), slice(0, None, None)),
            ),
        )
        self.assertEqual(self._actual[:].numpy().dtype, self._expected.dtype)
        for index in indexes:
            np.testing.assert_allclose(
                self._actual.__getitem__(index.value),
                self._expected.__getitem__(index.value),
                rtol=self._rtol,
                atol=self._atol,
                err_msg=f'Testcase {index.type} index not passed, value is {index.value}',
            )

    def test_jacobian_attribute_operator(self):
        # test for attribute operator "."
        xs = (
            [paddle.to_tensor(x, stop_gradient=False) for x in self.xs]
            if isinstance(self.xs, typing.Sequence)
            else paddle.to_tensor(self.xs, stop_gradient=False)
        )
        ys = (
            self.func(*xs) if isinstance(xs, typing.Sequence) else self.func(xs)
        )
        self._actual = paddle.autograd.jacobian(ys, xs, batch_axis=0)
        self._expected = self._get_expected()

        Index = collections.namedtuple('Index', ('type', 'value'))
        indexes = (
            Index(
                'all',
                (
                    slice(0, None, None),
                    slice(0, None, None),
                    slice(0, None, None),
                ),
            ),
            Index('row', (slice(0, None, None), 0, slice(0, None, None))),
            Index('col', (slice(0, None, None), slice(0, None, None), 0)),
            Index(
                'batch',
                (slice(0, 2, None), slice(0, None, None), slice(0, None, None)),
            ),
            Index(
                'multi_row',
                (slice(0, 1, None), slice(0, 2, 1), slice(0, None, None)),
            ),
        )
        self.assertEqual(self._actual.numpy().dtype, self._expected.dtype)
        for index in indexes:
            np.testing.assert_allclose(
                self._actual.__getitem__(index.value),
                self._expected.__getitem__(index.value),
                rtol=self._rtol,
                atol=self._atol,
                err_msg=f'Testcase {index.type} index not passed, value is {index.value}',
            )

    def _get_expected(self):
        xs = (
            [paddle.to_tensor(x, stop_gradient=False) for x in self.xs]
            if isinstance(self.xs, typing.Sequence)
            else paddle.to_tensor(self.xs, stop_gradient=False)
        )
        jac = utils._compute_numerical_batch_jacobian(
            self.func, xs, self._eps, self._dtype, False
        )
        jac = utils._np_concat_matrix_sequence(jac, utils.MatrixFormat.NBM)
        return utils._np_transpose_matrix_format(
            jac, utils.MatrixFormat.NBM, utils.MatrixFormat.BNM
        )


class TestHessianNoBatch(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.shape = (4,)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = (
            config.TOLERANCE.get(self.dtype).get("second_order_grad").get("eps")
        )
        self.rtol = (
            config.TOLERANCE.get(self.dtype)
            .get("second_order_grad")
            .get("rtol")
        )
        self.atol = (
            config.TOLERANCE.get(self.dtype)
            .get("second_order_grad")
            .get("atol")
        )
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)

    def func_create_graph_true(self):
        def func(x):
            return paddle.sum(F.sigmoid(x))

        numerical_hessian = utils._compute_numerical_hessian(
            func, self.x, self.numerical_delta, self.np_dtype
        )
        numerical_hessian = utils._np_concat_matrix_sequence(numerical_hessian)
        self.x.stop_gradient = False
        hessian = paddle.autograd.hessian(func(self.x), self.x, batch_axis=None)
        assert not hessian[:].stop_gradient
        np.testing.assert_allclose(
            hessian[:].numpy(), numerical_hessian, self.rtol, self.atol
        )

    def func_out_not_single(self):
        def func(x):
            return x * x

        with self.assertRaises(ValueError):
            x = paddle.ones([3])
            paddle.autograd.hessian(func(x), x, batch_axis=None)

    def func_add(self):
        def func(x):
            return (x * x).sum()

        H = paddle.autograd.hessian(func(self.x), self.x)
        expected = np.diag(np.full((self.x.size,), 2.0))

        expected = expected + 1.0
        actual = H + 1.0
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_sub(self):
        def func(x):
            return (x * x).sum()

        H = paddle.autograd.hessian(func(self.x), self.x)
        expected = np.diag(np.full((self.x.size,), 2.0))

        expected = expected - 1.0
        actual = H - 1.0
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_mul(self):
        def func(x):
            return (x * x).sum()

        H = paddle.autograd.hessian(func(self.x), self.x)
        expected = np.diag(np.full((self.x.size,), 2.0))

        expected = expected * 2.0
        actual = H * 2.0
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_div(self):
        def func(x):
            return (x * x).sum()

        H = paddle.autograd.hessian(func(self.x), self.x)
        expected = np.diag(np.full((self.x.size,), 2.0))

        expected = expected / 2.0
        actual = H / 2.0
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_truediv(self):
        def func(x):
            return (x * x).sum()

        H = paddle.autograd.hessian(func(self.x), self.x)
        expected = np.diag(np.full((self.x.size,), 2.0))

        expected = expected / 2.0
        actual = H / 2.0
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_pow(self):
        def func(x):
            return (x * x).sum()

        H = paddle.autograd.hessian(func(self.x), self.x)
        expected = np.diag(np.full((self.x.size,), 2.0))

        expected = expected**3.0
        actual = H**3.0
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_mod(self):
        def func(x):
            return (x * x).sum()

        H = paddle.autograd.hessian(func(self.x), self.x)
        expected = np.diag(np.full((self.x.size,), 2.0))

        expected = expected % 1.2
        actual = H % 1.2
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_matmul(self):
        def func(x):
            return (x * x).sum()

        H = paddle.autograd.hessian(func(self.x), self.x)
        expected = np.diag(np.full((self.x.size,), 2.0))

        expected = expected @ expected
        actual = H @ H
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_eq(self):
        def func(x):
            return (x * x).sum()

        H = paddle.autograd.hessian(func(self.x), self.x)
        expected = np.diag(np.full((self.x.size,), 2.0))

        expected = expected == expected
        actual = H == H
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_ne(self):
        def func(x):
            return (x * x).sum()

        H = paddle.autograd.hessian(func(self.x), self.x)
        expected = np.diag(np.full((self.x.size,), 2.0))

        expected = expected != expected
        actual = H != H
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_lt(self):
        def func(x):
            return (x * x).sum()

        H = paddle.autograd.hessian(func(self.x), self.x)
        expected = np.diag(np.full((self.x.size,), 2.0))

        expected = expected < expected
        actual = H < H
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_le(self):
        def func(x):
            return (x * x).sum()

        H = paddle.autograd.hessian(func(self.x), self.x)
        expected = np.diag(np.full((self.x.size,), 2.0))

        expected = expected <= expected
        actual = H <= H
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_gt(self):
        def func(x):
            return (x * x).sum()

        H = paddle.autograd.hessian(func(self.x), self.x)
        expected = np.diag(np.full((self.x.size,), 2.0))

        expected = expected > expected
        actual = H > H
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_ge(self):
        def func(x):
            return (x * x).sum()

        H = paddle.autograd.hessian(func(self.x), self.x)
        expected = np.diag(np.full((self.x.size,), 2.0))

        expected = expected >= expected
        actual = H >= H
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_0Dtensor_index(self):
        x_0d = self.x[0].reshape([])

        def func(x):
            return x * x

        with self.assertRaises(IndexError):
            H = paddle.autograd.hessian(func(x_0d), x_0d)
            H = H[:]

    def func_2Dtensor(self):
        x_2d = self.x.reshape([self.x.shape[0] // 2, 2])

        def func(x):
            return (x * x).sum()

        with self.assertRaises(ValueError):
            H = paddle.autograd.hessian(func(x_2d), x_2d)

    def test_all_cases(self):
        self.setUpClass()
        self.func_create_graph_true()
        self.func_out_not_single()
        self.func_add()
        self.func_sub()
        self.func_mul()
        self.func_div()
        self.func_truediv()
        self.func_pow()
        self.func_mod()
        self.func_matmul()
        self.func_eq()
        self.func_ne()
        self.func_lt()
        self.func_le()
        self.func_gt()
        self.func_ge()
        self.func_0Dtensor_index()
        self.func_2Dtensor()


class TestHessianBatchFirst(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.x_shape = (5, 2)
        self.weight_shape = (2, 4)
        self.y_shape = (5, 2)
        self.nbatch, self.nrow = 5, 2
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = (
            config.TOLERANCE.get(self.dtype).get('second_order_grad').get('eps')
        )
        self.rtol = (
            config.TOLERANCE.get(self.dtype)
            .get('second_order_grad')
            .get('rtol')
        )
        self.atol = (
            config.TOLERANCE.get(self.dtype)
            .get('second_order_grad')
            .get('atol')
        )
        self.x = paddle.rand(shape=self.x_shape, dtype=self.dtype)
        self.x.stop_gradient = False
        self.weight = paddle.rand(shape=self.weight_shape, dtype=self.dtype)
        self.weight.stop_gradient = False
        self.y = paddle.rand(shape=self.y_shape, dtype=self.dtype)
        self.y.stop_gradient = False

    def func_allow_unused(self):
        def func(x, y):
            return paddle.matmul(x * x, self.weight)[:, 0:1]

        xs_len = 2
        expected = utils._compute_numerical_batch_hessian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype
        )
        expected = np.reshape(
            np.array(expected),
            (xs_len, xs_len, self.nrow, self.nbatch, self.nrow),
        )
        expected = [list(row) for row in expected]
        expected = utils._np_concat_matrix_sequence(expected)
        expected = utils._np_transpose_matrix_format(
            expected, utils.MatrixFormat.NBM, utils.MatrixFormat.BNM
        )

        actual = paddle.autograd.hessian(
            func(self.x, self.y), [self.x, self.y], batch_axis=0
        )
        actual = paddle.concat(
            [
                paddle.concat([actual[i][j][:] for j in range(2)], axis=2)
                for i in range(2)
            ],
            axis=1,
        )

        np.testing.assert_allclose(
            actual.shape, expected.shape, rtol=self.rtol, atol=self.atol
        )

    def func_stop_gradient(self):
        def func(x):
            return paddle.matmul(x * x, self.weight)[:, 0:1]

        expected = utils._compute_numerical_batch_hessian(
            func, self.x, self.numerical_delta, self.np_dtype
        )

        x = self.x.clone()
        x.stop_gradient = True
        H = paddle.autograd.hessian(func(self.x), self.x, batch_axis=0)[:]
        actual = utils._np_transpose_matrix_format(
            H[:].numpy(), utils.MatrixFormat.BNM, utils.MatrixFormat.NBM
        )
        actual = actual.reshape((H.shape[1], -1))

        np.testing.assert_allclose(
            actual.shape, np.asarray(expected).shape, self.rtol, self.atol
        )

    def func_out_not_single(self):
        def func(x):
            return x * x

        with self.assertRaises(ValueError):
            x = paddle.ones((3, 3))
            paddle.autograd.hessian(func(x), x, batch_axis=0)

    def func_batch_axis_except_0(self):
        def func(x):
            return x * x

        with self.assertRaises(ValueError):
            x = paddle.ones([3])
            paddle.autograd.hessian(func(x), x, batch_axis=2)

    def func_ndim_bigger_than_2(self):
        def func(x):
            return (x * x).sum([1, 2, 3])

        with self.assertRaises(ValueError):
            x = paddle.ones([3, 3, 3, 3])
            paddle.autograd.hessian(func(x), x, batch_axis=0)

    def func_batch_axis_str(self):
        def func(x):
            return (x * x).sum()

        with self.assertRaises(ValueError):
            x = paddle.ones([3, 3, 3, 3])
            paddle.autograd.hessian(func(x), x, batch_axis="0")

    def func_ellipsis_index(self):
        def func(x):
            return (x * x).sum()

        with self.assertRaises(IndexError):
            x = paddle.ones([2, 3])
            H = paddle.autograd.hessian(func(x), x, batch_axis=0)[..., 1]

    def test_all_cases(self):
        self.setUpClass()
        self.func_allow_unused()
        self.func_stop_gradient()
        self.func_out_not_single()
        self.func_batch_axis_except_0()
        self.func_ndim_bigger_than_2()
        self.func_batch_axis_str()
        self.func_ellipsis_index()


if __name__ == "__main__":
    np.random.seed(2022)
    unittest.main()
