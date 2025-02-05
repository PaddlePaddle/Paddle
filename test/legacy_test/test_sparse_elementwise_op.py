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
from operator import __add__, __mul__, __sub__, __truediv__

import numpy as np

import paddle
from paddle.base.framework import in_pir_mode

op_list = [__add__, __sub__, __mul__, __truediv__]


def get_actual_res(x, y, op):
    if op == __add__:
        res = paddle.sparse.add(x, y)
    elif op == __sub__:
        res = paddle.sparse.subtract(x, y)
    elif op == __mul__:
        res = paddle.sparse.multiply(x, y)
    elif op == __truediv__:
        res = paddle.sparse.divide(x, y)
    else:
        raise ValueError("unsupported op")
    return res


def mask_to_zero(x, mask):
    x[mask == 0] = 0
    return x


class TestSparseElementWiseAPI(unittest.TestCase):
    """
    test paddle.sparse.add, subtract, multiply, divide
    """

    def setUp(self):
        np.random.seed(2022)
        self.op_list = op_list
        self.csr_shape = [8, 8]
        self.coo_shape = [4, 8, 3, 5]
        self.support_dtypes = ['float32', 'float64', 'int32', 'int64']

    def func_test_csr(self, op):
        for dtype in self.support_dtypes:
            x = np.random.randint(-255, 255, size=self.csr_shape)
            y = np.random.randint(-255, 255, size=self.csr_shape)
            mask_x = x / x
            mask_y = y / y
            mask_x[mask_x != 1] = 0
            mask_y[mask_y != 1] = 0
            x = x.astype(dtype)
            y = y.astype(dtype)

            dense_x = paddle.to_tensor(x, dtype=dtype, stop_gradient=False)
            dense_y = paddle.to_tensor(y, dtype=dtype, stop_gradient=False)

            s_dense_x = paddle.to_tensor(x, dtype=dtype, stop_gradient=False)
            s_dense_y = paddle.to_tensor(y, dtype=dtype, stop_gradient=False)
            csr_x = s_dense_x.to_sparse_csr()
            csr_y = s_dense_y.to_sparse_csr()

            actual_res = get_actual_res(csr_x, csr_y, op)
            actual_res.backward()

            expect_res = op(dense_x, dense_y)
            expect_res.backward()

            np.testing.assert_allclose(
                expect_res.numpy(),
                actual_res.to_dense().numpy(),
                rtol=1e-05,
                equal_nan=True,
            )
            if not (op == __truediv__ and dtype in ['int32', 'int64']):
                np.testing.assert_allclose(
                    mask_to_zero(dense_x.grad.numpy(), mask_x),
                    csr_x.grad.to_dense().numpy(),
                    rtol=1e-05,
                    equal_nan=True,
                )
                np.testing.assert_allclose(
                    mask_to_zero(dense_y.grad.numpy(), mask_y),
                    csr_y.grad.to_dense().numpy(),
                    rtol=1e-05,
                    equal_nan=True,
                )

    def func_test_coo(self, op):
        for sparse_dim in range(len(self.coo_shape) - 1, len(self.coo_shape)):
            for dtype in self.support_dtypes:
                x = np.random.randint(-255, 255, size=self.coo_shape).astype(
                    dtype
                )
                y = np.random.randint(-255, 255, size=self.coo_shape).astype(
                    dtype
                )

                dense_x = paddle.to_tensor(x, dtype=dtype, stop_gradient=False)
                dense_y = paddle.to_tensor(y, dtype=dtype, stop_gradient=False)

                s_dense_x = paddle.to_tensor(
                    x, dtype=dtype, stop_gradient=False
                )
                s_dense_y = paddle.to_tensor(
                    y, dtype=dtype, stop_gradient=False
                )
                coo_x = s_dense_x.to_sparse_coo(sparse_dim)
                coo_x.retain_grads()
                coo_y = s_dense_y.to_sparse_coo(sparse_dim)
                coo_y.retain_grads()

                actual_res = get_actual_res(coo_x, coo_y, op)
                actual_res.backward(actual_res)

                expect_res = op(dense_x, dense_y)
                expect_res.backward(expect_res)

                np.testing.assert_allclose(
                    expect_res.numpy(),
                    actual_res.to_dense().numpy(),
                    rtol=1e-05,
                    equal_nan=True,
                )
                np.testing.assert_allclose(coo_x.shape, coo_x.grad.shape)
                np.testing.assert_allclose(
                    dense_x.grad.numpy(),
                    coo_x.grad.to_dense().numpy(),
                    rtol=1e-05,
                    equal_nan=True,
                )
                np.testing.assert_allclose(coo_y.shape, coo_y.grad.shape)
                np.testing.assert_allclose(
                    dense_y.grad.numpy(),
                    coo_y.grad.to_dense().numpy(),
                    rtol=1e-05,
                    equal_nan=True,
                )

    def test_support_dtypes_csr(self):
        paddle.device.set_device('cpu')
        if paddle.device.get_device() == "cpu":
            for op in self.op_list:
                self.func_test_csr(op)

    def test_support_dtypes_coo(self):
        paddle.device.set_device('cpu')
        if paddle.device.get_device() == "cpu":
            for op in self.op_list:
                self.func_test_coo(op)

    def test_add_same_indices(self):
        indices_data = [[0, 1], [0, 3]]
        values1_data = [[1.0], [2.0]]
        values2_data = [[1.0], [2.0]]
        shape = [2, 4, 2]

        sp_a = paddle.sparse.sparse_coo_tensor(
            indices_data, values1_data, shape, stop_gradient=False
        )
        sp_a.retain_grads()

        sp_b = paddle.sparse.sparse_coo_tensor(
            indices_data, values2_data, shape, stop_gradient=False
        )
        sp_b.retain_grads()

        values1 = paddle.to_tensor(values1_data, stop_gradient=False)
        values2 = paddle.to_tensor(values2_data, stop_gradient=False)

        # c.values() = a.values() + b.values()
        sp_c = paddle.sparse.add(sp_a, sp_b)
        sp_c.backward()
        ref_c = values1 + values2
        ref_c.backward()
        np.testing.assert_allclose(sp_c.values().numpy(), ref_c.numpy())
        np.testing.assert_allclose(
            sp_a.grad.values().numpy(), values1.grad.numpy()
        )
        np.testing.assert_allclose(
            sp_b.grad.values().numpy(), values2.grad.numpy()
        )

    def test_add_bias(self):
        indices_data = [[0, 1], [0, 3]]
        values_data = [[1.0, 1.0], [2.0, 2.0]]
        shape = [2, 4, 2]

        sp_a = paddle.sparse.sparse_coo_tensor(
            indices_data, values_data, shape, stop_gradient=False
        )
        sp_a.retain_grads()

        bias_values = [1.0, 2.0]

        values1 = paddle.to_tensor(values_data, stop_gradient=False)
        values2 = paddle.to_tensor(bias_values, stop_gradient=False)
        values3 = paddle.to_tensor(bias_values, stop_gradient=False)

        # c.values() = a.values() + b
        sp_c = paddle.sparse.add(sp_a, values2)
        sp_c.backward()
        ref_c = values1 + values3
        ref_c.backward()
        np.testing.assert_allclose(sp_c.values().numpy(), ref_c.numpy())
        np.testing.assert_allclose(
            sp_a.grad.values().numpy(), values1.grad.numpy()
        )
        np.testing.assert_allclose(values2.grad.numpy(), values3.grad.numpy())


class TestSparseElementWiseAPIComplex(unittest.TestCase):
    def setUp(self):
        np.random.seed(2022)
        self.op_list = op_list
        self.csr_shape = [8, 10]
        self.coo_shape = [3, 7, 2, 9]
        self.support_dtypes = ['complex64', 'complex128']

    def func_test_csr(self, op):
        for dtype in self.support_dtypes:
            x = np.vectorize(complex)(
                np.random.randint(-255, 255, size=self.csr_shape),
                np.random.randint(-255, 255, size=self.csr_shape),
            )
            y = np.vectorize(complex)(
                np.random.randint(-255, 255, size=self.csr_shape),
                np.random.randint(-255, 255, size=self.csr_shape),
            )
            mask_x = x / x
            mask_y = y / y
            mask_x[mask_x > 10000.0] = 0
            mask_y[mask_y > 10000.0] = 0
            x = x.astype(dtype)
            y = y.astype(dtype)

            dense_x = paddle.to_tensor(x, dtype=dtype, stop_gradient=False)
            dense_y = paddle.to_tensor(y, dtype=dtype, stop_gradient=False)

            s_dense_x = paddle.to_tensor(x, dtype=dtype, stop_gradient=False)
            s_dense_y = paddle.to_tensor(y, dtype=dtype, stop_gradient=False)
            csr_x = s_dense_x.to_sparse_csr()
            csr_y = s_dense_y.to_sparse_csr()

            actual_res = get_actual_res(csr_x, csr_y, op)
            actual_res.backward()

            expect_res = op(dense_x, dense_y)
            expect_res.backward()

            np.testing.assert_allclose(
                expect_res.numpy(),
                actual_res.to_dense().numpy(),
                rtol=1e-05,
                equal_nan=True,
            )
            np.testing.assert_allclose(
                mask_to_zero(dense_x.grad.numpy(), mask_x),
                csr_x.grad.to_dense().numpy(),
                rtol=1e-05,
                equal_nan=True,
            )
            np.testing.assert_allclose(
                mask_to_zero(dense_y.grad.numpy(), mask_y),
                csr_y.grad.to_dense().numpy(),
                rtol=1e-05,
                equal_nan=True,
            )

    def func_test_coo(self, op):
        for sparse_dim in range(len(self.coo_shape) - 1, len(self.coo_shape)):
            for dtype in self.support_dtypes:
                x = np.vectorize(complex)(
                    np.random.randint(-255, 255, size=self.coo_shape),
                    np.random.randint(-255, 255, size=self.coo_shape),
                )
                y = np.vectorize(complex)(
                    np.random.randint(-255, 255, size=self.coo_shape),
                    np.random.randint(-255, 255, size=self.coo_shape),
                )

                dense_x = paddle.to_tensor(x, dtype=dtype, stop_gradient=False)
                dense_y = paddle.to_tensor(y, dtype=dtype, stop_gradient=False)

                s_dense_x = paddle.to_tensor(
                    x, dtype=dtype, stop_gradient=False
                )
                s_dense_y = paddle.to_tensor(
                    y, dtype=dtype, stop_gradient=False
                )
                coo_x = s_dense_x.to_sparse_coo(sparse_dim)
                coo_x.retain_grads()
                coo_y = s_dense_y.to_sparse_coo(sparse_dim)
                coo_y.retain_grads()

                actual_res = get_actual_res(coo_x, coo_y, op)
                actual_res.backward(actual_res)

                expect_res = op(dense_x, dense_y)
                expect_res.backward(expect_res)

                np.testing.assert_allclose(
                    expect_res.numpy(),
                    actual_res.to_dense().numpy(),
                    rtol=1e-05,
                    equal_nan=True,
                )
                np.testing.assert_allclose(coo_x.shape, coo_x.grad.shape)
                np.testing.assert_allclose(
                    dense_x.grad.numpy(),
                    coo_x.grad.to_dense().numpy(),
                    rtol=1e-05,
                    equal_nan=True,
                )
                np.testing.assert_allclose(coo_y.shape, coo_y.grad.shape)
                np.testing.assert_allclose(
                    dense_y.grad.numpy(),
                    coo_y.grad.to_dense().numpy(),
                    rtol=1e-05,
                    equal_nan=True,
                )

    def test_support_dtypes_csr(self):
        paddle.device.set_device('cpu')
        if paddle.device.get_device() == "cpu":
            for op in self.op_list:
                self.func_test_csr(op)

    def test_support_dtypes_coo(self):
        paddle.device.set_device('cpu')
        if paddle.device.get_device() == "cpu":
            for op in self.op_list:
                self.func_test_coo(op)

    def test_add_same_indices(self):
        indices_data = [[0, 1], [0, 3]]
        values1_data = [[(1.0 + 0.2j)], [(2.0 + 0.3j)]]
        values2_data = [[(1.0 + 0.2j)], [(2.0 - 0.3j)]]
        shape = [2, 4, 2]

        sp_a = paddle.sparse.sparse_coo_tensor(
            indices_data, values1_data, shape, stop_gradient=False
        )
        sp_a.retain_grads()

        sp_b = paddle.sparse.sparse_coo_tensor(
            indices_data, values2_data, shape, stop_gradient=False
        )
        sp_b.retain_grads()

        values1 = paddle.to_tensor(values1_data, stop_gradient=False)
        values2 = paddle.to_tensor(values2_data, stop_gradient=False)

        # c.values() = a.values() + b.values()
        sp_c = paddle.sparse.add(sp_a, sp_b)
        sp_c.backward()
        ref_c = values1 + values2
        ref_c.backward()
        np.testing.assert_allclose(sp_c.values().numpy(), ref_c.numpy())
        np.testing.assert_allclose(
            sp_a.grad.values().numpy(), values1.grad.numpy()
        )
        np.testing.assert_allclose(
            sp_b.grad.values().numpy(), values2.grad.numpy()
        )

    def test_add_bias(self):
        indices_data = [[0, 1], [0, 3]]
        values_data = [
            [(1.0 + 0.2j), (1.0 - 0.1j)],
            [(2.0 + 0.3j), (2.0 - 0.4j)],
        ]
        shape = [2, 4, 2]

        sp_a = paddle.sparse.sparse_coo_tensor(
            indices_data, values_data, shape, stop_gradient=False
        )
        sp_a.retain_grads()

        bias_values = [(1.0 + 0.1j), (2.0 - 0.1j)]

        values1 = paddle.to_tensor(values_data, stop_gradient=False)
        values2 = paddle.to_tensor(bias_values, stop_gradient=False)
        values3 = paddle.to_tensor(bias_values, stop_gradient=False)

        # c.values() = a.values() + b
        sp_c = paddle.sparse.add(sp_a, values2)
        sp_c.backward()
        ref_c = values1 + values3
        ref_c.backward()
        np.testing.assert_allclose(sp_c.values().numpy(), ref_c.numpy())
        np.testing.assert_allclose(
            sp_a.grad.values().numpy(), values1.grad.numpy()
        )
        np.testing.assert_allclose(values2.grad.numpy(), values3.grad.numpy())


class TestSparseAddStaticAPI(unittest.TestCase):
    """
    test paddle.sparse.add
    """

    def setUp(self):
        np.random.seed(2022)
        self.op_list = op_list
        self.coo_shape = [4, 8, 3, 5]
        self.support_dtypes = [
            'float32',
            'float64',
            'int32',
            'int64',
            'complex64',
            'complex128',
        ]

    def test_coo(self):
        if in_pir_mode:
            sparse_dim = len(self.coo_shape) - 1
            op = __add__
            for dtype in self.support_dtypes:
                if 'complex' in dtype:
                    x = np.vectorize(complex)(
                        np.random.randint(-255, 255, size=self.coo_shape),
                        np.random.randint(-255, 255, size=self.coo_shape),
                    ).astype(dtype)
                    y = np.vectorize(complex)(
                        np.random.randint(-255, 255, size=self.coo_shape),
                        np.random.randint(-255, 255, size=self.coo_shape),
                    ).astype(dtype)
                else:
                    x = np.random.randint(
                        -255, 255, size=self.coo_shape
                    ).astype(dtype)
                    y = np.random.randint(
                        -255, 255, size=self.coo_shape
                    ).astype(dtype)

                self.dense_x = paddle.to_tensor(
                    x, dtype=dtype, stop_gradient=True
                )
                self.dense_y = paddle.to_tensor(
                    y, dtype=dtype, stop_gradient=True
                )

                self.expect_res = op(self.dense_x, self.dense_y)

                self.x_indices_data, self.x_values_data = (
                    self.dense_x.detach().to_sparse_coo(sparse_dim).indices(),
                    self.dense_x.detach().to_sparse_coo(sparse_dim).values(),
                )

                self.y_indices_data, self.y_values_data = (
                    self.dense_y.detach().to_sparse_coo(sparse_dim).indices(),
                    self.dense_y.detach().to_sparse_coo(sparse_dim).values(),
                )
                paddle.enable_static()
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x_indices = paddle.static.data(
                        name='x_indices',
                        shape=self.x_indices_data.shape,
                        dtype=self.x_indices_data.dtype,
                    )
                    x_values = paddle.static.data(
                        name='x_values',
                        shape=self.x_values_data.shape,
                        dtype=self.x_values_data.dtype,
                    )
                    sp_x = paddle.sparse.sparse_coo_tensor(
                        x_indices,
                        x_values,
                        shape=self.dense_x.shape,
                        dtype=self.dense_x.dtype,
                    )

                    y_indices = paddle.static.data(
                        name='y_indices',
                        shape=self.y_indices_data.shape,
                        dtype=self.y_indices_data.dtype,
                    )
                    y_values = paddle.static.data(
                        name='y_values',
                        shape=self.y_values_data.shape,
                        dtype=self.y_values_data.dtype,
                    )
                    sp_y = paddle.sparse.sparse_coo_tensor(
                        y_indices,
                        y_values,
                        shape=self.dense_y.shape,
                        dtype=self.dense_y.dtype,
                    )

                    sp_out = paddle.sparse.add(sp_x, sp_y)

                    sp_dense_out = sp_out.to_dense()

                    sparse_exe = paddle.static.Executor()
                    sparse_fetch = sparse_exe.run(
                        feed={
                            'x_indices': self.x_indices_data.numpy(),
                            "x_values": self.x_values_data.numpy(),
                            'y_indices': self.y_indices_data.numpy(),
                            "y_values": self.y_values_data.numpy(),
                        },
                        fetch_list=[sp_dense_out],
                        return_numpy=True,
                    )

                    np.testing.assert_allclose(
                        self.expect_res.numpy(), sparse_fetch[0], rtol=1e-5
                    )
                paddle.disable_static()

    def test_coo_dense(self):
        # currently only support 1-D dense y input and need y.shape == x.to_dense().shape[-1]
        if in_pir_mode:
            sparse_dim = len(self.coo_shape) - 1
            op = __add__
            for dtype in self.support_dtypes:
                if 'complex' in dtype:
                    x = np.vectorize(complex)(
                        np.random.randint(-255, 255, size=self.coo_shape),
                        np.random.randint(-255, 255, size=self.coo_shape),
                    ).astype(dtype)
                    y = np.vectorize(complex)(
                        np.random.randint(-255, 255, size=self.coo_shape[-1]),
                        np.random.randint(-255, 255, size=self.coo_shape[-1]),
                    ).astype(dtype)
                else:
                    x = np.random.randint(
                        -255, 255, size=self.coo_shape
                    ).astype(dtype)
                    y = np.random.randint(
                        -255, 255, size=self.coo_shape[-1]
                    ).astype(dtype)

                self.dense_x = paddle.to_tensor(
                    x, dtype=dtype, stop_gradient=True
                )
                self.dense_y = paddle.to_tensor(
                    y, dtype=dtype, stop_gradient=True
                )

                self.expect_res = op(self.dense_x, self.dense_y)

                self.x_indices_data, self.x_values_data = (
                    self.dense_x.detach().to_sparse_coo(sparse_dim).indices(),
                    self.dense_x.detach().to_sparse_coo(sparse_dim).values(),
                )

                paddle.enable_static()
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x_indices = paddle.static.data(
                        name='x_indices',
                        shape=self.x_indices_data.shape,
                        dtype=self.x_indices_data.dtype,
                    )
                    x_values = paddle.static.data(
                        name='x_values',
                        shape=self.x_values_data.shape,
                        dtype=self.x_values_data.dtype,
                    )
                    sp_x = paddle.sparse.sparse_coo_tensor(
                        x_indices,
                        x_values,
                        shape=self.dense_x.shape,
                        dtype=self.dense_x.dtype,
                    )

                    y = paddle.static.data(
                        name='y',
                        shape=self.dense_y.shape,
                        dtype=self.dense_y.dtype,
                    )

                    sp_out = paddle.sparse.add(sp_x, y)

                    sp_dense_out = sp_out.to_dense()

                    sparse_exe = paddle.static.Executor()
                    sparse_fetch = sparse_exe.run(
                        feed={
                            'x_indices': self.x_indices_data.numpy(),
                            "x_values": self.x_values_data.numpy(),
                            'y': self.dense_y.numpy(),
                        },
                        fetch_list=[sp_dense_out],
                        return_numpy=True,
                    )

                    np.testing.assert_allclose(
                        self.expect_res.numpy(), sparse_fetch[0], rtol=1e-5
                    )
                paddle.disable_static()


class TestSparseSubStaticAPI(unittest.TestCase):
    """
    test paddle.sparse.subtract
    """

    def setUp(self):
        np.random.seed(2022)
        self.op_list = op_list
        self.coo_shape = [4, 8, 3, 5]
        self.support_dtypes = [
            'float32',
            'float64',
            'int32',
            'int64',
            'complex64',
            'complex128',
        ]

    def test_coo(self):
        if in_pir_mode():
            sparse_dim = len(self.coo_shape) - 1
            op = __sub__
            for dtype in self.support_dtypes:
                if 'complex' in dtype:
                    x = np.vectorize(complex)(
                        np.random.randint(-255, 255, size=self.coo_shape),
                        np.random.randint(-255, 255, size=self.coo_shape),
                    ).astype(dtype)
                    y = np.vectorize(complex)(
                        np.random.randint(-255, 255, size=self.coo_shape),
                        np.random.randint(-255, 255, size=self.coo_shape),
                    ).astype(dtype)
                else:
                    x = np.random.randint(
                        -255, 255, size=self.coo_shape
                    ).astype(dtype)
                    y = np.random.randint(
                        -255, 255, size=self.coo_shape
                    ).astype(dtype)

                self.dense_x = paddle.to_tensor(
                    x, dtype=dtype, stop_gradient=True
                )
                self.dense_y = paddle.to_tensor(
                    y, dtype=dtype, stop_gradient=True
                )

                self.expect_res = op(self.dense_x, self.dense_y)

                self.x_indices_data, self.x_values_data = (
                    self.dense_x.detach().to_sparse_coo(sparse_dim).indices(),
                    self.dense_x.detach().to_sparse_coo(sparse_dim).values(),
                )

                self.y_indices_data, self.y_values_data = (
                    self.dense_y.detach().to_sparse_coo(sparse_dim).indices(),
                    self.dense_y.detach().to_sparse_coo(sparse_dim).values(),
                )
                paddle.enable_static()
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x_indices = paddle.static.data(
                        name='x_indices',
                        shape=self.x_indices_data.shape,
                        dtype=self.x_indices_data.dtype,
                    )
                    x_values = paddle.static.data(
                        name='x_values',
                        shape=self.x_values_data.shape,
                        dtype=self.x_values_data.dtype,
                    )
                    sp_x = paddle.sparse.sparse_coo_tensor(
                        x_indices,
                        x_values,
                        shape=self.dense_x.shape,
                        dtype=self.dense_x.dtype,
                    )

                    y_indices = paddle.static.data(
                        name='y_indices',
                        shape=self.y_indices_data.shape,
                        dtype=self.y_indices_data.dtype,
                    )
                    y_values = paddle.static.data(
                        name='y_values',
                        shape=self.y_values_data.shape,
                        dtype=self.y_values_data.dtype,
                    )
                    sp_y = paddle.sparse.sparse_coo_tensor(
                        y_indices,
                        y_values,
                        shape=self.dense_y.shape,
                        dtype=self.dense_y.dtype,
                    )

                    sp_out = paddle.sparse.subtract(sp_x, sp_y)

                    sp_dense_out = sp_out.to_dense()

                    sparse_exe = paddle.static.Executor()
                    sparse_fetch = sparse_exe.run(
                        feed={
                            'x_indices': self.x_indices_data.numpy(),
                            "x_values": self.x_values_data.numpy(),
                            'y_indices': self.y_indices_data.numpy(),
                            "y_values": self.y_values_data.numpy(),
                        },
                        fetch_list=[sp_dense_out],
                        return_numpy=True,
                    )

                    np.testing.assert_allclose(
                        self.expect_res.numpy(), sparse_fetch[0], rtol=1e-5
                    )
                paddle.disable_static()


class TestSparseMulStaticAPI(unittest.TestCase):
    """
    test paddle.sparse.multiply
    """

    def setUp(self):
        np.random.seed(2022)
        self.op_list = op_list
        self.coo_shape = [4, 8, 3, 5]
        self.support_dtypes = [
            'float32',
            'float64',
            'int32',
            'int64',
            'complex64',
            'complex128',
        ]

    def test_coo(self):
        if in_pir_mode():
            sparse_dim = len(self.coo_shape) - 1
            op = __mul__
            for dtype in self.support_dtypes:
                if 'complex' in dtype:
                    x = np.vectorize(complex)(
                        np.random.randint(-255, 255, size=self.coo_shape),
                        np.random.randint(-255, 255, size=self.coo_shape),
                    ).astype(dtype)
                    y = np.vectorize(complex)(
                        np.random.randint(-255, 255, size=self.coo_shape),
                        np.random.randint(-255, 255, size=self.coo_shape),
                    ).astype(dtype)
                else:
                    x = np.random.randint(
                        -255, 255, size=self.coo_shape
                    ).astype(dtype)
                    y = np.random.randint(
                        -255, 255, size=self.coo_shape
                    ).astype(dtype)

                self.dense_x = paddle.to_tensor(
                    x, dtype=dtype, stop_gradient=True
                )
                self.dense_y = paddle.to_tensor(
                    y, dtype=dtype, stop_gradient=True
                )

                self.expect_res = op(self.dense_x, self.dense_y)

                self.x_indices_data, self.x_values_data = (
                    self.dense_x.detach().to_sparse_coo(sparse_dim).indices(),
                    self.dense_x.detach().to_sparse_coo(sparse_dim).values(),
                )

                self.y_indices_data, self.y_values_data = (
                    self.dense_y.detach().to_sparse_coo(sparse_dim).indices(),
                    self.dense_y.detach().to_sparse_coo(sparse_dim).values(),
                )
                paddle.enable_static()
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x_indices = paddle.static.data(
                        name='x_indices',
                        shape=self.x_indices_data.shape,
                        dtype=self.x_indices_data.dtype,
                    )
                    x_values = paddle.static.data(
                        name='x_values',
                        shape=self.x_values_data.shape,
                        dtype=self.x_values_data.dtype,
                    )
                    sp_x = paddle.sparse.sparse_coo_tensor(
                        x_indices,
                        x_values,
                        shape=self.dense_x.shape,
                        dtype=self.dense_x.dtype,
                    )

                    y_indices = paddle.static.data(
                        name='y_indices',
                        shape=self.y_indices_data.shape,
                        dtype=self.y_indices_data.dtype,
                    )
                    y_values = paddle.static.data(
                        name='y_values',
                        shape=self.y_values_data.shape,
                        dtype=self.y_values_data.dtype,
                    )
                    sp_y = paddle.sparse.sparse_coo_tensor(
                        y_indices,
                        y_values,
                        shape=self.dense_y.shape,
                        dtype=self.dense_y.dtype,
                    )

                    sp_out = paddle.sparse.multiply(sp_x, sp_y)

                    sp_dense_out = sp_out.to_dense()

                    sparse_exe = paddle.static.Executor()
                    sparse_fetch = sparse_exe.run(
                        feed={
                            'x_indices': self.x_indices_data.numpy(),
                            "x_values": self.x_values_data.numpy(),
                            'y_indices': self.y_indices_data.numpy(),
                            "y_values": self.y_values_data.numpy(),
                        },
                        fetch_list=[sp_dense_out],
                        return_numpy=True,
                    )

                    np.testing.assert_allclose(
                        self.expect_res.numpy(), sparse_fetch[0], rtol=1e-5
                    )
                paddle.disable_static()


class TestSparseDivStaticAPI(unittest.TestCase):
    """
    test paddle.sparse.divide
    """

    def setUp(self):
        np.random.seed(2022)
        self.op_list = op_list
        self.coo_shape = [4, 8, 3, 5]
        self.support_dtypes = [
            'float32',
            'float64',
            'int32',
            'int64',
            'complex64',
            'complex128',
        ]

    def test_coo(self):
        if in_pir_mode():
            sparse_dim = len(self.coo_shape) - 1
            op = __truediv__
            for dtype in self.support_dtypes:
                if 'complex' in dtype:
                    x = np.vectorize(complex)(
                        np.random.randint(-255, 255, size=self.coo_shape),
                        np.random.randint(-255, 255, size=self.coo_shape),
                    ).astype(dtype)
                    y = np.vectorize(complex)(
                        np.random.randint(-255, 255, size=self.coo_shape),
                        np.random.randint(-255, 255, size=self.coo_shape),
                    ).astype(dtype)
                else:
                    x = np.random.randint(
                        -255, 255, size=self.coo_shape
                    ).astype(dtype)
                    y = np.random.randint(
                        -255, 255, size=self.coo_shape
                    ).astype(dtype)

                self.dense_x = paddle.to_tensor(
                    x, dtype=dtype, stop_gradient=True
                )
                self.dense_y = paddle.to_tensor(
                    y, dtype=dtype, stop_gradient=True
                )

                self.expect_res = op(self.dense_x, self.dense_y)

                self.x_indices_data, self.x_values_data = (
                    self.dense_x.detach().to_sparse_coo(sparse_dim).indices(),
                    self.dense_x.detach().to_sparse_coo(sparse_dim).values(),
                )

                self.y_indices_data, self.y_values_data = (
                    self.dense_y.detach().to_sparse_coo(sparse_dim).indices(),
                    self.dense_y.detach().to_sparse_coo(sparse_dim).values(),
                )
                paddle.enable_static()
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x_indices = paddle.static.data(
                        name='x_indices',
                        shape=self.x_indices_data.shape,
                        dtype=self.x_indices_data.dtype,
                    )
                    x_values = paddle.static.data(
                        name='x_values',
                        shape=self.x_values_data.shape,
                        dtype=self.x_values_data.dtype,
                    )
                    sp_x = paddle.sparse.sparse_coo_tensor(
                        x_indices,
                        x_values,
                        shape=self.dense_x.shape,
                        dtype=self.dense_x.dtype,
                    )

                    y_indices = paddle.static.data(
                        name='y_indices',
                        shape=self.y_indices_data.shape,
                        dtype=self.y_indices_data.dtype,
                    )
                    y_values = paddle.static.data(
                        name='y_values',
                        shape=self.y_values_data.shape,
                        dtype=self.y_values_data.dtype,
                    )
                    sp_y = paddle.sparse.sparse_coo_tensor(
                        y_indices,
                        y_values,
                        shape=self.dense_y.shape,
                        dtype=self.dense_y.dtype,
                    )

                    sp_out = paddle.sparse.divide(sp_x, sp_y)

                    sp_dense_out = sp_out.to_dense()

                    sparse_exe = paddle.static.Executor()
                    sparse_fetch = sparse_exe.run(
                        feed={
                            'x_indices': self.x_indices_data.numpy(),
                            "x_values": self.x_values_data.numpy(),
                            'y_indices': self.y_indices_data.numpy(),
                            "y_values": self.y_values_data.numpy(),
                        },
                        fetch_list=[sp_dense_out],
                        return_numpy=True,
                    )

                    np.testing.assert_allclose(
                        self.expect_res.numpy(), sparse_fetch[0], rtol=1e-5
                    )
                paddle.disable_static()


if __name__ == "__main__":
    devices = []
    if paddle.device.get_device() != "cpu":
        devices.append(paddle.device.get_device())
    else:
        devices.append('cpu')
    for device in devices:
        paddle.device.set_device(device)
    unittest.main()
