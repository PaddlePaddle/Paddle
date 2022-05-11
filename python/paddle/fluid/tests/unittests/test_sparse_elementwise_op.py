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

from __future__ import print_function
import unittest
from operator import __add__, __sub__, __mul__, __truediv__

import numpy as np
import paddle
from paddle.fluid.framework import _test_eager_guard


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


class TestSparseElementWiseAPI(unittest.TestCase):
    """
    test paddle.sparse.add, subtract, multiply, divide
    """

    def setUp(self):
        np.random.seed(2022)
        self.op_list = [__add__, __sub__, __mul__, __truediv__]
        self.csr_shape = [128, 256]
        self.coo_shape = [4, 8, 3, 5]
        self.support_dtypes = ['float32','float64','int32', 'int64']

    def func_test_csr(self, op):
        for dtype in self.support_dtypes:
            x = np.random.randint(-255, 255, size=self.csr_shape).astype(dtype)
            y = np.random.randint(-255, 255, size=self.csr_shape).astype(dtype)
            dense_x = paddle.to_tensor(x).astype(dtype)
            dense_y = paddle.to_tensor(y).astype(dtype)
            csr_x = dense_x.to_sparse_csr()
            csr_y = dense_y.to_sparse_csr()

            actual_res = get_actual_res(csr_x, csr_y, op)
            expect_res = op(dense_x, dense_y)

            self.assertTrue(
                np.allclose(
                    expect_res.numpy(),
                    actual_res.to_dense().numpy(),
                    equal_nan=True))

    def func_test_coo(self, op):
        for sparse_dim in range(2, len(self.coo_shape) + 1):
            for dtype in self.support_dtypes:
                x = np.random.randint(
                    -255, 255, size=self.coo_shape).astype(dtype)
                y = np.random.randint(
                    -255, 255, size=self.coo_shape).astype(dtype)

                dense_x = paddle.to_tensor(x, dtype=dtype, stop_gradient=False)
                dense_y = paddle.to_tensor(y, dtype=dtype, stop_gradient=False)

                s_dense_x = paddle.to_tensor(x, dtype=dtype, stop_gradient=False)
                s_dense_y = paddle.to_tensor(y, dtype=dtype, stop_gradient=False)
                coo_x = s_dense_x.to_sparse_coo(sparse_dim)
                coo_y = s_dense_y.to_sparse_coo(sparse_dim)

                actual_res = get_actual_res(coo_x, coo_y, op)
                actual_res.backward(actual_res)

                expect_res = op(dense_x, dense_y)
                expect_res.backward(expect_res)

                self.assertTrue(
                    np.allclose(
                        expect_res.numpy(),
                        actual_res.to_dense().numpy(),
                        equal_nan=True))
                self.assertTrue(
                    np.allclose(
                        dense_x.grad.numpy(),
                        coo_x.grad.to_dense().numpy(),
                        equal_nan=True))
                self.assertTrue(
                    np.allclose(
                        dense_y.grad.numpy(),
                        coo_y.grad.to_dense().numpy(),
                        equal_nan=True))

    def test_coo_add(self):
        if paddle.device.get_device() == "cpu":
            with _test_eager_guard():
                self.func_test_coo(__add__)

    def test_coo_sub(self):
        if paddle.device.get_device() == "cpu":
            with _test_eager_guard():
                self.func_test_coo(__sub__)

    def test_coo_mul(self):
        if paddle.device.get_device() == "cpu":
            with _test_eager_guard():
                self.func_test_coo(__mul__)

    def test_coo_div(self):
        if paddle.device.get_device() == "cpu":
            with _test_eager_guard():
                self.func_test_coo(__truediv__)

    def test_csr_add(self):
        if paddle.device.get_device() == "cpu":
            with _test_eager_guard():
                self.func_test_csr(__add__)

    def test_csr_sub(self):
        if paddle.device.get_device() == "cpu":
            with _test_eager_guard():
                self.func_test_csr(__sub__)

    def test_csr_mul(self):
        if paddle.device.get_device() == "cpu":
            with _test_eager_guard():
                self.func_test_csr(__mul__)

    def test_csr_div(self):
        if paddle.device.get_device() == "cpu":
            with _test_eager_guard():
                self.func_test_csr(__truediv__)


if __name__ == "__main__":
    unittest.main()
