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
        self.shape = [128, 256]
        self.support_dtypes = ['float32', 'float64']

    def func_support_dtypes_csr(self):
        for op in self.op_list:
            for dtype in self.support_dtypes:
                x = np.random.randint(-255, 255, size=self.shape).astype(dtype)
                y = np.random.randint(-255, 255, size=self.shape).astype(dtype)
                dense_x = paddle.to_tensor(x).astype(dtype)
                dense_y = paddle.to_tensor(y).astype(dtype)
                csr_x = dense_x.to_sparse_csr()
                csr_y = dense_y.to_sparse_csr()

                actual_res = get_actual_res(csr_x, csr_y, op)
                expect_res = op(dense_x, dense_y)

                self.assertTrue(
                    np.allclose(expect_res.numpy(),
                                actual_res.to_dense().numpy()))

    def func_support_dtypes_coo(self):
        for op in self.op_list:
            for dtype in self.support_dtypes:
                x = np.random.randint(-255, 255, size=self.shape).astype(dtype)
                y = np.random.randint(-255, 255, size=self.shape).astype(dtype)
                dense_x = paddle.to_tensor(x).astype(dtype)
                dense_y = paddle.to_tensor(y).astype(dtype)
                coo_x = dense_x.to_sparse_coo(2)
                coo_y = dense_y.to_sparse_coo(2)

                actual_res = get_actual_res(coo_x, coo_y, op)
                expect_res = op(dense_x, dense_y)

                self.assertTrue(
                    np.allclose(expect_res.numpy(),
                                actual_res.to_dense().numpy()))

    def test_support_dtypes_csr(self):
        if paddle.device.get_device() == "cpu":
            with _test_eager_guard():
                self.func_support_dtypes_csr()

    def test_support_dtypes_coo(self):
        if paddle.device.get_device() == "cpu":
            with _test_eager_guard():
                self.func_support_dtypes_coo()


if __name__ == "__main__":
    unittest.main()
