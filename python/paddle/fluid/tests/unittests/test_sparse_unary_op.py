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
from paddle.fluid.framework import convert_np_dtype_to_dtype_


class TestSparseUnary(unittest.TestCase):

    def to_sparse(self, x, format):
        if format == 'coo':
            return x.detach().to_sparse_coo(sparse_dim=x.ndim)
        elif format == 'csr':
            return x.detach().to_sparse_csr()

    def check_result(self, dense_func, sparse_func, format, *args):
        origin_x = paddle.rand([8, 16, 32], dtype='float32')
        mask = paddle.randint(0, 2, [8, 16, 32]).astype('float32')

        ### check sparse coo with dense ###
        dense_x = origin_x * mask
        sp_x = self.to_sparse(dense_x, format)

        sp_x.stop_gradient = False
        if len(args) == 0:
            sp_out = sparse_func(sp_x)
        elif len(args) == 1:
            sp_out = sparse_func(sp_x, args[0])
        elif len(args) == 2:
            sp_out = sparse_func(sp_x, args[0], args[1])
        sp_out.backward()

        dense_x.stop_gradient = False
        if len(args) == 0:
            dense_out = dense_func(dense_x)
        elif len(args) == 1:
            dense_out = dense_func(dense_x, args[0])
        elif len(args) == 2:
            if dense_func == paddle.cast:
                dense_out = dense_func(dense_x, args[1])

                int_dtype = convert_np_dtype_to_dtype_(args[0])
                if sp_out.is_sparse_csr():
                    self.assertEqual(sp_out.crows().dtype, int_dtype)
                    self.assertEqual(sp_out.cols().dtype, int_dtype)
                elif sp_out.is_sparse_coo():
                    self.assertEqual(sp_out.indices().dtype, int_dtype)
            else:
                dense_out = dense_func(dense_x, args[0], args[1])
        dense_out.backward()

        # compare forward
        self.assertTrue(
            np.allclose(sp_out.to_dense().numpy(), dense_out.numpy()))

        # compare backward
        if dense_func == paddle.sqrt:
            expect_grad = np.nan_to_num(dense_x.grad.numpy(), 0., 0., 0.)
        else:
            expect_grad = (dense_x.grad * mask).numpy()
        self.assertTrue(np.allclose(sp_x.grad.to_dense().numpy(), expect_grad))

    def compare_with_dense(self, dense_func, sparse_func):
        self.check_result(dense_func, sparse_func, 'coo')
        self.check_result(dense_func, sparse_func, 'csr')

    def compare_with_dense_one_attr(self, dense_func, sparse_func, attr1):
        self.check_result(dense_func, sparse_func, 'coo', attr1)
        self.check_result(dense_func, sparse_func, 'csr', attr1)

    def compare_with_dense_two_attr(self, dense_func, sparse_func, attr1,
                                    attr2):
        self.check_result(dense_func, sparse_func, 'coo', attr1, attr2)
        self.check_result(dense_func, sparse_func, 'csr', attr1, attr2)

    def test_sparse_sin(self):
        self.compare_with_dense(paddle.sin, paddle.incubate.sparse.sin)

    def test_sparse_tan(self):
        self.compare_with_dense(paddle.tan, paddle.incubate.sparse.tan)

    def test_sparse_asin(self):
        self.compare_with_dense(paddle.asin, paddle.incubate.sparse.asin)

    def test_sparse_atan(self):
        self.compare_with_dense(paddle.atan, paddle.incubate.sparse.atan)

    def test_sparse_sinh(self):
        self.compare_with_dense(paddle.sinh, paddle.incubate.sparse.sinh)

    def test_sparse_tanh(self):
        self.compare_with_dense(paddle.tanh, paddle.incubate.sparse.tanh)

    def test_sparse_asinh(self):
        self.compare_with_dense(paddle.asinh, paddle.incubate.sparse.asinh)

    def test_sparse_atanh(self):
        self.compare_with_dense(paddle.atanh, paddle.incubate.sparse.atanh)

    def test_sparse_sqrt(self):
        self.compare_with_dense(paddle.sqrt, paddle.incubate.sparse.sqrt)

    def test_sparse_square(self):
        self.compare_with_dense(paddle.square, paddle.incubate.sparse.square)

    def test_sparse_log1p(self):
        self.compare_with_dense(paddle.log1p, paddle.incubate.sparse.log1p)

    def test_sparse_relu(self):
        self.compare_with_dense(paddle.nn.ReLU(),
                                paddle.incubate.sparse.nn.ReLU())

    def test_sparse_abs(self):
        self.compare_with_dense(paddle.abs, paddle.incubate.sparse.abs)

    def test_sparse_neg(self):
        self.compare_with_dense(paddle.neg, paddle.incubate.sparse.neg)

    def test_sparse_pow(self):
        self.compare_with_dense_one_attr(paddle.pow, paddle.incubate.sparse.pow,
                                         3)

    def test_sparse_mul_scalar(self):
        self.compare_with_dense_one_attr(paddle.Tensor.__mul__,
                                         paddle.incubate.sparse.multiply, 3)

    def test_sparse_div_scalar(self):
        self.compare_with_dense_one_attr(paddle.Tensor.__div__,
                                         paddle.incubate.sparse.divide, 2)

    def test_sparse_cast(self):
        self.compare_with_dense_two_attr(paddle.cast,
                                         paddle.incubate.sparse.cast, 'int16',
                                         'float32')
        self.compare_with_dense_two_attr(paddle.cast,
                                         paddle.incubate.sparse.cast, 'int32',
                                         'float64')


if __name__ == "__main__":
    unittest.main()
