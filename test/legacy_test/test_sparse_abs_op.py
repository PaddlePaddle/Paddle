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


class TestSparseUnary(unittest.TestCase):
    def to_sparse(self, x, format):
        if format == 'coo':
            return x.detach().to_sparse_coo(sparse_dim=x.ndim)
        elif format == 'csr':
            return x.detach().to_sparse_csr()

    def check_result(self, dense_func, sparse_func, format, dtype):
        if dtype == 'float32' or dtype == 'float64':
            origin_x = paddle.rand([8, 16, 32], dtype)
            mask = paddle.randint(0, 2, [8, 16, 32]).astype(dtype)
            while paddle.sum(mask) == 0:
                mask = paddle.randint(0, 2, [8, 16, 32]).astype(dtype)
        if dtype == 'complex64':
            origin_x_real = paddle.rand([8, 16, 32], 'float32')
            origin_x_com = paddle.rand([8, 16, 32], 'float32')
            origin_x = (origin_x_real + 1j * origin_x_com).astype('complex64')
            mask = paddle.randint(0, 2, [8, 16, 32]).astype("float32")
            while paddle.sum(mask) == 0:
                mask = paddle.randint(0, 2, [8, 16, 32]).astype("float32")
        if dtype == 'complex128':
            origin_x_real = paddle.rand([8, 16, 32], 'float64')
            origin_x_com = paddle.rand([8, 16, 32], 'float64')
            origin_x = (origin_x_real + 1j * origin_x_com).astype('complex128')
            mask = paddle.randint(0, 2, [8, 16, 32]).astype("float64")
            while paddle.sum(mask) == 0:
                mask = paddle.randint(0, 2, [8, 16, 32]).astype("float64")

        # --- check sparse coo with dense --- #
        dense_x = origin_x * mask

        sp_x = self.to_sparse(dense_x, format)

        sp_x.stop_gradient = False
        sp_out = sparse_func(sp_x)
        sp_out.backward()

        dense_x.stop_gradient = False
        dense_out = dense_func(dense_x)
        dense_out.backward()

        # compare forward
        np.testing.assert_allclose(
            sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-05
        )

        # compare backward
        if dense_func == paddle.sqrt:
            expect_grad = np.nan_to_num(dense_x.grad.numpy(), 0.0, 0.0, 0.0)
        else:
            expect_grad = (dense_x.grad * mask).numpy()
        np.testing.assert_allclose(
            sp_x.grad.to_dense().numpy(), expect_grad, rtol=1e-05
        )

    def compare_with_dense(self, dense_func, sparse_func, dtype):
        self.check_result(dense_func, sparse_func, 'coo', dtype)
        self.check_result(dense_func, sparse_func, 'csr', dtype)

    def test_sparse_abs(self):
        self.compare_with_dense(paddle.abs, paddle.sparse.abs, 'float32')
        self.compare_with_dense(paddle.abs, paddle.sparse.abs, 'float64')
        self.compare_with_dense(paddle.abs, paddle.sparse.abs, 'complex64')
        self.compare_with_dense(paddle.abs, paddle.sparse.abs, 'complex128')


if __name__ == "__main__":
    unittest.main()
