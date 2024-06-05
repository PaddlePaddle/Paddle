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
import paddle.sparse

devices = ['cpu']


class TestSparseUnary(unittest.TestCase):
    def to_sparse(self, x, format):
        if format == 'coo':
            return x.detach().to_sparse_coo(sparse_dim=x.ndim)
        elif format == 'csr':
            return x.detach().to_sparse_csr()

    def generate_data(self, dtype, shape):
        origin_x = paddle.rand(shape, dtype)
        mask = paddle.randint(0, 2, shape).astype(dtype)

        return {'origin': origin_x, 'mask': mask}

    def check_result(
        self,
        format,
        shape,
        axis,
        device='cpu',
        dtype='float64',
    ):
        x = self.generate_data(dtype, shape[0])
        y = self.generate_data(dtype, shape[1])

        # --- check sparse coo with dense --- #
        dense_x = x['origin'] * x['mask']
        dense_x.to(device)
        sp_x = self.to_sparse(dense_x, format)
        sp_x.stop_gradient = False
        dense_x.stop_gradient = False

        dense_y = y['origin'] * y['mask']
        dense_y.to(device)
        sp_y = self.to_sparse(dense_y, format)
        sp_y.stop_gradient = False
        dense_y.stop_gradient = False

        sp_out = paddle.sparse.concat((sp_x, sp_y), axis)
        sp_out.backward()

        dense_out = paddle.concat((dense_x, dense_y), axis)
        dense_out.backward()

        # compare forward
        np.testing.assert_allclose(
            sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-05
        )

        # compare backward
        expect_grad = (dense_x.grad * x['mask']).numpy()
        np.testing.assert_allclose(
            sp_x.grad.to_dense().numpy(), expect_grad, rtol=1e-05
        )

    def compare_with_dense(self, shape, axis, format, dtype='float64'):
        for device in devices:
            # The sparse unary op is only compatible with float16 on the CUDA.
            if (device == 'cpu' and dtype != 'float16') or (
                device == 'gpu' and paddle.is_compiled_with_cuda()
            ):
                self.check_result(format, shape, axis, device, dtype)
                self.check_result(format, shape, axis, device, dtype)

    def test_sparse_concat(self):
        self.compare_with_dense([[3, 4, 5], [3, 5, 5]], 1, 'coo', 'float64')
        self.compare_with_dense([[3, 4, 5], [3, 4, 6]], 2, 'coo', 'float64')
        self.compare_with_dense([[5, 6], [6, 6]], 0, 'csr', 'float64')
        self.compare_with_dense([[6, 7], [6, 8]], 1, 'csr', 'float64')
        self.compare_with_dense([[3, 5, 6], [4, 5, 6]], 0, 'csr', 'float64')
        self.compare_with_dense([[3, 4, 5], [3, 5, 5]], 1, 'csr', 'float64')
        self.compare_with_dense([[3, 4, 5], [3, 4, 6]], 2, 'csr', 'float64')


if __name__ == "__main__":
    unittest.main()
