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

devices = ['cpu', 'gpu']


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
        dtype='float64',
    ):
        x = self.generate_data(dtype, shape[0])
        y = self.generate_data(dtype, shape[1])
        z = self.generate_data(dtype, shape[2])

        # --- check sparse coo with dense --- #
        dense_x = x['origin'] * x['mask']

        sp_x = self.to_sparse(dense_x, format)
        sp_x.stop_gradient = False
        dense_x.stop_gradient = False

        dense_y = y['origin'] * y['mask']

        sp_y = self.to_sparse(dense_y, format)
        sp_y.stop_gradient = False
        dense_y.stop_gradient = False

        dense_z = z['origin'] * z['mask']

        sp_z = self.to_sparse(dense_z, format)
        sp_z.stop_gradient = False
        dense_z.stop_gradient = False

        sp_out = paddle.sparse.concat((sp_x, sp_y, sp_z), axis)
        sp_out.backward()

        dense_out = paddle.concat((dense_x, dense_y, dense_z), axis)
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
        # compare backward
        expect_grad_y = (dense_y.grad * y['mask']).numpy()
        np.testing.assert_allclose(
            sp_y.grad.to_dense().numpy(), expect_grad_y, rtol=1e-05
        )

    def compare_with_dense(self, shape, axis, format, dtype='float64'):
        self.check_result(format, shape, axis, dtype)

    def test_sparse_concat(self):
        for device in devices:
            if device == 'cpu' or (
                device == 'gpu' and paddle.is_compiled_with_cuda()
            ):
                paddle.device.set_device(device)
                self.compare_with_dense(
                    [[4, 8, 16], [4, 5, 16], [4, 12, 16]], 1, 'coo', 'float64'
                )
                self.compare_with_dense(
                    [[4, 8, 15], [4, 8, 16], [4, 8, 8]], 2, 'coo', 'float64'
                )
                self.compare_with_dense(
                    [[8, 16], [16, 16], [24, 16]], 0, 'csr', 'float64'
                )
                self.compare_with_dense(
                    [[8, 8], [8, 16], [8, 8]], 1, 'csr', 'float64'
                )
                self.compare_with_dense(
                    [[4, 8, 16], [8, 8, 16], [16, 8, 16]], 0, 'csr', 'float64'
                )
                self.compare_with_dense(
                    [[4, 16, 8], [4, 8, 8], [4, 24, 8]], 1, 'csr', 'float64'
                )
                self.compare_with_dense(
                    [[4, 8, 16], [4, 8, 24], [4, 8, 8]], 2, 'csr', 'float64'
                )


if __name__ == "__main__":
    unittest.main()
