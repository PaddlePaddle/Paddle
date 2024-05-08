# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from utils import compare_legacy_with_pt

import paddle

devices = ['cpu']
if paddle.device.get_device() != "cpu":
    devices.append(paddle.device.get_device())


class TestSparseSum(unittest.TestCase):
    """
    Test the API paddle.sparse.sum on some sparse tensors.
    x: sparse tensor, out: sparse tensor
    """

    def to_sparse(self, x, format, sparse_dim=None):
        if format == 'coo':
            if sparse_dim:
                return x.detach().to_sparse_coo(sparse_dim=sparse_dim)
            else:
                return x.detach().to_sparse_coo(sparse_dim=x.ndim)
        elif format == 'csr':
            return x.detach().to_sparse_csr()

    def check_result(
        self, x_shape, dims, keepdim, format, sparse_dim=None, dtype=None
    ):
        for device in devices:
            paddle.device.set_device(device)
            if sparse_dim:
                mask_shape = [*x_shape[:sparse_dim]] + [1] * (
                    len(x_shape) - sparse_dim
                )
                mask = paddle.randint(0, 2, mask_shape)
            else:
                mask = paddle.randint(0, 2, x_shape)

            while paddle.sum(mask) == 0:
                if sparse_dim:
                    mask_shape = [*x_shape[:sparse_dim]] + [1] * (
                        len(x_shape) - sparse_dim
                    )
                    mask = paddle.randint(0, 2, mask_shape)
                else:
                    mask = paddle.randint(0, 2, x_shape)
            # "+ 1" to make sure that all zero elements in "origin_x" is caused by multiplying by "mask",
            # or the backward checks may fail.
            origin_x = (
                paddle.rand(x_shape, dtype='float64') + 1
            ) * mask.astype('float64')
            dense_x = origin_x.detach()
            dense_x.stop_gradient = False
            dense_out = paddle.sum(dense_x, dims, keepdim=keepdim, dtype=dtype)
            sp_x = self.to_sparse(origin_x, format, sparse_dim)
            sp_x.stop_gradient = False
            sp_out = paddle.sparse.sum(sp_x, dims, keepdim=keepdim, dtype=dtype)
            np.testing.assert_allclose(
                sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-05
            )
            dense_out.backward()
            sp_out.backward()
            np.testing.assert_allclose(
                sp_x.grad.to_dense().numpy(),
                (dense_x.grad * mask.astype(dense_x.grad.dtype)).numpy(),
                rtol=1e-05,
            )

    def test_sum_1d(self):
        self.check_result([5], None, False, 'coo')
        self.check_result([5], None, True, 'coo')
        self.check_result([5], 0, False, 'coo')
        self.check_result([5], 0, True, 'coo')

    def test_sum_2d(self):
        self.check_result([2, 5], None, False, 'coo', dtype="float32")
        self.check_result([2, 5], None, True, 'coo')
        self.check_result([2, 5], 0, True, 'coo', dtype="float32")
        self.check_result([2, 5], 0, False, 'coo')
        self.check_result([2, 5], 1, False, 'coo')
        self.check_result([2, 5], None, True, 'csr', dtype="float32")
        self.check_result([2, 5], -1, True, 'csr', dtype="float32")
        self.check_result([2, 5], 0, False, 'coo')
        self.check_result([2, 5], -1, True, 'csr')

    def test_sum_3d(self):
        self.check_result([6, 2, 3], -1, True, 'csr')
        for i in [0, 1, -2, None]:
            self.check_result([6, 2, 3], i, False, 'coo')
            self.check_result([6, 2, 3], i, True, 'coo')

    def test_sum_nd(self):
        for i in range(6):
            self.check_result([8, 3, 4, 4, 5, 3], i, False, 'coo')
            self.check_result([8, 3, 4, 4, 5, 3], i, True, 'coo')
            # Randint now only supports access to dimension 0 to 9.
            self.check_result([2, 3, 4, 2, 3, 4, 2, 3, 4], i, False, 'coo')

    def test_sum_sparse_dim(self):
        for i in range(6):
            self.check_result([8, 3, 4, 4, 5, 3], i, False, 'coo', sparse_dim=3)
            self.check_result([8, 3, 4, 4, 5, 3], i, True, 'coo', sparse_dim=3)


class TestSparseSumStatic(unittest.TestCase):
    def check_result_coo(self, x_shape, dims, keepdim, dtype=None):
        for device in devices:
            paddle.device.set_device(device)
            mask = paddle.randint(0, 2, x_shape)
            while paddle.sum(mask) == 0:
                mask = paddle.randint(0, 2, x_shape)
            origin_data = (
                paddle.rand(x_shape, dtype='float32') + 1
            ) * mask.astype('float32')
            sparse_data = origin_data.detach().to_sparse_coo(
                sparse_dim=len(x_shape)
            )
            indices_data = sparse_data.indices()
            values_data = sparse_data.values()

            dense_x = origin_data
            dense_out = paddle.sum(dense_x, dims, keepdim=keepdim, dtype=dtype)

            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                indices = paddle.static.data(
                    name='indices',
                    shape=indices_data.shape,
                    dtype=indices_data.dtype,
                )
                values = paddle.static.data(
                    name='values',
                    shape=values_data.shape,
                    dtype=values_data.dtype,
                )
                sp_x = paddle.sparse.sparse_coo_tensor(
                    indices,
                    values,
                    shape=origin_data.shape,
                    dtype=origin_data.dtype,
                )
                sp_out = paddle.sparse.sum(
                    sp_x, dims, keepdim=keepdim, dtype=dtype
                )
                sp_dense_out = sp_out.to_dense()

                sparse_exe = paddle.static.Executor()
                sparse_fetch = sparse_exe.run(
                    feed={
                        'indices': indices_data.numpy(),
                        "values": values_data.numpy(),
                    },
                    fetch_list=[sp_dense_out],
                    return_numpy=True,
                )

                np.testing.assert_allclose(
                    dense_out.numpy(), sparse_fetch[0], rtol=1e-5
                )
            paddle.disable_static()

    @compare_legacy_with_pt
    def test_sum(self):
        # 1d
        self.check_result_coo([5], None, False)
        self.check_result_coo([5], None, True)
        self.check_result_coo([5], 0, True)
        self.check_result_coo([5], 0, False)

        # 2d
        self.check_result_coo([2, 5], None, False, dtype="float32")
        self.check_result_coo([2, 5], None, True)
        self.check_result_coo([2, 5], 0, True, dtype="float32")
        self.check_result_coo([2, 5], 0, False)
        self.check_result_coo([2, 5], 1, False)
        self.check_result_coo([2, 5], 0, False)

        # 3d
        for i in [0, 1, -2, None]:
            self.check_result_coo([6, 2, 3], i, False)
            self.check_result_coo([6, 2, 3], i, True)

        # nd
        for i in range(6):
            self.check_result_coo([8, 3, 4, 4, 5, 3], i, False)
            self.check_result_coo([8, 3, 4, 4, 5, 3], i, True)
            # Randint now only supports access to dimension 0 to 9.
            self.check_result_coo([2, 3, 4, 2, 3, 4, 2, 3, 4], i, False)


if __name__ == "__main__":
    unittest.main()
