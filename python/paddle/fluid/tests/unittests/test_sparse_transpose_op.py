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


class TestTranspose(unittest.TestCase):
    # x: sparse, out: sparse
    def check_result(self, x_shape, dims, format):
        mask = paddle.randint(0, 2, x_shape).astype("float32")
        # "+ 1" to make sure that all zero elements in "origin_x" is caused by multiplying by "mask",
        # or the backward checks may fail.
        origin_x = (paddle.rand(x_shape, dtype='float32') + 1) * mask
        dense_x = origin_x.detach()
        dense_x.stop_gradient = False
        dense_out = paddle.transpose(dense_x, dims)

        if format == "coo":
            sp_x = origin_x.detach().to_sparse_coo(len(x_shape))
        else:
            sp_x = origin_x.detach().to_sparse_csr()
        sp_x.stop_gradient = False
        sp_out = paddle.sparse.transpose(sp_x, dims)

        np.testing.assert_allclose(
            sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-05
        )
        dense_out.backward()
        sp_out.backward()
        np.testing.assert_allclose(
            sp_x.grad.to_dense().numpy(),
            (dense_x.grad * mask).numpy(),
            rtol=1e-05,
        )

    def test_transpose_2d(self):
        self.check_result([2, 5], [0, 1], 'coo')
        self.check_result([2, 5], [0, 1], 'csr')
        self.check_result([2, 5], [1, 0], 'coo')
        self.check_result([2, 5], [1, 0], 'csr')

    def test_transpose_3d(self):
        self.check_result([6, 2, 3], [0, 1, 2], 'coo')
        self.check_result([6, 2, 3], [0, 1, 2], 'csr')
        self.check_result([6, 2, 3], [0, 2, 1], 'coo')
        self.check_result([6, 2, 3], [0, 2, 1], 'csr')
        self.check_result([6, 2, 3], [1, 0, 2], 'coo')
        self.check_result([6, 2, 3], [1, 0, 2], 'csr')
        self.check_result([6, 2, 3], [2, 0, 1], 'coo')
        self.check_result([6, 2, 3], [2, 0, 1], 'csr')
        self.check_result([6, 2, 3], [2, 1, 0], 'coo')
        self.check_result([6, 2, 3], [2, 1, 0], 'csr')
        self.check_result([6, 2, 3], [1, 2, 0], 'coo')
        self.check_result([6, 2, 3], [1, 2, 0], 'csr')

    def test_transpose_nd(self):
        self.check_result([8, 3, 4, 4, 5, 3], [5, 3, 4, 1, 0, 2], 'coo')
        # Randint now only supports access to dimension 0 to 9.
        self.check_result(
            [2, 3, 4, 2, 3, 4, 2, 3, 4], [2, 3, 4, 5, 6, 7, 8, 0, 1], 'coo'
        )


if __name__ == "__main__":
    unittest.main()
