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


class TestSlice(unittest.TestCase):
    """
    Test the API paddle.sparse.slice on some sparse tensors.
    x: sparse, out: sparse
    """

    def _check_result(self, np_x, axes, starts, ends):
        x_shape = np_x.shape
        dense_x = paddle.to_tensor(np_x, place=paddle.CPUPlace())
        dense_x.stop_gradient = False
        dense_out = paddle.slice(dense_x, axes, starts, ends)

        sp_x = paddle.to_tensor(np_x, place=paddle.CPUPlace()).to_sparse_coo(
            len(x_shape)
        )
        sp_x.stop_gradient = False
        sp_out = paddle.sparse.slice(sp_x, axes, starts, ends)
        np.testing.assert_allclose(
            sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-5
        )

        dense_out.backward()
        sp_out.backward()
        np.testing.assert_allclose(
            sp_x.grad.to_dense().numpy(),
            dense_x.grad.numpy() * np_x.astype('bool').astype('int'),
            rtol=1e-5,
        )

    def check_result_with_shape(self, x_shape, axes, starts, ends):
        mask = np.random.randint(0, 2, x_shape)
        np_x = np.random.randint(-100, 100, x_shape) * mask
        self._check_result(np_x, axes, starts, ends)

    def check_result_with_list(self, x, axes, starts, ends):
        np_x = np.array(x)
        self._check_result(np_x, axes, starts, ends)

    def test_coo_3d(self):
        self.check_result_with_shape([3, 4, 5], [0, 1], [1, 2], [3, 3])

    def test_coo_2d(self):
        self.check_result_with_shape([3, 4], [0], [0], [2])

    def test_coo_1d(self):
        x = [-49, 55, -5, 0, 3, 0, 0, -60, -21, 0, 0, 0]
        self.check_result_with_list(x, [0], [-3], [-1])


if __name__ == "__main__":
    unittest.main()
