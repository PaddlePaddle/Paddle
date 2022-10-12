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

import paddle
import numpy as np
import unittest


class TestReshape(unittest.TestCase):
    """
    Test the API paddle.incubate.sparse.reshape on some sparse tensors.
    x: sparse, out: sparse
    """

    def check_result(self, x_shape, new_shape, format):
        """
        x_shape: original shape
        new_shape: new shape
        format: "coo" or "csr"
        Transform a sparse tensor with shape "x_shape" to
        a sparse tensor with shape "new_shape".
        Compare the output of paddle.reshape and the output of
        paddle.incubate.sparse.reshape.
        """
        mask = np.random.randint(0, 2, x_shape)
        np_x = np.random.randint(-100, 100, x_shape) * mask

        dense_x = paddle.to_tensor(np_x)
        dense_x.stop_gradient = False
        dense_out = paddle.reshape(dense_x, new_shape)

        if format == "coo":
            sp_x = paddle.to_tensor(np_x).to_sparse_coo(len(x_shape))
        else:
            sp_x = paddle.to_tensor(np_x).to_sparse_csr()
        sp_x.stop_gradient = False
        sp_out = paddle.incubate.sparse.reshape(sp_x, new_shape)

        np.testing.assert_allclose(sp_out.to_dense().numpy(),
                                   dense_out.numpy(),
                                   rtol=1e-05)

        dense_out.backward()
        sp_out.backward()
        np.testing.assert_allclose(sp_x.grad.to_dense().numpy(),
                                   dense_x.grad.numpy() *
                                   np_x.astype('bool').astype('int'),
                                   rtol=1e-05)

    def test_reshape_2d(self):
        self.check_result([2, 5], [
            10,
        ], 'coo')
        self.check_result([12, 5], [15, 4], 'coo')

        self.check_result([10, 5], [2, 25], 'csr')
        self.check_result([9, 8], [18, 4], 'csr')

    def test_reshape_3d(self):
        self.check_result([6, 2, 3], [6, 2, 3], 'coo')
        self.check_result([6, 2, 3], [2, 3, 3, 2], 'coo')
        self.check_result([6, 2, 3], [1, 18, 2], 'coo')
        self.check_result([6, 2, 3], [2, 9, 2], 'coo')
        self.check_result([6, 2, 3], [2, 1, 18], 'coo')
        self.check_result([6, 2, 3], [1, 2, 2, 3, 3], 'coo')

        self.check_result([6, 2, 3], [6, 2, 3], 'csr')
        self.check_result([6, 2, 3], [6, 3, 2], 'csr')
        self.check_result([6, 2, 3], [2, 6, 3], 'csr')
        self.check_result([6, 2, 3], [3, 6, 2], 'csr')
        self.check_result([6, 2, 3], [4, 9, 1], 'csr')
        self.check_result([6, 2, 3], [12, 1, 3], 'csr')

    def test_reshape_nd(self):
        self.check_result([8, 3, 4, 4, 5, 3], [24, 8, 10, 3], 'coo')
        self.check_result([3, 4, 4, 5, 7], [1, 12, 2, 5, 14], 'coo')

    def test_reshape_with_zero_or_minus_one_in_new_shape(self):
        self.check_result([6, 2, 3], [-1, 0, 3], 'coo')
        self.check_result([6, 2, 3], [2, 3, 0, -1], 'coo')
        self.check_result([6, 2, 3], [1, -1, 2], 'coo')
        self.check_result([6, 2, 3], [-1, 9, 2], 'coo')
        self.check_result([6, 2, 3], [2, -1, 18], 'coo')
        self.check_result([6, 2, 3], [1, 0, 2, -1, 3], 'coo')

        self.check_result([6, 2, 3], [0, 0, -1], 'csr')
        self.check_result([6, 2, 3], [-1, 3, 2], 'csr')
        self.check_result([6, 2, 3], [2, -1, 0], 'csr')
        self.check_result([6, 2, 3], [-1, 6, 2], 'csr')
        self.check_result([6, 2, 3], [-1, 9, 1], 'csr')
        self.check_result([6, 2, 3], [-1, 1, 3], 'csr')


if __name__ == "__main__":
    unittest.main()
