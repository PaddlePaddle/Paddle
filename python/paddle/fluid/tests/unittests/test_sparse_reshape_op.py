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
from paddle.fluid.framework import _test_eager_guard


class TestReshape(unittest.TestCase):
    # x: sparse, out: sparse
    def check_result(self, x_shape, new_shape, format):
        with _test_eager_guard():
            # mask = paddle.randint(0, 2, x_shape).astype("float32")
            # origin_x = paddle.rand(x_shape, dtype='float32') * mask
            mask = paddle.randint(0, 2, x_shape)
            origin_x = paddle.randint(-100, 100, x_shape) * mask

            dense_x = origin_x.detach()
            dense_x.stop_gradient = False
            # dense_out = paddle.transpose(dense_x, dims)
            dense_out = paddle.reshape(dense_x, new_shape)

            if format == "coo":
                sp_x = origin_x.detach().to_sparse_coo(len(x_shape))
            else:
                sp_x = origin_x.detach().to_sparse_csr()
            sp_x.stop_gradient = False
            # sp_out = paddle.incubate.sparse.transpose(sp_x, dims)
            sp_out = paddle.incubate.sparse.reshape(sp_x, new_shape)

            np.testing.assert_allclose(sp_out.to_dense().numpy(),
                                       dense_out.numpy(),
                                       rtol=1e-05)
            dense_out.backward()
            sp_out.backward()
            np.testing.assert_allclose(sp_x.grad.to_dense().numpy(),
                                    #    (dense_x.grad * mask).numpy(),
                                       dense_x.grad.numpy(),
                                       rtol=1e-05)

    def test_reshape_2d(self):
        self.check_result([2, 5], [10,], 'coo')
    #     self.check_result([10, 5], [2, 25], 'csr')
    #     self.check_result([12, 5], [15, 4], 'coo')
    #     self.check_result([9, 8], [18, 4], 'csr')

    # def test_transpose_3d(self):
    #     self.check_result([6, 2, 3], [6, 2, 3], 'coo')
    #     self.check_result([6, 2, 3], [6, 2, 3], 'csr')
    #     self.check_result([6, 2, 3], [2, 3, 3, 2], 'coo')
    #     self.check_result([6, 2, 3], [6, 3, 2], 'csr')
    #     # self.check_result([6, 2, 3], [1, 0, 2], 'coo')
    #     # self.check_result([6, 2, 3], [1, 0, 2], 'csr')
    #     # self.check_result([6, 2, 3], [2, 0, 1], 'coo')
    #     # self.check_result([6, 2, 3], [2, 0, 1], 'csr')
    #     # self.check_result([6, 2, 3], [2, 1, 0], 'coo')
    #     # self.check_result([6, 2, 3], [2, 1, 0], 'csr')
    #     # self.check_result([6, 2, 3], [1, 2, 0], 'coo')
    #     # self.check_result([6, 2, 3], [1, 2, 0], 'csr')

    # def test_transpose_nd(self):
    #     self.check_result([8, 3, 4, 4, 5, 3], [24, 8, 10, 3], 'coo')
    #     # Randint now only supports access to dimension 0 to 9.
    #     # self.check_result([i % 3 + 2 for i in range(9)],
    #     #                   [(i + 2) % 9 for i in range(9)], 'coo')


if __name__ == "__main__":
    unittest.main()
