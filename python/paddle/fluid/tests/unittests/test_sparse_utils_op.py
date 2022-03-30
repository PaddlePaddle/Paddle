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
import numpy as np
import paddle
from paddle import _C_ops
from paddle.fluid import core
from paddle.fluid.framework import _test_eager_guard


class TestSparseUtils(unittest.TestCase):
    def test_create_sparse_coo_tensor(self):
        with _test_eager_guard():
            non_zero_indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
            non_zero_elements = [1, 2, 3, 4, 5]
            dense_shape = [3, 4]
            dense_indices = paddle.to_tensor(non_zero_indices)
            dense_elements = paddle.to_tensor(
                non_zero_elements, dtype='float32')
            stop_gradient = False
            coo = core.eager.sparse_coo_tensor(dense_indices, dense_elements,
                                               dense_shape, stop_gradient)
            print(coo)

    def test_create_sparse_csr_tensor(self):
        with _test_eager_guard():
            non_zero_crows = [0, 2, 3, 5]
            non_zero_cols = [1, 3, 2, 0, 1]
            non_zero_elements = [1, 2, 3, 4, 5]
            dense_shape = [3, 4]
            dense_crows = paddle.to_tensor(non_zero_crows)
            dense_cols = paddle.to_tensor(non_zero_cols)
            dense_elements = paddle.to_tensor(
                non_zero_elements, dtype='float32')
            stop_gradient = False
            csr = core.eager.sparse_csr_tensor(dense_crows, dense_cols,
                                               dense_elements, dense_shape,
                                               stop_gradient)
            print(csr)

    def test_to_sparse_coo(self):
        with _test_eager_guard():
            x = [[0, 1, 0, 2], [0, 0, 3, 0], [4, 5, 0, 0]]
            non_zero_indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
            non_zero_elements = [1, 2, 3, 4, 5]
            dense_x = paddle.to_tensor(x)
            out = dense_x.to_sparse_coo(2)
            assert np.array_equal(out.non_zero_indices().numpy(),
                                  non_zero_indices)
            assert np.array_equal(out.non_zero_elements().numpy(),
                                  non_zero_elements)

            dense_tensor = out.to_dense()
            assert np.array_equal(dense_tensor.numpy(), x)

    def test_to_sparse_csr(self):
        with _test_eager_guard():
            x = [[0, 1, 0, 2], [0, 0, 3, 0], [4, 5, 0, 0]]
            non_zero_crows = [0, 2, 3, 5]
            non_zero_cols = [1, 3, 2, 0, 1]
            non_zero_elements = [1, 2, 3, 4, 5]
            dense_x = paddle.to_tensor(x)
            out = dense_x.to_sparse_csr()
            print(out)
            assert np.array_equal(out.non_zero_crows().numpy(), non_zero_crows)
            assert np.array_equal(out.non_zero_cols().numpy(), non_zero_cols)
            assert np.array_equal(out.non_zero_elements().numpy(),
                                  non_zero_elements)

            dense_tensor = out.to_dense()
            assert np.array_equal(dense_tensor.numpy(), x)


if __name__ == "__main__":
    unittest.main()
