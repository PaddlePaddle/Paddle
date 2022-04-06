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
import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard


class TestSparseCreate(unittest.TestCase):
    def test_create_coo_by_tensor(self):
        with _test_eager_guard():
            non_zero_indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
            non_zero_elements = [1, 2, 3, 4, 5]
            dense_shape = [3, 4]
            dense_indices = paddle.to_tensor(non_zero_indices)
            dense_elements = paddle.to_tensor(
                non_zero_elements, dtype='float32')
            coo = paddle.sparse.sparse_coo_tensor(
                dense_indices, dense_elements, dense_shape, stop_gradient=False)
            assert np.array_equal(non_zero_indices,
                                  coo.non_zero_indices().numpy())
            assert np.array_equal(non_zero_elements,
                                  coo.non_zero_elements().numpy())

    def test_create_coo_by_np(self):
        with _test_eager_guard():
            indices = [[0, 1, 2], [1, 2, 0]]
            values = [1.0, 2.0, 3.0]
            dense_shape = [2, 3]
            coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
            print(coo)
            assert np.array_equal(indices, coo.non_zero_indices().numpy())
            assert np.array_equal(values, coo.non_zero_elements().numpy())

    def test_create_csr_by_tensor(self):
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
            csr = paddle.sparse.sparse_csr_tensor(
                dense_crows,
                dense_cols,
                dense_elements,
                dense_shape,
                stop_gradient=stop_gradient)
            print(csr)

    def test_create_csr_by_np(self):
        with _test_eager_guard():
            crows = [0, 2, 3, 5]
            cols = [1, 3, 2, 0, 1]
            values = [1, 2, 3, 4, 5]
            dense_shape = [3, 4]
            csr = paddle.sparse.sparse_csr_tensor(crows, cols, values,
                                                  dense_shape)
            assert np.array_equal(crows, csr.non_zero_crows().numpy())
            assert np.array_equal(cols, csr.non_zero_cols().numpy())
            assert np.array_equal(values, csr.non_zero_elements().numpy())

    def test_place(self):
        with _test_eager_guard():
            place = core.CPUPlace()
            indices = [[0, 1], [0, 1]]
            values = [1.0, 2.0]
            dense_shape = [2, 2]
            coo = paddle.sparse.sparse_coo_tensor(
                indices, values, dense_shape, place=place)
            assert coo.place.is_cpu_place()
            assert coo.non_zero_elements().place.is_cpu_place()
            assert coo.non_zero_indices().place.is_cpu_place()

            crows = [0, 2, 3, 5]
            cols = [1, 3, 2, 0, 1]
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            csr = paddle.sparse.sparse_csr_tensor(
                crows, cols, values, [3, 5], place=place)
            assert csr.place.is_cpu_place()
            assert csr.non_zero_crows().place.is_cpu_place()
            assert csr.non_zero_cols().place.is_cpu_place()
            assert csr.non_zero_elements().place.is_cpu_place()

    def test_dtype(self):
        with _test_eager_guard():
            indices = [[0, 1], [0, 1]]
            values = [1.0, 2.0]
            dense_shape = [2, 2]
            indices = paddle.to_tensor(indices, dtype='int32')
            values = paddle.to_tensor(values, dtype='float32')
            coo = paddle.sparse.sparse_coo_tensor(
                indices, values, dense_shape, dtype='float64')
            assert coo.dtype == paddle.float64

            crows = [0, 2, 3, 5]
            cols = [1, 3, 2, 0, 1]
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            csr = paddle.sparse.sparse_csr_tensor(
                crows, cols, values, [3, 5], dtype='float16')
            assert csr.dtype == paddle.float16

    def test_create_coo_no_shape(self):
        with _test_eager_guard():
            indices = [[0, 1], [0, 1]]
            values = [1.0, 2.0]
            indices = paddle.to_tensor(indices, dtype='int32')
            values = paddle.to_tensor(values, dtype='float32')
            coo = paddle.sparse.sparse_coo_tensor(indices, values)
            assert [2, 2] == coo.shape


class TestSparseConvert(unittest.TestCase):
    def test_to_sparse_coo(self):
        with _test_eager_guard():
            x = [[0, 1, 0, 2], [0, 0, 3, 0], [4, 5, 0, 0]]
            non_zero_indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
            non_zero_elements = [1, 2, 3, 4, 5]
            dense_x = paddle.to_tensor(x)
            out = dense_x.to_sparse_coo(2)
            print(out)
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
            print(dense_tensor)
            assert np.array_equal(dense_tensor.numpy(), x)


if __name__ == "__main__":
    unittest.main()
