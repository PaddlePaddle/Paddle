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

import paddle


class TestSparseCreate(unittest.TestCase):
    def test_create_coo_by_tensor(self):
        indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        dense_indices = paddle.to_tensor(indices)
        dense_elements = paddle.to_tensor(values, dtype='float32')
        coo = paddle.sparse.sparse_coo_tensor(
            dense_indices, dense_elements, dense_shape, stop_gradient=False
        )
        np.testing.assert_array_equal(indices, coo.indices().numpy())
        np.testing.assert_array_equal(values, coo.values().numpy())

    def test_create_coo_by_np(self):
        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3]
        coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        np.testing.assert_array_equal(3, coo.nnz())
        np.testing.assert_array_equal(indices, coo.indices().numpy())
        np.testing.assert_array_equal(values, coo.values().numpy())

    def test_place(self):
        indices = [[0, 1], [0, 1]]
        values = [1.0, 2.0]
        dense_shape = [2, 2]
        coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        assert coo.place.is_xpu_place()
        assert coo.values().place.is_xpu_place()
        assert coo.indices().place.is_xpu_place()

    def test_dtype(self):
        indices = [[0, 1], [0, 1]]
        values = [1.0, 2.0]
        dense_shape = [2, 2]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        coo = paddle.sparse.sparse_coo_tensor(
            indices, values, dense_shape, dtype='float64'
        )
        assert coo.dtype == paddle.float64

    def test_create_coo_no_shape(self):
        indices = [[0, 1], [0, 1]]
        values = [1.0, 2.0]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        coo = paddle.sparse.sparse_coo_tensor(indices, values)
        assert [2, 2] == coo.shape


if __name__ == "__main__":
    unittest.main()
