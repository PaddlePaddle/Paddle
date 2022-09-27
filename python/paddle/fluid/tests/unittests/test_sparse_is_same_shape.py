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

import paddle
from paddle.incubate.sparse.binary import is_same_shape


class TestSparseIsSameShapeAPI(unittest.TestCase):
    """
    test paddle.incubate.sparse.is_same_shape
    """

    def setUp(self):
        self.shapes = [[2, 5, 8], [3, 4]]
        self.tensors = [
            paddle.rand(self.shapes[0]),
            paddle.rand(self.shapes[0]),
            paddle.rand(self.shapes[1])
        ]
        self.sparse_dim = 2

    def test_dense_dense(self):
        self.assertTrue(is_same_shape(self.tensors[0], self.tensors[1]))
        self.assertFalse(is_same_shape(self.tensors[0], self.tensors[2]))
        self.assertFalse(is_same_shape(self.tensors[1], self.tensors[2]))

    def test_dense_csr(self):
        self.assertTrue(
            is_same_shape(self.tensors[0], self.tensors[1].to_sparse_csr()))
        self.assertFalse(
            is_same_shape(self.tensors[0], self.tensors[2].to_sparse_csr()))
        self.assertFalse(
            is_same_shape(self.tensors[1], self.tensors[2].to_sparse_csr()))

    def test_dense_coo(self):
        self.assertTrue(
            is_same_shape(self.tensors[0],
                          self.tensors[1].to_sparse_coo(self.sparse_dim)))
        self.assertFalse(
            is_same_shape(self.tensors[0],
                          self.tensors[2].to_sparse_coo(self.sparse_dim)))
        self.assertFalse(
            is_same_shape(self.tensors[1],
                          self.tensors[2].to_sparse_coo(self.sparse_dim)))

    def test_csr_dense(self):
        self.assertTrue(
            is_same_shape(self.tensors[0].to_sparse_csr(), self.tensors[1]))
        self.assertFalse(
            is_same_shape(self.tensors[0].to_sparse_csr(), self.tensors[2]))
        self.assertFalse(
            is_same_shape(self.tensors[1].to_sparse_csr(), self.tensors[2]))

    def test_csr_csr(self):
        self.assertTrue(
            is_same_shape(self.tensors[0].to_sparse_csr(),
                          self.tensors[1].to_sparse_csr()))
        self.assertFalse(
            is_same_shape(self.tensors[0].to_sparse_csr(),
                          self.tensors[2].to_sparse_csr()))
        self.assertFalse(
            is_same_shape(self.tensors[1].to_sparse_csr(),
                          self.tensors[2].to_sparse_csr()))

    def test_csr_coo(self):
        self.assertTrue(
            is_same_shape(self.tensors[0].to_sparse_csr(),
                          self.tensors[1].to_sparse_coo(self.sparse_dim)))
        self.assertFalse(
            is_same_shape(self.tensors[0].to_sparse_csr(),
                          self.tensors[2].to_sparse_coo(self.sparse_dim)))
        self.assertFalse(
            is_same_shape(self.tensors[1].to_sparse_csr(),
                          self.tensors[2].to_sparse_coo(self.sparse_dim)))

    def test_coo_dense(self):
        self.assertTrue(
            is_same_shape(self.tensors[0].to_sparse_coo(self.sparse_dim),
                          self.tensors[1]))
        self.assertFalse(
            is_same_shape(self.tensors[0].to_sparse_coo(self.sparse_dim),
                          self.tensors[2]))
        self.assertFalse(
            is_same_shape(self.tensors[1].to_sparse_coo(self.sparse_dim),
                          self.tensors[2]))

    def test_coo_csr(self):
        self.assertTrue(
            is_same_shape(self.tensors[0].to_sparse_coo(self.sparse_dim),
                          self.tensors[1].to_sparse_csr()))
        self.assertFalse(
            is_same_shape(self.tensors[0].to_sparse_coo(self.sparse_dim),
                          self.tensors[2].to_sparse_csr()))
        self.assertFalse(
            is_same_shape(self.tensors[1].to_sparse_coo(self.sparse_dim),
                          self.tensors[2].to_sparse_csr()))

    def test_coo_coo(self):
        self.assertTrue(
            is_same_shape(self.tensors[0].to_sparse_coo(self.sparse_dim),
                          self.tensors[1].to_sparse_coo(self.sparse_dim)))
        self.assertFalse(
            is_same_shape(self.tensors[0].to_sparse_coo(self.sparse_dim),
                          self.tensors[2].to_sparse_coo(self.sparse_dim)))
        self.assertFalse(
            is_same_shape(self.tensors[1].to_sparse_coo(self.sparse_dim),
                          self.tensors[2].to_sparse_coo(self.sparse_dim)))


if __name__ == "__main__":
    unittest.main()
