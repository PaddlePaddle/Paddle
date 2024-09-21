# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.sparse
from paddle.base import core


def is_coalesced_naive(x):
    if not x.is_sparse_coo():
        return False
    indices = x.indices().numpy()
    indices = list(zip(*indices))
    duplicated_len = len(indices)
    remove_duplicated_len = len(set(indices))
    return duplicated_len == remove_duplicated_len


def is_coalesced_naive_static(indices):
    indices = list(zip(*indices))
    duplicated_len = len(indices)
    remove_duplicated_len = len(set(indices))
    return duplicated_len == remove_duplicated_len


class TestSparseIsCoalescedAPI(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        coo_indices = [[0, 0, 0, 1], [0, 0, 1, 2]]
        coo_values = [1.0, 2.0, 3.0, 4.0]
        coo_tenosr = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        )
        csr_crows = [0, 2, 3, 5]
        csr_cols = [1, 3, 2, 0, 1]
        csr_values = [1, 2, 3, 4, 5]
        csr_shape = [3, 4]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            csr_crows, csr_cols, csr_values, csr_shape, dtype=self.dtype
        )
        other_tensor = paddle.to_tensor([1, 2, 3, 4], dtype=self.dtype)
        self.tensors = [coo_tenosr, csr_tensor, other_tensor]

    def test_is_coalesced(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        excepted = [is_coalesced_naive(t) for t in self.tensors]
        for place in places:
            paddle.disable_static(place)
            for i in range(len(self.tensors)):
                self.assertEqual(self.tensors[i].is_coalesced(), excepted[i])

        paddle.enable_static()


class TestSparseIsCoalescedAPI1(unittest.TestCase):
    def setUp(self):
        self.dtype = "float64"
        coo_indices = [[0, 0, 1, 2], [0, 1, 1, 2]]
        coo_values = [1.0, 2.0, 3.0, 4.0]
        coo_tenosr = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        )
        csr_crows = [0, 2, 3, 5]
        csr_cols = [1, 3, 2, 0, 1]
        csr_values = [1, 2, 3, 4, 5]
        csr_shape = [3, 4]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            csr_crows, csr_cols, csr_values, csr_shape, dtype=self.dtype
        )
        other_tensor = paddle.to_tensor([1, 2, 3, 4], dtype=self.dtype)
        self.tensors = [coo_tenosr, csr_tensor, other_tensor]


class TestSparseIsCoalescedAPI2(unittest.TestCase):
    def setUp(self):
        coo_indices = [[0, 0, 1, 2], [0, 1, 1, 2], [0, 1, 1, 2]]
        coo_values = [1.0, 2.0, 3.0, 4.0]
        self.dtype = "int8"
        coo_tenosr = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        )
        csr_crows = [0, 2, 3, 5]
        csr_cols = [1, 3, 2, 0, 1]
        csr_values = [1, 2, 3, 4, 5]
        csr_shape = [3, 4]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            csr_crows, csr_cols, csr_values, csr_shape, dtype=self.dtype
        )
        other_tensor = paddle.to_tensor([1, 2, 3, 4], dtype=self.dtype)
        self.tensors = [coo_tenosr, csr_tensor, other_tensor]


class TestSparseIsCoalescedAPI3(unittest.TestCase):
    def setUp(self):
        coo_indices = [[0, 0, 1, 2], [0, 2, 0, 2], [0, 1, 1, 0]]
        coo_values = [1.0, 2.0, 3.0, 4.0]
        self.dtype = "int16"
        coo_tenosr = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        ).coalesce()
        csr_crows = [0, 2, 3, 5]
        csr_cols = [1, 3, 2, 0, 1]
        csr_values = [1, 2, 3, 4, 5]
        csr_shape = [3, 4]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            csr_crows, csr_cols, csr_values, csr_shape, dtype=self.dtype
        )
        other_tensor = paddle.to_tensor([1, 2, 3, 4], dtype=self.dtype)
        self.tensors = [coo_tenosr, csr_tensor, other_tensor]


class TestSparseIsCoalescedAPI4(unittest.TestCase):
    def setUp(self):
        coo_indices = [[0, 0, 0, 1], [0, 0, 1, 2]]
        coo_values = [1.0, 2.0, 3.0, 4.0]
        self.dtype = "int32"
        coo_tenosr = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        ).coalesce()
        csr_crows = [0, 2, 3, 5]
        csr_cols = [1, 3, 2, 0, 1]
        csr_values = [1, 2, 3, 4, 5]
        csr_shape = [3, 4]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            csr_crows, csr_cols, csr_values, csr_shape, dtype=self.dtype
        )
        other_tensor = paddle.to_tensor([1, 2, 3, 4], dtype=self.dtype)
        self.tensors = [coo_tenosr, csr_tensor, other_tensor]


class TestSparseIsCoalescedAPI5(unittest.TestCase):
    def setUp(self):
        coo_indices = [[0, 0, 0, 1], [0, 0, 1, 2]]
        coo_values = [1.0, 2.0, 3.0, 4.0]
        self.dtype = "int64"
        coo_tenosr = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        )
        csr_crows = [0, 2, 3, 5]
        csr_cols = [1, 3, 2, 0, 1]
        csr_values = [1, 2, 3, 4, 5]
        csr_shape = [3, 4]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            csr_crows, csr_cols, csr_values, csr_shape, dtype=self.dtype
        )
        other_tensor = paddle.to_tensor([1, 2, 3, 4], dtype=self.dtype)
        self.tensors = [coo_tenosr, csr_tensor, other_tensor]


class TestSparseIsCoalescedAPI6(unittest.TestCase):
    def setUp(self):
        coo_indices = [[0, 0, 0, 1], [0, 0, 1, 2]]
        coo_values = [1.0, 2.0, 3.0, 4.0]
        self.dtype = "uint8"
        coo_tenosr = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        )
        csr_crows = [0, 2, 3, 5]
        csr_cols = [1, 3, 2, 0, 1]
        csr_values = [1, 2, 3, 4, 5]
        csr_shape = [3, 4]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            csr_crows, csr_cols, csr_values, csr_shape, dtype=self.dtype
        )
        other_tensor = paddle.to_tensor([1, 2, 3, 4], dtype=self.dtype)
        self.tensors = [coo_tenosr, csr_tensor, other_tensor]


class TestSparseIsCoalescedAPI7(unittest.TestCase):
    def setUp(self):
        coo_indices = [[0, 0, 1, 2], [0, 1, 1, 2], [0, 1, 1, 2]]
        coo_values = [1.0, 0.0, 0.0, 1.0]
        self.dtype = "bool"
        coo_tenosr = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        )
        csr_crows = [0, 2, 3, 5]
        csr_cols = [1, 3, 2, 0, 1]
        csr_values = [1, 0, 1, 0, 0]
        csr_shape = [3, 4]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            csr_crows, csr_cols, csr_values, csr_shape, dtype=self.dtype
        )
        other_tensor = paddle.to_tensor([1, 2, 3, 4], dtype=self.dtype)
        self.tensors = [coo_tenosr, csr_tensor, other_tensor]


class TestSparseIsCoalescedAPI8(unittest.TestCase):
    def setUp(self):
        coo_indices = [[0, 0, 1, 2], [0, 1, 1, 2], [0, 1, 1, 2]]
        coo_values = [1.0, 2.0, 3.0, 4.0]
        self.dtype = "complex64"
        coo_tenosr = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        )
        csr_crows = [0, 2, 3, 5]
        csr_cols = [1, 3, 2, 0, 1]
        csr_values = [1, 2, 3, 4, 5]
        csr_shape = [3, 4]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            csr_crows, csr_cols, csr_values, csr_shape, dtype=self.dtype
        )
        other_tensor = paddle.to_tensor([1, 2, 3, 4], dtype=self.dtype)
        self.tensors = [coo_tenosr, csr_tensor, other_tensor]


class TestSparseIsCoalescedAPI9(unittest.TestCase):
    def setUp(self):
        coo_indices = [[0, 0, 1, 2], [0, 1, 1, 2], [1, 0, 1, 2]]
        coo_values = [1.0, 2.0, 3.0, 4.0]
        self.dtype = "complex128"
        coo_tenosr = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        )
        csr_crows = [0, 2, 3, 5]
        csr_cols = [1, 3, 2, 0, 1]
        csr_values = [1, 2, 3, 4, 5]
        csr_shape = [3, 4]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            csr_crows, csr_cols, csr_values, csr_shape, dtype=self.dtype
        )
        other_tensor = paddle.to_tensor([1, 2, 3, 4], dtype=self.dtype)
        self.tensors = [coo_tenosr, csr_tensor, other_tensor]


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the float16",
)
class TestSparseIsCoalescedFP16API(unittest.TestCase):
    def setUp(self):
        self.dtype = "float16"
        coo_indices = [[0, 0, 0, 1], [0, 0, 1, 2]]
        coo_values = [1.0, 2.0, 3.0, 4.0]
        coo_tenosr = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        ).coalesce()
        csr_crows = [0, 2, 3, 5]
        csr_cols = [1, 3, 2, 0, 1]
        csr_values = [1, 2, 3, 4, 5]
        csr_shape = [3, 4]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            csr_crows, csr_cols, csr_values, csr_shape, dtype=self.dtype
        )
        other_tensor = paddle.to_tensor([1, 2, 3, 4], dtype=self.dtype)
        self.tensors = [coo_tenosr, csr_tensor, other_tensor]


if __name__ == "__main__":
    unittest.main()
