#  Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import numpy as np

import paddle


class TestPaddleSize(unittest.TestCase):
    def test_tensor_size(self):
        x = paddle.empty(3, 4, 5)
        size = x.size()
        self.assertEqual(size, (3, 4, 5))
        self.assertIsInstance(size, paddle.Size)

        int_size = x.size(dim=1)
        self.assertEqual(int_size, 4)
        self.assertIsInstance(int_size, int)

    def test_creation_size(self):
        size = paddle.Size()
        self.assertEqual(size, ())
        self.assertIsInstance(size, list)
        self.assertIsInstance(size, paddle.Size)

        size = paddle.Size([2, 3, 4])
        self.assertEqual(size, (2, 3, 4))
        self.assertIsInstance(size, paddle.Size)

        size = paddle.Size((2, 3, 4))
        self.assertEqual(size, (2, 3, 4))
        self.assertIsInstance(size, paddle.Size)

        tensor1 = paddle.to_tensor(2)
        tensor2 = paddle.to_tensor(3)
        size = paddle.Size([tensor1, tensor2])
        self.assertEqual(size, (2, 3))
        self.assertIsInstance(size, paddle.Size)

        tensor3 = paddle.to_tensor([2, 3])
        size = paddle.Size(tensor3)
        self.assertEqual(size, (2, 3))
        self.assertIsInstance(size, paddle.Size)

        size = paddle.Size([True, False])
        self.assertEqual(size, (1, 0))
        self.assertIsInstance(size, paddle.Size)

        size = paddle.Size([np.int64(8), np.int64(8)])
        self.assertEqual(size, (8, 8))
        self.assertIsInstance(size, paddle.Size)

    def test_creation_invalid_type(self):
        with self.assertRaises(TypeError):
            paddle.Size([1.5, 2.5])  # float not allowed
        with self.assertRaises(TypeError):
            paddle.Size(["a", "b"])  # string not allowed

    def test_creation_from_mixed_types(self):
        size = paddle.Size([1, paddle.to_tensor(2), 3])
        self.assertEqual(size, (1, 2, 3))
        self.assertIsInstance(size, paddle.Size)

    def test_getitem_int(self):
        size = paddle.Size([2, 3, 4])
        self.assertEqual(size[0], 2)
        self.assertEqual(size[1], 3)
        self.assertEqual(size[2], 4)
        self.assertIsInstance(size[0], int)

    def test_getitem_slice(self):
        size = paddle.Size([2, 3, 4, 5])
        sliced = size[1:3]
        self.assertEqual(sliced, (3, 4))
        self.assertIsInstance(sliced, paddle.Size)

    def test_addition(self):
        size1 = paddle.Size([2, 3])
        size2 = (4, 5)
        result = size1 + size2
        self.assertEqual(result, (2, 3, 4, 5))
        self.assertIsInstance(result, paddle.Size)

    def test_raddition(self):
        size1 = paddle.Size([2, 3])
        size2 = (4, 5)
        result = size2 + size1
        self.assertEqual(result, (4, 5, 2, 3))
        self.assertIsInstance(result, paddle.Size)

    def test_addition_invalid_type(self):
        size = paddle.Size([2, 3])
        with self.assertRaises(TypeError):
            size + "abc"  # string not allowed

    def test_multiplication(self):
        size = paddle.Size([2, 3])
        result = size * 2
        self.assertEqual(result, (2, 3, 2, 3))
        self.assertIsInstance(result, paddle.Size)

    def test_rmultiplication(self):
        size = paddle.Size([2, 3])
        result = 2 * size
        self.assertEqual(result, (2, 3, 2, 3))
        self.assertIsInstance(result, paddle.Size)

    def test_multiplication_invalid_type(self):
        size = paddle.Size([2, 3])
        with self.assertRaises(TypeError):
            size * 2.5  # float not allowed
        with self.assertRaises(TypeError):
            size * "a"  # string not allowed

    def test_repr(self):
        size = paddle.Size([2, 3, 4])
        size1 = paddle.Size()
        self.assertEqual(repr(size), "paddle.Size([2, 3, 4])")
        self.assertEqual(str(size), "paddle.Size([2, 3, 4])")
        self.assertEqual(str(size1), "paddle.Size([])")

    def test_numel(self):
        size = paddle.Size([2, 3, 4])
        self.assertEqual(size.numel(), 24)  # 2*3*4=24

    def test_empty_size_numel(self):
        size = paddle.Size([])
        self.assertEqual(size.numel(), 1)  # Empty size has numel=1

    def test_reduce(self):
        size = paddle.Size([2, 3])
        reduced = size.__reduce__()
        self.assertEqual(reduced, (paddle.Size, ((2, 3),)))
        # Test reconstruction
        new_size = reduced[0](*reduced[1])
        self.assertEqual(new_size, size)
        self.assertIsInstance(new_size, paddle.Size)

    def test_count_index(self):
        x = paddle.Size([2, 3]).count(2)
        y = paddle.Size([2, 3]).index(3, 0)
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)


class TestTensorShapeBehavior(unittest.TestCase):
    def setUp(self):
        self.tensor = paddle.randn([10, 20, 30])

    def test_01_type_and_value(self):
        s = self.tensor.shape

        self.assertIsInstance(
            s, paddle.Size, "Tensor.shape should be instance of paddle.Size"
        )

        self.assertEqual(
            type(s),
            paddle.Size,
            "The exact type of Tensor.shape should be paddle.Size",
        )

        self.assertIsInstance(s, list, "paddle.Size should inherit from list")

        self.assertEqual(s, [10, 20, 30])

        self.assertEqual(len(s), 3)

    def test_02_edge_cases_0d_and_1d(self):
        scalar = paddle.to_tensor(100)
        s_scalar = scalar.shape

        self.assertEqual(type(s_scalar), paddle.Size)
        self.assertEqual(s_scalar, [])
        self.assertEqual(len(s_scalar), 0)

        vector = paddle.to_tensor([1, 2, 3])
        s_vector = vector.shape

        self.assertEqual(type(s_vector), paddle.Size)
        self.assertEqual(s_vector, [3])
        self.assertEqual(len(s_vector), 1)

    def test_03_indexing_and_slicing(self):
        s = self.tensor.shape

        self.assertEqual(s[0], 10)
        self.assertEqual(s[-1], 30)
        self.assertIsInstance(s[0], int)

        s_slice = s[1:3]
        self.assertEqual(s_slice, [20, 30])
        self.assertIsInstance(s_slice, paddle.Size)

        s_full_slice = s[:]
        self.assertEqual(s_full_slice, [10, 20, 30])
        self.assertIsInstance(s_full_slice, paddle.Size)
        self.assertIsNot(s, s_full_slice)

    def test_04_concatenation_add(self):
        s = paddle.Size([1, 2])
        result_ss = s + paddle.Size([3, 4])
        self.assertEqual(result_ss, [1, 2, 3, 4])
        self.assertEqual(type(result_ss), paddle.Size)

        result_sl = s + [3, 4]  # noqa: RUF005
        self.assertEqual(result_sl, [1, 2, 3, 4])
        self.assertEqual(type(result_sl), paddle.Size)

        result_ls = [0, 0] + s  # noqa: RUF005
        self.assertEqual(result_ls, [0, 0, 1, 2])
        self.assertEqual(type(result_ls), paddle.Size)

        result_st = s + (3, 4)  # noqa: RUF005
        self.assertEqual(result_st, [1, 2, 3, 4])
        self.assertEqual(type(result_st), paddle.Size)

        result_ts = (0, 0) + s  # noqa: RUF005
        self.assertEqual(result_ts, [0, 0, 1, 2])
        self.assertEqual(type(result_ts), paddle.Size)

    def test_05_repetition_mul(self):
        s = paddle.Size([1, 2])

        result_sm = s * 3
        self.assertEqual(result_sm, [1, 2, 1, 2, 1, 2])
        self.assertEqual(type(result_sm), paddle.Size)

        result_ms = 3 * s
        self.assertEqual(result_ms, [1, 2, 1, 2, 1, 2])
        self.assertEqual(type(result_ms), paddle.Size)

    def test_06_custom_methods(self):
        s = self.tensor.shape
        self.assertTrue(hasattr(s, "numel"))
        self.assertEqual(s.numel(), 10 * 20 * 30)

        s_scalar = paddle.to_tensor(100).shape
        self.assertEqual(s_scalar.numel(), 1)

    def test_07_mutability_and_independence(self):
        s = self.tensor.shape
        original_shape_copy = list(s)  # [10, 20, 30]

        try:
            s.append(40)
            s[0] = 99
        except Exception as e:
            self.fail(f"paddle.Size should support methods like list: {e}")

        self.assertEqual(
            s, [99, 20, 30, 40], "paddle.Size should support methods like list"
        )

        self.assertEqual(
            self.tensor.shape,
            original_shape_copy,
            "Modifying a Size object should not modify the original tensor shape",
        )

        s_new = self.tensor.shape
        self.assertEqual(
            s_new,
            original_shape_copy,
            "Calling tensor.shape again should return the unmodified shape",
        )
        self.assertIsNot(
            s, s_new, "Calling tensor.shape should return a new object"
        )


if __name__ == "__main__":
    unittest.main()
