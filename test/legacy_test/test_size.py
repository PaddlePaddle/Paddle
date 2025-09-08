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
    # TODO: enable when paddle.Tensor.size() is implemented
    # def test_tensor_size(self):
    #     x = paddle.empty(3, 4, 5)
    #     size = x.size()
    #     self.assertEqual(size, (3, 4, 5))
    #     self.assertIsInstance(size, paddle.Size)

    #     int_size = x.size(dim=1)
    #     self.assertEqual(int_size, 3)
    #     self.assertIsInstance(int_size, int)

    def test_creation_size(self):
        size = paddle.Size()
        self.assertEqual(size, ())
        self.assertIsInstance(size, tuple)
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

    def test_concat_method(self):
        size1 = paddle.Size([1, 2])
        size2 = (3, 4)
        result = size1.__concat__(size2)
        self.assertEqual(result, (1, 2, 3, 4))
        self.assertIsInstance(result, paddle.Size)

    def test_concat_invalid_type(self):
        size = paddle.Size([1, 2])
        with self.assertRaises(TypeError):
            size.__concat__("invalid")  # string not allowed

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


if __name__ == "__main__":
    unittest.main()
