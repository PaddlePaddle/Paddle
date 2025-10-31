# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class TestSplitWithSizes(unittest.TestCase):
    def setUp(self):
        self.x = paddle.arange(12).reshape([3, 4])
        self.split_sizes = [1, 2]
        self.dim = 0

    def test_basic_functionality(self):
        splits = paddle.Tensor.split_with_sizes(
            self.x, self.split_sizes, dim=self.dim
        )

        self.assertEqual(len(splits), len(self.split_sizes))

        expected_shapes = [[1, 4], [2, 4]]
        for s, shape in zip(splits, expected_shapes):
            self.assertListEqual(list(s.shape), shape)

        np_x = self.x.numpy()
        start = 0
        for i, size in enumerate(self.split_sizes):
            np_ref = np_x[start : start + size, :]
            np.testing.assert_array_equal(splits[i].numpy(), np_ref)
            start += size

    def test_ValueError_raises(self):
        invalid_split_sizes = [1, -2]
        with self.assertRaises(ValueError) as cm:
            paddle.Tensor.split_with_sizes(
                self.x, invalid_split_sizes, dim=self.dim
            )

        invalid_split_sizes = [1, 1]
        with self.assertRaises(ValueError) as cm:
            paddle.Tensor.split_with_sizes(
                self.x, invalid_split_sizes, dim=self.dim
            )


if __name__ == "__main__":
    unittest.main()
