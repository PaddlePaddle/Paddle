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

import paddle


class TestBroadcastShapes(unittest.TestCase):
    def test_result(self):
        shape = paddle.broadcast_shapes(
            [5, 1, 3, 10],
            [5, 4, 1, 1],
            [1, 1, 3, 10],
            [1, 4, 3, 1],
            [1, 4, 1, 10],
        )
        self.assertEqual(shape, [5, 4, 3, 10])

        shape = paddle.broadcast_shapes([-1, 1, 3], [1, 6, 1], [1, 1, 3])
        self.assertEqual(shape, [-1, 6, 3])

        shape = paddle.broadcast_shapes([8, 3])

        self.assertEqual(shape, [8, 3])

        shape = paddle.broadcast_shapes([2, 3, 1], [6], [3, 1])
        self.assertEqual(shape, [2, 3, 6])

    def test_empty(self):
        shape = paddle.broadcast_shapes([])
        self.assertEqual(shape, [])

        shape = paddle.broadcast_shapes([], [2, 3, 4])
        self.assertEqual(shape, [2, 3, 4])

        shape = paddle.broadcast_shapes([10, 1, 7], [], [1, 6, 1], [1, 1, 7])
        self.assertEqual(shape, [10, 6, 7])

    def test_complex_case(self):
        test_cases = [
            ([0], [1], [], [0]),
            ([2, -1], [0], [2, 0]),
            ([0, 3], [3], [0, 3]),
            ([0, 1, 3], [0, 1, 0, 3], [1, 0, -1], [0, 0, 0, 3]),
            ([0, 1, 3], [0, 1, 1, 5, 3], [], [0, 1, 0, 5, 3]),
        ]

        for shape_list in test_cases:
            expected = shape_list[-1]
            result = paddle.broadcast_shapes(*shape_list[:-1])
            self.assertEqual(result, expected)

    def test_error(self):
        self.assertRaises(
            ValueError, paddle.broadcast_shapes, [5, 1, 3], [1, 4, 1], [1, 2, 3]
        )
        self.assertRaises(ValueError, paddle.broadcast_shapes, [0], [0, 2])


if __name__ == "__main__":
    unittest.main()
