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


def wrap_place(place):
    p = paddle.base.libpaddle.Place()
    p.set_place(place)
    return p


class TestPlace(unittest.TestCase):
    def test_eq(self):
        x = paddle.to_tensor([1, 2, 3], place=paddle.CPUPlace())
        y = paddle.to_tensor([1, 2, 3], place=paddle.CPUPlace())
        self.assertEqual(x.place, y.place)
        self.assertEqual(x.place, wrap_place(paddle.CPUPlace()))

    def test_ne(self):
        if not paddle.is_compiled_with_cuda():
            return
        x = paddle.to_tensor([1, 2, 3], place=paddle.CPUPlace())
        y = paddle.to_tensor([1, 2, 3], place=paddle.CUDAPlace(0))
        self.assertNotEqual(x.place, y.place)
        self.assertNotEqual(x.place, wrap_place(paddle.CUDAPlace(0)))
        self.assertNotEqual(y.place, wrap_place(paddle.CPUPlace()))
        self.assertEqual(y.place, wrap_place(paddle.CUDAPlace(0)))


if __name__ == "__main__":
    unittest.main()
