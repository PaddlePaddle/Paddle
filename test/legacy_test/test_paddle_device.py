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


class TestPaddleDevice(unittest.TestCase):
    def test_str_only(self):
        d = paddle.device("cpu")
        self.assertEqual(str(d), "cpu")
        self.assertEqual(d.type, "cpu")
        self.assertIsNone(d.index)
        if (
            paddle.is_compiled_with_cuda()
            and paddle.device.get_device().startswith('gpu')
        ):
            d = paddle.device("cuda")
            self.assertEqual(str(d), "cuda:0")
            self.assertEqual(d.type, "cuda")
            self.assertEqual(d.index, 0)

            d = paddle.device("gpu")
            self.assertEqual(str(d), "cuda:0")
            self.assertEqual(d.type, "cuda")
            self.assertEqual(d.index, 0)
        if paddle.is_compiled_with_xpu():
            d = paddle.device("xpu")
            self.assertEqual(str(d), "xpu:0")
            self.assertEqual(d.type, "xpu")
            self.assertEqual(d.index, 0)

    def test_with_device(self):
        if paddle.device.cuda.device_count() >= 1:
            with paddle.device("cpu"):
                a = paddle.empty([2])
                assert str(a.place) == "Place(cpu)"

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            paddle.device(None, 1)

        with self.assertRaises(ValueError):
            paddle.device("abc")

        with self.assertRaises(AttributeError):
            paddle.device(3.14)

        with self.assertRaises(ValueError):
            paddle.device("abc:0")


if __name__ == "__main__":
    unittest.main()
