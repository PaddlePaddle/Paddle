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
from paddle import device as Device


class TestDevice(unittest.TestCase):
    def test_str_only(self):
        d = Device("cpu")
        self.assertEqual(str(d), "cpu")
        self.assertEqual(d.type, "cpu")
        self.assertIsNone(d.index)

        d = Device("cuda")
        self.assertEqual(str(d), "cuda:0")
        self.assertEqual(d.type, "cuda")
        self.assertEqual(d.index, 0)

        d = Device("gpu")
        self.assertEqual(str(d), "gpu:0")
        self.assertEqual(d.type, "gpu")
        self.assertEqual(d.index, 0)

        d = Device("xpu")
        self.assertEqual(str(d), "xpu:0")
        self.assertEqual(d.type, "xpu")
        self.assertEqual(d.index, 0)

    def test_str_with_index(self):
        d = Device("cuda", 1)
        self.assertEqual(str(d), "cuda:1")
        self.assertEqual(d.type, "cuda")
        self.assertEqual(d.index, 1)

        d = Device("gpu", 2)
        self.assertEqual(str(d), "gpu:2")
        self.assertEqual(d.type, "gpu")
        self.assertEqual(d.index, 2)

        d = Device("cpu", 0)
        self.assertEqual(str(d), "cpu")
        self.assertEqual(d.type, "cpu")
        self.assertIsNone(d.index)

    def test_str_colon(self):
        d = Device("cuda:3")
        self.assertEqual(str(d), "cuda:3")
        self.assertEqual(d.type, "cuda")
        self.assertEqual(d.index, 3)

        d = Device("gpu:5")
        self.assertEqual(str(d), "gpu:5")
        self.assertEqual(d.type, "gpu")
        self.assertEqual(d.index, 5)

    def test_int_legacy(self):
        d = Device(4)
        self.assertEqual(str(d), "cuda:4")
        self.assertEqual(d.type, "cuda")
        self.assertEqual(d.index, 4)

    def test_device_copy(self):
        original = Device("cuda:2")
        d = Device(original)
        self.assertEqual(str(d), "cuda:2")
        self.assertEqual(d.type, "cuda")
        self.assertEqual(d.index, 2)

    def test_with_device(self):
        if paddle.device.cuda.device_count() >= 1:
            with Device("cpu"):
                a = paddle.empty([2])
                assert str(a.place) == "Place(cpu)"

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            Device(None, 1)

        with self.assertRaises(ValueError):
            Device("abc")

        with self.assertRaises(TypeError):
            Device(3.14)

        with self.assertRaises(ValueError):
            Device("abc:0")


if __name__ == "__main__":
    unittest.main()
