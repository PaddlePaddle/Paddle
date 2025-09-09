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

# test_cuda_unittest.py
import unittest

import paddle


class TestCudaCompat(unittest.TestCase):
    # ---------------------
    # paddle.device compatibility tests
    # ---------------------

    def test_paddle_device_cpu(self):
        d = paddle.device("cpu")
        self.assertTrue(d == "cpu")
        self.assertEqual(str(d), "cpu")
        self.assertEqual(d(), "cpu")

    def test_paddle_device_gpu_variants(self):
        cases = [
            (("cuda", 2), "gpu:2"),
            (("gpu", 1), "gpu:1"),
            (("cuda:3",), "gpu:3"),
            (("gpu:4",), "gpu:4"),
            ((5,), "gpu:5"),  # int -> gpu
            (("gpu", None), "gpu:0"),  # None index defaults to 0
        ]
        for args, expected in cases:
            d = paddle.device(*args)
            self.assertEqual(str(d), expected)
            self.assertEqual(d(), expected)  # __call__ path
            self.assertTrue(d == expected)  # __eq__ with str

    def test_paddle_device_xpu_variants(self):
        cases = [
            (("xpu", 2), "xpu:2"),
            (("xpu:3",), "xpu:3"),
            (("xpu", None), "xpu:0"),
        ]
        for args, expected in cases:
            d = paddle.device(*args)
            self.assertEqual(str(d), expected)

    def test_paddle_device_copy(self):
        d1 = paddle.device("gpu:1")
        d2 = paddle.device(d1)
        self.assertEqual(d1, d2)

    def test_paddle_device_invalid(self):
        with self.assertRaises(ValueError):
            paddle.device("cpu", 2)

        with self.assertRaises(ValueError):
            paddle.device("tpu")

        with self.assertRaises(TypeError):
            paddle.device(3.14)

    def test_device_eq(self):
        d1 = paddle.device("cuda:1")
        d2 = paddle.device("gpu:1")
        d3 = paddle.device("gpu:2")
        self.assertTrue(d1 == d2)
        self.assertFalse(d1 == d3)
        self.assertFalse(d1 == "gpu:2")  # mismatch

    def test_device_module_getattr_success(self):
        mod = paddle.device.cuda
        self.assertIs(mod, paddle.device.cuda)

    def test_device_module_getattr_fail(self):
        with self.assertRaises(AttributeError):
            _ = paddle.device.foobar


if __name__ == '__main__':
    unittest.main()
