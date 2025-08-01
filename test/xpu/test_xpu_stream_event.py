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

import unittest

import paddle
from paddle.device import xpu


class TestCurrentStream(unittest.TestCase):
    def test_current_stream(self):
        if paddle.is_compiled_with_xpu():
            s = xpu.current_stream()
            self.assertTrue(isinstance(s, xpu.Stream))

            s1 = xpu.current_stream(0)
            self.assertTrue(isinstance(s1, xpu.Stream))

            s2 = xpu.current_stream(paddle.XPUPlace(0))
            self.assertTrue(isinstance(s2, xpu.Stream))
            self.assertEqual(s1, s2)
            self.assertRaises(ValueError, xpu.current_stream, "xpu:0")


class TestSynchronize(unittest.TestCase):
    def test_synchronize(self):
        if paddle.is_compiled_with_xpu():
            self.assertIsNone(xpu.synchronize())
            self.assertIsNone(xpu.synchronize(0))
            self.assertIsNone(xpu.synchronize(paddle.XPUPlace(0)))

            self.assertRaises(ValueError, xpu.synchronize, "xpu:0")


class TestXPUStream(unittest.TestCase):
    def test_xpu_stream(self):
        if paddle.is_compiled_with_xpu():
            s = paddle.device.xpu.Stream()
            self.assertIsNotNone(s)

    def test_xpu_stream_synchronize(self):
        if paddle.is_compiled_with_xpu():
            s = paddle.device.xpu.Stream()
            e1 = paddle.device.xpu.Event()
            e2 = paddle.device.xpu.Event()

            e1.record(s)
            print("1111")
            e1.query()
            tensor1 = paddle.to_tensor(paddle.rand([1000, 1000]))
            tensor2 = paddle.matmul(tensor1, tensor1)
            s.synchronize()
            e2.record(s)
            e2.synchronize()

            self.assertTrue(e2.query())

    def test_xpu_stream_wait_event_and_record_event(self):
        if paddle.is_compiled_with_xpu():
            s1 = xpu.Stream(0)
            tensor1 = paddle.to_tensor(paddle.rand([1000, 1000]))
            tensor2 = paddle.matmul(tensor1, tensor1)
            e1 = xpu.Event()
            s1.record_event(e1)

            s2 = xpu.Stream(0)
            s2.wait_event(e1)
            s2.synchronize()

            self.assertTrue(e1.query() and s1.query() and s2.query())


if __name__ == "__main__":
    unittest.main()
