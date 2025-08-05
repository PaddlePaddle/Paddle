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

import ctypes
import unittest

import numpy as np

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

            self.assertTrue(e1.query())


class TestXPUEvent(unittest.TestCase):
    def test_xpu_event(self):
        if paddle.is_compiled_with_xpu():
            e = paddle.device.xpu.Event()
            self.assertIsNotNone(e)
            s = paddle.device.xpu.current_stream()

    def test_xpu_event_methods(self):
        if paddle.is_compiled_with_xpu():
            e = paddle.device.xpu.Event()
            s = paddle.device.xpu.current_stream()
            event_query_1 = e.query()
            tensor1 = paddle.to_tensor(paddle.rand([1000, 1000]))
            tensor2 = paddle.matmul(tensor1, tensor1)
            s.record_event(e)
            e.synchronize()
            event_query_2 = e.query()

            self.assertTrue(event_query_1)
            self.assertTrue(event_query_2)


class TestStreamGuard(unittest.TestCase):
    '''
    Note:
        The asynchronous execution property of XPU Stream can only be tested offline.
    '''

    def test_stream_guard_normal(self):
        if paddle.is_compiled_with_xpu():
            s = paddle.device.Stream()
            a = paddle.to_tensor(np.array([0, 2, 4], dtype="int32"))
            b = paddle.to_tensor(np.array([1, 3, 5], dtype="int32"))
            c = a + b
            with paddle.device.stream_guard(s):
                d = a + b
                s.synchronize()

            np.testing.assert_array_equal(np.array(c), np.array(d))

    def test_stream_guard_default_stream(self):
        if paddle.is_compiled_with_xpu():
            s1 = paddle.device.current_stream()
            with paddle.device.stream_guard(s1):
                pass
            s2 = paddle.device.current_stream()

            self.assertTrue(id(s1.stream_base) == id(s2.stream_base))

    def test_set_current_stream_default_stream(self):
        if paddle.is_compiled_with_xpu():
            cur_stream = paddle.device.current_stream()
            new_stream = paddle.device.set_stream(cur_stream)

            self.assertTrue(
                id(cur_stream.stream_base) == id(new_stream.stream_base)
            )

    def test_stream_guard_raise_error(self):
        if paddle.is_compiled_with_xpu():

            def test_not_correct_stream_guard_input():
                tmp = np.zeros(5)
                with paddle.device.stream_guard(tmp):
                    pass

            self.assertRaises(TypeError, test_not_correct_stream_guard_input)


class TestRawStream(unittest.TestCase):
    def test_xpu_stream(self):
        if paddle.is_compiled_with_xpu():
            xpu_stream = paddle.device.xpu.current_stream().xpu_stream
            print(xpu_stream)
            self.assertTrue(type(xpu_stream) is int)
            ptr = ctypes.c_void_p(xpu_stream)


if __name__ == "__main__":
    unittest.main()
