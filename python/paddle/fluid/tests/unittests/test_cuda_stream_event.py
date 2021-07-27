# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.device import cuda
import paddle

import unittest


class TestCurrentStream(unittest.TestCase):
    def test_current_stream(self):
        if paddle.is_compiled_with_cuda():
            s = cuda.current_stream()
            self.assertTrue(isinstance(s, cuda.Stream))

            s1 = cuda.current_stream(0)
            self.assertTrue(isinstance(s1, cuda.Stream))

            s2 = cuda.current_stream(paddle.CUDAPlace(0))
            self.assertTrue(isinstance(s2, cuda.Stream))

            self.assertEqual(s1, s2)

            self.assertRaises(ValueError, cuda.current_stream, "gpu:0")


class TestSynchronize(unittest.TestCase):
    def test_synchronize(self):
        if paddle.is_compiled_with_cuda():
            self.assertIsNone(cuda.synchronize())
            self.assertIsNone(cuda.synchronize(0))
            self.assertIsNone(cuda.synchronize(paddle.CUDAPlace(0)))

            self.assertRaises(ValueError, cuda.synchronize, "gpu:0")


class TestCUDAStream(unittest.TestCase):
    def test_cuda_stream(self):
        if paddle.is_compiled_with_cuda():
            s = paddle.device.cuda.Stream()
            self.assertIsNotNone(s)

    def test_cuda_stream_synchronize(self):
        if paddle.is_compiled_with_cuda():
            s = paddle.device.cuda.Stream()
            e1 = paddle.device.cuda.Event(True, False, False)
            e2 = paddle.device.cuda.Event(True, False, False)

            e1.record(s)
            e1.query()
            tensor1 = paddle.to_tensor(paddle.rand([1000, 1000]))
            tensor2 = paddle.matmul(tensor1, tensor1)
            s.synchronize()
            e2.record(s)
            e2.synchronize()

            self.assertTrue(s.query())

    def test_cuda_stream_wait_event_and_record_event(self):
        if paddle.is_compiled_with_cuda():
            s1 = cuda.Stream(0)
            tensor1 = paddle.to_tensor(paddle.rand([1000, 1000]))
            tensor2 = paddle.matmul(tensor1, tensor1)
            e1 = cuda.Event(False, False, False)
            s1.record_event(e1)

            s2 = cuda.Stream(0)
            s2.wait_event(e1)
            s2.synchronize()

            self.assertTrue(e1.query() and s1.query() and s2.query())


class TestCUDAEvent(unittest.TestCase):
    def test_cuda_event(self):
        if paddle.is_compiled_with_cuda():
            e = paddle.device.cuda.Event(True, False, False)
            self.assertIsNotNone(e)
            s = paddle.device.cuda.current_stream()

    def test_cuda_event_methods(self):
        if paddle.is_compiled_with_cuda():
            e = paddle.device.cuda.Event(True, False, False)
            s = paddle.device.cuda.current_stream()
            event_query_1 = e.query()
            tensor1 = paddle.to_tensor(paddle.rand([1000, 1000]))
            tensor2 = paddle.matmul(tensor1, tensor1)
            s.record_event(e)
            e.synchronize()
            event_query_2 = e.query()

            self.assertTrue(event_query_1)
            self.assertTrue(event_query_2)


if __name__ == "__main__":
    unittest.main()
