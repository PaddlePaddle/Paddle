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
from paddle.cuda import (
    Stream,
    StreamContext,
    _device_to_paddle,
    current_stream,
    get_device_capability,
    get_device_name,
    get_device_properties,
    is_available,
    stream,
    synchronize,
)


class TestCudaCompat(unittest.TestCase):
    # ---------------------
    # _device_to_paddle test
    # ---------------------
    def test_device_to_paddle_none(self):
        self.assertIsNone(_device_to_paddle(None))

    def test_device_to_paddle_int(self):
        self.assertEqual(_device_to_paddle(0), 'gpu:0')
        self.assertEqual(_device_to_paddle(2), 'gpu:2')

    def test_device_to_paddle_str(self):
        self.assertEqual(_device_to_paddle('cuda:0'), 'gpu:0')
        self.assertEqual(_device_to_paddle('gpu:1'), 'gpu:1')

    def test_device_to_paddle_invalid(self):
        with self.assertRaises(TypeError):
            _device_to_paddle(1.5)

    # ---------------------
    # is_available test
    # ---------------------
    def test_is_available(self):
        self.assertIsInstance(is_available(), bool)

    # ---------------------
    # synchronize test
    # ---------------------
    def test_synchronize(self):
        try:
            synchronize(None)
            synchronize(0)
            synchronize('cuda:0')
            synchronize('gpu:0')
        except Exception as e:
            self.fail(f"synchronize raised Exception {e}")

    # ---------------------
    # current_stream test
    # ---------------------
    def test_current_stream(self):
        stream = current_stream(None)
        self.assertIsNotNone(stream)
        stream = current_stream(0)
        self.assertIsNotNone(stream)

    # ---------------------
    # get_device_properties test
    # ---------------------
    def test_get_device_properties(self):
        props = get_device_properties(0)
        self.assertTrue(hasattr(props, 'name'))
        self.assertTrue(hasattr(props, 'total_memory'))

    # ---------------------
    # get_device_name / get_device_capability test
    # ---------------------
    def test_device_name_and_capability(self):
        name = get_device_name(0)
        self.assertIsInstance(name, str)

        cap = get_device_capability(0)
        self.assertIsInstance(cap, tuple)
        self.assertEqual(len(cap), 2)

    def test_stream_creation(self):
        s = Stream()
        s1 = paddle.Stream()  # test paddle.Stream
        self.assertIsInstance(s, paddle.device.Stream)
        self.assertIsInstance(s1, paddle.device.Stream)

    def test_stream_context(self):
        s = Stream(device='gpu', priority=2)
        with stream(s):
            ctx = stream(s)
            self.assertIsInstance(ctx, StreamContext)
            current = current_stream()
            self.assertEqual(current.stream_base, s.stream_base)

    def test_nested_streams(self):
        s1 = Stream()
        s2 = Stream()
        with stream(s1):
            with stream(s2):
                current = paddle.cuda.current_stream()
                self.assertEqual(current.stream_base, s2.stream_base)
            current = paddle.cuda.current_stream()
            self.assertEqual(current.stream_base, s1.stream_base)


if __name__ == '__main__':
    unittest.main()
