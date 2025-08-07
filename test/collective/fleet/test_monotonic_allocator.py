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
from paddle.distributed.fleet.utils import MonotonicAllocatorManager


class MonotonicAllocatorTest(unittest.TestCase):
    def test_monotonic_allocator_basic(self):
        if paddle.device.cuda.device_count() < 1:
            return

        paddle.seed(42)
        data_ref = paddle.randn(shape=[1024, 1024], dtype='float32')

        paddle.seed(42)
        MonotonicAllocatorManager.allocate_buffer(1024 * 1024 * 6)
        with MonotonicAllocatorManager.switch_to_monotonic_allocator():
            data = paddle.randn(shape=[1024, 1024], dtype='float32')

        MonotonicAllocatorManager.deallocate_buffer()
        np.testing.assert_array_equal(data_ref.numpy(), data.numpy())

    def test_monotonic_allocator_out_of_memory(self):
        if paddle.device.cuda.device_count() < 1:
            return

        paddle.seed(42)
        data_ref = paddle.randn(shape=[1024, 1024], dtype='float32')

        paddle.seed(42)
        MonotonicAllocatorManager.allocate_buffer(1024 * 1024 * 2)
        with MonotonicAllocatorManager.switch_to_monotonic_allocator():
            data = paddle.randn(shape=[1024, 1024], dtype='float32')

        MonotonicAllocatorManager.deallocate_buffer()
        np.testing.assert_array_equal(data_ref.numpy(), data.numpy())

    def test_monotonic_allocator_reset(self):
        if paddle.device.cuda.device_count() < 1:
            return

        paddle.seed(42)
        MonotonicAllocatorManager.allocate_buffer(1024 * 1024 * 6)
        with MonotonicAllocatorManager.switch_to_monotonic_allocator():
            data = paddle.randn(shape=[1024, 1024], dtype='float32')

        data_clone = data.clone()
        MonotonicAllocatorManager.reset_buffer()

        paddle.seed(42)
        with MonotonicAllocatorManager.switch_to_monotonic_allocator():
            new_data = paddle.randn(shape=[1024, 1024], dtype='float32')

        np.testing.assert_array_equal(data_clone.numpy(), new_data.numpy())
        MonotonicAllocatorManager.deallocate_buffer()

    def test_monotonic_allocator_non_cuda(self):
        paddle.seed(42)
        data_ref = paddle.randn(shape=[1024, 1024], dtype='float32').cpu()

        paddle.seed(42)
        with MonotonicAllocatorManager.switch_to_monotonic_allocator():
            data = paddle.randn(shape=[1024, 1024], dtype='float32').cpu()
            np.testing.assert_array_equal(data_ref.numpy(), data.numpy())


if __name__ == '__main__':
    unittest.main()
