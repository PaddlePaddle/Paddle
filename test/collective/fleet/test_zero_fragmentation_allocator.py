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
from paddle.distributed.fleet.utils import ZeroFragmentationAllocatorManager


def prealloc(size):
    paddle.empty([size], dtype='uint8')


class ZeroFragmentationAllocatorTest(unittest.TestCase):
    def setUp(self):
        prealloc(1024 * 1024)
        paddle.base.set_flags({'FLAGS_use_zero_fragmentation': True})

    def test_zero_fragmentation_allocator_basic(self):
        if paddle.device.cuda.device_count() < 1:
            return

        paddle.seed(42)
        data_ref = paddle.randn(shape=[1024, 1024], dtype='float32')

        paddle.seed(42)
        ZeroFragmentationAllocatorManager.allocate_buffer(1024 * 1024 * 6)
        with (
            ZeroFragmentationAllocatorManager.zero_fragmentation_allocator_context()
        ):
            data = paddle.randn(shape=[1024, 1024], dtype='float32')

        np.testing.assert_array_equal(data_ref.numpy(), data.numpy())

        ZeroFragmentationAllocatorManager.allocate_buffer(1024 * 1024 * 6)
        np.testing.assert_array_equal(data_ref.numpy(), data.numpy())

    def test_zero_fragmentation_allocator_fallback(self):
        if paddle.device.cuda.device_count() < 1:
            return

        paddle.seed(42)
        data_ref = paddle.randn(shape=[1024, 1024], dtype='float32')

        paddle.seed(42)
        ZeroFragmentationAllocatorManager.allocate_buffer(1024 * 1024 * 2)
        with (
            ZeroFragmentationAllocatorManager.zero_fragmentation_allocator_context()
        ):
            data = paddle.randn(shape=[1024, 1024], dtype='float32')

    def test_zero_fragmentation_allocator_non_cuda(self):
        paddle.seed(42)
        data_ref = paddle.randn(shape=[1024, 1024], dtype='float32').cpu()

        paddle.seed(42)
        with (
            ZeroFragmentationAllocatorManager.zero_fragmentation_allocator_context()
        ):
            data = paddle.randn(shape=[1024, 1024], dtype='float32').cpu()
            np.testing.assert_array_equal(data_ref.numpy(), data.numpy())

    def test_zero_fragmentation_allocator_multitensors(self):
        ZeroFragmentationAllocatorManager.allocate_buffer(1024 * 1024 * 10)

        tensor_list = []
        with (
            ZeroFragmentationAllocatorManager.zero_fragmentation_allocator_context()
        ):
            for _ in range(10):
                tensor_list.append(paddle.ones((1024, 1024)))

        for _ in range(10):
            tensor_list.append(paddle.ones((1024, 1024)))

        with (
            ZeroFragmentationAllocatorManager.zero_fragmentation_allocator_context()
        ):
            for _ in range(10):
                tensor_list.append(paddle.ones((1024, 1024)))

        paddle.core.allocator_dump_fragmentation_metric(
            paddle.framework._current_expected_place()
        )


if __name__ == '__main__':
    unittest.main()
