# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base

# it should be set at the beginning
if base.is_compiled_with_cuda():
    paddle.set_flags(
        {
            'FLAGS_allocator_strategy': 'auto_growth',
            'FLAGS_auto_growth_chunk_size_in_mb': 10,
            # Async allocator does not support auto growth allocator.
            'FLAGS_use_cuda_malloc_async_allocator': 0,
        }
    )


class TestMemoryLimit(unittest.TestCase):
    def setUp(self):
        self._limit = 10
        if base.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_gpu_memory_limit_mb': 10})

    def test_allocate(self):
        if not base.is_compiled_with_cuda():
            return

        other_dim = int(1024 * 1024 / 4)

        place = base.CUDAPlace(0)
        t = base.DenseTensor()
        t.set(
            np.ndarray([int(self._limit / 2), other_dim], dtype='float32'),
            place,
        )
        del t

        t = base.DenseTensor()
        large_np = np.ndarray([2 * self._limit, other_dim], dtype='float32')

        try:
            t.set(large_np, place)
            self.assertTrue(False)
        except:
            self.assertTrue(True)


class TestChunkSize(unittest.TestCase):
    def test_allocate(self):
        if not base.is_compiled_with_cuda():
            return

        paddle.rand([1024])
        reserved, allocated = (
            paddle.device.cuda.max_memory_reserved(),
            paddle.device.cuda.max_memory_allocated(),
        )

        self.assertEqual(reserved, 1024 * 1024 * 10)
        self.assertEqual(allocated, 1024 * 4)


if __name__ == '__main__':
    unittest.main()
