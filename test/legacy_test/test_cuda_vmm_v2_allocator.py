# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import gc
import unittest

import paddle


class TestCUDAVMMV2Allocator(unittest.TestCase):
    def test_allocator_facade_creates_vmm_v2_allocator(self):
        flags = paddle.get_flags(
            [
                "FLAGS_allocator_strategy",
                "FLAGS_use_vmm_auto_growth_best_fit_allocator_v2",
                "FLAGS_vmm_v2_large_pool_handle_size_in_mb",
            ]
        )
        self.assertEqual(flags["FLAGS_allocator_strategy"], "auto_growth")
        self.assertTrue(
            flags["FLAGS_use_vmm_auto_growth_best_fit_allocator_v2"]
        )
        self.assertGreater(
            flags["FLAGS_vmm_v2_large_pool_handle_size_in_mb"], 0
        )

        x = paddle.zeros([1024], dtype="float32")
        paddle.device.synchronize()
        self.assertEqual(x.shape, [1024])
        self.assertGreater(paddle.device.cuda.memory_reserved(), 0)

    def test_public_api_allocate_free_path(self):
        # Exercise the real allocator stack instead of constructing allocator
        # internals directly:
        #   AllocatorFacade -> RetryAllocator -> StreamSafeCUDAAllocator
        #   -> VMMAutoGrowthBestFitMultiPoolAllocatorV2.
        # Explicitly dropping tensors covers the stream-safe free path that
        # records VMM v2 remap-safety metadata for later OOM compaction.
        tensors = [
            paddle.zeros([1024 * 1024], dtype="float32") for _ in range(4)
        ]
        paddle.device.synchronize()
        reserved_before_free = paddle.device.cuda.memory_reserved()
        self.assertGreater(reserved_before_free, 0)

        del tensors
        gc.collect()
        paddle.device.synchronize()

        y = paddle.zeros([1024 * 1024], dtype="float32")
        paddle.device.synchronize()
        self.assertEqual(y.shape, [1024 * 1024])


if __name__ == "__main__":
    unittest.main()
