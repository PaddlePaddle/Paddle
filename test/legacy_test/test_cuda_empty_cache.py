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

import platform
import unittest

import paddle
from paddle.base import core


def get_process_memory_mb():
    """Get current process memory usage in MB from /proc/self/status (Linux only)"""
    if platform.system() == "Windows":
        raise NotImplementedError(
            "get_process_memory_mb is not supported on Windows"
        )
    with open('/proc/self/status', 'r') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                # VmRSS is in kB, convert to MB
                return int(line.split()[1]) / 1024
    return 0


class TestEmptyCache(unittest.TestCase):
    def test_empty_cache(self):
        x = paddle.randn((2, 10, 12)).astype('float32')
        del x
        self.assertIsNone(paddle.device.cuda.empty_cache())


@unittest.skipIf(
    platform.system() == "Windows",
    "Skip on Windows because /proc/self/status is not available.",
)
class TestEmptyPinnedCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if paddle.is_compiled_with_cuda():
            # Enable auto_growth pinned allocator
            paddle.set_flags({'FLAGS_use_auto_growth_pinned_allocator': True})

    def test_empty_pinned_cache(self):
        if not paddle.is_compiled_with_cuda():
            return

        # Allocate a large amount of pinned memory (~2GB) to reduce measurement fluctuation
        # 512 * 1024 * 1024 * 4 bytes = 2GB
        mem_before = get_process_memory_mb()
        x = paddle.randn((512, 1024, 1024)).astype('float32')
        x = x.pin_memory()
        mem_after_alloc = get_process_memory_mb()

        # Verify memory increased after allocation
        self.assertGreater(mem_after_alloc, mem_before)

        # Delete the tensor, memory should be held by allocator pool
        del x
        mem_after_del = get_process_memory_mb()

        # Memory should still be held (not returned to OS)
        # Allow 50MB tolerance for measurement fluctuation
        self.assertGreaterEqual(mem_after_del, mem_after_alloc - 50)

        # Call empty_pinned_cache, memory should be returned to OS
        core.cuda_pinned_empty_cache()
        mem_after_empty = get_process_memory_mb()

        # Memory should be released (significantly reduced)
        # Expect at least 1.5GB reduction (allowing some overhead)
        self.assertLess(mem_after_empty, mem_after_del - 1500)

    def test_empty_pinned_cache_no_crash(self):
        """Test that empty_pinned_cache doesn't crash when no pinned memory allocated"""
        if not paddle.is_compiled_with_cuda():
            return

        # Call empty_pinned_cache without any pinned memory allocation
        # Should not crash
        core.cuda_pinned_empty_cache()


if __name__ == '__main__':
    unittest.main()
