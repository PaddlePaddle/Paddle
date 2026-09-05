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

import unittest

import paddle
from paddle.base import core


class TestEmptyCache(unittest.TestCase):
    def test_empty_cache(self):
        x = paddle.randn((2, 10, 12)).astype('float32')
        del x
        self.assertIsNone(paddle.device.cuda.empty_cache())


class TestEmptyPinnedCache(unittest.TestCase):
    def test_empty_pinned_cache_no_crash(self):
        """Test that empty_pinned_cache doesn't crash when no pinned memory allocated"""
        if not paddle.is_compiled_with_cuda():
            return

        # Call empty_pinned_cache without any pinned memory allocation
        # Should not crash
        core.cuda_pinned_empty_cache()

    def test_empty_pinned_cache_with_allocation(self):
        """Test that empty_pinned_cache works correctly with pinned memory"""
        if not paddle.is_compiled_with_cuda():
            return

        # Allocate pinned memory
        x = paddle.randn((128, 1024, 1024)).astype('float32')
        x = x.pin_memory()

        # Delete the tensor
        del x

        # Call empty_pinned_cache, should not crash
        core.cuda_pinned_empty_cache()


if __name__ == '__main__':
    unittest.main()
