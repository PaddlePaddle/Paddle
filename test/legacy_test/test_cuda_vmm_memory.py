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


class TestMemoryreserved(unittest.TestCase):
    def setUp(self):
        if paddle.base.is_compiled_with_cuda():
            paddle.set_flags(
                {
                    'FLAGS_use_virtual_memory_auto_growth': 1,
                }
            )

    def func_test_memory_stats(self):
        if core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm():
            # 256 float32 data, with 4 bytes for each one
            alloc_size = 4 * 256
            # The chunk size of VMM allocator is aligned to granularity, which is at least 2 MB.
            reserved_size = 2 * 1024 * 1024

            tensor1 = paddle.zeros(shape=[256])
            tensor2 = paddle.zeros(shape=[256])
            self.assertEqual(
                paddle.device.cuda.memory_reserved(), reserved_size
            )
            self.assertEqual(
                paddle.device.cuda.memory_allocated(), 2 * alloc_size
            )

            del tensor1
            self.assertEqual(
                paddle.device.cuda.memory_reserved(), reserved_size
            )
            self.assertEqual(paddle.device.cuda.memory_allocated(), alloc_size)
            del tensor2
            self.assertEqual(
                paddle.device.cuda.memory_reserved(), reserved_size
            )
            self.assertEqual(paddle.device.cuda.memory_allocated(), 0)
            self.assertEqual(
                paddle.device.cuda.max_memory_reserved(), 2 * 1024 * 1024
            )
            self.assertEqual(
                paddle.device.cuda.max_memory_allocated(), 2 * 4 * 256
            )

    def test_memory_stats(self):
        self.func_test_memory_stats()


if __name__ == "__main__":
    unittest.main()
