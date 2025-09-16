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


class TestVMMAllocator(unittest.TestCase):
    def setUp(self):
        paddle.set_flags({'FLAGS_use_virtual_memory_auto_growth': True})
        paddle.set_flags({'FLAGS_enable_compact_mem': True})
        self.cmds = [
            ["Alloc", 15 * 1000**3, "0x100000000"],
            ["Alloc", 15 * 1000**3, "0x100000001"],
            ["Alloc", 15 * 1000**3, "0x100000002"],
            ["Alloc", 15 * 1000**3, "0x100000003"],
            ["Alloc", 15 * 1000**3, "0x100000004"],
            ["Free", 15 * 1000**3, "0x100000001"],
            ["Free", 15 * 1000**3, "0x100000003"],
            ["Alloc", 30 * 1000**3, "0x100000005"],
        ]

    def test_paddle(self):
        params = {}
        old_tensor4, old_tensor4_ptr, new_tensor4, new_tensor4_ptr = 0, 0, 0, 0

        for op, size, ptr in self.cmds:
            paddle.device.synchronize()
            paddle_reserved1 = paddle.device.cuda.memory_reserved() // (1000**2)

            if op == "Alloc":
                params[ptr] = paddle.randn(
                    [int(int(size) / 4)], dtype='float32'
                )
            if op == "Free" and ptr in params:
                del params[ptr]

            if ptr == '0x100000004':
                old_tensor4 = params['0x100000004'].numpy()[0:100]
                old_tensor4_ptr = hex(params['0x100000004'].data_ptr())

            paddle.device.synchronize()
            paddle_reserved2 = paddle.device.cuda.memory_reserved() // (1000**2)
            paddle_allocated2 = paddle.device.cuda.memory_allocated() // (
                1000**2
            )
            paddle_max_reserved = paddle.device.cuda.max_memory_reserved() // (
                1000**2
            )
            paddle_max_allocated = (
                paddle.device.cuda.max_memory_allocated() // (1000**2)
            )

            print(
                f"reserved = {paddle_reserved2} allocated = {paddle_allocated2} auto growth = {paddle_reserved2 - paddle_reserved1} max_allocated = {paddle_max_allocated} max_reserved = {paddle_max_reserved}"
            )
        new_tensor4 = params['0x100000004'].numpy()[0:100]
        new_tensor4_ptr = hex(params['0x100000004'].data_ptr())
        np.testing.assert_array_equal(old_tensor4, new_tensor4)
        assert old_tensor4_ptr != new_tensor4_ptr


if __name__ == '__main__':
    unittest.main()
