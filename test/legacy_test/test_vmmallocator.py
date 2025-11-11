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


@unittest.skipIf(
    (not paddle.is_compiled_with_cuda()) or paddle.is_compiled_with_rocm,
    'should compile with cuda.',
)
class TestVMMAllocator(unittest.TestCase):
    def setUp(self):
        self.GB = 1000**3
        self.MB = 1000**2
        paddle.set_flags({'FLAGS_dump_vmm_allocation_info': True})
        paddle.set_flags({'FLAGS_use_virtual_memory_auto_growth': True})
        paddle.set_flags({'FLAGS_native_compact': True})
        self.cmds = [
            ["Alloc", 6 * self.GB, "0x100000000"],
            ["Alloc", 11 * self.GB, "0x100000001"],
            ["Alloc", 1 * self.GB, "0x100000002"],
            ["Alloc", 10 * self.GB, "0x100000003"],
            ["Free", 6 * self.GB, "0x100000000"],
            ["Free", 10 * self.GB, "0x100000003"],
            ["Alloc", 17 * self.GB, "0x100000004"],
        ]

    def test_vmm_allocator(self):
        params = {}
        old_tensor1, old_tensor1_ptr, new_tensor1, new_tensor1_ptr = 0, 0, 0, 0

        for op, size, ptr in self.cmds:
            paddle.device.synchronize()
            paddle_reserved1 = paddle.device.cuda.memory_reserved() // self.MB

            if op == "Alloc":
                params[ptr] = paddle.randn(
                    [int(int(size) / 4)], dtype='float32'
                )
            if op == "Free" and ptr in params:
                del params[ptr]

            if ptr == '0x100000001':
                old_tensor1 = params['0x100000001'].numpy()[0:100]
                old_tensor1_ptr = hex(params['0x100000001'].data_ptr())

            paddle.device.synchronize()
            paddle_reserved2 = paddle.device.cuda.memory_reserved() // self.MB
            paddle_allocated2 = paddle.device.cuda.memory_allocated() // self.MB
            paddle_max_reserved = (
                paddle.device.cuda.max_memory_reserved() // self.MB
            )
            paddle_max_allocated = (
                paddle.device.cuda.max_memory_allocated() // self.MB
            )

            print(
                f"reserved = {paddle_reserved2} allocated = {paddle_allocated2} auto growth = {paddle_reserved2 - paddle_reserved1} max_allocated = {paddle_max_allocated} max_reserved = {paddle_max_reserved}"
            )
        new_tensor1 = params['0x100000001'].numpy()[0:100]
        new_tensor1_ptr = hex(params['0x100000001'].data_ptr())
        np.testing.assert_array_equal(old_tensor1, new_tensor1)
        assert old_tensor1_ptr != new_tensor1_ptr


if __name__ == '__main__':
    unittest.main()
