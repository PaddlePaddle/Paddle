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

import paddle


@unittest.skipIf(
    (not paddle.is_compiled_with_cuda()) or paddle.is_compiled_with_rocm(),
    'should compile with cuda.',
)
class TestAllocatorVisitor(unittest.TestCase):
    def setUp(self):
        self.GB = 1000**3
        self.MB = 1000**2
        self.KB = 1000
        self.cmds = [
            ["Alloc", 1 * self.GB, "0x100000000"],
            ["Alloc", 2 * self.GB, "0x100000001"],
            ["Alloc", 1 * self.GB, "0x100000002"],
            ["Alloc", 2 * self.GB, "0x100000003"],
            ["Free", 1 * self.GB, "0x100000000"],
            ["Free", 2 * self.GB, "0x100000003"],
        ]
        self.cmds2 = [
            ["Alloc", 1 * self.KB, "0x100000000"],
            ["Alloc", 2 * self.MB, "0x100000001"],
            ["Alloc", 1 * self.GB, "0x100000002"],
            ["Alloc", 2 * self.GB, "0x100000003"],
            ["Free", 1 * self.KB, "0x100000000"],
            ["Free", 2 * self.GB, "0x100000003"],
            ["Alloc", 1.5 * self.KB, "0x100000000"],
        ]
        paddle.set_flags(
            {'FLAGS_use_multi_scale_virtual_memory_auto_growth': True}
        )
        paddle.set_flags({'FLAGS_vmm_small_pool_size_in_mb': 1})

    def allocate_cmds(self, cmds):
        params = {}
        for op, size, ptr in cmds:
            paddle.device.synchronize()
            paddle_reserved1 = paddle.device.cuda.memory_reserved() // self.MB

            if op == "Alloc":
                params[ptr] = paddle.randn(
                    [int(int(size) / 4)], dtype='float32'
                )
            if op == "Free" and ptr in params:
                del params[ptr]

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
        return params

    def test_multi_scale_alloc_free(self):
        params = self.allocate_cmds(self.cmds)
        paddle.device.cuda.vmm_max_free_size()

    def test_block_info(self):
        paddle.device.cuda.memory_summary()
        params = self.allocate_cmds(self.cmds)
        x = paddle.device.cuda.vmm_free_block_info()
        y = paddle.device.cuda.vmm_all_block_info()
        self.assertEqual(x[0][0][0], 1000000000)
        self.assertEqual(x[0][1][0], 2002049024)
        self.assertEqual(len(y), 1)  # 1 allocators
        self.assertEqual(len(y[0]), 4)  # 4 blocks

    def test_memory_summary(self):
        paddle.set_flags(
            {'FLAGS_use_multi_scale_virtual_memory_auto_growth': True}
        )
        paddle.set_flags({'FLAGS_vmm_small_pool_size_in_mb': 1})
        paddle.device.cuda.memory_summary()
        params = self.allocate_cmds(self.cmds2)
        paddle.device.cuda.memory_summary()


if __name__ == '__main__':
    unittest.main()
