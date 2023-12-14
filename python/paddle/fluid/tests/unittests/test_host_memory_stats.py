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
from paddle.fluid import core

paddle.set_device('cpu')


class TestHostMemoryStats(unittest.TestCase):
    def test_memory_allocated_with_pinned(self, device=None):
        if core.is_compiled_with_cuda():
            tensor = paddle.zeros(shape=[256])
            tensor_pinned = tensor.pin_memory()
            alloc_size = 4 * 256  # 256 float32 data, with 4 bytes for each one
            memory_allocated_size = core.host_memory_stat_current_value(
                "Allocated", 0
            )
            pinned_memory_allocated_size = (
                core.pinned_memory_stat_current_value("Allocated", 0)
            )
            self.assertEqual(memory_allocated_size, alloc_size)
            self.assertEqual(pinned_memory_allocated_size, alloc_size)

            def foo():
                tensor = paddle.zeros(shape=[256])
                tensor_pinned = tensor.pin_memory()
                memory_allocated_size = core.host_memory_stat_current_value(
                    "Allocated", 0
                )
                self.assertEqual(memory_allocated_size, alloc_size * 2)
                max_allocated_size = core.host_memory_stat_peak_value(
                    "Allocated", 0
                )
                self.assertEqual(max_allocated_size, alloc_size * 2)

                pinned_memory_allocated_size = (
                    core.pinned_memory_stat_current_value("Allocated", 0)
                )
                self.assertEqual(pinned_memory_allocated_size, alloc_size * 2)
                pinned_max_allocated_size = core.pinned_memory_stat_peak_value(
                    "Allocated", 0
                )
                self.assertEqual(pinned_max_allocated_size, alloc_size * 2)

            foo()

            memory_allocated_size = core.host_memory_stat_current_value(
                "Allocated", 0
            )
            self.assertEqual(memory_allocated_size, alloc_size)

            max_allocated_size = core.host_memory_stat_peak_value(
                "Allocated", 0
            )
            self.assertEqual(max_allocated_size, alloc_size * 2)

            pinned_memory_allocated_size = (
                core.pinned_memory_stat_current_value("Allocated", 0)
            )
            self.assertEqual(pinned_memory_allocated_size, alloc_size)

            pinned_max_allocated_size = core.pinned_memory_stat_peak_value(
                "Allocated", 0
            )
            self.assertEqual(pinned_max_allocated_size, alloc_size * 2)


class TestHostMemoryStatsAPI(unittest.TestCase):
    def test_memory_allocated_with_pinned(self, device=None):
        if core.is_compiled_with_cuda():
            tensor = paddle.zeros(shape=[256])
            tensor_pinned = tensor.pin_memory()
            alloc_size = 4 * 256  # 256 float32 data, with 4 bytes for each one

            cpu_memory_allocated_size = paddle.device.cpu.memory_allocated()
            cpu_memory_reserved_size = paddle.device.cpu.memory_reserved()
            cpu_max_memory_allocated_size = (
                paddle.device.cpu.max_memory_allocated()
            )
            cpu_max_memory_reserved_size = (
                paddle.device.cpu.max_memory_reserved()
            )

            pinned_memory_allocated_size = (
                paddle.device.cuda.pinned_memory_allocated()
            )
            pinned_memory_reserved_size = (
                paddle.device.cuda.pinned_memory_reserved()
            )
            pinned_max_memory_allocated_size = (
                paddle.device.cuda.max_pinned_memory_allocated()
            )
            pinned_max_memory_reserved_size = (
                paddle.device.cuda.max_pinned_memory_reserved()
            )

            self.assertEqual(cpu_memory_allocated_size, alloc_size)
            self.assertEqual(cpu_memory_reserved_size, alloc_size)
            # since had allocated in TestHostMemoryStats
            self.assertEqual(cpu_max_memory_allocated_size, 2 * alloc_size)
            self.assertEqual(cpu_max_memory_reserved_size, 2 * alloc_size)

            self.assertEqual(pinned_memory_allocated_size, alloc_size)
            # since had allocated in TestHostMemoryStats
            self.assertEqual(pinned_max_memory_allocated_size, 2 * alloc_size)

            def foo():
                tensor = paddle.zeros(shape=[256])
                tensor_pinned = tensor.pin_memory()

                cpu_memory_allocated_size = paddle.device.cpu.memory_allocated()
                cpu_memory_reserved_size = paddle.device.cpu.memory_reserved()
                cpu_max_memory_allocated_size = (
                    paddle.device.cpu.max_memory_allocated()
                )
                cpu_max_memory_reserved_size = (
                    paddle.device.cpu.max_memory_reserved()
                )

                pinned_memory_allocated_size = (
                    paddle.device.cuda.pinned_memory_allocated()
                )
                pinned_memory_reserved_size = (
                    paddle.device.cuda.pinned_memory_reserved()
                )
                pinned_max_memory_allocated_size = (
                    paddle.device.cuda.max_pinned_memory_allocated()
                )
                pinned_max_memory_reserved_size = (
                    paddle.device.cuda.max_pinned_memory_reserved()
                )

                self.assertEqual(cpu_memory_allocated_size, 2 * alloc_size)
                self.assertEqual(cpu_memory_reserved_size, 2 * alloc_size)
                self.assertEqual(cpu_max_memory_allocated_size, 2 * alloc_size)
                self.assertEqual(cpu_max_memory_reserved_size, 2 * alloc_size)

                self.assertEqual(pinned_memory_allocated_size, 2 * alloc_size)
                self.assertEqual(
                    pinned_max_memory_allocated_size, 2 * alloc_size
                )

            foo()

            cpu_memory_allocated_size = paddle.device.cpu.memory_allocated()
            cpu_memory_reserved_size = paddle.device.cpu.memory_reserved()
            cpu_max_memory_allocated_size = (
                paddle.device.cpu.max_memory_allocated()
            )
            cpu_max_memory_reserved_size = (
                paddle.device.cpu.max_memory_reserved()
            )

            pinned_memory_allocated_size = (
                paddle.device.cuda.pinned_memory_allocated()
            )
            pinned_memory_reserved_size = (
                paddle.device.cuda.pinned_memory_reserved()
            )
            pinned_max_memory_allocated_size = (
                paddle.device.cuda.max_pinned_memory_allocated()
            )
            pinned_max_memory_reserved_size = (
                paddle.device.cuda.max_pinned_memory_reserved()
            )

            self.assertEqual(cpu_memory_allocated_size, alloc_size)
            self.assertEqual(cpu_memory_reserved_size, alloc_size)
            self.assertEqual(cpu_max_memory_allocated_size, 2 * alloc_size)
            self.assertEqual(cpu_max_memory_reserved_size, 2 * alloc_size)

            self.assertEqual(pinned_memory_allocated_size, alloc_size)
            self.assertEqual(pinned_max_memory_allocated_size, 2 * alloc_size)


if __name__ == "__main__":
    unittest.main()
