# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.device.cuda import (
    device_count,
    max_memory_allocated,
    memory_allocated,
    reset_max_memory_allocated,
)


class TestResetMaxMemoryAllocated(unittest.TestCase):
    def func_test_reset_max_memory_allocated(self, device=None):
        if core.is_compiled_with_cuda():
            alloc_time = 100
            max_alloc_size = 10000
            for i in range(alloc_time):
                # first alloc
                shape = paddle.randint(
                    low=max_alloc_size, high=max_alloc_size * 2
                )
                tensor = paddle.zeros(shape)
                peak_memory_allocated_size_first = max_memory_allocated(device)

                del shape
                del tensor

                # second alloc
                shape = paddle.randint(low=0, high=max_alloc_size)
                tensor = paddle.zeros(shape)

                # reset peak memory stats
                reset_max_memory_allocated(device)

                peak_memory_allocated_size_second = max_memory_allocated(device)
                self.assertEqual(
                    peak_memory_allocated_size_second, memory_allocated(device)
                )
                self.assertLess(
                    peak_memory_allocated_size_second,
                    peak_memory_allocated_size_first,
                )

                del shape
                del tensor

    def test_reset_max_memory_allocated_for_all_places(self):
        if core.is_compiled_with_cuda():
            gpu_num = device_count()
            for i in range(gpu_num):
                paddle.device.set_device("gpu:" + str(i))
                self.func_test_reset_max_memory_allocated(core.CUDAPlace(i))
                self.func_test_reset_max_memory_allocated(i)
                self.func_test_reset_max_memory_allocated("gpu:" + str(i))

    def test_reset_max_memory_allocated_exception(self):
        if core.is_compiled_with_cuda():
            wrong_device = [
                core.CPUPlace(),
                device_count() + 1,
                -2,
                0.5,
                "gpu1",
            ]
            for device in wrong_device:
                with self.assertRaises(BaseException):  # noqa: B017
                    reset_max_memory_allocated(device)
        else:
            with self.assertRaises(ValueError):
                reset_max_memory_allocated()


if __name__ == "__main__":
    unittest.main()
