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

import paddle
import unittest
from paddle.fluid import core
from paddle.device.cuda import device_count, memory_allocated, max_memory_allocated


class TestMaxMemoryAllocated(unittest.TestCase):
    def test_max_memory_allocated(self, device=None):
        if core.is_compiled_with_cuda():
            alloc_time = 100
            max_alloc_size = 10000
            peak_memory_allocated_size = max_memory_allocated(device)
            for i in range(alloc_time):
                shape = paddle.randint(max_alloc_size)
                tensor = paddle.zeros(shape)
                peak_memory_allocated_size = max(peak_memory_allocated_size,
                                                 memory_allocated(device))
                del shape
                del tensor

            self.assertEqual(peak_memory_allocated_size,
                             max_memory_allocated(device))

    def test_max_memory_allocated_for_all_places(self):
        if core.is_compiled_with_cuda():
            gpu_num = device_count()
            for i in range(gpu_num):
                paddle.device.set_device("gpu:" + str(i))
                self.test_max_memory_allocated(core.CUDAPlace(i))
                self.test_max_memory_allocated(i)
                self.test_max_memory_allocated("gpu:" + str(i))

    def test_max_memory_allocated_exception(self):
        if core.is_compiled_with_cuda():
            wrong_device = [
                core.CPUPlace(), device_count() + 1, -2, 0.5, "gpu1", "npu"
            ]
            for device in wrong_device:
                with self.assertRaises(BaseException):
                    max_memory_allocated(device)
        else:
            with self.assertRaises(BaseException):
                max_memory_allocated()


if __name__ == "__main__":
    unittest.main()
