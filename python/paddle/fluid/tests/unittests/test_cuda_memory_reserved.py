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
import numpy as np
from paddle.fluid import core
from paddle.device.cuda import device_count, memory_reserved
from paddle.fluid.framework import _test_eager_guard, _in_legacy_dygraph


class TestMemoryreserved(unittest.TestCase):
    def func_test_memory_reserved(self, device=None):
        if core.is_compiled_with_cuda():
            tensor = paddle.zeros(shape=[256])
            alloc_size = 4 * 256  # 256 float32 data, with 4 bytes for each one
            memory_reserved_size = memory_reserved(device)
            self.assertEqual(memory_reserved_size, alloc_size)

    def test_memory_reserved(self):
        with _test_eager_guard():
            self.func_test_memory_reserved()
        self.func_test_memory_reserved()

    def func_test_memory_reserved_for_all_places(self):
        if core.is_compiled_with_cuda():
            gpu_num = device_count()
            for i in range(gpu_num):
                paddle.device.set_device("gpu:" + str(i))
                self.func_test_memory_reserved(core.CUDAPlace(i))
                self.func_test_memory_reserved(i)
                self.func_test_memory_reserved("gpu:" + str(i))

    def test_memory_reserved_for_all_places(self):
        with _test_eager_guard():
            self.func_test_memory_reserved_for_all_places()
        self.func_test_memory_reserved_for_all_places()

    def func_test_memory_reserved_exception(self):
        if core.is_compiled_with_cuda():
            wrong_device = [
                core.CPUPlace(), device_count() + 1, -2, 0.5, "gpu1", "npu"
            ]
            for device in wrong_device:
                with self.assertRaises(BaseException):
                    memory_reserved(device)
        else:
            with self.assertRaises(BaseException):
                memory_reserved()

    def test_memory_reserved_exception(self):
        with _test_eager_guard():
            self.func_test_memory_reserved_exception()
        self.func_test_memory_reserved_exception()


if __name__ == "__main__":
    unittest.main()
