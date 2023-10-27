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
from paddle.device.cuda import device_count, memory_reserved


class TestMemoryreserved(unittest.TestCase):
    def func_test_memory_reserved(self, device=None):
        if core.is_compiled_with_cuda():
            tensor = paddle.zeros(shape=[256])
            alloc_size = 4 * 256  # 256 float32 data, with 4 bytes for each one
            memory_reserved_size = memory_reserved(device)
            self.assertEqual(memory_reserved_size, alloc_size)

    def test_memory_reserved_for_all_places(self):
        if core.is_compiled_with_cuda():
            gpu_num = device_count()
            for i in range(gpu_num):
                paddle.device.set_device("gpu:" + str(i))
                self.func_test_memory_reserved(core.CUDAPlace(i))
                self.func_test_memory_reserved(i)
                self.func_test_memory_reserved("gpu:" + str(i))

    def test_memory_reserved_exception(self):
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
                    memory_reserved(device)
        else:
            with self.assertRaises(ValueError):
                memory_reserved()


if __name__ == "__main__":
    unittest.main()
