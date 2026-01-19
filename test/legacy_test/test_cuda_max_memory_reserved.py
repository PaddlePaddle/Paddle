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

from op_test import get_device_place, is_custom_device

import paddle
from paddle.base import core
from paddle.device.cuda import (
    device_count,
    max_memory_reserved,
    memory_reserved,
)


class TestMaxMemoryreserved(unittest.TestCase):
    def test_max_memory_reserved(self, device=None):
        if core.is_compiled_with_cuda() or is_custom_device():
            alloc_time = 100
            max_alloc_size = 10000
            peak_memory_reserved_size = max_memory_reserved(device)
            for i in range(alloc_time):
                shape = paddle.randint(max_alloc_size)
                tensor = paddle.zeros(shape)
                peak_memory_reserved_size = max(
                    peak_memory_reserved_size, memory_reserved(device)
                )
                del shape
                del tensor

            self.assertEqual(
                peak_memory_reserved_size, max_memory_reserved(device)
            )

    def test_max_memory_reserved_for_all_places(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            gpu_num = device_count()
            for i in range(gpu_num):
                paddle.device.set_device("gpu:" + str(i))
                self.test_max_memory_reserved(get_device_place(i))
                self.test_max_memory_reserved(i)
                self.test_max_memory_reserved("gpu:" + str(i))

    def test_max_memory_reserved_exception(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            wrong_device = [
                core.CPUPlace(),
                device_count() + 1,
                -2,
                0.5,
                "gpu1",
            ]
            for device in wrong_device:
                with self.assertRaises(BaseException):  # noqa: B017
                    max_memory_reserved(device)
        else:
            with self.assertRaises(ValueError):
                max_memory_reserved()


if __name__ == "__main__":
    unittest.main()
