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
from paddle.base import core
from paddle.device.xpu import device_count, memory_used


class TestMemoryUsed(unittest.TestCase):
    def test_memory_used(self, device=None):
        if core.is_compiled_with_xpu():
            tensor = paddle.zeros(shape=[256])
            alloc_size = 4 * 256  # 256 float32 data, with 4 bytes for each one
            memory_used_size = memory_used(device)
            self.assertGreaterEqual(memory_used_size, alloc_size)

    def test_memory_used_for_all_places(self):
        if core.is_compiled_with_xpu():
            xpu_num = device_count()
            for i in range(xpu_num):
                paddle.device.set_device("xpu:" + str(i))
                self.test_memory_used(core.XPUPlace(i))
                self.test_memory_used(i)
                self.test_memory_used("xpu:" + str(i))

    def test_memory_used_exception(self):
        if core.is_compiled_with_xpu():
            wrong_device = [
                core.CPUPlace(),
                device_count() + 1,
                -2,
                0.5,
                "xpu1",
            ]
            for device in wrong_device:
                with self.assertRaises(BaseException):  # noqa: B017
                    memory_used(device)
        else:
            with self.assertRaises(ValueError):
                memory_used()


if __name__ == "__main__":
    unittest.main()
