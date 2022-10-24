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


class TestDeviceName(unittest.TestCase):
    def test_device_name_default(self):
        if paddle.is_compiled_with_cuda():
            name = paddle.device.cuda.get_device_name()
            self.assertIsNotNone(name)

    def test_device_name_int(self):
        if paddle.is_compiled_with_cuda():
            name = paddle.device.cuda.get_device_name(0)
            self.assertIsNotNone(name)

    def test_device_name_CUDAPlace(self):
        if paddle.is_compiled_with_cuda():
            name = paddle.device.cuda.get_device_name(paddle.CUDAPlace(0))
            self.assertIsNotNone(name)


class TestDeviceCapability(unittest.TestCase):
    def test_device_capability_default(self):
        if paddle.is_compiled_with_cuda():
            capability = paddle.device.cuda.get_device_capability()
            self.assertIsNotNone(capability)

    def test_device_capability_int(self):
        if paddle.is_compiled_with_cuda():
            capability = paddle.device.cuda.get_device_capability(0)
            self.assertIsNotNone(capability)

    def test_device_capability_CUDAPlace(self):
        if paddle.is_compiled_with_cuda():
            capability = paddle.device.cuda.get_device_capability(
                paddle.CUDAPlace(0)
            )
            self.assertIsNotNone(capability)


if __name__ == "__main__":
    unittest.main()
