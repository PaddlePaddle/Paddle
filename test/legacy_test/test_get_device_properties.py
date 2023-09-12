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

from paddle.base import core
from paddle.device.cuda import device_count, get_device_properties


class TestGetDeviceProperties(unittest.TestCase):
    def test_get_device_properties_default(self):
        if core.is_compiled_with_cuda():
            props = get_device_properties()
            self.assertIsNotNone(props)

    def test_get_device_properties_str(self):
        if core.is_compiled_with_cuda():
            props = get_device_properties('gpu:0')
            self.assertIsNotNone(props)

    def test_get_device_properties_int(self):
        if core.is_compiled_with_cuda():
            gpu_num = device_count()
            for i in range(gpu_num):
                props = get_device_properties(i)
                self.assertIsNotNone(props)

    def test_get_device_properties_CUDAPlace(self):
        if core.is_compiled_with_cuda():
            device = core.CUDAPlace(0)
            props = get_device_properties(device)
            self.assertIsNotNone(props)


class TestGetDevicePropertiesError(unittest.TestCase):
    def test_error_api(self):
        if core.is_compiled_with_cuda():

            def test_device_indexError_error():
                device_error = device_count() + 1
                props = get_device_properties(device_error)

            self.assertRaises(IndexError, test_device_indexError_error)

            def test_device_value_error1():
                device_error = 'gpu1'
                props = get_device_properties(device_error)

            self.assertRaises(ValueError, test_device_value_error1)

            def test_device_value_error2():
                device_error = float(device_count())
                props = get_device_properties(device_error)

            self.assertRaises(ValueError, test_device_value_error2)


if __name__ == "__main__":
    unittest.main()
