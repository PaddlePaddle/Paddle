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


@unittest.skipIf(paddle.device.get_device() == "cpu", "Skip AMP test on CPU")
class TestAutocast(unittest.TestCase):
    def setUp(self) -> None:
        paddle.disable_static()
        self.device_list = [None, paddle.device.get_device()]
        self.default_dtype = "float16"

    def do_test(self, device, expected_type):
        self.assertTrue(paddle.get_autocast_dtype(device) == expected_type)
        self.assertTrue(paddle.get_autocast_gpu_dtype() == expected_type)
        self.assertTrue(paddle.amp.get_autocast_dtype(device) == expected_type)
        self.assertTrue(paddle.amp.get_autocast_gpu_dtype() == expected_type)
        self.assertTrue(paddle.amp.get_autocast_cpu_dtype() == expected_type)
        self.assertTrue(
            paddle.amp.get_autocast_cpu_dtype(device) == expected_type
        )

    def test_amp_default(self):
        for device in self.device_list:
            self.do_test(device, self.default_dtype)

    def test_amp_autocast_fp16(self):
        for device in self.device_list:
            with paddle.amp.auto_cast(True, dtype="float16"):
                self.do_test(device, "float16")
            self.do_test(device, self.default_dtype)

    def test_amp_autocast_bf16(self):
        for device in self.device_list:
            with paddle.amp.auto_cast(True, dtype="bfloat16"):
                self.do_test(device, "bfloat16")
            self.do_test(device, self.default_dtype)

    def test_amp_autocast_false_bf16(self):
        for device in self.device_list:
            with paddle.amp.auto_cast(True, dtype="bfloat16"):
                self.do_test(device, "bfloat16")
            self.do_test(device, self.default_dtype)

    def test_amp_nested_context(self):
        for device in self.device_list:
            with paddle.amp.auto_cast(True, dtype="bfloat16"):
                self.do_test(device, "bfloat16")
                with paddle.amp.auto_cast(True, dtype="float16"):
                    self.do_test(device, "float16")
                self.do_test(device, "bfloat16")
            self.do_test(device, self.default_dtype)


class TestAutocastStatic(TestAutocast):
    def setUp(self) -> None:
        paddle.enable_static()
        self.device_list = [None, paddle.device.get_device()]
        self.default_dtype = "float16"


if __name__ == "__main__":
    unittest.main()
