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

    def test_amp_default(self):
        for device in self.device_list:
            self.assertFalse(paddle.is_autocast_enabled(device))
            self.assertFalse(paddle.amp.is_autocast_enabled(device))

    def test_amp_autocast_true(self):
        for device in self.device_list:
            with paddle.amp.auto_cast(True):
                self.assertTrue(paddle.is_autocast_enabled(device))
                self.assertTrue(paddle.amp.is_autocast_enabled(device))

            self.assertFalse(paddle.is_autocast_enabled(device))
            self.assertFalse(paddle.amp.is_autocast_enabled(device))

    def test_amp_autocast_false(self):
        for device in self.device_list:
            with paddle.amp.auto_cast(False):
                self.assertFalse(paddle.is_autocast_enabled(device))
                self.assertFalse(paddle.amp.is_autocast_enabled(device))

            self.assertFalse(paddle.is_autocast_enabled(device))
            self.assertFalse(paddle.amp.is_autocast_enabled(device))

    def test_amp_nested_context(self):
        for device in self.device_list:
            with paddle.amp.auto_cast(True):
                self.assertTrue(paddle.is_autocast_enabled(device))
                self.assertTrue(paddle.amp.is_autocast_enabled(device))

                with paddle.amp.auto_cast(False):
                    self.assertFalse(paddle.is_autocast_enabled(device))
                    self.assertFalse(paddle.amp.is_autocast_enabled(device))

                self.assertTrue(paddle.is_autocast_enabled(device))
                self.assertTrue(paddle.amp.is_autocast_enabled(device))
            self.assertFalse(paddle.is_autocast_enabled(device))
            self.assertFalse(paddle.amp.is_autocast_enabled(device))


class TestAutocastStatic(TestAutocast):
    def setUp(self) -> None:
        paddle.enable_static()
        self.device_list = [None, paddle.device.get_device()]


if __name__ == "__main__":
    unittest.main()
