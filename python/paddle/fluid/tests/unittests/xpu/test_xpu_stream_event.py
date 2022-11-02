# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.device import xpu
import paddle

import unittest


class TestSynchronize(unittest.TestCase):
    def test_synchronize(self):
        if paddle.is_compiled_with_xpu():
            self.assertIsNone(xpu.synchronize())
            self.assertIsNone(xpu.synchronize(0))
            self.assertIsNone(xpu.synchronize(paddle.XPUPlace(0)))

            self.assertRaises(ValueError, xpu.synchronize, "xpu:0")


if __name__ == "__main__":
    unittest.main()
