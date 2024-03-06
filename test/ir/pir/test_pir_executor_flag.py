# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

from paddle.base.framework import in_cinn_mode, in_pir_executor_mode


class TestPIRModeFlags(unittest.TestCase):
    def test_pir_mode_flags(self):
        self.assertTrue(in_pir_executor_mode())
        os.environ["FLAGS_enable_pir_in_executor"] = "false"
        self.assertFalse(in_pir_executor_mode())


class TestCinnModeFlags(unittest.TestCase):
    def test_cinn_mode_flags(self):
        self.assertFalse(in_cinn_mode())
        os.environ["FLAGS_use_cinn"] = "true"
        self.assertTrue(in_cinn_mode())


if __name__ == '__main__':
    unittest.main()
