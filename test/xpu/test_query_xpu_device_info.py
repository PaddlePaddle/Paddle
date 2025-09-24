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
from paddle.device import core


class TestQueryXPUDeviceInfo(unittest.TestCase):
    def test_dygraph(self):
        if core.is_compiled_with_xpu():
            paddle.disable_static()
            dev_num = core.get_xpu_device_count()
            self.assertGreater(
                dev_num,
                0,
                "The environment you run this test does not have any xpu device.",
            )
            for dev_id in range(dev_num):
                self.assertGreaterEqual(
                    core.get_xpu_device_utilization_rate(dev_id), 0
                )
                self.assertGreater(core.get_xpu_device_total_memory(dev_id), 0)
                self.assertGreaterEqual(
                    core.get_xpu_device_used_memory(dev_id), 0
                )
            paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
