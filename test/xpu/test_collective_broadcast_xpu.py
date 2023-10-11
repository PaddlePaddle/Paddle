#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from get_test_cover_info import XPUOpTestWrapper, create_test_class
from test_collective_base_xpu import TestDistBase

import paddle
from paddle.base import core

paddle.enable_static()


class XPUTestCBroadcastOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'c_broadcast'
        self.use_dynamic_create_class = False

    class TestCBroadcastOp(TestDistBase):
        def _setup_config(self):
            pass

        def test_broadcast(self):
            self.check_with_place(
                "collective_broadcast_op_xpu.py", "broadcast", self.in_type_str
            )


support_types = ["float32"]
for stype in support_types:
    create_test_class(
        globals(),
        XPUTestCBroadcastOP,
        stype,
        ignore_device_version=[core.XPUVersion.XPU1],
    )

if __name__ == '__main__':
    unittest.main()
