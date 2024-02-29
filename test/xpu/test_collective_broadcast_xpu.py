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

from get_test_cover_info import get_xpu_op_support_types
from xpu.test_collective_api_base import TestDistBase

import paddle
from paddle import core

paddle.enable_static()


class TestCBroadcastOp(TestDistBase):
    def _setup_config(self):
        pass

    @unittest.skipIf(
        not core.is_compiled_with_xpu() or paddle.device.xpu.device_count() < 2,
        "run test when having at least 2 XPUs.",
    )
    def test_broadcast(self):
        support_types = get_xpu_op_support_types('c_broadcast')
        for dtype in support_types:
            self.check_with_place(
                "collective_broadcast_api.py",
                "broadcast",
                dtype=dtype,
            )

    @unittest.skipIf(
        not core.is_compiled_with_xpu() or paddle.device.xpu.device_count() < 2,
        "run test when having at least 2 XPUs.",
    )
    def test_broadcast_dygraph(self):
        support_types = get_xpu_op_support_types('c_broadcast')
        for dtype in support_types:
            self.check_with_place(
                "collective_broadcast_api_dygraph.py",
                "broadcast",
                static_mode="0",
                dtype=dtype,
            )


if __name__ == '__main__':
    unittest.main()
