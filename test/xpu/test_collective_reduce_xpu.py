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

import os

os.environ['FLAGS_enable_pir_api'] = '0'

import unittest

from get_test_cover_info import get_xpu_op_support_types
from xpu.test_collective_api_base import TestDistBase

import paddle
from paddle import core

paddle.enable_static()


class TestCollectiveReduceAPI(TestDistBase):
    def _setup_config(self):
        pass

    # TODO(lijin23): fix bugs in KL3
    @unittest.skipIf(
        not core.is_compiled_with_xpu() or paddle.device.xpu.device_count() < 2,
        "run test when having at least 2 XPUs.",
    )
    def test_reduce(self):
        support_types = get_xpu_op_support_types('reduce')
        for dtype in support_types:
            self.check_with_place(
                "collective_reduce_api.py",
                "reduce",
                dtype=dtype,
            )

    @unittest.skipIf(
        not core.is_compiled_with_xpu() or paddle.device.xpu.device_count() < 2,
        "run test when having at least 2 XPUs.",
    )
    def test_reduce_dygraph(self):
        support_types = get_xpu_op_support_types('reduce')
        for dtype in support_types:
            self.check_with_place(
                "collective_reduce_api_dygraph.py",
                "reduce",
                static_mode="0",
                dtype=dtype,
            )


if __name__ == '__main__':
    unittest.main()
