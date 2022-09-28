#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid import core

from test_collective_base_xpu import TestDistBase

import sys

sys.path.append("..")

from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestCAllgatherOP(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'c_allgather'
        self.use_dynamic_create_class = False

    class TestCAllgatherOp(TestDistBase):

        def _setup_config(self):
            pass

        def test_allgather(self):
            self.check_with_place("collective_allgather_op_xpu.py", "allgather",
                                  self.in_type_str)


support_types = get_xpu_op_support_types('c_allgather')
for stype in support_types:
    create_test_class(globals(),
                      XPUTestCAllgatherOP,
                      stype,
                      ignore_device_version=[core.XPUVersion.XPU1])

if __name__ == '__main__':
    unittest.main()
