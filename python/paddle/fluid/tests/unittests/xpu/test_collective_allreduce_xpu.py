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


class XPUTestCAllreduceOP(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'c_allreduce_sum'
        self.use_dynamic_create_class = False

    class TestCAllreduceOp(TestDistBase):

        def _setup_config(self):
            pass

        def test_allreduce(self):
            self.check_with_place("collective_allreduce_op_xpu.py", "allreduce",
                                  self.in_type_str)


support_types = get_xpu_op_support_types('c_allreduce_sum')
for stype in support_types:
    create_test_class(globals(),
                      XPUTestCAllreduceOP,
                      stype,
                      ignore_device_version=[core.XPUVersion.XPU1])

if __name__ == '__main__':
    unittest.main()
