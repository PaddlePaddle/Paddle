#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from test_collective_base_xpu import TestDistBase

import paddle
from paddle.base import core

paddle.enable_static()


class XPUTestCConcatOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'c_concat'
        self.use_dynamic_create_class = False

    class TestConcatOp(TestDistBase):
        def _setup_config(self):
            pass

        def test_concat(self, col_type="c_concat"):
            self.check_with_place(
                "collective_concat_op.py", col_type, self.in_type_str
            )


support_types = get_xpu_op_support_types('c_concat')
for stype in support_types:
    create_test_class(
        globals(),
        XPUTestCConcatOp,
        stype,
        ignore_device_version=[core.XPUVersion.XPU1],
    )

if __name__ == '__main__':
    unittest.main()
