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

# Test set_value op in static mode

import sys
import unittest

import numpy as np

sys.path.append("../")
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle
from paddle.fluid.framework import _test_eager_guard


class XPUTestSetValueOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'set_value'
        self.use_dynamic_create_class = False

    class TestSetValueOp(XPUOpTest):
        def setUp(self):
            paddle.enable_static()
            self.__class__.op_type = "set_value"
            self.place = paddle.XPUPlace(0)
            self.shape = [2]
            self.value = 6
            self.dtype = "float32"
            self.__class__.dtype = self.dtype
            self.data = np.ones(self.shape).astype(self.dtype)
            self.program = paddle.static.Program()

        def _call_setitem(self, x):
            x[0] = self.value

        def _get_answer(self):
            self.data[0] = self.value

    class TestSetValueOp1(TestSetValueOp):
        def _run_static(self):
            paddle.enable_static()
            with paddle.static.program_guard(self.program):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                self._call_setitem(x)

            exe = paddle.static.Executor(paddle.XPUPlace(0))
            out = exe.run(self.program, fetch_list=[x])
            paddle.disable_static()
            return out

        def func_test_api(self):
            static_out = self._run_static()
            self._get_answer()

            error_msg = (
                "\nIn {} mode: \nExpected res = \n{}, \n\nbut received : \n{}"
            )
            self.assertTrue(
                (self.data == static_out).all(),
                msg=error_msg.format("static", self.data, static_out),
            )

        def test_api(self):
            with _test_eager_guard():
                self.func_test_api()
            self.func_test_api()


support_types = get_xpu_op_support_types('set_value')
for stype in support_types:
    create_test_class(globals(), XPUTestSetValueOp, stype)

if __name__ == '__main__':
    unittest.main()
