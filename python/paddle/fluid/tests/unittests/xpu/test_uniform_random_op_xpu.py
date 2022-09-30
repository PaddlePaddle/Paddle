#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys

sys.path.append("..")
import subprocess
import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from test_uniform_random_op import TestUniformRandomOp, TestUniformRandomOpSelectedRows

paddle.enable_static()


class TestXPUUniformRandomOp(TestUniformRandomOp):

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            outs = self.calc_output(place)
            outs = [np.array(out) for out in outs]
            outs.sort(key=len)
            self.verify_output(outs)


class TestXPUUniformRandomOpSelectedRows(TestUniformRandomOpSelectedRows):

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
