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

<<<<<<< HEAD
import sys
import unittest

sys.path.append("..")

from test_merged_momentum_op_xpu_base import TestMergedMomentumBase
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle
=======
from __future__ import print_function

import unittest
import numpy as np
import sys

sys.path.append("..")

import paddle
import paddle.fluid.core as core

from op_test import OpTest
from op_test_xpu import XPUOpTest
from test_merged_momentum_op_xpu_base import TestMergedMomentumBase
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


class XPUTestMergedMomentumOP(XPUOpTestWrapper):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'merged_momentum'
        self.use_dynamic_create_class = False

    class TestMergedMomentumOp(TestMergedMomentumBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            super().setUp()
            self.set_case()

        def set_case(self):
            self.shapes = [[3, 4], [2, 7], [5, 6, 8]]
            self.place = paddle.fluid.XPUPlace(0)
            self.seed = 1

        def testalltype(self):
            self.check_with_place(self.place, self.in_type)

    class TestMergedMomentum1(TestMergedMomentumOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def set_case(self):
            self.shapes = [[3, 4], [2, 7], [5, 6, 8]]

    class TestMergedMomentum2(TestMergedMomentumOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def set_case(self):
            self.shapes = [[3, 4], [2, 7]]

    class TestMergedMomentum3(TestMergedMomentumOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def set_case(self):
            self.shapes = [[3, 4]]

    class TestMergedMomentum4(TestMergedMomentumOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def set_case(self):
            self.shapes = [[3, 4], [2, 7], [5, 6, 7], [9, 9], [10, 12]]


support_types = get_xpu_op_support_types('merged_momentum')
for stype in support_types:
    create_test_class(globals(), XPUTestMergedMomentumOP, stype)

if __name__ == '__main__':
    unittest.main()
