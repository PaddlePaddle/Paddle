# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

sys.path.append("..")
import unittest

import numpy as np
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle
=======
from __future__ import print_function

import sys

sys.path.append("..")
import paddle
import paddle.fluid as fluid
import numpy as np
import unittest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


def bce_loss(input, label):
<<<<<<< HEAD
    return -1 * (label * np.log(input) + (1.0 - label) * np.log(1.0 - input))


class XPUTestBceLossOp(XPUOpTestWrapper):
=======
    return -1 * (label * np.log(input) + (1. - label) * np.log(1. - input))


class XPUTestBceLossOp(XPUOpTestWrapper):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'bce_loss'
        self.use_dynamic_create_class = False

    class TestBceLossOp(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.op_type = "bce_loss"
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.init_test_case()
<<<<<<< HEAD
            input_np = np.random.uniform(0.1, 0.8, self.shape).astype(
                self.dtype
            )
=======
            input_np = np.random.uniform(0.1, 0.8,
                                         self.shape).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            label_np = np.random.randint(0, 2, self.shape).astype(self.dtype)
            output_np = bce_loss(input_np, label_np)

            self.inputs = {'X': input_np, 'Label': label_np}
            self.outputs = {'Out': output_np}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def init_test_case(self):
            self.shape = [10, 10]

    class TestBceLossOpCase1(TestBceLossOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_cast(self):
            self.shape = [2, 3, 4, 5]

    class TestBceLossOpCase2(TestBceLossOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_cast(self):
            self.shape = [2, 3, 20]


support_types = get_xpu_op_support_types('bce_loss')
for stype in support_types:
    create_test_class(globals(), XPUTestBceLossOp, stype)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
