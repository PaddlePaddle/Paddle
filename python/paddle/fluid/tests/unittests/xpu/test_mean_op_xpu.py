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

<<<<<<< HEAD
import sys
import unittest

import numpy as np

import paddle

sys.path.append("..")
from op_test_xpu import XPUOpTest

=======
from __future__ import print_function

import unittest
import numpy as np
import sys

sys.path.append("..")
from op_test_xpu import XPUOpTest
from op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from paddle.fluid import Program, program_guard

np.random.seed(10)

<<<<<<< HEAD
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
=======
import op_test
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


class XPUTestMeanOp(XPUOpTestWrapper):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'mean'
        self.use_dynamic_create_class = False

    class TestMeanOp(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "mean"
            self.place = paddle.XPUPlace(0)
            self.set_shape()
            self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
            self.outputs = {'Out': np.mean(self.inputs["X"]).astype(np.float16)}

        def init_dtype(self):
            self.dtype = self.in_type

        def set_shape(self):
            self.shape = (10, 10)

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.dtype

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_checkout_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestMeanOp1(TestMeanOp):
<<<<<<< HEAD
        def set_shape(self):
            self.shape = 5

    class TestMeanOp2(TestMeanOp):
=======

        def set_shape(self):
            self.shape = (5)

    class TestMeanOp2(TestMeanOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def set_shape(self):
            self.shape = (5, 7, 8)

    class TestMeanOp3(TestMeanOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def set_shape(self):
            self.shape = (10, 5, 7, 8)

    class TestMeanOp4(TestMeanOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def set_shape(self):
            self.shape = (2, 2, 3, 3, 3)


class TestMeanOpError(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of mean_op must be Variable.
            input1 = 12
            self.assertRaises(TypeError, paddle.mean, input1)
            # The input dtype of mean_op must be float16, float32, float64.
<<<<<<< HEAD
            input2 = paddle.static.data(
                name='input2', shape=[-1, 12, 10], dtype="int32"
            )
            self.assertRaises(TypeError, paddle.mean, input2)
            input3 = paddle.static.data(
                name='input3', shape=[-1, 4], dtype="float16"
            )
            paddle.nn.functional.softmax(input3)
=======
            input2 = fluid.layers.data(name='input2',
                                       shape=[12, 10],
                                       dtype="int32")
            self.assertRaises(TypeError, paddle.mean, input2)
            input3 = fluid.layers.data(name='input3',
                                       shape=[4],
                                       dtype="float16")
            fluid.layers.softmax(input3)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


support_types = get_xpu_op_support_types('mean')
for stype in support_types:
    create_test_class(globals(), XPUTestMeanOp, stype)

if __name__ == "__main__":
    unittest.main()
