#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import unittest
import sys
sys.path.append("..")

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from op_test import OpTest
from op_test_xpu import XPUOpTest

paddle.enable_static()


# Correct: General.
class TestSqueezeOp(XPUOpTest):
    def setUp(self):
        self.op_type = "squeeze"
        self.use_xpu = True
        self.use_mkldnn = False
        self.init_test_case()
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.init_attrs()
        self.outputs = {"Out": self.inputs["X"].reshape(self.new_shape), }

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


# Correct: There is mins axis.
class TestSqueezeOp1(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, -2)
        self.new_shape = (3, 40)


# Correct: No axes input.
class TestSqueezeOp2(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)


# Correct: Just part of axes be squeezed. 
class TestSqueezeOp3(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (6, 5, 1, 4)


# Correct: The demension of axis is not of size 1 remains unchanged.
class TestSqueezeOp4(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, 2)
        self.new_shape = (6, 5, 1, 4, 1)


class TestSqueezeOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # The input type of softmax_op must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], paddle.XPUPlace(0))
            self.assertRaises(TypeError, paddle.squeeze, x1)
            # The input axes of squeeze must be list.
            x2 = paddle.static.data(name='x2', shape=[4], dtype="int32")
            self.assertRaises(TypeError, paddle.squeeze, x2, axes=0)
            # The input dtype of squeeze not support float16.
            x3 = paddle.static.data(name='x3', shape=[4], dtype="float16")
            self.assertRaises(TypeError, paddle.squeeze, x3, axes=0)


if __name__ == "__main__":
    unittest.main()
