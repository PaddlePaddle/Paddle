# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

paddle.enable_static()


# Correct: General.
class TestUnsqueezeOp(OpTest):
    def setUp(self):
        self.init_test_case()
        self.set_mlu()
        self.op_type = "unsqueeze2"
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32")
        }

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place, no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')

    def init_test_case(self):
        self.ori_shape = (3, 40)
        self.axes = (1, 2)
        self.new_shape = (3, 1, 1, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


# Correct: Single input index.
class TestUnsqueezeOp1(TestUnsqueezeOp):
    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = (-1, )
        self.new_shape = (20, 5, 1)


# Correct: Mixed input axis.
class TestUnsqueezeOp2(TestUnsqueezeOp):
    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = (0, -1)
        self.new_shape = (1, 20, 5, 1)


# Correct: There is duplicated axis.
class TestUnsqueezeOp3(TestUnsqueezeOp):
    def init_test_case(self):
        self.ori_shape = (10, 2, 5)
        self.axes = (0, 3, 3)
        self.new_shape = (1, 10, 2, 1, 1, 5)


# Correct: Reversed axes.
class TestUnsqueezeOp4(TestUnsqueezeOp):
    def init_test_case(self):
        self.ori_shape = (10, 2, 5)
        self.axes = (3, 1, 1)
        self.new_shape = (10, 1, 1, 2, 5, 1)


# axes is a list(with tensor)
class TestUnsqueezeOp_AxesTensorList(OpTest):
    def setUp(self):
        self.init_test_case()
        self.set_mlu()
        self.op_type = "unsqueeze2"

        axes_tensor_list = []
        for index, ele in enumerate(self.axes):
            axes_tensor_list.append(("axes" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            "AxesTensorList": axes_tensor_list
        }
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32")
        }

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place, no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')

    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = (1, 2)
        self.new_shape = (20, 1, 1, 5)

    def init_attrs(self):
        self.attrs = {}


class TestUnsqueezeOp1_AxesTensorList(TestUnsqueezeOp_AxesTensorList):
    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = (-1, )
        self.new_shape = (20, 5, 1)


class TestUnsqueezeOp2_AxesTensorList(TestUnsqueezeOp_AxesTensorList):
    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = (0, -1)
        self.new_shape = (1, 20, 5, 1)


class TestUnsqueezeOp3_AxesTensorList(TestUnsqueezeOp_AxesTensorList):
    def init_test_case(self):
        self.ori_shape = (10, 2, 5)
        self.axes = (0, 3, 3)
        self.new_shape = (1, 10, 2, 1, 1, 5)


class TestUnsqueezeOp4_AxesTensorList(TestUnsqueezeOp_AxesTensorList):
    def init_test_case(self):
        self.ori_shape = (10, 2, 5)
        self.axes = (3, 1, 1)
        self.new_shape = (10, 1, 1, 2, 5, 1)


# axes is a Tensor
class TestUnsqueezeOp_AxesTensor(OpTest):
    def setUp(self):
        self.init_test_case()
        self.set_mlu()
        self.op_type = "unsqueeze2"

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            "AxesTensor": np.array(self.axes).astype("int32")
        }
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32")
        }

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place, no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')

    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = (1, 2)
        self.new_shape = (20, 1, 1, 5)

    def init_attrs(self):
        self.attrs = {}


class TestUnsqueezeOp1_AxesTensor(TestUnsqueezeOp_AxesTensor):
    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = (-1, )
        self.new_shape = (20, 5, 1)


class TestUnsqueezeOp2_AxesTensor(TestUnsqueezeOp_AxesTensor):
    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = (0, -1)
        self.new_shape = (1, 20, 5, 1)


class TestUnsqueezeOp3_AxesTensor(TestUnsqueezeOp_AxesTensor):
    def init_test_case(self):
        self.ori_shape = (10, 2, 5)
        self.axes = (0, 3, 3)
        self.new_shape = (1, 10, 2, 1, 1, 5)


class TestUnsqueezeOp4_AxesTensor(TestUnsqueezeOp_AxesTensor):
    def init_test_case(self):
        self.ori_shape = (10, 2, 5)
        self.axes = (3, 1, 1)
        self.new_shape = (10, 1, 1, 2, 5, 1)


if __name__ == "__main__":
    unittest.main()
