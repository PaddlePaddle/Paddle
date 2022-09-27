#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard

paddle.enable_static()


# unsqueeze
class TestUnsqueezeOp(OpTest):

    def setUp(self):
        self.set_npu()
        self.op_type = "unsqueeze"
        self.place = paddle.NPUPlace(0)
        self.init_test_case()
        self.x = np.random.random(self.ori_shape).astype("float32")
        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(self.x)}
        self.init_attrs()
        self.outputs = {
            "Out": self.x.reshape(self.new_shape),
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")

    def init_test_case(self):
        self.ori_shape = (3, 40)
        self.axes = (0, 2)
        self.new_shape = (1, 3, 1, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


class TestUnsqueezeOp1(TestUnsqueezeOp):

    def init_test_case(self):
        self.ori_shape = (3, 40)
        self.axes = (0, -2)
        self.new_shape = (1, 3, 1, 40)


# No axes input.
class TestUnsqueezeOp2(TestUnsqueezeOp):

    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = ()
        self.new_shape = (1, 20, 5)


# Just part of axes be squeezed.
class TestUnsqueezeOp3(TestUnsqueezeOp):

    def init_test_case(self):
        self.ori_shape = (6, 5, 1, 4)
        self.axes = (1, -1)
        self.new_shape = (6, 1, 5, 1, 4, 1)


# unsqueeze 2
class TestUnsqueeze2Op(OpTest):

    def setUp(self):
        self.set_npu()
        self.op_type = "unsqueeze2"
        self.place = paddle.NPUPlace(0)
        self.init_test_case()
        self.x = np.random.random(self.ori_shape).astype("float32")
        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(self.x)}
        self.init_attrs()
        self.outputs = {
            "Out": self.x.reshape(self.new_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32")
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")

    def init_test_case(self):
        self.ori_shape = (3, 40)
        self.axes = (0, 2)
        self.new_shape = (1, 3, 1, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


# Correct: There is mins axis.
class TestUnsqueeze2Op1(TestUnsqueeze2Op):

    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = (0, -2)
        self.new_shape = (1, 20, 1, 5)


# Correct: No axes input.
class TestUnsqueeze2Op2(TestUnsqueeze2Op):

    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = ()
        self.new_shape = (1, 20, 5)


# Correct: Just part of axes be squeezed.
class TestUnsqueeze2Op3(TestUnsqueeze2Op):

    def init_test_case(self):
        self.ori_shape = (6, 5, 1, 4)
        self.axes = (1, -1)
        self.new_shape = (6, 1, 5, 1, 4, 1)


if __name__ == "__main__":
    unittest.main()
