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
from __future__ import print_function

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
import unittest
import numpy as np
import sys

sys.path.append('..')
from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
import paddle
import paddle.nn.functional as F

paddle.enable_static()
np.random.seed(10)


class TestAbs(OpTest):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def setUp(self):
        self.op_type = "abs"
        self.set_mlu()
        self.dtype = 'float32'
        self.shape = [4, 25]

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # Because we set delta = 0.005 in calculating numeric gradient,
        # if x is too small, such as 0.002, x_neg will be -0.003
        # x_pos will be 0.007, so the numeric gradient is inaccurate.
        # we should avoid this
        x[np.abs(x) < 0.005] = 0.02
        out = np.abs(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad_with_place(self.place, ['X'], ['Out'],
                                   check_eager=False)


class TestAbsHalf(OpTest):

=======
        self.check_grad_with_place(
            self.place, ['X'], ['Out'], check_eager=False
        )


class TestAbsHalf(OpTest):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def setUp(self):
        self.op_type = "abs"
        self.set_mlu()
        self.dtype = 'float16'
        self.shape = [7, 9, 13, 19]

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # Because we set delta = 0.005 in calculating numeric gradient,
        # if x is too small, such as 0.002, x_neg will be -0.003
        # x_pos will be 0.007, so the numeric gradient is inaccurate.
        # we should avoid this
        x[np.abs(x) < 0.005] = 0.02
        out = np.abs(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad_with_place(self.place, ['X'], ['Out'],
                                   check_eager=False)
=======
        self.check_grad_with_place(
            self.place, ['X'], ['Out'], check_eager=False
        )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91


if __name__ == "__main__":
    unittest.main()
