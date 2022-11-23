#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

sys.path.append("..")
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci
from paddle.fluid.tests.unittests.test_norm_op import l2_norm


class TestNPUNormOp(OpTest):

    def setUp(self):
        paddle.enable_static()
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "norm"
        self.init_dtype()
        self.init_test_case()

        x = np.random.random(self.shape).astype(self.dtype)
        y, norm = l2_norm(x, self.axis, self.epsilon)
        self.inputs = {'X': x}
        self.attrs = {'epsilon': self.epsilon, 'axis': self.axis}
        self.outputs = {'Out': y, 'Norm': norm}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_test_case(self):
        self.axis = 1
        self.epsilon = 1e-10
        self.shape = (2, 3, 4, 5)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'],
                                   'Out',
                                   max_relative_error=0.006)


class TestNPUNormOp2(TestNPUNormOp):

    def init_test_case(self):
        self.shape = [5, 3, 9, 7]
        self.axis = 0
        self.epsilon = 1e-8


class TestNPUNormOp3(TestNPUNormOp):

    def init_test_case(self):
        self.shape = [5, 3, 2, 7]
        self.axis = -1
        self.epsilon = 1e-8


@skip_check_grad_ci(reason="'check_grad' on large inputs is too slow, " +
                    "however it is desirable to cover the forward pass")
class TestNPUNormOp4(TestNPUNormOp):

    def init_test_case(self):
        self.shape = [128, 1024, 14, 14]
        self.axis = 2
        self.epsilon = 1e-8

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="'check_grad' on large inputs is too slow, " +
                    "however it is desirable to cover the forward pass")
class TestNPUNormOp5(TestNPUNormOp):

    def init_test_case(self):
        self.shape = [2048, 2048]
        self.axis = 1
        self.epsilon = 1e-8

    def test_check_grad(self):
        pass


class API_NormTest(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):

            def test_norm_x_type():
                data = fluid.data(name="x", shape=[3, 3], dtype="float64")
                out = fluid.layers.l2_normalize(data)

            self.assertRaises(TypeError, test_norm_x_type)


class TestNPUNormOpFP16(TestNPUNormOp):

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16

    def init_test_case(self):
        self.axis = -1
        self.epsilon = 1e-10
        self.shape = (2, 3, 100)


if __name__ == '__main__':
    unittest.main()
