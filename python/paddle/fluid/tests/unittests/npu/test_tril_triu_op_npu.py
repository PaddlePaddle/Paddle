#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at #
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.tensor as tensor
from paddle.fluid.framework import Program, program_guard

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUTrilTriu(OpTest):
    """ the base class of other op testcases
    """

    def setUp(self):
        self.op_type = "tril_triu"
        self.set_npu()
        self.init_dtype()
        self.initTestCase()

        self.real_np_op = getattr(np, self.real_op_type)

        self.inputs = {'X': self.X}
        self.attrs = {
            'diagonal': self.diagonal,
            'lower': True if self.real_op_type == 'tril' else False,
        }
        self.outputs = {
            'Out': self.real_np_op(self.X, self.diagonal)
            if self.diagonal else self.real_np_op(self.X)
        }

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    # def test_check_grad_normal(self):
    #     self.check_grad_with_place(self.place, ['X'], 'Out', check_dygraph=False)

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def initTestCase(self):
        self.real_op_type = np.random.choice(['tril', 'tril', 'triu', 'tril'])
        self.diagonal = None
        self.X = np.arange(1, 101, dtype=self.dtype).reshape([10, -1])

        print('real_op_type: ', self.real_op_type)
        print('X.shape: ', self.X.shape)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUTrilTriu_Tril1(TestNPUTrilTriu):
    """ tril
    """

    def initTestCase(self):
        self.real_op_type = 'tril'
        self.diagonal = None
        self.X = np.arange(1, 101, dtype=self.dtype).reshape([10, -1])

        print('real_op_type: ', self.real_op_type)
        print('X.shape: ', self.X.shape)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUTrilTriu_Tril2(TestNPUTrilTriu):
    """ tril
    """

    def initTestCase(self):
        self.real_op_type = 'tril'
        self.diagonal = 3
        self.X = np.arange(1, 101, dtype=self.dtype).reshape([10, -1])

        print('real_op_type: ', self.real_op_type)
        print('X.shape: ', self.X.shape)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUTrilTriu_Triu1(TestNPUTrilTriu):
    """ tril
    """

    def initTestCase(self):
        self.real_op_type = 'triu'
        self.diagonal = None
        self.X = np.arange(1, 101, dtype=self.dtype).reshape([10, -1])

        print('real_op_type: ', self.real_op_type)
        print('X.shape: ', self.X.shape)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUTrilTriu_Triu2(TestNPUTrilTriu):
    """ tril
    """

    def initTestCase(self):
        self.real_op_type = 'triu'
        self.diagonal = 3
        self.X = np.arange(1, 101, dtype=self.dtype).reshape([10, -1])

        print('real_op_type: ', self.real_op_type)
        print('X.shape: ', self.X.shape)


if __name__ == '__main__':
    unittest.main()
