#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at #
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

sys.path.append("..")

import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.tensor as tensor
import unittest
import numpy as np
from op_test import OpTest
from op_test_xpu import XPUOpTest
from paddle.fluid.framework import Program, program_guard
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestTrilTriuOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'tril_triu'
        self.use_dynamic_create_class = False

    class TestTrilTriuOp(XPUOpTest):

        def setUp(self):
            self.init_dtype()
            self.initTestCase()
            self.real_op_type = np.random.choice(['triu', 'tril'])
            self.real_np_op = getattr(np, self.real_op_type)
            self.set_xpu()
            self.op_type = "tril_triu"
            self.place = paddle.XPUPlace(0)
            if self.dtype == np.int32:
                self.X = np.arange(1,
                                   self.get_Xshape_prod() + 1,
                                   dtype=self.dtype).reshape(self.Xshape)
            else:
                self.X = np.random.random(self.Xshape).astype(dtype=self.dtype)
            self.inputs = {'X': self.X}
            self.attrs = {
                'diagonal': self.diagonal,
                'lower': True if self.real_op_type == 'tril' else False,
            }
            self.outputs = {
                'Out':
                self.real_np_op(self.X, self.diagonal)
                if self.diagonal else self.real_np_op(self.X)
            }

        def init_dtype(self):
            self.dtype = self.in_type

        def get_Xshape_prod(self):
            ret = 1
            for v in self.Xshape:
                ret *= v
            return ret

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = False
            self.__class__.op_type = self.real_op_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad_normal(self):
            if self.dtype == np.int32:
                user_defined_grad_outputs = np.random.random(
                    self.Xshape).astype('float32')
                self.check_grad_with_place(
                    self.place, ['X'],
                    'Out',
                    user_defined_grad_outputs=user_defined_grad_outputs)
            else:
                self.check_grad_with_place(self.place, ['X'], 'Out')

        def initTestCase(self):
            self.diagonal = None
            self.Xshape = (10, 10)

    class TestTrilTriuOp1(TestTrilTriuOp):

        def initTestCase(self):
            self.diagonal = -3
            self.Xshape = (5, 5)

    class TestTrilTriuOp2(TestTrilTriuOp):

        def initTestCase(self):
            self.diagonal = 4
            self.Xshape = (11, 17)

    class TestTrilTriuOp3(TestTrilTriuOp):

        def initTestCase(self):
            self.diagonal = 10
            self.Xshape = (2, 25, 25)

    class TestTrilTriuOp4(TestTrilTriuOp):

        def initTestCase(self):
            self.diagonal = -10
            self.Xshape = (1, 2, 33, 11)

    class TestTrilTriuOp5(TestTrilTriuOp):

        def initTestCase(self):
            self.diagonal = 11
            self.Xshape = (1, 1, 99)

    class TestTrilTriuOp6(TestTrilTriuOp):

        def initTestCase(self):
            self.diagonal = 5
            self.Xshape = (1, 2, 3, 5, 99)

    class TestTrilTriuOp7(TestTrilTriuOp):

        def initTestCase(self):
            self.diagonal = -100
            self.Xshape = (2, 2, 3, 4, 5)


class TestTrilTriuOpError(unittest.TestCase):

    def test_errors1(self):
        paddle.enable_static()
        data = fluid.data(shape=(20, 22), dtype='float32', name="data1")
        op_type = np.random.choice(['triu', 'tril'])
        errmsg = {
            "diagonal: TypeError":
            "diagonal in {} must be a python Int".format(op_type),
        }
        expected = list(errmsg.keys())[0]
        with self.assertRaisesRegex(eval(expected.split(':')[-1]),
                                    errmsg[expected]):
            getattr(tensor, op_type)(x=data, diagonal='2022')

    def test_errors2(self):
        paddle.enable_static()
        data = fluid.data(shape=(200, ), dtype='float32', name="data2")
        op_type = np.random.choice(['triu', 'tril'])
        errmsg = {
            "input: ValueError":
            "x shape in {} must be at least 2-D".format(op_type),
        }
        expected = list(errmsg.keys())[0]
        with self.assertRaisesRegex(eval(expected.split(':')[-1]),
                                    errmsg[expected]):
            getattr(tensor, op_type)(x=data, diagonal=[None])


support_types = get_xpu_op_support_types('tril_triu')
for stype in support_types:
    create_test_class(globals(), XPUTestTrilTriuOp, stype)

if __name__ == '__main__':
    unittest.main()
