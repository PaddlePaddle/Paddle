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

import unittest
import numpy as np
import sys

sys.path.append("..")

import paddle

from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestReduceProdOP(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'reduce_prod'
        self.use_dynamic_create_class = False

    class TestXPUReduceProdOp(XPUOpTest):

        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.op_type = 'reduce_prod'
            self.use_mkldnn = False
            self.keep_dim = False
            self.reduce_all = False
            self.initTestCase()
            self.attrs = {
                'dim': self.axis,
                'keep_dim': self.keep_dim,
                'reduce_all': self.reduce_all
            }
            self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
            if self.attrs['reduce_all']:
                self.outputs = {'Out': self.inputs['X'].prod()}
            else:
                self.outputs = {
                    'Out':
                    self.inputs['X'].prod(axis=self.axis,
                                          keepdims=self.attrs['keep_dim'])
                }

        def initTestCase(self):
            self.shape = (5, 6, 10)
            self.axis = (0, )

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestProdOp5D(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (1, 2, 5, 6, 10)
            self.axis = (0, )

    class TestProdOp6D(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (1, 1, 2, 5, 6, 10)
            self.axis = (0, )

    class TestProdOp8D(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (1, 3, 1, 2, 1, 4, 3, 10)
            self.axis = (0, 3)

    class Test1DReduce(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = 120
            self.axis = (0, )

    class Test2DReduce0(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (20, 10)
            self.axis = (0, )

    class Test2DReduce1(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (20, 10)
            self.axis = (1, )

    class Test3DReduce0(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (5, 6, 7)
            self.axis = (1, )

    class Test3DReduce1(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (5, 6, 7)
            self.axis = (2, )

    class Test3DReduce2(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (5, 6, 7)
            self.axis = (-2, )

    class Test3DReduce3(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (5, 6, 7)
            self.axis = (1, 2)

    class TestKeepDimReduce(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (5, 6, 10)
            self.axis = (1, )
            self.keep_dim = True

    class TestKeepDim8DReduce(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (2, 5, 3, 2, 2, 3, 4, 2)
            self.axis = (3, 4, 5)
            self.keep_dim = True

    class TestReduceAll(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (5, 6, 2, 10)
            self.axis = (0, )
            self.reduce_all = True


support_types = get_xpu_op_support_types('reduce_prod')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceProdOP, stype)

if __name__ == '__main__':
    unittest.main()
