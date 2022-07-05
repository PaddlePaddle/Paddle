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
            self.shape = (8, 10800, 50, 2)
            self.axis = (3, )

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class Test3DReduce36(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = 124332
            self.axis = (0, )

    class Test3DReduce39(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = 1270080
            self.axis = (0,)

    class Test3DReduce33(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = 6553600
            self.axis = (0,)

    class Test3DReduce35(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = 2235392
            self.axis = (0,)

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

    class Test3DReduce4(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (512, 26, 1)
            self.axis = (1,)

    class Test3DReduce5(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (512, 13, 1)
            self.axis = (1,)

    class Test3DReduce6(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (512, 9)
            self.axis = (1,)

    class Test3DReduce10(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (8, 138240)
            self.axis = (1,)


    class Test3DReduce11(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (307200 , 2)
            self.axis = (1,)

    class Test3DReduce12(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (8, 432)
            self.axis = (1,)

    class Test3DReduce13(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (5548800, 2)
            self.axis = (1,)

    class Test3DReduce14(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (172800, 2)
            self.axis = (1,)

    class Test3DReduce15(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (172800, 2)
            self.axis = (1,)

    class Test3DReduce16(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (27744, 50)
            self.axis = (1,)

    class Test3DReduce17(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (3456, 2)
            self.axis = (1,)

    class Test3DReduce18(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (1080000, 2)
            self.axis = (1,)

    class Test3DReduce19(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (400, 2)
            self.axis = (1,)

    class Test3DReduce20(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (4320000, 2)
            self.axis = (1,)

    class Test3DReduce21(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (8, 8640)
            self.axis = (1,)

    class Test3DReduce22(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (8, 311040)
            self.axis = (1,)

    class Test3DReduce23(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (8, 1728)
            self.axis = (1,)

    class Test3DReduce24(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (691200, 2)
            self.axis = (1,)

    class Test3DReduce25(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (13824, 2)
            self.axis = (1,)

    class Test3DReduce26(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (13824, 2)
            self.axis = (1,)

    class Test3DReduce27(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (400, 2)
            self.axis = (1,)

    class Test3DReduce28(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (13824, 2)
            self.axis = (1,)

    class Test3DReduce29(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (8, 34560)
            self.axis = (1,)

    class Test3DReduce30(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (8, 6912)
            self.axis = (1,)

    class Test3DReduce31(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (2764800, 2)
            self.axis = (1,)

    class Test3DReduce32(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (55296, 2)
            self.axis = (1,)

    class Test3DReduce34(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (64, 90, 8732)
            self.axis = (1,)

    class Test3DReduce37(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (64, 8732)
            self.axis = (1,)

    class Test3DReduce38(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (5760, 8732)
            self.axis = (1,)

    class Test3DReduce40(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (64, 8732)
            self.axis = (1,)

    class Test3DReduce41(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (512, 39, 1)
            self.axis = (1,)

    class Test3DReduce3_2(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (5, 6, 7)
            self.axis = (1, 2)

    class TestKeepDimReduce(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (5, 6, 10)
            self.axis = (1, )
            self.keep_dim = True

'''
    big t (m == 1 && t <= 1024)
    class Test3DReduce9(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (8, )
            self.axis = (0,)

    class Test1DReduce(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (120, )
            self.axis = (0, )

    class Test3DReduce8(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (50, )
            self.axis = (0,)

    class Test3DReduce7(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = 512
            self.axis = (0,)

    class Test3DReduce3(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = 1
            self.axis = (0,)

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

    class Test2DReduce0(TestXPUReduceProdOp):

        def initTestCase(self):
            self.shape = (20, 10)
            self.axis = (0, )
'''

support_types = get_xpu_op_support_types('reduce_prod')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceProdOP, stype)

if __name__ == '__main__':
    unittest.main()
