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

from __future__ import print_function

import unittest
import numpy as np
import sys
sys.path.append("..")

import paddle
from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestReduceSumOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'reduce_sum'

    class XPUTestReduceSumBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'reduce_sum'
            self.attrs = {
                'use_xpu': True,
                'reduce_all': self.reduce_all,
                'keep_dim': self.keep_dim
            }
            self.inputs = {'X': np.random.random(self.shape).astype("float32")}
            if self.attrs['reduce_all']:
                self.outputs = {'Out': self.inputs['X'].sum()}
            else:
                self.outputs = {
                    'Out': self.inputs['X'].sum(axis=self.axis,
                                                keepdims=self.attrs['keep_dim'])
                }

        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (0, )
            self.reduce_all = False
            self.keep_dim = False

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            pass

    class XPUTestReduceSumCase1(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (0, )
            self.reduce_all = False
            self.keep_dim = True


support_types = get_xpu_op_support_types('reduce_sum')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceSumOp, stype)

if __name__ == '__main__':
    unittest.main()
