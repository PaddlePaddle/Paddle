#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestReduceMaxOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'reduce_max'

    class XPUTestReduceMaxBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'reduce_max'
            self.attrs = {
                'use_xpu': True,
                'reduce_all': self.reduce_all,
                'keep_dim': self.keep_dim,
                'dim': self.axis,
            }
            self.temp_x = np.random.random(self.shape)
            self.dtype = self.in_type
            if self.dtype == np.uint16:  # bfloat16 actually
                self.x = convert_float_to_uint16(self.temp_x)
            else:
                self.x = self.temp_x.astype(self.dtype)
            self.inputs = {'X': self.x}
            if self.attrs['reduce_all']:
                self.outputs = {'Out': self.inputs['X'].max()}
            else:
                self.outputs = {
                    'Out': self.inputs['X'].max(
                        axis=self.axis, keepdims=self.attrs['keep_dim']
                    )
                }

        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = False

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class XPUTestReduceMaxCase1(XPUTestReduceMaxBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = True

    class XPUTestReduceMaxCase2(XPUTestReduceMaxBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = False
            self.keep_dim = True

    class XPUTestReduceMaxCase3(XPUTestReduceMaxBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = False
            self.keep_dim = False

    class XPUTestReduceMaxCase4(XPUTestReduceMaxBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = True
            self.keep_dim = False


support_types = get_xpu_op_support_types('reduce_max')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceMaxOp, stype)

if __name__ == '__main__':
    unittest.main()
