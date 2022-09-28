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


class XPUTestReduceAllOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'reduce_all'

    class XPUTestReduceAllBase(XPUOpTest):

        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.set_case()

        def set_case(self):
            self.op_type = 'reduce_all'
            self.attrs = {
                'use_xpu': True,
                'reduce_all': True,
                'keep_dim': True,
                'dim': (3, 5, 4)
            }
            self.inputs = {
                'X':
                np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
            }
            self.outputs = {'Out': self.inputs['X'].all(axis=self.attrs['dim'])}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            pass

    class XPUTestReduceAllCase1(XPUTestReduceAllBase):

        def set_case(self):
            self.op_type = 'reduce_all'
            self.attrs = {
                'use_xpu': True,
                'reduce_all': True,
                'keep_dim': True,
                'dim': [1]
            }
            self.inputs = {
                'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")
            }
            self.outputs = {'Out': self.inputs['X'].all()}

    class XPUTestReduceAllCase2(XPUTestReduceAllBase):

        def set_case(self):
            self.op_type = 'reduce_all'
            self.attrs = {
                'use_xpu': True,
                'reduce_all': True,
                'keep_dim': False,
                'dim': (3, 6)
            }
            self.inputs = {
                'X':
                np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
            }
            self.outputs = {'Out': self.inputs['X'].all(axis=self.attrs['dim'])}

    class XPUTestReduceAllCase3(XPUTestReduceAllBase):

        def set_case(self):
            self.op_type = 'reduce_all'
            self.attrs = {
                'use_xpu': True,
                'keep_dim': True,
                'dim': [1]
                # 'reduce_all': True,
            }
            self.inputs = {
                'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")
            }
            self.outputs = {
                'Out': np.expand_dims(self.inputs['X'].all(axis=1), axis=1)
            }


support_types = get_xpu_op_support_types('reduce_all')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceAllOp, stype)

if __name__ == '__main__':
    unittest.main()
