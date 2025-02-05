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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestReduceAnyOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'reduce_any'

    class XPUTestReduceAnyBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.set_case()

        def set_case(self):
            self.op_type = 'reduce_any'
            self.attrs = {
                'use_xpu': True,
                'reduce_all': False,
                'keep_dim': False,
                'dim': (3, 5, 4),
            }
            self.inputs = {
                'X': np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype(
                    "bool"
                )
            }
            self.outputs = {'Out': self.inputs['X'].any(axis=self.attrs['dim'])}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            pass

    class XPUTestReduceAnyCase1(XPUTestReduceAnyBase):
        def set_case(self):
            self.op_type = 'reduce_any'
            self.attrs = {
                'use_xpu': True,
                'dim': [1],
                # 'reduce_all': True,
                # 'keep_dim': True,
            }
            self.inputs = {
                'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")
            }
            self.outputs = {'Out': self.inputs['X'].any(axis=1)}

    class XPUTestReduceAnyCase2(XPUTestReduceAnyBase):
        def set_case(self):
            self.op_type = 'reduce_any'
            self.attrs = {
                'use_xpu': True,
                'reduce_all': False,
                'keep_dim': False,
                'dim': (3, 6),
            }
            self.inputs = {
                'X': np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype(
                    "bool"
                )
            }
            self.outputs = {'Out': self.inputs['X'].any(axis=self.attrs['dim'])}


support_types = get_xpu_op_support_types('reduce_any')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceAnyOp, stype)

if __name__ == '__main__':
    unittest.main()
