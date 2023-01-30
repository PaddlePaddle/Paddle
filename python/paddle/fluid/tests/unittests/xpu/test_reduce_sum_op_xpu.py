<<<<<<< HEAD
#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
=======
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
import sys
import unittest

import numpy as np

sys.path.append("..")

from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle
=======
from __future__ import print_function

import unittest
import numpy as np
import sys

sys.path.append("..")

import paddle
from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


class XPUTestReduceSumOp(XPUOpTestWrapper):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'reduce_sum'

    class XPUTestReduceSumBase(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'reduce_sum'
            self.attrs = {
                'use_xpu': True,
                'reduce_all': self.reduce_all,
<<<<<<< HEAD
                'keep_dim': self.keep_dim,
            }
            self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
=======
                'keep_dim': self.keep_dim
            }
            self.inputs = {'X': np.random.random(self.shape).astype("float32")}
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            if self.attrs['reduce_all']:
                self.outputs = {'Out': self.inputs['X'].sum()}
            else:
                self.outputs = {
<<<<<<< HEAD
                    'Out': self.inputs['X'].sum(
                        axis=self.axis, keepdims=self.attrs['keep_dim']
                    )
=======
                    'Out':
                    self.inputs['X'].sum(axis=self.axis,
                                         keepdims=self.attrs['keep_dim'])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                }

        def init_case(self):
            self.shape = (5, 6, 10)
<<<<<<< HEAD
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = False
            self.dtype = self.in_type
=======
            self.axis = (0, )
            self.reduce_all = False
            self.keep_dim = False
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
<<<<<<< HEAD
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class XPUTestReduceSumCase1(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = False

    class XPUTestReduceSumCase2(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = True

    class XPUTestReduceSumCase3(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = True
            self.keep_dim = False

    class XPUTestReduceSumCase4(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = False
            self.keep_dim = False

    class XPUTestReduceSumCase5(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = False
            self.keep_dim = True

    class XPUTestReduceSumCase6(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = True
            self.keep_dim = False

=======
            pass

    class XPUTestReduceSumCase1(XPUTestReduceSumBase):

        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (0, )
            self.reduce_all = False
            self.keep_dim = True

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

support_types = get_xpu_op_support_types('reduce_sum')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceSumOp, stype)

if __name__ == '__main__':
    unittest.main()
