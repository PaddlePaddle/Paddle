# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()


class XPUTestUnStackOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'unstack'
        self.use_dynamic_create_class = False

    class TestUnStackOpBase(XPUOpTest):
        def initDefaultParameters(self):
            self.input_dim = (5, 6, 7)
            self.axis = 0
            self.dtype = 'float32'

        def initParameters(self):
            pass

        def get_y_names(self):
            y_names = []
            for i in range(self.input_dim[self.axis]):
                y_names.append('y{}'.format(i))
            return y_names

        def setUp(self):
            self.initDefaultParameters()
            self.initParameters()
            self.op_type = 'unstack'
            self.python_api = paddle.unstack
            self.x = np.random.random(size=self.input_dim).astype(self.dtype)

            outs = np.split(self.x, self.input_dim[self.axis], self.axis)
            new_shape = list(self.input_dim)
            del new_shape[self.axis]
            y_names = self.get_y_names()
            tmp = []
            tmp_names = []
            for i in range(self.input_dim[self.axis]):
                tmp.append((y_names[i], np.reshape(outs[i], new_shape)))
                tmp_names.append(y_names[i])

            self.python_out_sig = tmp_names
            self.inputs = {'X': self.x}
            self.outputs = {'Y': tmp}
            self.attrs = {'axis': self.axis, 'num': self.input_dim[self.axis]}

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def test_check_grad(self):
            self.check_grad_with_place(
                paddle.XPUPlace(0), self.get_y_names, 'Y'
            )

    class TestStackOp3(TestUnStackOpBase):
        def initParameters(self):
            self.axis = -1

    class TestStackOp4(TestUnStackOpBase):
        def initParameters(self):
            self.axis = -3

    class TestStackOp5(TestUnStackOpBase):
        def initParameters(self):
            self.axis = 1

    class TestStackOp6(TestUnStackOpBase):
        def initParameters(self):
            self.axis = 2

    class TestUnstackZeroInputOp(unittest.TestCase):
        def unstack_zero_input_static(self):

            paddle.enable_static()

            array = np.array([], dtype=np.float32)
            x = paddle.to_tensor(np.reshape(array, [0]), dtype='float32')
            paddle.unstack(x, axis=1)

        def unstack_zero_input_dynamic(self):

            array = np.array([], dtype=np.float32)
            x = paddle.to_tensor(np.reshape(array, [0]), dtype='float32')
            paddle.unstack(x, axis=1)

        def test_type_error(self):
            paddle.disable_static()

            self.assertRaises(ValueError, self.unstack_zero_input_dynamic)
            self.assertRaises(ValueError, self.unstack_zero_input_static)

            paddle.disable_static()


support_types = get_xpu_op_support_types('unstack')
for stype in support_types:
    create_test_class(globals(), XPUTestUnStackOp, stype)


if __name__ == '__main__':
    unittest.main()
