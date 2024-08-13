#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


def huber_loss_forward(val, delta):
    abs_val = abs(val)
    if abs_val <= delta:
        return 0.5 * val * val
    else:
        return delta * (abs_val - 0.5 * delta)


class XPUTestHuberLossOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'huber_loss'
        self.use_dynamic_create_class = False

    class TestHuberLossOp(XPUOpTest):
        def setUp(self):
            self.set_xpu()
            self.op_type = 'huber_loss'
            self.place = paddle.XPUPlace(0)

            self.init_dtype()
            self.set_inputs()
            self.set_attrs()
            self.set_outputs()

        def set_inputs(self):
            shape = self.set_shape()
            x = np.random.uniform(0, 1.0, shape).astype(self.dtype)
            y = np.random.uniform(0, 1.0, shape).astype(self.dtype)
            self.inputs = {
                'X': OpTest.np_dtype_to_base_dtype(x),
                'Y': OpTest.np_dtype_to_base_dtype(y),
            }

        def set_attrs(self):
            self.attrs = {'delta': 0.5}

        def set_outputs(self):
            delta = self.attrs['delta']
            shape = self.set_shape()
            residual = self.inputs['Y'] - self.inputs['X']
            loss = np.vectorize(huber_loss_forward)(residual, delta).astype(
                self.dtype
            )
            self.outputs = {'Residual': residual, 'Out': loss.reshape(shape)}

        def set_shape(self):
            return (100, 1)

        def set_xpu(self):
            self.__class__.use_xpu = True

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad_normal(self):
            self.check_grad_with_place(self.place, ['X', 'Y'], 'Out')

        def test_check_grad_ignore_x(self):
            self.check_grad_with_place(
                self.place, ['Y'], 'Out', no_grad_set=set("residual")
            )

        def test_check_grad_ignore_y(self):
            self.check_grad_with_place(
                self.place, ['X'], 'Out', no_grad_set=set('residual')
            )

    class TestHuberLossOp1(TestHuberLossOp):
        def set_shape(self):
            return 640

    class TestHuberLossOp2(TestHuberLossOp):
        def set_shape(self):
            return (10, 10)

    class TestHuberLossOp3(TestHuberLossOp):
        def set_shape(self):
            return (10, 10, 1)


support_types = get_xpu_op_support_types('huber_loss')
for stype in support_types:
    create_test_class(globals(), XPUTestHuberLossOp, stype)

if __name__ == '__main__':
    unittest.main()
