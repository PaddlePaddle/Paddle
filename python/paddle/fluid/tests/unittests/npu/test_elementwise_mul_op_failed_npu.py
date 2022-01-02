#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid

paddle.enable_static()


class ElementwiseMulOp(OpTest):
    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_mul"
        self.dtype = np.float32
        self.axis = -1
        self.init_dtype()
        self.init_input_output()
        self.init_axis()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.outputs = {'Out': self.out}
        self.attrs = {'axis': self.axis}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        # if True:
        #     return
        
        self.check_grad_with_place(self.place, ['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        # if True:
        #     return
        
        self.check_grad_with_place(
            self.place, ['Y'], 'Out', no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        # if True:
        #     return
        
        self.check_grad_with_place(
            self.place, ['X'], 'Out', no_grad_set=set('Y'))

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [110, 1]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [110, 2]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def init_dtype(self):
        pass

    def init_axis(self):
        pass


@unittest.skipIf(False, "tmp")
class ElementwiseMulOp2(ElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [110, 2]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [110, 1]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)


if __name__ == '__main__':
    unittest.main()
