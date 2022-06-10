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


class XPUTestMergedMomentumeOp(XPUOpTestWrapper):
  def __init__(self):
        self.op_name = "merged_momentom"
        self.use_dynamic_create_class = False

  class TestMergedMomentomOP(XPUOpTest):
    def setUp(self):
        self.op_type = "merged_momentom"
        self.place = paddle.XPUPlace(0)
        self.init_dtype()
        self.set_case()
        self.set_xpu()

    def set_case(self):
        #self.shapes=[[3], [2], [6], [56]]
        self.param = self.gen_rand_data(self.shapes,self.dtype)
        self.grad = self.gen_rand_data(self.shapes,self.dtype)
        self.velocity = self.gen_rand_data(self.shapes,self.dtype)
        mu=[0.0001]*19
        self.mu = np.array(mu).astype(self.dtype)
        self.lr = np.array([0.001]).astype(self.dtype)
        self.use_nesterov = False
        
        self.attrs = {
                'use_xpu': True,
                'mu': self.mu,
                'use_nesterov': use_nesterov,
                'regularization_method':['l2_decay' for i in range(len(param_vars))],
                'regularization_coeff':[2.0 for i in range(len(param_vars))],
            }
        self.inputs = {'Param': OpTest.np_dtype_to_fluid_dtype(self.param),
                       'Grad':OpTest.np_dtype_to_fluid_dtype(self.grad),
                       'Velocity': OpTest.np_dtype_to_fluid_dtype(self.velocity),
                       'LearningRate':OpTest.np_dtype_to_fluid_dtype(self.lr)}
        self.outputs = {'Param_out': OpTest.np_dtype_to_fluid_dtype(self.param),
                        'Velocity_out': OpTest.np_dtype_to_fluid_dtype(self.velocity) }

    def set_data(self):
        self.shapes=[[3], [2], [6], [56]]

    def set_xpu(self):
        self.__class__.use_xpu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def gen_rand_data(self, shapes, dtype):
        return [np.random.random(s).astype(dtype) for s in shapes]

    

  class TestMergedMomentomOP2(TestMergedMomentomOP):
        def set_case(self):
            self.shapes=[[4], [5], [6], [56],[78]]

  class TestMergedMomentomOP3(TestMergedMomentomOP):
        def set_case(self):
            self.shapes=[[1076], [3018], [6], [56],[78]]

  class TestMergedMomentomOP4(TestMergedMomentomOP):
        def set_case(self):
            self.shapes=[[7], [8], [9], [56],[78]]

  class TestMergedMomentomOP5(TestMergedMomentomOP):
        def set_case(self):
            self.shapes=[[1076], [30086], [895666], [56],[78]]


support_types = get_xpu_op_support_types("merged_momentum")
for stype in support_types:
    create_test_class(globals(), XPUTestMergedMomentumOp, stype)

if __name__ == "__main__":
    unittest.main()
