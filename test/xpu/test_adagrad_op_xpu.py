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
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestAdagradOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'adagrad'
        self.use_dynamic_create_class = False

    class TestAdagradOp1(XPUOpTest):
        '''Test Adagrad operator with explicit attributes'''

        def setUp(self):
            self.op_type = "adagrad"
            self.dtype = self.in_type
            param = np.random.random((123, 321)).astype(self.in_type)
            grad = np.random.random((123, 321)).astype(self.in_type)
            moment = np.zeros((123, 321)).astype(self.in_type)
            lr = 0.01
            epsilon = 1e-8

            self.inputs = {
                'Param': param,
                'Grad': grad,
                'Moment': moment,
                'LearningRate': np.array([lr]).astype(self.in_type),
            }

            self.attrs = {'epsilon': epsilon}

            moment_out = moment + grad * grad
            param_out = param - lr * grad / (np.sqrt(moment_out) + epsilon)

            self.outputs = {'ParamOut': param_out, 'MomentOut': moment_out}

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0))

    class TestAdagradOp2(XPUOpTest):
        '''Test Adagrad operator with default attributes'''

        def setUp(self):
            self.op_type = "adagrad"

            param = np.random.random((123, 321)).astype(self.in_type)
            grad = np.random.random((123, 321)).astype(self.in_type)
            moment = np.zeros((123, 321)).astype(self.in_type)
            lr = 0.01
            epsilon = 1e-6

            self.inputs = {
                'Param': param,
                'Grad': grad,
                'Moment': moment,
                'LearningRate': np.array([lr]).astype(self.in_type),
            }

            self.attrs = {'epsilon': epsilon}

            moment_out = moment + grad * grad
            param_out = param - lr * grad / (np.sqrt(moment_out) + epsilon)

            self.outputs = {'ParamOut': param_out, 'MomentOut': moment_out}

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0))


support_types = get_xpu_op_support_types('adagrad')
for stype in support_types:
    create_test_class(globals(), XPUTestAdagradOp, stype)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
