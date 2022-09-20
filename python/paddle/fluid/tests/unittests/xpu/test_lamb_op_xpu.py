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
import sys

sys.path.append("..")
import unittest
import numpy as np
from op_test_xpu import XPUOpTest
from paddle.fluid import core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
import paddle

from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper


def lamb_step(inputs, attributes):
    '''
    Simulate one step of the lamb optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    '''
    param = inputs['Param']
    grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']

    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    epsilon = attributes['epsilon']
    weight_decay = attributes['weight_decay']

    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)

    moment1_unbiased = moment1_out / (1 - beta1_pow)
    moment2_unbiased = moment2_out / (1 - beta2_pow)

    r_1 = np.linalg.norm(param)
    r_2 = np.linalg.norm(moment1_unbiased /
                         (np.sqrt(moment2_unbiased) + epsilon) +
                         weight_decay * param)
    lr_t = lr * r_1 / r_2

    param_out = param - lr_t * (moment1_unbiased /
                                (np.sqrt(moment2_unbiased) + epsilon) +
                                weight_decay * param)

    beta1_pow_out = beta1_pow * beta1
    beta2_pow_out = beta2_pow * beta2

    return param_out, moment1_out, moment2_out, beta1_pow_out, beta2_pow_out


class XPUTestLambOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'lamb'
        self.use_dynamic_create_class = False

    class TestLambOp1(XPUOpTest):

        def set_attrs(self):
            self.attrs = {
                'epsilon': 1e-4,
                'beta1': 0.78,
                'beta2': 0.836,
                'weight_decay': 0.01
            }

        def setUp(self):
            '''Test Lamb Op with supplied attributes
            '''
            # self.op_type = self.op_name
            self.__class__.op_type = 'lamb'
            self.dtype = self.in_type
            param = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
            grad = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
            moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
            moment2 = np.random.random((102, 105)).astype("float32")

            learning_rate = 0.001
            self.set_attrs()
            beta1_pow = self.attrs['beta1']
            beta2_pow = self.attrs['beta2']

            self.inputs = {
                'Param': param,
                'Grad': grad,
                'Moment1': moment1,
                'Moment2': moment2,
                'LearningRate': np.array([learning_rate]).astype("float32"),
                'Beta1Pow': np.array([beta1_pow]).astype("float32"),
                'Beta2Pow': np.array([beta2_pow]).astype("float32")
            }


            param_out, moment1_out, moment2_out, \
                beta1_pow_out, beta2_pow_out = lamb_step(self.inputs, self.attrs)

            self.outputs = {
                'Moment1Out': moment1_out,
                'Moment2Out': moment2_out,
                'ParamOut': param_out,
                'Beta1PowOut': beta1_pow_out,
                'Beta2PowOut': beta2_pow_out
            }

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0))

    class TestLambOp2(TestLambOp1):

        def set_attrs(self):
            self.attrs = {
                'epsilon': 1e-8,
                'beta1': 0.9,
                'beta2': 0.999,
                'weight_decay': 0.01
            }

    class TestLambOpMultipleSteps(TestLambOp1):

        def set_attrs(self):
            self.attrs = {
                'epsilon': 1e-8,
                'beta1': 0.9,
                'beta2': 0.999,
                'weight_decay': 0.01
            }
            self.num_steps = 10

        def test_check_output(self):
            for i in range(self.num_steps):
                param_out, moment1_out, moment2_out, \
                    beta1_pow_out, beta2_pow_out = lamb_step(self.inputs, self.attrs)

                self.outputs = {
                    'Moment1Out': moment1_out,
                    'Moment2Out': moment2_out,
                    'ParamOut': param_out,
                    'Beta1PowOut': beta1_pow_out,
                    'Beta2PowOut': beta2_pow_out
                }

                # Verify output for this step
                self.check_output()

                # Output of this step becomes input for next step
                self.inputs['Param'] = param_out
                self.inputs['Moment1'] = moment1_out
                self.inputs['Moment2'] = moment2_out

                # Update powers of Beta1 and Beta2 for next time step
                self.inputs['Beta1Pow'] = beta1_pow_out
                self.inputs['Beta2Pow'] = beta2_pow_out

                # Randomize gradient for next step
                self.inputs['Grad'] = np.random.uniform(
                    -1, 1, (102, 105)).astype("float32")


support_types = get_xpu_op_support_types('lamb')
for stype in support_types:
    create_test_class(globals(), XPUTestLambOp, stype)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
