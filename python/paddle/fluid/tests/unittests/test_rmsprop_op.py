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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
from paddle.fluid.op import Operator

import paddle.fluid.core as core


class TestBase(unittest.TestCase):
    def setup(self):
        self.param_name = "param"
        self.param = np.random.random((123, 321)).astype("float32")

        self.mean_square_name = "mean_square"
        self.mean_square = np.random.random((123, 321)).astype("float32")

        self.mean_grad_name = "mean_grad"
        self.mean_grad = np.random.random((123, 321)).astype("float32")

        self.lr_name = "lr"
        self.learning_rate = np.array([0.01]).astype("float32")

        self.grad_name = "grad"
        self.grad = np.random.random((123, 321)).astype("float32")

        self.moment_name = "moment"
        self.moment = np.zeros((123, 321)).astype("float32")

        self.epsilon = 1e-6
        self.decay = 0.9
        self.momentum = 0.0
        self.centered = False

        self.ms_out = self.decay * self.mean_square + (1 - self.decay
                                                       ) * self.grad * self.grad
        self.moment_out = self.momentum * self.moment + \
                          self.learning_rate * self.grad / np.sqrt(self.ms_out + self.epsilon)
        self.param_out = self.param - self.moment_out

    def check(self,
              actual_t,
              expect_t,
              place,
              out_name,
              atol=1e-5,
              equal_nan=False):
        self.assertTrue(
            np.allclose(
                actual_t, expect_t, atol=atol, equal_nan=equal_nan),
            "Output (" + out_name + ") has diff at " + str(place) + "\nExpect "
            + str(expect_t) + "\n" + "But Got" + str(actual_t))


class TestRmspropOp3(TestBase):
    def check_with_place(self, place):
        self.setup()
        scope = core.Scope()

        # create and initialize Param Variable
        param = scope.var(self.param_name).get_tensor()
        param.set(self.param, place)

        mean_square = scope.var(self.mean_square_name).get_tensor()
        mean_square.set(self.mean_square, place)

        lr = scope.var(self.lr_name).get_tensor()
        lr.set(self.learning_rate, place)

        grad = scope.var(self.grad_name).get_tensor()
        grad.set(self.grad, place)

        moment = scope.var(self.moment_name).get_tensor()
        moment.set(self.moment, place)

        # create and run sgd operator

        if self.centered:
            mean_grad = scope.var(self.mean_grad_name).get_tensor()
            mean_grad.set(self.mean_grad, place)

            rmsprop_op = Operator(
                "rmsprop",
                Param=self.param_name,
                Grad=self.grad_name,
                MeanSquare=self.mean_square_name,
                MeanGrad=self.mean_grad_name,
                Moment=self.moment_name,
                LearningRate=self.lr_name,
                ParamOut=self.param_name,
                MeanSquareOut=self.mean_square_name,
                MomentOut=self.moment_name,
                MeanGradOut=self.mean_grad,
                epsilon=self.epsilon,
                decay=self.decay,
                momentum=self.momentum,
                centered=True)
        else:
            rmsprop_op = Operator(
                "rmsprop",
                Param=self.param_name,
                Grad=self.grad_name,
                MeanSquare=self.mean_square_name,
                Moment=self.moment_name,
                LearningRate=self.lr_name,
                ParamOut=self.param_name,
                MeanSquareOut=self.mean_square_name,
                MomentOut=self.moment_name,
                epsilon=self.epsilon,
                decay=self.decay,
                momentum=self.momentum,
                centered=False)

        rmsprop_op.run(scope, place)

        self.check(np.array(param), self.param_out, place, self.param_name)
        self.check(
            np.array(mean_square), self.ms_out, place, self.mean_square_name)
        self.check(np.array(moment), self.moment_out, place, self.moment_name)

    def test_rmsprop(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)


class TestRmspropOp1(OpTest):
    ''' Test RMSProp with explicit inputs
    '''

    def setUp(self):
        self.op_type = "rmsprop"

        param = np.random.random((123, 321)).astype("float32")
        mean_square = np.random.random((123, 321)).astype("float32")
        learning_rate = np.array([0.01]).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")

        epsilon = 1e-6
        decay = 0.9
        momentum = 0.0

        self.inputs = {
            'Param': param,
            'MeanSquare': mean_square,
            'MeanGrad': mean_square,  # just a place holder
            'LearningRate': learning_rate,
            'Grad': grad,
            'Moment': moment,
        }

        self.attrs = {'epsilon': epsilon, 'decay': decay, 'momentum': momentum}

        ms_out = decay * mean_square + (1 - decay) * grad * grad
        moment_out = momentum * moment + \
            learning_rate * grad / np.sqrt(ms_out + epsilon)
        param_out = param - moment_out

        self.outputs = {
            'ParamOut': param_out,
            'MomentOut': moment_out,
            'MeanSquareOut': ms_out
        }

    def non_test_check_output(self):
        self.check_output()


class TestRmspropOp2(OpTest):
    '''Test RMSProp with default values for attributes
    '''

    def setUp(self):
        self.op_type = "rmsprop"

        param = np.random.random((123, 321)).astype("float32")
        mean_square = np.random.random((123, 321)).astype("float32")
        learning_rate = np.array([0.01]).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")

        epsilon = 1.0e-10
        decay = 0.9
        momentum = 0.0

        self.inputs = {
            'Param': param,
            'MeanSquare': mean_square,
            'MeanGrad': mean_square,  # just a place holder
            'LearningRate': learning_rate,
            'Grad': grad,
            'Moment': moment,
        }

        ms_out = decay * mean_square + (1 - decay) * grad * grad
        moment_out = momentum * moment + \
            learning_rate * grad / np.sqrt(ms_out + epsilon)
        param_out = param - moment_out

        self.outputs = {
            'ParamOut': param_out,
            'MomentOut': moment_out,
            'MeanSquareOut': ms_out
        }

    def non_test_check_output(self):
        self.check_output()


class TestRmspropOpV1Mode(OpTest):
    ''' Test RMSProp with explicit inputs
    '''

    def setUp(self):
        self.op_type = "rmsprop"

        param = np.random.random((123, 321)).astype("float32")
        mean_square = np.random.random((123, 321)).astype("float32")
        mean_grad = np.random.random((123, 321)).astype("float32")
        learning_rate = np.array([0.01]).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")

        epsilon = 1e-6
        decay = 0.9
        momentum = 0.0

        self.inputs = {
            'Param': param,
            'MeanSquare': mean_square,
            'MeanGrad': mean_grad,
            'LearningRate': learning_rate,
            'Grad': grad,
            'Moment': moment,
        }

        self.attrs = {
            'epsilon': epsilon,
            'decay': decay,
            'momentum': momentum,
            'centered': True
        }

        ms_out = decay * mean_square + (1 - decay) * grad * grad
        mg_out = decay * mean_grad + (1 - decay) * grad
        moment_out = momentum * moment + \
                     learning_rate * grad / np.sqrt(ms_out - np.square(mg_out) + epsilon)
        param_out = param - moment_out

        self.outputs = {
            'ParamOut': param_out,
            'MomentOut': moment_out,
            'MeanSquareOut': ms_out,
            'MeanGradOut': mg_out
        }

    def test_check_output(self):
        self.check_output(atol=1e-3, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
