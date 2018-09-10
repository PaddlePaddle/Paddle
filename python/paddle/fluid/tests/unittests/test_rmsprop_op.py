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
import paddle.fluid.core as core
from paddle.fluid.op import Operator


class TestBase(unittest.TestCase):
    def setup(self, centered, epsilon=1e-6):
        np.random.seed(5)  # fix seed

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

        self.epsilon = epsilon
        self.decay = 0.9
        self.momentum = 0.0
        self.centered = centered

        self.ms_out = self.decay * self.mean_square + (1 - self.decay
                                                       ) * self.grad * self.grad
        if centered:
            self.mg_out = self.decay * self.mean_grad + (1 - self.decay
                                                         ) * self.grad
            self.moment_out = self.momentum * self.moment + \
                              self.learning_rate * self.grad / np.sqrt(self.ms_out - np.square(self.mg_out) + self.epsilon)
        else:
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


class TestRmspropOp(TestBase):
    def check_with_place(self, place, centered, epsilon):
        self.setup(centered, epsilon)
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
                MeanGradOut=self.mean_grad_name,
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

        atol = 1e-5
        equal_nan = False

        if self.centered:
            atol = 1e-3
            equal_nan = True

        self.check(
            np.array(mean_square), self.ms_out, place, self.mean_square_name)
        self.check(
            np.array(moment),
            self.moment_out,
            place,
            self.moment_name,
            atol=atol,
            equal_nan=equal_nan)
        self.check(
            np.array(param),
            self.param_out,
            place,
            self.param_name,
            atol=atol,
            equal_nan=equal_nan)

        if self.centered:
            self.check(
                np.array(mean_grad), self.mg_out, place, self.mean_grad_name)

    def test_rmsprop(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place, False, 1e-6)
            self.check_with_place(place, False, 1e-10)
            self.check_with_place(place, True, 1e-6)
            self.check_with_place(place, True, 1e-10)


if __name__ == "__main__":
    unittest.main()
