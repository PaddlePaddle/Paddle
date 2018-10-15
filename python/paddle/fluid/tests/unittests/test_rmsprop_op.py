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
import paddle.fluid as fluid


def create_selected_rows_and_tensor(scope, place, height, row_num,
                                    embedding_size):
    sr = scope.var("@selected_rows@").get_selected_rows()
    tensor = scope.var("grad").get_tensor()

    rows = np.random.random_integers(
        low=0, high=height - 1, size=[row_num, ]).astype('int64')
    sr_val = np.random.random(size=[row_num, embedding_size]).astype('float32')

    sr.set_height(height)
    sr.set_rows(rows)
    sr.get_tensor().set(sr_val, place)

    tensor_val = np.zeros(shape=[height, embedding_size], dtype='float32')
    for i in range(row_num):
        row = rows[i]
        tensor_val[row, :] = tensor_val[row, :] + sr_val[i, :]

    tensor.set(tensor_val, place)
    return tensor_val, sr_val


class TestBase(unittest.TestCase):
    def setup(self,
              place,
              is_sparse,
              centered,
              size,
              row_num=None,
              epsilon=1e-6):
        np.random.seed(5)  # fix seed

        self.scope = fluid.global_scope()
        self.place = place

        self.param_name = "param"
        self.param = np.random.random(size).astype("float32")

        self.mean_square_name = "mean_square"
        self.mean_square = np.random.uniform(
            low=1, high=2, size=size).astype("float32")

        self.mean_grad_name = "mean_grad"
        self.mean_grad = np.random.random(size).astype("float32")

        self.lr_name = "lr"
        self.learning_rate = np.array([0.01]).astype("float32")

        self.grad_name = "grad"

        self.is_sparse = is_sparse
        if self.is_sparse:
            self.grad_sr_name = "@selected_rows@"
            self.grad, self.grad_sr = create_selected_rows_and_tensor(
                self.scope, place, size[0], row_num, size[1])
        else:
            self.grad = np.random.random(size).astype("float32")
            grad_tensor = self.scope.var(self.grad_name).get_tensor()
            grad_tensor.set(self.grad, place)

        self.moment_name = "moment"
        self.moment = np.random.uniform(
            low=0, high=1, size=size).astype("float32")

        self.epsilon = epsilon
        self.decay = 0.9
        self.momentum = 0.1
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

        # create and initialize Param Variable
        self.param_tensor = self.scope.var(self.param_name).get_tensor()
        self.param_tensor.set(self.param, place)

        self.mean_square_tensor = self.scope.var(
            self.mean_square_name).get_tensor()
        self.mean_square_tensor.set(self.mean_square, place)

        lr = self.scope.var(self.lr_name).get_tensor()
        lr.set(self.learning_rate, place)

        self.moment_tensor = self.scope.var(self.moment_name).get_tensor()
        self.moment_tensor.set(self.moment, place)

        if self.centered:
            self.mean_grad_tensor = self.scope.var(
                self.mean_grad_name).get_tensor()
            self.mean_grad_tensor.set(self.mean_grad, place)

    def check(self, actual_t, expect_t, place, out_name, atol=1e-5):
        self.assertTrue(
            np.allclose(
                actual_t, expect_t, atol=atol),
            "Output (" + out_name + ") has diff at " + str(place) + "\nExpect "
            + str(expect_t) + "\n" + "But Got" + str(actual_t))


class TestRmspropOp(TestBase):
    def check_with_place(self,
                         place,
                         is_sparse,
                         centered,
                         size,
                         row_num=None,
                         epsilon=1e-6):
        self.setup(place, is_sparse, centered, size, row_num, epsilon)
        self.run_and_check()

    def run_and_check(self):
        grad_name = self.grad_sr_name if self.is_sparse else self.grad_name

        kwargs = {
            'Param': self.param_name,
            'Grad': grad_name,
            'MeanSquare': self.mean_square_name,
            'Moment': self.moment_name,
            'LearningRate': self.lr_name,
            'ParamOut': self.param_name,
            'MeanSquareOut': self.mean_square_name,
            'MomentOut': self.moment_name,
            'epsilon': self.epsilon,
            'decay': self.decay,
            'momentum': self.momentum,
            'centered': self.centered
        }

        if self.centered:
            kwargs['MeanGrad'] = self.mean_grad_name
            kwargs['MeanGradOut'] = self.mean_grad_name

        rmsprop_op = Operator('rmsprop', **kwargs)
        atol = 1e-6

        rmsprop_op.run(self.scope, self.place)

        self.check(
            np.array(self.mean_square_tensor),
            self.ms_out,
            self.place,
            self.mean_square_name,
            atol=atol)
        self.check(
            np.array(self.moment_tensor),
            self.moment_out,
            self.place,
            self.moment_name,
            atol=atol)
        self.check(
            np.array(self.param_tensor),
            self.param_out,
            self.place,
            self.param_name,
            atol=atol)

        if self.centered:
            self.check(
                np.array(self.mean_grad_tensor), self.mg_out, self.place,
                self.mean_grad_name)

    def test_rmsprop(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        size = (128, 320)
        for place in places:
            for centered in [False, True]:
                with fluid.scope_guard(core.Scope()):
                    self.check_with_place(
                        place, is_sparse=False, centered=centered, size=size)

                with fluid.scope_guard(core.Scope()):
                    self.check_with_place(
                        place,
                        is_sparse=True,
                        centered=centered,
                        row_num=512,
                        size=size)

                with fluid.scope_guard(core.Scope()):
                    self.check_with_place(
                        place,
                        is_sparse=True,
                        centered=centered,
                        row_num=60,
                        size=size)


if __name__ == "__main__":
    unittest.main()
