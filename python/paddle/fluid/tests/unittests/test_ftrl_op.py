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
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from op_test import OpTest


def ftrl_step(param, grad, rows, sq_accum, lin_accum, lr, l1, l2, lr_power):
    l1 += 1e-10
    l2 += 1e-10

    param_hit = param[rows]
    sq_accum_hit = sq_accum[rows]
    lin_accum_hit = lin_accum[rows]

    new_accum = sq_accum_hit + grad * grad
    if lr_power == -0.5:
        lin_accum_updated = lin_accum_hit + grad - (
            (np.sqrt(new_accum) - np.sqrt(sq_accum_hit)) / lr) * param_hit
    else:
        lin_accum_updated = lin_accum_hit + grad - (
            (np.power(new_accum, -lr_power) - np.power(sq_accum_hit, -lr_power))
            / lr) * param_hit

    x = l1 * np.sign(lin_accum_updated) - lin_accum_updated
    if lr_power == -0.5:
        y = (np.sqrt(new_accum) / lr) + (2 * l2)
        pre_shrink = x / y
        param_updated = np.where(
            np.abs(lin_accum_updated) > l1, pre_shrink, 0.0)
    else:
        y = (np.power(new_accum, -lr_power) / lr) + (2 * l2)
        pre_shrink = x / y
        param_updated = np.where(
            np.abs(lin_accum_updated) > l1, pre_shrink, 0.0)

    sq_accum_updated = sq_accum_hit + grad * grad

    param_out = param.copy()
    sq_accum_out = sq_accum.copy()
    lin_accum_out = lin_accum.copy()

    for i in range(len(rows)):
        param_out[rows[i]] = param_updated[i]
        sq_accum_out[rows[i]] = sq_accum_updated[i]
        lin_accum_out[rows[i]] = lin_accum_updated[i]

    return param_out, sq_accum_out, lin_accum_out


class TestFTRLOp(OpTest):

    def setUp(self):
        self.op_type = "ftrl"
        rows = 102
        w = np.random.random((rows, 105)).astype("float32")
        g = np.random.random((rows, 105)).astype("float32")
        sq_accum = np.full((rows, 105), 0.1).astype("float32")
        linear_accum = np.full((rows, 105), 0.1).astype("float32")
        lr = np.array([0.01]).astype("float32")
        l1 = 0.1
        l2 = 0.2
        lr_power = -0.5

        self.inputs = {
            'Param': w,
            'SquaredAccumulator': sq_accum,
            'LinearAccumulator': linear_accum,
            'Grad': g,
            'LearningRate': lr
        }
        self.attrs = {
            'l1': l1,
            'l2': l2,
            'lr_power': lr_power,
            'learning_rate': lr
        }

        param_out, sq_accum_out, lin_accum_out = ftrl_step(
            w, g, range(rows), sq_accum, linear_accum, lr, l1, l2, lr_power)

        self.outputs = {
            'ParamOut': param_out,
            'SquaredAccumOut': sq_accum_out,
            'LinearAccumOut': lin_accum_out
        }

    def test_check_output(self):
        self.check_output(check_eager=True)


class TestSparseFTRLOp(unittest.TestCase):

    def setUp(self):
        self.lr_power = -0.5

    def check_with_place(self, place):
        self.init_kernel()
        scope = core.Scope()

        height = 10
        rows = [0, 4, 7]
        row_numel = 12
        l1 = 0.1
        l2 = 0.2
        lr_power = self.lr_power

        # create and initialize Param Variable
        param = scope.var('Param').get_tensor()
        param_array = np.random.random((height, row_numel)).astype("float32")
        param.set(param_array, place)

        # create and initialize Grad Variable
        grad = scope.var('Grad').get_selected_rows()
        grad.set_height(height)
        grad.set_rows(rows)
        grad_array = np.random.random((len(rows), row_numel)).astype("float32")

        grad_tensor = grad.get_tensor()
        grad_tensor.set(grad_array, place)

        # create and initialize SquaredAccumulator Variable
        sq_accum = scope.var('SquaredAccumulator').get_tensor()
        sq_accum_array = np.full((height, row_numel), 0.1).astype("float32")
        sq_accum.set(sq_accum_array, place)

        # create and initialize LinearAccumulator Variable
        lin_accum = scope.var('LinearAccumulator').get_tensor()
        lin_accum_array = np.full((height, row_numel), 0.1).astype("float32")
        lin_accum.set(lin_accum_array, place)

        # create and initialize LeraningRate Variable
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.array([0.01]).astype("float32")
        lr.set(lr_array, place)

        # calculate ground-truth answer
        param_out, sq_accum_out, lin_accum_out = ftrl_step(
            param_array, grad_array, rows, sq_accum_array, lin_accum_array, lr,
            l1, l2, lr_power)

        # create and run operator
        op = Operator("ftrl",
                      Param='Param',
                      Grad='Grad',
                      ParamOut='Param',
                      SquaredAccumulator='SquaredAccumulator',
                      SquaredAccumOut='SquaredAccumulator',
                      LinearAccumulator='LinearAccumulator',
                      LinearAccumOut='LinearAccumulator',
                      LearningRate='LearningRate',
                      l1=l1,
                      l2=l2,
                      lr_power=lr_power)

        op.run(scope, place)

        # get and compare param result
        param_array = np.array(param)
        sq_accum_array = np.array(sq_accum)
        lin_accum_array = np.array(lin_accum)

        for i in range(height):
            for j in range(row_numel):
                self.assertAlmostEqual(param_out[i][j],
                                       param_array[i][j],
                                       places=4)
                self.assertAlmostEqual(sq_accum_out[i][j],
                                       sq_accum_array[i][j],
                                       places=4)
                self.assertAlmostEqual(lin_accum_out[i][j],
                                       lin_accum_array[i][j],
                                       places=4)

    def init_kernel(self):
        pass

    def test_sparse_ftrl(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)


class TestSparseFTRLOp2(TestSparseFTRLOp):

    def init_kernel(self):
        self.lr_power = -0.6


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    unittest.main()
