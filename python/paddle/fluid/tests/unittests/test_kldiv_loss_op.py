#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software # distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

<<<<<<< HEAD
import unittest

import numpy as np
from op_test import OpTest

import paddle
=======
from __future__ import division

import paddle
import unittest
import numpy as np
from op_test import OpTest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from paddle.nn.functional import kl_div


def kldiv_loss(x, target, reduction):
    output = target * (np.log(target) - x)
    loss = np.where(target >= 0, output, np.zeros_like(x))

    if reduction == "batchmean":
        if len(x.shape) > 0:
            return loss.sum() / x.shape[0]
        else:
            return loss.sum()
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()

    return loss


class TestKLDivLossOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.initTestCase()
        self.op_type = 'kldiv_loss'
        self.python_api = kl_div
        x = np.random.uniform(-10, 10, self.x_shape).astype('float64')
        target = np.random.uniform(-10, 10, self.x_shape).astype('float64')

        self.attrs = {"reduction": self.reduction}

        self.inputs = {
            'X': x,
            'Target': target,
        }
        loss = kldiv_loss(x, target, self.reduction)
        self.outputs = {'Loss': loss.astype('float64')}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad(
            ['X'], 'Loss', no_grad_set=set(["Target"]), check_eager=True
        )
=======
        self.check_grad(['X'],
                        'Loss',
                        no_grad_set=set(["Target"]),
                        check_eager=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def initTestCase(self):
        self.x_shape = (4, 5, 5)
        self.reduction = 'batchmean'


class TestKLDivLossOp2(TestKLDivLossOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.x_shape = (3, 2, 7, 7)
        self.reduction = 'none'


class TestKLDivLossOp3(TestKLDivLossOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.x_shape = (2, 3, 5, 7, 9)
        self.reduction = 'mean'


class TestKLDivLossOp4(TestKLDivLossOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.x_shape = (5, 20)
        self.reduction = 'sum'


class TestKLDivLossDygraph(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def run_kl_loss(self, reduction, shape=(5, 20)):
        x = np.random.uniform(-10, 10, shape).astype('float64')
        target = np.random.uniform(-10, 10, shape).astype('float64')
        gt_loss = kldiv_loss(x, target, reduction)

        with paddle.fluid.dygraph.guard():
            kldiv_criterion = paddle.nn.KLDivLoss(reduction)
<<<<<<< HEAD
            pred_loss = kldiv_criterion(
                paddle.to_tensor(x), paddle.to_tensor(target)
            )
=======
            pred_loss = kldiv_criterion(paddle.to_tensor(x),
                                        paddle.to_tensor(target))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            np.testing.assert_allclose(pred_loss.numpy(), gt_loss, rtol=1e-05)

    def test_kl_loss_batchmean(self):
        self.run_kl_loss('batchmean')

    def test_kl_loss_batchmean_shape(self):
        self.run_kl_loss('batchmean', ())

    def test_kl_loss_mean(self):
        self.run_kl_loss('mean')

    def test_kl_loss_sum(self):
        self.run_kl_loss('sum')

    def test_kl_loss_none(self):
        self.run_kl_loss('none')

    def test_kl_loss_static_api(self):
        input = paddle.fluid.data(name='input', shape=[5, 20])
        label = paddle.fluid.data(name='label', shape=[5, 20])

        paddle.nn.functional.kl_div(input, label)
        paddle.nn.functional.kl_div(input, label, 'sum')
        paddle.nn.functional.kl_div(input, label, 'batchmean')


class TestKLDivLossTypePromotion(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_kl_div_promotion(self):

        with paddle.fluid.dygraph.guard():
            x1 = paddle.rand([5, 20], dtype='float32')
            target1 = paddle.rand([5, 20], dtype='float64')

            kldiv_criterion = paddle.nn.KLDivLoss()
            pred_loss1 = kldiv_criterion(x1, target1)

            x2 = paddle.rand([5, 20], dtype='float64')
            target2 = paddle.rand([5, 20], dtype='float32')
            pred_loss2 = paddle.nn.functional.kl_div(x2, target2)


if __name__ == "__main__":
    unittest.main()
