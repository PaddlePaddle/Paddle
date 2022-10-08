# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from test_kldiv_loss_op import kldiv_loss

paddle.enable_static()


class TestKLDivLossOp(OpTest):

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = 'float32'

    def setUp(self):
        self.set_npu()
        self.init_dtype()
        self.initTestCase()
        self.op_type = 'kldiv_loss'
        x = np.random.uniform(-10, 10, self.x_shape).astype(self.dtype)
        target = np.random.uniform(-10, 10, self.x_shape).astype(self.dtype)

        self.attrs = {"reduction": self.reduction}

        self.inputs = {
            'X': x,
            'Target': target,
        }
        loss = kldiv_loss(x, target, self.reduction)
        self.outputs = {'Loss': loss.astype(self.dtype)}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'],
                                   'Loss',
                                   no_grad_set=set(["Target"]),
                                   max_relative_error=0.15)

    def initTestCase(self):
        self.x_shape = (4, 5, 5)
        self.reduction = 'batchmean'


class TestKLDivLossOp2(TestKLDivLossOp):

    def initTestCase(self):
        self.x_shape = (3, 2, 7, 7)
        self.reduction = 'none'


class TestKLDivLossOp3(TestKLDivLossOp):

    def initTestCase(self):
        self.x_shape = (2, 3, 5, 7, 9)
        self.reduction = 'mean'


class TestKLDivLossOp4(TestKLDivLossOp):

    def initTestCase(self):
        self.x_shape = (5, 20)
        self.reduction = 'sum'


class TestKLDivLossOp_fp16(TestKLDivLossOp):

    def init_dtype(self):
        self.dtype = 'float16'

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=3e-1)

    def test_check_grad(self):
        input_grad = -self.inputs['Target'] * (
            self.inputs['Target'] > 0) / self.inputs['Target'].shape[0]
        self.check_grad_with_place(self.place, ['X'],
                                   'Loss',
                                   no_grad_set=set(["Target"]),
                                   max_relative_error=0.2,
                                   user_defined_grads=[input_grad])


class TestKLDivLossDygraph(unittest.TestCase):

    def run_kl_loss(self, reduction, shape=(5, 20)):
        x = np.random.uniform(-10, 10, shape).astype('float32')
        target = np.random.uniform(-10, 10, shape).astype('float32')
        gt_loss = kldiv_loss(x, target, reduction)

        with paddle.fluid.dygraph.guard(paddle.NPUPlace(0)):
            kldiv_criterion = paddle.nn.KLDivLoss(reduction)
            pred_loss = kldiv_criterion(paddle.to_tensor(x),
                                        paddle.to_tensor(target))
            np.testing.assert_allclose(pred_loss.numpy(), gt_loss, rtol=1e-6)

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

        pred_loss = paddle.nn.functional.kl_div(input, label)


class TestKLDivLossTypePromotion(unittest.TestCase):

    def test_kl_div_promotion(self):
        with paddle.fluid.dygraph.guard(paddle.NPUPlace(0)):
            x1 = paddle.rand([5, 20], dtype='float32')
            target1 = paddle.rand([5, 20], dtype='float32')

            kldiv_criterion = paddle.nn.KLDivLoss()
            pred_loss1 = kldiv_criterion(x1, target1)

            x2 = paddle.rand([5, 20], dtype='float32')
            target2 = paddle.rand([5, 20], dtype='float32')
            pred_loss2 = paddle.nn.functional.kl_div(x2, target2)


if __name__ == "__main__":
    unittest.main()
