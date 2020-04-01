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

from __future__ import division

import unittest
import numpy as np
from op_test import OpTest


def kldiv_loss(x, target, reduction):
    output = target * (np.log(target) - x)
    loss = np.where(target >= 0, output, np.zeros_like(x))

    if reduction == "batchmean":
        return loss.sum() / x.shape[0]
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()

    return loss


class TestKLDivLossOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'kldiv_loss'
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
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Loss', no_grad_set=set(["Target"]))

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


if __name__ == "__main__":
    unittest.main()
