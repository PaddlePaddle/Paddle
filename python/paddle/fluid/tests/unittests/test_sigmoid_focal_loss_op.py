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
import math
import copy
from op_test import OpTest
from paddle.fluid import core


def sigmoid_focal_loss_forward(x_data, label_data, fg_num_data, gamma, alpha,
                               num_classes):
    x_data_t = copy.deepcopy(x_data)
    out_data = copy.deepcopy(x_data)
    x_width = len(x_data)
    x_height = len(x_data[0, :])
    x_data_t = x_data_t.flatten()
    out_data = out_data.flatten()
    for idx in range(len(x_data_t)):
        x = x_data_t[idx]
        a = int(idx / num_classes)
        d = int(idx % num_classes)
        label = label_data[a]
        c_pos = float((int(label) == int(d + 1)))
        c_neg = float(((int(label) != -1) & (int(label) != (d + 1))))
        fg_num = max(fg_num_data, 1)
        z_neg = (1.0 - alpha) / fg_num
        z_pos = alpha / fg_num

        p = 1. / (1. + math.exp(-x))
        FLT_MIN = 1.175494351e-38
        term_pos = math.pow((1. - p), gamma) * math.log(max(FLT_MIN, p))
        term_neg = math.pow(p, gamma) * (
            -1. * x * (x >= 0) - math.log(1. + math.exp(x - 2. * x * (x >= 0))))
        out_data[idx] = 0.0
        out_data[idx] += -c_pos * term_pos * z_pos
        out_data[idx] += -c_neg * term_neg * z_neg

    out_data = out_data.reshape(x_width, x_height)
    return out_data


class TestSigmoidFocalLossOp1(OpTest):
    def set_argument(self):
        self.num_anchors = 10
        self.num_classes = 10
        self.gamma = 2.0
        self.alpha = 0.25

    def setUp(self):
        self.set_argument()

        dims = (self.num_anchors, self.num_classes)
        X = np.random.standard_normal(dims).astype("float64")
        L = np.random.randint(0, self.num_classes + 1,
                              (dims[0], 1)).astype("int32")
        F = np.zeros(1)
        F[0] = len(np.where(L > 0)[0])
        F = F.astype("int32")

        self.op_type = "sigmoid_focal_loss"
        self.inputs = {
            'X': X,
            'Label': L,
            'FgNum': F,
        }
        self.attrs = {
            'gamma': self.gamma,
            'alpha': self.alpha,
        }
        loss = sigmoid_focal_loss_forward(
            self.inputs['X'], self.inputs['Label'], self.inputs['FgNum'],
            self.gamma, self.alpha, self.num_classes)
        self.outputs = {'Out': loss.astype('float64')}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSigmoidFocalLossOp2(TestSigmoidFocalLossOp1):
    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=2e-3)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place, ['X'], 'Out', max_relative_error=0.002)


class TestSigmoidFocalLossOp3(TestSigmoidFocalLossOp1):
    def set_argument(self):
        self.num_anchors = 200
        self.num_classes = 10
        self.gamma = 1.0
        self.alpha = 0.5


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSigmoidFocalLossOp4(TestSigmoidFocalLossOp3):
    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=2e-3)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place, ['X'], 'Out', max_relative_error=0.002)


if __name__ == '__main__':
    unittest.main()
