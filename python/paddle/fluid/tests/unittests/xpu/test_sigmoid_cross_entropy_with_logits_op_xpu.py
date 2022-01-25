#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("..")
from op_test_xpu import OpTest, XPUOpTest
from op_test import skip_check_grad_ci
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.framework import convert_np_dtype_to_dtype_

from scipy.special import logit
from scipy.special import expit

paddle.enable_static()


class TestSigmoidCrossEntropyWithLogitsOp1(XPUOpTest):
    """Test sigmoid_cross_entropy_with_logit_op with binary label
    """

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.set_xpu()
        self.init_dtype()

        batch_size = 64
        num_classes = 20
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (batch_size, num_classes))
                .astype(self.dtype)),
            'Label': np.random.randint(0, 2, (batch_size, num_classes))
            .astype(self.dtype)
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')

    def set_xpu(self):
        self.__class__.use_xpu = True
        self.place = paddle.XPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32


class TestSigmoidCrossEntropyWithLogitsOp3(
        TestSigmoidCrossEntropyWithLogitsOp1):
    """Test sigmoid_cross_entropy_with_logit_op with probabalistic label
    """

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.set_xpu()
        self.init_dtype()

        batch_size = 64
        num_classes = 20
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (batch_size, num_classes))
                .astype(self.dtype)),
            'Label': np.random.uniform(0, 1, (batch_size, num_classes))
            .astype(self.dtype)
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}


class TestSigmoidCrossEntropyWithLogitsOp5(
        TestSigmoidCrossEntropyWithLogitsOp1):
    """Test sigmoid_cross_entropy_with_logit_op with probabalistic label
    """

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.set_xpu()
        self.init_dtype()

        batch_size = [10, 10]
        num_classes = 20
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, tuple(batch_size + [num_classes]))
                .astype(self.dtype)),
            'Label': np.random.uniform(0, 1, tuple(batch_size + [num_classes]))
            .astype(self.dtype)
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}


class TestSigmoidCrossEntropyWithLogitsOp6(
        TestSigmoidCrossEntropyWithLogitsOp1):
    """Test sigmoid_cross_entropy_with_logit_op with binary label
    """

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.set_xpu()
        self.init_dtype()

        batch_size = [10, 10]
        num_classes = 20
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, tuple(batch_size + [num_classes]))
                .astype(self.dtype)),
            'Label': np.random.randint(0, 2, tuple(batch_size + [num_classes]))
            .astype(self.dtype)
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}


if __name__ == '__main__':
    unittest.main()
