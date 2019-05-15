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

import numpy as np
from op_test import OpTest
from scipy.special import logit
from scipy.special import expit
import paddle.fluid.core as core
import unittest


class TestSigmoidCrossEntropyWithLogitsOp1(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with binary label
    """

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        batch_size = 64
        num_classes = 20
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (batch_size, num_classes))
                .astype("float32")),
            'Label': np.random.randint(0, 2, (batch_size, num_classes))
            .astype("float32")
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSigmoidCrossEntropyWithLogitsOp2(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with probabalistic label
    """

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        batch_size = 64
        num_classes = 20
        ignore_index = -1
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (batch_size, num_classes))
                .astype("float32")),
            'Label': np.random.randint(-1, 2, (batch_size, num_classes))
            .astype("float32")
        }
        self.attrs = {'ignore_index': ignore_index, }
        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        out = -term1 - term2
        out[np.where(self.inputs['Label'] == ignore_index)] = 0
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSigmoidCrossEntropyWithLogitsOp3(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with probabalistic label
    """

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        batch_size = 64
        num_classes = 20
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (batch_size, num_classes))
                .astype("float32")),
            'Label': np.random.uniform(0, 1, (batch_size, num_classes))
            .astype("float32")
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSigmoidCrossEntropyWithNorm(OpTest):
    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        batch_size = 64
        num_classes = 20
        ignore_index = -1
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (batch_size, num_classes))
                .astype("float32")),
            'Label': np.random.randint(-1, 2, (batch_size, num_classes))
            .astype("float32")
        }
        self.attrs = {'ignore_index': ignore_index, 'normalize': True}
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        out = -term1 - term2
        out[np.where(self.inputs['Label'] == ignore_index)] = 0
        if self.attrs['normalize']:
            out = out / float(
                np.where(self.inputs['Label'] != ignore_index)[0].size)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSigmoidCrossEntropyWithLogitsOp5(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with probabalistic label
    """

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        batch_size = [10, 10]
        num_classes = 20
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, tuple(batch_size + [num_classes]))
                .astype("float32")),
            'Label': np.random.uniform(0, 1, tuple(batch_size + [num_classes]))
            .astype("float32")
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSigmoidCrossEntropyWithNorm2(OpTest):
    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        batch_size = [10, 10]
        num_classes = 20
        ignore_index = -1
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, tuple(batch_size + [num_classes]))
                .astype("float32")),
            'Label': np.random.randint(-1, 2, tuple(batch_size + [num_classes]))
            .astype("float32")
        }
        self.attrs = {'ignore_index': ignore_index, 'normalize': True}
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        out = -term1 - term2
        out[np.where(self.inputs['Label'] == ignore_index)] = 0
        if self.attrs['normalize']:
            out = out / float(
                np.where(self.inputs['Label'] != ignore_index)[0].size)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSigmoidCrossEntropyWithLogitsOp6(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with binary label
    """

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        batch_size = [10, 10]
        num_classes = 20
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, tuple(batch_size + [num_classes]))
                .astype("float32")),
            'Label': np.random.randint(0, 2, tuple(batch_size + [num_classes]))
            .astype("float32")
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
