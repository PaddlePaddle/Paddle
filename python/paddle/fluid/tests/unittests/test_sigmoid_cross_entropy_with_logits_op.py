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

import numpy as np
from op_test import OpTest
from scipy.special import logit
from scipy.special import expit
import paddle.fluid.core as core
import unittest
from paddle.fluid import compiler, Program, program_guard
import paddle.fluid as fluid
import paddle


def test_fluid_sigmoid(x, label, normalize=False, ignore_index=-100):
    return paddle.fluid.layers.sigmoid_cross_entropy_with_logits(
        x, label, int(ignore_index), normalize=normalize)


class TestSigmoidCrossEntropyWithLogitsOp1(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with binary label
    """

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = test_fluid_sigmoid
        batch_size = 64
        num_classes = 20
        self.inputs = {
            'X':
            logit(
                np.random.uniform(0, 1,
                                  (batch_size, num_classes)).astype("float64")),
            'Label':
            np.random.randint(0, 2, (batch_size, num_classes)).astype("float64")
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=True)


class TestSigmoidCrossEntropyWithLogitsOp2(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with probabalistic label
    """

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = test_fluid_sigmoid
        batch_size = 64
        num_classes = 20
        ignore_index = -1
        self.inputs = {
            'X':
            logit(
                np.random.uniform(0, 1,
                                  (batch_size, num_classes)).astype("float64")),
            'Label':
            np.random.randint(-1, 2,
                              (batch_size, num_classes)).astype("float64")
        }
        self.attrs = {
            'ignore_index': ignore_index,
        }
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
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=True)


class TestSigmoidCrossEntropyWithLogitsOp3(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with probabalistic label
    """

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = test_fluid_sigmoid
        batch_size = 64
        num_classes = 20
        self.inputs = {
            'X':
            logit(
                np.random.uniform(0, 1,
                                  (batch_size, num_classes)).astype("float64")),
            'Label':
            np.random.uniform(0, 1, (batch_size, num_classes)).astype("float64")
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=True)


class TestSigmoidCrossEntropyWithNorm(OpTest):

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = test_fluid_sigmoid
        batch_size = 64
        num_classes = 20
        ignore_index = -1
        self.inputs = {
            'X':
            logit(
                np.random.uniform(0, 1,
                                  (batch_size, num_classes)).astype("float64")),
            'Label':
            np.random.randint(-1, 2,
                              (batch_size, num_classes)).astype("float64")
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
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=True)


class TestSigmoidCrossEntropyWithLogitsOp5(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with probabalistic label
    """

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = test_fluid_sigmoid
        batch_size = [10, 10]
        num_classes = 20
        self.inputs = {
            'X':
            logit(
                np.random.uniform(
                    0, 1, tuple(batch_size + [num_classes])).astype("float64")),
            'Label':
            np.random.uniform(0, 1, tuple(batch_size +
                                          [num_classes])).astype("float64")
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=True)


class TestSigmoidCrossEntropyWithNorm2(OpTest):

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = test_fluid_sigmoid
        batch_size = [10, 10]
        num_classes = 20
        ignore_index = -1
        self.inputs = {
            'X':
            logit(
                np.random.uniform(
                    0, 1, tuple(batch_size + [num_classes])).astype("float64")),
            'Label':
            np.random.randint(-1, 2, tuple(batch_size +
                                           [num_classes])).astype("float64")
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
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=True)

    class TestSigmoidCrossEntropyWithLogitsOp6(OpTest):
        """Test sigmoid_cross_entropy_with_logit_op with binary label
        """

        def setUp(self):
            self.op_type = "sigmoid_cross_entropy_with_logits"
            self.python_api = test_fluid_sigmoid
            batch_size = [10, 10]
            num_classes = 20
            self.inputs = {
                'X':
                logit(
                    np.random.uniform(0, 1,
                                      tuple(batch_size +
                                            [num_classes])).astype("float64")),
                'Label':
                np.random.randint(0, 2, tuple(batch_size +
                                              [num_classes])).astype("float64")
            }

            # Fw Pass is implemented as elementwise sigmoid followed by
            # elementwise logistic loss
            # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
            sigmoid_X = expit(self.inputs['X'])
            term1 = self.inputs['Label'] * np.log(sigmoid_X)
            term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
            self.outputs = {'Out': -term1 - term2}

        def test_check_output(self):
            self.check_output(check_eager=True)

        def test_check_grad(self):
            self.check_grad(['X'], 'Out', check_eager=True)

    class TestSigmoidCrossEntropyWithLogitsOpError(unittest.TestCase):

        def test_errors(self):
            with program_guard(Program(), Program()):

                def test_Variable():
                    # the input of sigmoid_cross_entropy_with_logits must be Variable.
                    x1 = fluid.create_lod_tensor(np.array([-1, 3, 5,
                                                           5]), [[1, 1, 1, 1]],
                                                 fluid.CPUPlace())
                    lab1 = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]),
                                                   [[1, 1, 1, 1]],
                                                   fluid.CPUPlace())
                    fluid.layers.sigmoid_cross_entropy_with_logits(x1, lab1)

                self.assertRaises(TypeError, test_Variable)

                def test_dtype():
                    # the input dtype of sigmoid_cross_entropy_with_logits must be float16 or float32 or float64
                    # float16 only can be set on GPU place
                    x2 = fluid.layers.data(name='x2',
                                           shape=[3, 4, 5, 6],
                                           dtype="int32")
                    lab2 = fluid.layers.data(name='lab2',
                                             shape=[3, 4, 5, 6],
                                             dtype="int32")
                    fluid.layers.sigmoid_cross_entropy_with_logits(x2, lab2)

                self.assertRaises(TypeError, test_dtype)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
