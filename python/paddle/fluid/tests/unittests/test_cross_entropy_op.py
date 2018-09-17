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
from op_test import OpTest, randomize_probability


class TestCrossEntropyOp1(OpTest):
    """Test cross-entropy with discrete one-hot labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        batch_size = 30
        class_num = 10

        X = randomize_probability(batch_size, class_num, dtype='float64')

        label = np.random.randint(0, class_num, (batch_size, 1), dtype="int64")
        cross_entropy = np.asmatrix(
            [[-np.log(X[i][label[i][0]])] for i in range(X.shape[0])],
            dtype="float64")

        self.inputs = {"X": X, "Label": label}
        self.outputs = {"Y": cross_entropy}
        self.attrs = {"soft_label": False}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Y", numeric_grad_delta=0.001)


class TestCrossEntropyOp2(OpTest):
    """Test cross-entropy with vectorized soft labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        batch_size = 5
        class_num = 37

        X = randomize_probability(batch_size, class_num)
        label = np.random.uniform(0.1, 1.0,
                                  [batch_size, class_num]).astype("float32")
        label /= label.sum(axis=1, keepdims=True)
        cross_entropy = (-label * np.log(X)).sum(
            axis=1, keepdims=True).astype("float32")

        self.inputs = {"X": X, "Label": label}
        self.outputs = {"Y": cross_entropy}
        self.attrs = {"soft_label": True}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ["X"], "Y", max_relative_error=0.05, numeric_grad_delta=0.001)


class TestCrossEntropyOp3(OpTest):
    """Test cross-entropy with vectorized one-hot representation of labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        batch_size = 5
        class_num = 17

        X = randomize_probability(batch_size, class_num)
        label_index = np.random.randint(
            0, class_num, (batch_size), dtype="int32")
        label = np.zeros(X.shape)
        label[np.arange(batch_size), label_index] = 1

        cross_entropy = np.asmatrix(
            [[-np.log(X[i][label_index[i]])] for i in range(X.shape[0])],
            dtype="float32")
        cross_entropy2 = (-label * np.log(X)).sum(
            axis=1, keepdims=True).astype("float32")

        self.inputs = {"X": X, "Label": label.astype(np.float32)}
        self.outputs = {"Y": cross_entropy}
        self.attrs = {"soft_label": True}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ["X"], "Y", max_relative_error=0.05, numeric_grad_delta=0.001)


class TestCrossEntropyOp4(OpTest):
    """Test high rank tensor cross-entropy with discrete one-hot labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        shape = [10, 2, 4]
        ins_num = np.prod(np.array(shape))
        class_num = 10

        X_2d = randomize_probability(ins_num, class_num, dtype='float64')

        label_2d = np.random.randint(0, class_num, (ins_num, 1), dtype="int64")
        cross_entropy_2d = np.asmatrix(
            [[-np.log(X_2d[i][label_2d[i][0]])] for i in range(X_2d.shape[0])],
            dtype="float64")

        X = X_2d.reshape(shape + [class_num])
        label = label_2d.reshape(shape + [1])
        cross_entropy = np.array(cross_entropy_2d).reshape(shape + [1])

        self.inputs = {"X": X, "Label": label}
        self.outputs = {"Y": cross_entropy}
        self.attrs = {"soft_label": False}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Y", numeric_grad_delta=0.001)


class TestCrossEntropyOp5(OpTest):
    """Test high rank tensor cross-entropy with vectorized soft labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        shape = [4, 3]
        ins_num = np.prod(np.array(shape))
        class_num = 37

        X_2d = randomize_probability(ins_num, class_num)
        label_2d = np.random.uniform(0.1, 1.0,
                                     [ins_num, class_num]).astype("float32")
        label_2d /= label_2d.sum(axis=1, keepdims=True)
        cross_entropy_2d = (-label_2d * np.log(X_2d)).sum(
            axis=1, keepdims=True).astype("float32")

        X = X_2d.reshape(shape + [class_num])
        label = label_2d.reshape(shape + [class_num])
        cross_entropy = np.array(cross_entropy_2d).reshape(shape + [1])

        self.inputs = {"X": X, "Label": label}
        self.outputs = {"Y": cross_entropy}
        self.attrs = {"soft_label": True}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ["X"], "Y", max_relative_error=0.05, numeric_grad_delta=0.001)


class TestCrossEntropyOp6(OpTest):
    """Test high rank tensor cross-entropy with vectorized one-hot representation of labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        shape = [4, 3, 2]
        ins_num = np.prod(np.array(shape))
        class_num = 17

        X_2d = randomize_probability(ins_num, class_num)
        label_index_2d = np.random.randint(
            0, class_num, (ins_num), dtype="int32")
        label_2d = np.zeros(X_2d.shape)
        label_2d[np.arange(ins_num), label_index_2d] = 1

        cross_entropy_2d = np.asmatrix(
            [[-np.log(X_2d[i][label_index_2d[i]])]
             for i in range(X_2d.shape[0])],
            dtype="float32")

        X = X_2d.reshape(shape + [class_num])
        label = label_2d.reshape(shape + [class_num])
        cross_entropy = np.array(cross_entropy_2d).reshape(shape + [1])

        self.inputs = {"X": X, "Label": label.astype(np.float32)}
        self.outputs = {"Y": cross_entropy}
        self.attrs = {"soft_label": True}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ["X"], "Y", max_relative_error=0.05, numeric_grad_delta=0.001)


class TestCrossEntropyOp7(OpTest):
    """Test cross-entropy with ignore index.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        batch_size = 30
        class_num = 10
        ignore_index = 3

        X = randomize_probability(batch_size, class_num, dtype='float64')

        label = np.random.randint(0, class_num, (batch_size, 1), dtype="int64")
        cross_entropy = np.asmatrix(
            [[-np.log(X[i][label[i][0]])]
             if label[i][0] != ignore_index else [0]
             for i in range(X.shape[0])],
            dtype="float64")
        self.inputs = {"X": X, "Label": label}
        self.outputs = {"Y": cross_entropy}
        self.attrs = {"soft_label": False, "ignore_index": ignore_index}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Y", numeric_grad_delta=0.001)


if __name__ == "__main__":
    unittest.main()
