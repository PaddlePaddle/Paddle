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

from op_test import OpTest
from test_softmax_op import stable_softmax


def cross_entropy(softmax, label, soft_label, axis, ignore_index=-1):
    if soft_label:
        return (-label * np.log(softmax)).sum(axis=axis, keepdims=True)

    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    axis_dim = shape[axis]
    remain = int(np.prod(shape[axis + 1:]))
    softmax_reshape = softmax.reshape((n, axis_dim, remain))
    label_reshape = label.reshape((n, 1, remain))
    result = np.zeros_like(label_reshape, dtype=softmax.dtype)
    for i in range(n):
        for j in range(remain):
            lbl = label_reshape[i, 0, j]
            if lbl != ignore_index:
                result[i, 0, j] -= np.log(softmax_reshape[i, lbl, j])
    return result.reshape(label.shape)


class TestSoftmaxWithCrossEntropyOp(OpTest):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = False
        self.soft_label = False
        self.dtype = np.float64
        self.axis = -1
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]

    def setUp(self):
        self.initParams()

        logits = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
        softmax = np.apply_along_axis(stable_softmax, self.axis, logits)

        if self.soft_label:
            labels = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
            labels /= np.sum(labels, axis=self.axis, keepdims=True)
        else:
            axis_dim = self.shape[self.axis]
            self.shape[self.axis] = 1
            labels = np.random.randint(0, axis_dim, self.shape, dtype="int64")

        loss = cross_entropy(softmax, labels, self.soft_label, self.axis,
                             self.ignore_index)

        self.inputs = {"Logits": logits, "Label": labels}
        self.outputs = {
            "Softmax": softmax.astype(self.dtype),
            "Loss": loss.astype(self.dtype)
        }
        self.attrs = {
            "numeric_stable_mode": self.numeric_stable_mode,
            "axis": self.axis,
            "soft_label": self.soft_label,
        }
        if self.ignore_index >= 0:
            self.attrs['ignore_index'] = self.ignore_index

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss", max_relative_error=0.05)


class TestSoftmaxWithCrossEntropyOpNoCudnn(TestSoftmaxWithCrossEntropyOp):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float64


class TestSoftmaxWithCrossEntropyOpFp16(TestSoftmaxWithCrossEntropyOp):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = False
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float16

    def setUp(self):
        self.initParams()
        self.op_type = "softmax_with_cross_entropy"

        # NOTE: numpy float16 have very low accuracy, use float32 for numpy check.
        logits = np.random.uniform(0.1, 1.0, self.shape).astype(np.float32)
        softmax = np.apply_along_axis(stable_softmax, self.axis, logits)

        axis_dim = self.shape[self.axis]
        self.shape[self.axis] = 1
        labels = np.random.randint(0, axis_dim, self.shape, dtype="int64")

        loss = cross_entropy(softmax, labels, self.soft_label, self.axis)

        self.inputs = {
            "Logits": logits.astype(self.dtype).view(np.uint16),
            "Label": labels
        }
        self.outputs = {
            "Softmax": softmax.astype(self.dtype),
            "Loss": loss.astype(self.dtype)
        }
        self.attrs = {
            "numeric_stable_mode": self.numeric_stable_mode,
            "axis": self.axis,
            "soft_label": self.soft_label,
        }

    def test_check_output(self):
        self.check_output(atol=1e-2)

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss", max_relative_error=0.1)


class TestSoftmaxWithCrossEntropyOpNoCudnnFp16(
        TestSoftmaxWithCrossEntropyOpFp16):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float16

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss", max_relative_error=0.1)


class TestSoftmaxWithCrossEntropyOp2(TestSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with soft labels.
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = True
        self.dtype = np.float64
        self.axis = -1
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss")


class TestSoftmaxWithCrossEntropyOp3(TestSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with ignore_index.
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = False
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.ignore_index = 5
        self.axis = -1
        self.dtype = np.float64


class TestSoftmaxWithCrossEntropyOp3NoCudnn(TestSoftmaxWithCrossEntropyOp3):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.ignore_index = 4
        self.axis = -1
        self.dtype = np.float64


class TestSoftmaxWithCrossEntropyOpAxis1(TestSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    Given axis != -1
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float64
        self.axis = 0
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]


class TestSoftmaxWithCrossEntropyOpAxis2(TestSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    Given axis != -1
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float64
        self.axis = 1
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]


class TestSoftmaxWithCrossEntropyOpAxis3(TestSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    Given axis != -1
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float64
        self.axis = 2
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]


class TestSoftmaxWithCrossEntropyOpNoCudnnFp16Axis1(
        TestSoftmaxWithCrossEntropyOpNoCudnnFp16):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = 0
        self.ignore_index = -1
        self.dtype = np.float16


class TestSoftmaxWithCrossEntropyOpNoCudnnFp16Axis2(
        TestSoftmaxWithCrossEntropyOpNoCudnnFp16):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = 1
        self.ignore_index = -1
        self.dtype = np.float16


class TestSoftmaxWithCrossEntropyOpNoCudnnFp16Axis3(
        TestSoftmaxWithCrossEntropyOpNoCudnnFp16):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = 2
        self.ignore_index = -1
        self.dtype = np.float16


class TestSoftmaxWithCrossEntropyOpSoftLabelAxis1(
        TestSoftmaxWithCrossEntropyOp2):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = True
        self.shape = [3, 5, 7, 11]
        self.axis = 0
        self.ignore_index = -1
        self.dtype = np.float64


class TestSoftmaxWithCrossEntropyOpSoftLabelAxis1(
        TestSoftmaxWithCrossEntropyOp2):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = True
        self.shape = [3, 5, 7, 11]
        self.axis = 1
        self.ignore_index = -1
        self.dtype = np.float64


class TestSoftmaxWithCrossEntropyOpSoftLabelAxis1(
        TestSoftmaxWithCrossEntropyOp2):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = True
        self.shape = [3, 5, 7, 11]
        self.axis = 2
        self.ignore_index = -1
        self.dtype = np.float64


class TestSoftmaxWithCrossEntropyOpIgnoreIndexNoCudnnAxis1(
        TestSoftmaxWithCrossEntropyOp3):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.ignore_index = 1
        self.axis = 0
        self.dtype = np.float64


class TestSoftmaxWithCrossEntropyOpIgnoreIndexNoCudnnAxis2(
        TestSoftmaxWithCrossEntropyOp3):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.ignore_index = 0
        self.axis = 1
        self.dtype = np.float64


class TestSoftmaxWithCrossEntropyOpIgnoreIndexNoCudnnAxis3(
        TestSoftmaxWithCrossEntropyOp3):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.ignore_index = 3
        self.axis = 2
        self.dtype = np.float64


if __name__ == "__main__":
    unittest.main()
