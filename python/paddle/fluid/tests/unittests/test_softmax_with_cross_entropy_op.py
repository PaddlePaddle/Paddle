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


def cross_entropy(softmax, label, soft_label, axis):
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
            result[i, 0, j] += -np.log(softmax_reshape[i, lbl, j])
    return result.reshape(label.shape)


class TestSoftmaxWithCrossEntropyOp(OpTest):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    """

    def initParams(self):
        self.numeric_stable_mode = False
        self.soft_label = False
        self.dtype = np.float64
        self.axis = -1
        self.shape = [2, 3, 5, 7]
        self.label_shape = self.shape[:]
        self.label_shape[self.axis] = 1

    def setUp(self):
        self.initParams()
        self.op_type = "softmax_with_cross_entropy"

        logits = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
        softmax = np.apply_along_axis(stable_softmax, self.axis, logits)

        if self.soft_label:
            labels = np.random.uniform(0.1, 1.0, self.shape).astype("float64")
            labels /= np.sum(labels, axis=1, keepdims=True)
        else:
            labels = np.random.randint(
                0, self.shape[self.axis], self.label_shape, dtype="int64")

        loss = cross_entropy(softmax, labels, self.soft_label, self.axis)

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

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss", max_relative_error=0.05)


class TestSoftmaxWithCrossEntropyOpNoCudnn(TestSoftmaxWithCrossEntropyOp):
    def initParams(self):
        self.numeric_stable_mode = True
        self.soft_label = False
        self.axis = -1


class TestSoftmaxWithCrossEntropyOpFp16(TestSoftmaxWithCrossEntropyOp):
    def initParams(self):
        self.numeric_stable_mode = False
        self.soft_label = False
        self.axis = -1
        self.dtype = np.float16

    def setUp(self):
        self.initParams()
        self.op_type = "softmax_with_cross_entropy"
        batch_size = 41
        class_num = 37

        # NOTE: numpy float16 have very low accuracy, use float32 for numpy check.
        logits = np.random.uniform(0.1, 1.0,
                                   [batch_size, class_num]).astype(np.float32)
        softmax = np.apply_along_axis(stable_softmax, 1, logits)
        labels = np.random.randint(0, class_num, [batch_size, 1], dtype="int64")

        cross_entropy = np.asmatrix(
            [[-np.log(softmax[i][labels[i][0]])]
             for i in range(softmax.shape[0])],
            dtype=np.float32)

        self.inputs = {
            "Logits": logits.astype(self.dtype).view(np.uint16),
            "Label": labels
        }
        self.outputs = {
            "Softmax": softmax.astype(self.dtype),
            "Loss": cross_entropy.astype(self.dtype)
        }
        self.attrs = {"numeric_stable_mode": self.numeric_stable_mode}

    def test_check_output(self):
        self.check_output(atol=1e-2)

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss", max_relative_error=0.1)


# class TestSoftmaxWithCrossEntropyOpNoCudnnFp16(
#         TestSoftmaxWithCrossEntropyOpFp16):
#     def initParams(self):
#         self.numeric_stable_mode = True
#         self.axis = -1
#         self.dtype = np.float16
#
#     def test_check_grad(self):
#         self.check_grad(["Logits"], "Loss", max_relative_error=0.1)
#
#
# class TestSoftmaxWithCrossEntropyOp2(OpTest):
#     """
#     Test softmax with cross entropy operator with soft labels.
#     """
#
#     def setUp(self):
#         self.op_type = "softmax_with_cross_entropy"
#         self.axis = -1
#         batch_size = 41
#         class_num = 37
#
#         logits = np.random.uniform(0.1, 1.0,
#                                    [batch_size, class_num]).astype("float64")
#         softmax = np.apply_along_axis(stable_softmax, 1, logits)
#         labels = np.random.uniform(0.1, 1.0,
#                                    [batch_size, class_num]).astype("float64")
#         labels /= np.sum(labels, axis=1, keepdims=True)
#
#         cross_entropy = (-labels * np.log(softmax)).sum(
#             axis=1, keepdims=True).astype("float64")
#
#         self.inputs = {"Logits": logits, "Label": labels}
#         self.outputs = {
#             "Softmax": softmax.astype("float64"),
#             "Loss": cross_entropy.astype("float64")
#         }
#         self.attrs = {"soft_label": True, "axis": self.axis}
#
#     def test_check_output(self):
#         self.check_output()
#
#     def test_check_grad(self):
#         self.check_grad(["Logits"], "Loss")
#
#
# class TestSoftmaxWithCrossEntropyOp3(OpTest):
#     """
#     Test softmax with cross entropy operator with ignore_index.
#     """
#
#     def initParams(self):
#         self.numeric_stable_mode = False
#         self.axis = -1
#
#     def setUp(self):
#         self.initParams()
#         self.op_type = "softmax_with_cross_entropy"
#         batch_size = 41
#         class_num = 37
#
#         logits = np.random.uniform(0.1, 1.0,
#                                    [batch_size, class_num]).astype("float64")
#         softmax = np.apply_along_axis(stable_softmax, 1, logits)
#         labels = np.random.randint(0, class_num, [batch_size, 1], dtype="int64")
#         ignore_index = 7
#         cross_entropy = np.asmatrix(
#             [[-np.log(softmax[i][labels[i][0]])]
#              if labels[i] != ignore_index else [0]
#              for i in range(softmax.shape[0])],
#             dtype="float64")
#
#         self.inputs = {"Logits": logits, "Label": labels}
#         self.outputs = {
#             "Softmax": softmax.astype("float64"),
#             "Loss": cross_entropy.astype("float64")
#         }
#         self.attrs = {
#             "ignore_index": ignore_index,
#             "numeric_stable_mode": self.numeric_stable_mode,
#             "axis": self.axis
#         }
#
#     def test_check_output(self):
#         self.check_output()
#
#     def test_check_grad(self):
#         self.check_grad(["Logits"], "Loss")
#
#
# class TestSoftmaxWithCrossEntropyOp3NoCudnn(TestSoftmaxWithCrossEntropyOp3):
#     def initParams(self):
#         self.numeric_stable_mode = True
#         self.axis = -1
#
#
# if __name__ == "__main__":
#     unittest.main()
