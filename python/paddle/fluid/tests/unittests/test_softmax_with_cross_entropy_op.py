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


class TestSoftmaxWithCrossEntropyOp(OpTest):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    """

    def initParams(self):
        self.numeric_stable_mode = False
        self.dtype = np.float64

    def setUp(self):
        self.initParams()
        self.op_type = "softmax_with_cross_entropy"
        batch_size = 41
        class_num = 37

        logits = np.random.uniform(0.1, 1.0,
                                   [batch_size, class_num]).astype(self.dtype)
        softmax = np.apply_along_axis(stable_softmax, 1, logits)
        labels = np.random.randint(0, class_num, [batch_size, 1], dtype="int64")

        cross_entropy = np.asmatrix(
            [[-np.log(softmax[i][labels[i][0]])]
             for i in range(softmax.shape[0])],
            dtype=self.dtype)

        self.inputs = {"Logits": logits, "Label": labels}
        self.outputs = {
            "Softmax": softmax.astype(self.dtype),
            "Loss": cross_entropy.astype(self.dtype)
        }
        self.attrs = {"numeric_stable_mode": self.numeric_stable_mode}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss", max_relative_error=0.05)


class TestSoftmaxWithCrossEntropyOpNoCudnn(TestSoftmaxWithCrossEntropyOp):
    def initParams(self):
        self.numeric_stable_mode = True


class TestSoftmaxWithCrossEntropyOpFp16(TestSoftmaxWithCrossEntropyOp):
    def initParams(self):
        self.numeric_stable_mode = False
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


class TestSoftmaxWithCrossEntropyOpNoCudnnFp16(
        TestSoftmaxWithCrossEntropyOpFp16):
    def initParams(self):
        self.numeric_stable_mode = True
        self.dtype = np.float16

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss", max_relative_error=0.1)


class TestSoftmaxWithCrossEntropyOp2(OpTest):
    """
    Test softmax with cross entropy operator with soft labels.
    """

    def setUp(self):
        self.op_type = "softmax_with_cross_entropy"
        batch_size = 41
        class_num = 37

        logits = np.random.uniform(0.1, 1.0,
                                   [batch_size, class_num]).astype("float64")
        softmax = np.apply_along_axis(stable_softmax, 1, logits)
        labels = np.random.uniform(0.1, 1.0,
                                   [batch_size, class_num]).astype("float64")
        labels /= np.sum(labels, axis=1, keepdims=True)

        cross_entropy = (-labels * np.log(softmax)).sum(
            axis=1, keepdims=True).astype("float64")

        self.inputs = {"Logits": logits, "Label": labels}
        self.outputs = {
            "Softmax": softmax.astype("float64"),
            "Loss": cross_entropy.astype("float64")
        }
        self.attrs = {"soft_label": True}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss")


class TestSoftmaxWithCrossEntropyOp3(OpTest):
    """
    Test softmax with cross entropy operator with ignore_index.
    """

    def initParams(self):
        self.numeric_stable_mode = False

    def setUp(self):
        self.initParams()
        self.op_type = "softmax_with_cross_entropy"
        batch_size = 41
        class_num = 37

        logits = np.random.uniform(0.1, 1.0,
                                   [batch_size, class_num]).astype("float64")
        softmax = np.apply_along_axis(stable_softmax, 1, logits)
        labels = np.random.randint(0, class_num, [batch_size, 1], dtype="int64")
        ignore_index = 7
        cross_entropy = np.asmatrix(
            [[-np.log(softmax[i][labels[i][0]])]
             if labels[i] != ignore_index else [0]
             for i in range(softmax.shape[0])],
            dtype="float64")

        self.inputs = {"Logits": logits, "Label": labels}
        self.outputs = {
            "Softmax": softmax.astype("float64"),
            "Loss": cross_entropy.astype("float64")
        }
        self.attrs = {
            "ignore_index": ignore_index,
            "numeric_stable_mode": self.numeric_stable_mode
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss")


class TestSoftmaxWithCrossEntropyOp3NoCudnn(TestSoftmaxWithCrossEntropyOp3):
    def initParams(self):
        self.numeric_stable_mode = True


class TestSoftmaxWithCrossEntropyOp5(OpTest):
    """
    Test softmax with cross entropy operator with ignore_index.
    """

    def initParams(self):
        self.numeric_stable_mode = False

    def setUp(self):
        self.initParams()
        self.op_type = "softmax_with_cross_entropy"
        batch_size = [6, 10]
        class_num = 47

        logits = np.random.uniform(
            0.1, 1.0, tuple(batch_size + [class_num])).astype("float64")
        softmax = np.apply_along_axis(stable_softmax, 2, logits)
        labels = np.random.randint(
            0, class_num, tuple(batch_size + [1]), dtype="int64")
        ignore_index = 7

        softmax_2d = np.reshape(softmax, [-1, class_num])
        labels_2d = np.reshape(labels, [-1, 1])
        cross_entropy = np.asmatrix(
            [[-np.log(softmax_2d[i][labels_2d[i][0]])]
             if labels_2d[i] != ignore_index else [0]
             for i in range(softmax_2d.shape[0])],
            dtype="float64")

        cross_entropy = np.reshape(cross_entropy, batch_size)

        output_shape = tuple(batch_size + [1])
        output_res = cross_entropy.astype("float64")
        output_res = np.expand_dims(output_res, axis=2)
        self.inputs = {"Logits": logits, "Label": labels}
        self.outputs = {
            "Softmax": softmax.astype("float64"),
            "Loss": output_res,
        }
        self.attrs = {
            "ignore_index": ignore_index,
            "numeric_stable_mode": self.numeric_stable_mode
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss")


class TestSoftmaxWithCrossEntropyOp5NoCudnn(TestSoftmaxWithCrossEntropyOp5):
    def initParams(self):
        self.numeric_stable_mode = True


class TestSoftmaxWithCrossEntropyOp6(OpTest):
    """
    Test softmax with cross entropy operator with soft labels.
    """

    def setUp(self):
        self.op_type = "softmax_with_cross_entropy"
        batch_size = [6, 10]
        class_num = 37

        logits = np.random.uniform(
            0.1, 1.0, tuple(batch_size + [class_num])).astype("float64")
        softmax = np.apply_along_axis(stable_softmax, 2, logits)
        labels = np.random.uniform(
            0.1, 1.0, tuple(batch_size + [class_num])).astype("float64")
        labels /= np.sum(labels, axis=2, keepdims=True)

        cross_entropy = (-labels * np.log(softmax)).sum(
            axis=2, keepdims=True).astype("float64")

        self.inputs = {"Logits": logits, "Label": labels}
        self.outputs = {
            "Softmax": softmax.astype("float64"),
            "Loss": cross_entropy.astype("float64")
        }
        self.attrs = {"soft_label": True}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss")


class TestSoftmaxWithCrossEntropyOpFp16_2(TestSoftmaxWithCrossEntropyOp):
    def initParams(self):
        self.numeric_stable_mode = False
        self.dtype = np.float16

    def setUp(self):
        self.initParams()
        self.op_type = "softmax_with_cross_entropy"
        batch_size = [64, 10]
        class_num = 37

        # NOTE: numpy float16 have very low accuracy, use float32 for numpy check.
        logits = np.random.uniform(
            0.1, 1.0, tuple(batch_size + [class_num])).astype(np.float32)
        softmax = np.apply_along_axis(stable_softmax, 2, logits)
        labels = np.random.randint(
            0, class_num, tuple(batch_size + [1]), dtype="int64")

        softmax_2d = np.reshape(softmax, [-1, class_num])
        labels_2d = np.reshape(labels, [-1, 1])

        cross_entropy = np.asmatrix(
            [[-np.log(softmax_2d[i][labels_2d[i][0]])]
             for i in range(softmax_2d.shape[0])],
            dtype=np.float32)

        cross_entropy = np.reshape(cross_entropy, batch_size)
        output_shape = tuple(batch_size + [1])
        output_res = cross_entropy.astype(self.dtype)
        output_res = np.expand_dims(output_res, axis=2)
        self.inputs = {"Logits": logits, "Label": labels}

        self.inputs = {
            "Logits": logits.astype(self.dtype).view(np.uint16),
            "Label": labels
        }
        self.outputs = {
            "Softmax": softmax.astype(self.dtype),
            "Loss": output_res,
        }
        self.attrs = {"numeric_stable_mode": self.numeric_stable_mode}

    def test_check_output(self):
        self.check_output(atol=1e-2)

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss", max_relative_error=0.1)


if __name__ == "__main__":
    unittest.main()
