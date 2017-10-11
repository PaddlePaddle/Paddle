import unittest
import numpy as np

from op_test import OpTest
from test_softmax_op import stable_softmax


class TestSoftmaxWithCrossEntropyOp(OpTest):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    """

    def setUp(self):
        self.op_type = "softmax_with_cross_entropy"
        batch_size = 3
        class_num = 37

        logits = np.random.uniform(0.1, 1.0,
                                   [batch_size, class_num]).astype("float32")
        softmax = np.apply_along_axis(stable_softmax, 1, logits)
        labels = np.random.randint(0, class_num, [batch_size, 1], dtype="int32")

        cross_entropy = np.asmatrix(
            [[-np.log(softmax[i][labels[i][0]])]
             for i in range(softmax.shape[0])],
            dtype="float32")

        self.inputs = {"Logits": logits, "Label": labels}
        self.outputs = {"Softmax": softmax, "Loss": cross_entropy}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss", max_relative_error=0.05)


class TestSoftmaxWithCrossEntropyOp2(OpTest):
    """
    Test softmax with cross entropy operator with soft labels.
    """

    def setUp(self):
        self.op_type = "softmax_with_cross_entropy"
        batch_size = 2
        class_num = 37

        logits = np.random.uniform(0.1, 1.0,
                                   [batch_size, class_num]).astype("float32")
        softmax = np.apply_along_axis(stable_softmax, 1, logits)
        labels = np.random.uniform(0.1, 1.0,
                                   [batch_size, class_num]).astype("float32")
        labels /= np.sum(labels, axis=1, keepdims=True)

        cross_entropy = (-labels * np.log(softmax)).sum(
            axis=1, keepdims=True).astype("float32")

        self.inputs = {"Logits": logits, "Label": labels}
        self.outputs = {"Softmax": softmax, "Loss": cross_entropy}
        self.attrs = {"softLabel": True}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss", max_relative_error=0.05)


if __name__ == "__main__":
    unittest.main()
