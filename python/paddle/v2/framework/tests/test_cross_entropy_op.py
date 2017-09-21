import unittest
import numpy as np
from op_test import OpTest


class TestCrossEntropyOp1(OpTest):
    """Test standard cross-entropy, with index representation of labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        batch_size = 30
        class_num = 10
        X = np.random.uniform(0.1, 1.0,
                              [batch_size, class_num]).astype("float32")
        label = np.random.randint(0, class_num, (batch_size, 1), dtype="int32")
        cross_entropy = np.asmatrix(
            [[-np.log(X[i][label[i][0]])] for i in range(X.shape[0])],
            dtype="float32")
        self.inputs = {"X": X, "Label": label}
        self.outputs = {"Y": cross_entropy}
        self.attrs = {'soft_label': 0}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Y")


class TestCrossEntropyOp2(OpTest):
    """Test soft-label cross-entropy, with vecterized soft labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        batch_size = 10
        class_num = 5
        X = np.random.uniform(0.1, 1.0,
                              [batch_size, class_num]).astype("float32")
        label = np.random.uniform(0.1, 1.0,
                                  [batch_size, class_num]).astype("float32")
        label /= label.sum(axis=1, keepdims=True)
        cross_entropy = (-label * np.log(X)).sum(
            axis=1, keepdims=True).astype("float32")
        self.inputs = {'X': X, 'Label': label}
        self.outputs = {'Y': cross_entropy}
        self.attrs = {'soft_label': 1}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y')


class TestCrossEntropyOp3(OpTest):
    """Test one-hot cross-entropy, with vecterized one-hot representation of
    labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        batch_size = 30
        class_num = 10
        X = np.random.uniform(0.1, 1.0,
                              [batch_size, class_num]).astype("float32")
        label_index = np.random.randint(
            0, class_num, (batch_size), dtype="int32")
        label = np.zeros(X.shape)
        label[np.arange(batch_size), label_index] = 1
        cross_entropy = np.asmatrix(
            [[-np.log(X[i][label_index[i]])] for i in range(X.shape[0])],
            dtype="float32")
        cross_entropy2 = (-label * np.log(X)).sum(
            axis=1, keepdims=True).astype("float32")
        self.inputs = {'X': X, 'Label': label}
        self.outputs = {'Y': cross_entropy}
        self.attrs = {'soft_label': 1}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y')


if __name__ == "__main__":
    unittest.main()
