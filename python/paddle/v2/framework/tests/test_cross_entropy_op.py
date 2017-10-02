import unittest
import numpy as np
from op_test import OpTest


class TestCrossEntropyOp1(OpTest):
    """Test cross-entropy with discrete one-hot labels.
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
        self.attrs = {"softLabel": False}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Y")


class TestCrossEntropyOp2(OpTest):
    """Test cross-entropy with vectorized soft labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        batch_size = 5
        class_num = 37

        X = np.random.uniform(0.1, 1.0,
                              [batch_size, class_num]).astype("float32")
        label = np.random.uniform(0.1, 1.0,
                                  [batch_size, class_num]).astype("float32")
        label /= label.sum(axis=1, keepdims=True)
        cross_entropy = (-label * np.log(X)).sum(
            axis=1, keepdims=True).astype("float32")

        self.inputs = {"X": X, "Label": label}
        self.outputs = {"Y": cross_entropy}
        self.attrs = {"softLabel": True}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Y", max_relative_error=0.05)


class TestCrossEntropyOp3(OpTest):
    """Test cross-entropy with vectorized one-hot representation of labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        batch_size = 5
        class_num = 17

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

        self.inputs = {"X": X, "Label": label.astype(np.float32)}
        self.outputs = {"Y": cross_entropy}
        self.attrs = {"softLabel": True}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Y", max_relative_error=0.05)


if __name__ == "__main__":
    unittest.main()
