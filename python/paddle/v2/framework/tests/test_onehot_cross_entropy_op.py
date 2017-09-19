import unittest
import numpy
from op_test import OpTest


class TestOnehotCrossEntropyOp(OpTest):
    def setUp(self):
        self.op_type = "onehot_cross_entropy"
        batch_size = 30
        class_num = 10

        X = numpy.random.uniform(0.1, 1.0,
                                 [batch_size, class_num]).astype("float32")
        labels = numpy.random.randint(0, class_num, batch_size, dtype="int32")

        cross_entropy = numpy.asmatrix(
            [[-numpy.log(X[i][labels[i]])] for i in range(X.shape[0])],
            dtype="float32")
        self.inputs = {"X": X, "label": labels}
        self.outputs = {"Y": cross_entropy}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Y")


if __name__ == "__main__":
    unittest.main()
