import unittest
import numpy
from op_test import OpTest


class TestCrossEntropy(OpTest):
    def setUp(self):
        self.op_type = "onehot_cross_entropy"
        batch_size = 30
        class_num = 10
        X = numpy.random.uniform(0.1, 1.0,
                                 [batch_size, class_num]).astype("float32")
        label = (class_num / 2) * numpy.ones(batch_size).astype("int32")
        self.inputs = {'X': X, 'label': label}
        Y = []
        for i in range(0, batch_size):
            Y.append(-numpy.log(X[i][label[i]]))
        self.outputs = {'Y': numpy.array(Y).astype("float32")}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Y")


if __name__ == "__main__":
    unittest.main()
