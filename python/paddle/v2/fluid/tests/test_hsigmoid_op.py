import unittest
import numpy as np
from op_test import OpTest


class TestHSigmoidOp(OpTest):
    def setUp(self):
        self.op_type = "hierarchical_sigmoid_op"
        num_classes = 6
        embded_size = 10
        batch_size = 5
        x = np.random.random((batch_size, embded_size)).astype("float32")
        parameter = np.random.random(
            (batch_size, num_classes - 1, embded_size)).astype("float32")
        label = np.random.randint(0, num_classes, batch_size).astype("int64")
        bias = np.random.random((1, num_classes - 1))
        self.inputs = {
            'X': x,
            'Parameters': parameter,
            'Label': label,
            'Bias': bias
        }
        self.attrs = {'num_classes': num_classes}
        self.outputs = {'Out': label}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out')


if __name__ == '__main__':
    unittest.main()
