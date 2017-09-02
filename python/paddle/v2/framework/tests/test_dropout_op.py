import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestDropoutOpProbZero(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 0.0}
        self.outputs = {'Out': self.inputs['X'], 'Mask': np.ones((32, 64))}


class TestDropoutOpProbOne(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 1.0}
        self.outputs = {'Out': np.zeros((32, 64)), 'Mask': np.zeros((32, 64))}


class TestDropoutGradOp(GradientChecker):
    def test_dropout_2d(self):
        op = create_op("dropout")
        inputs = {'X': np.random.random((10, 5)).astype("float32")}
        self.compare_grad(op, inputs)
        self.check_grad(op, inputs, set(["X"]), "Out")

    def test_dropout_3d(self):
        op = create_op("dropout")
        inputs = {'X': np.random.random((10, 5, 4)).astype("float32")}
        self.compare_grad(op, inputs)
        self.check_grad(op, inputs, set(["X"]), "Out")


if __name__ == '__main__':
    unittest.main()
