import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestDropoutOpWithProbZero(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 0.0}
        self.outputs = {'Out': self.inputs['X'], 'Mask': np.ones((32, 64))}


class TestDropoutOpWithProbOne(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 1.0}
        self.outputs = {'Out': np.zeros((32, 64)), 'Mask': np.zeros((32, 64))}


class TestDropoutOpWithRank3(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "dropout"
        self.inputs = {'X': np.random.random((32, 64, 8)).astype("float32")}
        self.attrs = {'dropout_prob': 0.0}
        self.outputs = {'Out': self.inputs['X'], 'Mask': np.ones((32, 64, 8))}


class TestDropoutGradOp(GradientChecker):
    def setUp(self):
        self.op = create_op("dropout")
        self.inputs = {'X': np.random.random((10, 5)).astype("float32")}

    def test_cpu_gpu_compare(self):
        self.compare_grad(self.op, self.inputs)

    def test_normal(self):
        self.check_grad(self.op, self.inputs, set(["X"]), "Out")


class TestDropoutGradOpWithRank3(TestDropoutGradOp):
    def setUp(self):
        self.op = create_op("dropout")
        self.inputs = {'X': np.random.random((10, 5, 4)).astype("float32")}


if __name__ == '__main__':
    unittest.main()
