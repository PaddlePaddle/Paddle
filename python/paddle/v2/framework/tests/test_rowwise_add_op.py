import unittest
import numpy as np
from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op


class TestRowwiseAddOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "rowwise_add"
        self.inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'b': np.random.random(84).astype("float32")
        }
        self.outputs = {'Out': np.add(self.inputs['X'], self.inputs['b'])}


class TestRowwiseAddOp2(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "rowwise_add"
        self.inputs = {
            'X': np.random.random((13, 6, 7, 8)).astype("float32"),
            'b': np.random.random((7, 8)).astype("float32")
        }
        self.outputs = {'Out': np.add(self.inputs['X'], self.inputs['b'])}


class TestRowwiseAddGradOp(GradientChecker):
    def setUp(self):
        self.op = create_op("rowwise_add")
        self.inputs = {
            "X": np.random.uniform(0.1, 1, [5, 10]).astype("float32"),
            "b": np.random.uniform(0.1, 1, [10]).astype("float32")
        }

    def test_normal(self):
        self.check_grad(self.op, self.inputs, ["X", "b"], "Out")

    def test_ignore_b(self):
        self.check_grad(self.op, self.inputs, ["X"], "Out", no_grad_set={"b"})

    def test_ignore_x(self):
        self.check_grad(self.op, self.inputs, ["b"], "Out", no_grad_set={"X"})


class TestRowwiseAddGradOp2(GradientChecker):
    def setUp(self):
        self.op = create_op("rowwise_add")
        self.inputs = {
            "X": np.random.uniform(0.1, 1, [2, 3, 2, 5]).astype("float32"),
            "b": np.random.uniform(0.1, 1, [2, 5]).astype("float32")
        }

    def test_normal(self):
        self.check_grad(self.op, self.inputs, ["X", "b"], "Out")

    def test_ignore_b(self):
        self.check_grad(self.op, self.inputs, ["X"], "Out", no_grad_set={"b"})

    def test_ignore_x(self):
        self.check_grad(self.op, self.inputs, ["b"], "Out", no_grad_set={"X"})


if __name__ == '__main__':
    unittest.main()
