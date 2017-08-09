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


class RowwiseAddGradOpTest(GradientChecker):
    def test_rowwise_add(self):
        op = create_op("rowwise_add")
        inputs = {
            "X": np.random.uniform(0.1, 1, [10, 10]).astype("float32"),
            "b": np.random.uniform(0.1, 1, [10, 1]).astype("float32")
        }
        self.check_grad(op, inputs, set(["X", "b"]), "Out")


#TODO(dzh): rowwise_grad check

if __name__ == '__main__':
    unittest.main()
