import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestCosSimOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "cos_sim"
        self.inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'Y': np.random.random((32, 84)).astype("float32")
        }
        expect = (self.inputs['X'] * self.inputs['Y']).sum(axis=1) / \
                 np.linalg.norm(self.inputs['X'], axis=1) / \
                 np.linalg.norm(self.inputs['Y'], axis=1)
        expect = np.expand_dims(expect, 1)
        self.outputs = {'Out': expect}


class CosSimGradOpTest(GradientChecker):
    def test_cos_sim(self):
        op = create_op("cos_sim")
        #inputs = {
        #'X': np.random.random((2, 2)).astype("float32"),
        #'Y': np.random.random((2, 2)).astype("float32")
        #}
        inputs = {
            'X': np.array([[0.9, 0.6], [1.9, 1.6]]).astype("float32"),
            'Y': np.array([[0.7, 0.8], [1.7, 1.8]]).astype("float32")
        }
        print(inputs)
        self.check_grad(
            op, inputs, set(["X", "Y"]), "Out", max_relative_error=0.5)


if __name__ == '__main__':
    unittest.main()
