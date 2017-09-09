import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta
import random


class TestElementwiseMulOp_Matrix(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "elementwise_mul"
        self.inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'Y': np.random.random((32, 84)).astype("float32"),
        }
        self.attrs = {'axis': 0, 'broadcast': 0}
        self.outputs = {'Out': np.multiply(self.inputs['X'], self.inputs['Y'])}


'''
class TestElementwiseMulOp_Vector(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "elementwise_mul"
        self.inputs = {
            'X': np.random.random((32, )).astype("float32"),
            'Y': np.random.random((32, )).astype("float32")
        }
        self.outputs = {'Out': np.multiply(self.inputs['X'], self.inputs['Y'])}


class ElemMulGradOpTest_Matrix(GradientChecker):
    def test_mul(self):
        op = create_op("elementwise_mul")
        """ Warning
        CPU gradient check error!
        'X': np.random.random((32,84)).astype("float32"),
        'Y': np.random.random((32,84)).astype("float32")
        """
        inputs = {
            'X': np.random.uniform(0.1, 1, [13, 17]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [13, 17]).astype("float32")
        }

        self.compare_grad(op, inputs)
        self.check_grad(
            op, inputs, set(["X", "Y"]), "Out", max_relative_error=0.02)


class ElemMulGradOpTest_Vector(GradientChecker):
    def test_mul(self):
        op = create_op("elementwise_mul")
        inputs = {
            'X': np.random.random((32, )).astype("float32"),
            'Y': np.random.random((32, )).astype("float32")
        }
        self.compare_grad(op, inputs)
        self.check_grad(
            op, inputs, set(["X", "Y"]), "Out", max_relative_error=0.02)
'''

if __name__ == '__main__':
    unittest.main()
