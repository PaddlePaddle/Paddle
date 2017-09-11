import unittest
import numpy as np
from op_test import OpTest


class TestRowwiseAddOp(OpTest):
    def setUp(self):
        self.op_type = "rowwise_add"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [5, 10]).astype("float32"),
            'b': np.random.uniform(0.1, 1, [10]).astype("float32")
        }
        self.outputs = {'Out': np.add(self.inputs['X'], self.inputs['b'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'b'], 'Out')

    def test_check_grad_ingore_b(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('b'))

    def test_check_grad_ingore_x(self):
        self.check_grad(['b'], 'Out', no_grad_set=set('X'))


class TestRowwiseAddOp2(OpTest):
    def setUp(self):
        self.op_type = "rowwise_add"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 2, 5]).astype("float32"),
            'b': np.random.uniform(0.1, 1, [2, 5]).astype("float32")
        }
        self.outputs = {'Out': np.add(self.inputs['X'], self.inputs['b'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'b'], 'Out')

    def test_check_grad_ignore_b(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('b'))

    def test_check_grad_ignore_x(self):
        self.check_grad(['b'], 'Out', no_grad_set=set('X'))


if __name__ == "__main__":
    unittest.main()
