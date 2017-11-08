import unittest
import numpy as np
from op_test import OpTest


class ElementwiseModOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_mod"
        """ Warning
        CPU gradient check error!
        'X': np.random.randint((32,84)).astype("int32"),
        'Y': np.random.randint((32,84)).astype("int32")
        """
        self.inputs = {
            'X': np.random.randint(1, 10, [13, 17]).astype("int32"),
            'Y': np.random.randint(1, 10, [13, 17]).astype("int32")
        }
        self.outputs = {'Out': np.mod(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.05)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.05, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.05, no_grad_set=set('Y'))


if __name__ == '__main__':
    unittest.main()
