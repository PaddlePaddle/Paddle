import unittest
import numpy as np
from op_test import OpTest


class TestCosSimOp(OpTest):
    def setUp(self):
        self.op_type = "cos_sim"
        self.inputs = {
            'X': np.random.random((10, 5)).astype("float32"),
            'Y': np.random.random((10, 5)).astype("float32")
        }
        expect_x_norm = np.linalg.norm(self.inputs['X'], axis=1)
        expect_y_norm = np.linalg.norm(self.inputs['Y'], axis=1)
        expect_out = (self.inputs['X'] * self.inputs['Y']).sum(axis=1) / \
            expect_x_norm / expect_y_norm
        self.outputs = {
            'XNorm': np.expand_dims(expect_x_norm, 1),
            'YNorm': np.expand_dims(expect_y_norm, 1),
            'Out': np.expand_dims(expect_out, 1)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.05)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.05, no_grad_set=set('X'))

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.05, no_grad_set=set('Y'))


if __name__ == "__main__":
    unittest.main()
