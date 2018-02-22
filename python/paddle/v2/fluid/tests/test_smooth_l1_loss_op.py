import unittest
import numpy as np
from op_test import OpTest


def smooth_l1_loss_forward(val, sigma2):
    abs_val = abs(val)
    if abs_val < 1.0 / sigma2:
        return 0.5 * val * val * sigma2
    else:
        return abs_val - 0.5 / sigma2


class TestSmoothL1LossOp1(OpTest):
    def setUp(self):
        self.op_type = "smooth_l1_loss"
        dims = (5, 10)
        self.inputs = {
            'X': np.random.random(dims).astype("float32"),
            'Y': np.random.random(dims).astype("float32")
        }
        sigma = 3.0
        self.attrs = {'sigma': sigma}
        sigma2 = sigma * sigma
        diff = self.inputs['X'] - self.inputs['Y']
        loss = np.vectorize(smooth_l1_loss_forward)(diff, sigma2).sum(1)
        loss = loss.reshape((dims[0], 1))
        self.outputs = {
            'Diff': diff.astype('float32'),
            'Out': loss.astype('float32')
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.02)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.03, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.03, no_grad_set=set('Y'))


class TestSmoothL1LossOp2(OpTest):
    def setUp(self):
        self.op_type = "smooth_l1_loss"
        dims = (5, 10)
        self.inputs = {
            'X': np.random.random(dims).astype("float32"),
            'Y': np.random.random(dims).astype("float32"),
            'InsideWeight': np.random.random(dims).astype("float32"),
            'OutsideWeight': np.random.random(dims).astype("float32")
        }
        sigma = 3.0
        self.attrs = {'sigma': sigma}
        sigma2 = sigma * sigma
        diff = self.inputs['X'] - self.inputs['Y']
        diff = diff * self.inputs['InsideWeight']
        loss = np.vectorize(smooth_l1_loss_forward)(diff, sigma2)
        loss = loss * self.inputs['OutsideWeight']
        loss = loss.sum(1).reshape((dims[0], 1))
        self.outputs = {
            'Diff': diff.astype('float32'),
            'Out': loss.astype('float32')
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.03)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            max_relative_error=0.03,
            no_grad_set=set(['X', 'InsideWeight', 'OutsideWeight']))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.03,
            no_grad_set=set(['Y', 'InsideWeight', 'OutsideWeight']))


if __name__ == '__main__':
    unittest.main()
