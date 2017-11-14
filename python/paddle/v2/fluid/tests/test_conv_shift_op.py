import unittest
import numpy as np
from op_test import OpTest


def conv_shift_forward(x, y):
    out = np.zeros_like(x)
    M = x.shape[1]
    N = y.shape[1]
    y_half_width = (N - 1) / 2
    for i in xrange(M):
        for j in xrange(N):
            out[:, i] += x[:, (i + j + M - y_half_width) % M] * y[:, j]
    return out


class TestConvShiftOp(OpTest):
    def setUp(self):
        self.op_type = "conv_shift"

        batch_size = 4
        x_dim = 17
        y_dim = 3  # must be odd and <= x_dim
        x = np.random.random((batch_size, x_dim)).astype("float32")
        y = np.random.random((batch_size, y_dim)).astype("float32")
        self.inputs = {'X': x, 'Y': y}

        out = conv_shift_forward(x, y)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.05)

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.05, no_grad_set=set("X"))

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.05, no_grad_set=set('Y'))


if __name__ == '__main__':
    unittest.main()
