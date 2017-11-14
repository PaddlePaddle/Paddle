import unittest
import numpy as np
from op_test import OpTest


def sigmoid_np(x):
    return 1. / (1. + np.exp(-x))


def tanh_np(x):
    return 2 * sigmoid_np(2. * x) - 1.


class LstmUnitTest(OpTest):
    def setUp(self):
        self.op_type = "lstm_unit"
        x_np = np.random.normal(size=(5, 16)).astype("float64")
        c_np = np.random.normal(size=(5, 4)).astype("float64")
        i_np, f_np, o_np, j_np = np.split(x_np, 4, axis=1)
        forget_bias_np = 0.
        self.attrs = {'forget_bias': 0.}

        new_c = c_np * sigmoid_np(f_np + forget_bias_np) + sigmoid_np(
            i_np) * tanh_np(j_np)
        new_h = tanh_np(new_c) * sigmoid_np(o_np)

        self.inputs = {'X': x_np, 'C_prev': c_np}
        self.outputs = {'C': new_c, 'H': new_h}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'C_prev'], ['C', 'H'])


if __name__ == "__main__":
    # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5185
    exit(0)
    unittest.main()
