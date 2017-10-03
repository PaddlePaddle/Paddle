import unittest
import numpy as np
from op_test import OpTest


class TestDecayedAdagradOp(OpTest):
    def setUp(self):
        self.op_type = "decayed_adagrad"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")

        learning_rate = 0.01
        epsilon = 1e-6

        self.inputs = {'param': param, 'grad': grad, 'moment': moment}

        self.attrs = {'learning_rate': learning_rate, 'epsilon': epsilon}

        moment_out = moment + grad * grad
        param_out = param - learning_rate * grad / (np.sqrt(moment_out) +
                                                    epsilon)

        self.outputs = {'param_out': param_out, 'moment_out': moment_out}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
