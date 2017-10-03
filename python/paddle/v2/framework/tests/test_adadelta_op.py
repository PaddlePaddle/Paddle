import unittest
import numpy as np
from op_test import OpTest


class TestAdadeltaOp(OpTest):
    def setUp(self):
        self.op_type = "adadelta"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The squared gradient is positive
        avg_squared_grad = np.random.random((102, 105)).astype("float32")
        # The squared update is positive
        avg_squared_update = np.random.random((102, 105)).astype("float32")

        rho = 0.95
        epsilon = 1e-6

        self.inputs = {
            'param': param,
            'grad': grad,
            'avg_squared_grad': avg_squared_grad,
            'avg_squared_update': avg_squared_update
        }

        self.attrs = {'rho': rho, 'epsilon': epsilon}

        avg_squared_grad_out = rho * avg_squared_grad + \
            (1 - rho) * np.square(grad)
        update = -np.multiply(
            np.sqrt(
                np.divide(avg_squared_update + epsilon, avg_squared_grad_out +
                          epsilon)), grad)

        avg_squared_update_out = rho * avg_squared_update + \
            (1 - rho) * np.square(update)

        param_out = param + update

        self.outputs = {
            'param_out': param_out,
            'avg_squared_grad_out': avg_squared_grad_out,
            'avg_squared_update_out': avg_squared_update_out
        }

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
