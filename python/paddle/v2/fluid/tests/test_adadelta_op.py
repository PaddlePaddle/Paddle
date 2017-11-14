import unittest
import numpy as np
from op_test import OpTest


class TestAdadeltaOp1(OpTest):
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
            'Param': param,
            'Grad': grad,
            'AvgSquaredGrad': avg_squared_grad,
            'AvgSquaredUpdate': avg_squared_update
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
            'ParamOut': param_out,
            'AvgSquaredGradOut': avg_squared_grad_out,
            'AvgSquaredUpdateOut': avg_squared_update_out
        }

    def test_check_output(self):
        self.check_output()


class TestAdadeltaOp2(OpTest):
    '''Test Adadelta op with default attribute values
    '''

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
            'Param': param,
            'Grad': grad,
            'AvgSquaredGrad': avg_squared_grad,
            'AvgSquaredUpdate': avg_squared_update
        }

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
            'ParamOut': param_out,
            'AvgSquaredGradOut': avg_squared_grad_out,
            'AvgSquaredUpdateOut': avg_squared_update_out
        }

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
