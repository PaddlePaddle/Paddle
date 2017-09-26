import unittest
import numpy as np
from op_test import OpTest


class TestAdamOp(OpTest):
    def setUp(self):
        self.op_type = "adam"

        param = np.random.random((123, 42)).astype("float32")
        grad = np.random.random((123, 42)).astype("float32")
        moment1 = np.zeros((123, 42)).astype("float32")
        moment2 = np.zeros((123, 42)).astype("float32")

        t = 7
        learning_rate = 0.001
        epsilon = 1e-8
        beta1 = 0.9
        beta2 = 0.999

        self.inputs = {'param': param, 'grad': grad,
            'moment1': moment1, 'moment2': moment2}
        self.attrs = {'time_step': t,'learning_rate': learning_rate,
            'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

        moment1_out = beta1 * moment1 + (1 - beta1) * grad
        moment2_out = beta2 * moment2 + (1 - beta2) * grad
        moment1_hat = moment1_out / (1 - beta1**t)
        moment2_hat = moment2_out / (1 - beta2**t)
        param_out = param - learning_rate * moment1_hat / (np.sqrt(moment2_hat) + epsilon)

        self.outputs = {'param_out': param_out, "moment1_out": moment1_out,
                'moment2_out': moment2_out}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
