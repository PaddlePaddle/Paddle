import unittest
import numpy as np
from op_test import OpTest


class TestAdamaxOp(OpTest):
    def setUp(self):
        self.op_type = "adamax"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The infinity norm is positive
        inf_norm = np.random.random((102, 105)).astype("float32")

        time_step = 9
        learning_rate = 0.002
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8

        self.inputs = {
            'param': param,
            'grad': grad,
            'moment': moment,
            'inf_norm': inf_norm,
            'time_step': time_step,
            'learning_rate': learning_rate
        }

        self.attrs = {'beta_1': beta_1, 'beta_2': beta_2, 'epsilon': epsilon}

        moment_out = beta_1 * moment + (1 - beta_1) * grad
        inf_norm_out = np.maximum(beta_2 * inf_norm + epsilon, np.abs(grad))
        lr_t = (learning_rate / (1 - beta_1**time_step))
        param_out = param - lr_t * np.divide(moment_out, inf_norm_out)

        self.outputs = {
            'param_out': param_out,
            'moment_out': moment_out,
            'inf_norm_out': inf_norm_out
        }

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
