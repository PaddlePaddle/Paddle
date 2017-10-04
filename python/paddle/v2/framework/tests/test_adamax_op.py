import unittest
import numpy as np
from op_test import OpTest


class TestAdamaxOp1(OpTest):
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
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'InfNorm': inf_norm,
            'TimeStep': np.array([time_step]).astype("int32"),
            'LearningRate': np.array([learning_rate]).astype("float32")
        }

        self.attrs = {'beta1': beta_1, 'beta2': beta_2, 'epsilon': epsilon}

        moment_out = beta_1 * moment + (1 - beta_1) * grad
        inf_norm_out = np.maximum(beta_2 * inf_norm + epsilon, np.abs(grad))
        lr_t = (learning_rate / (1 - beta_1**time_step))
        param_out = param - lr_t * np.divide(moment_out, inf_norm_out)

        self.outputs = {
            'ParamOut': param_out,
            'MomentOut': moment_out,
            'InfNormOut': inf_norm_out
        }

    def test_check_output(self):
        self.check_output()


class TestAdamaxOp2(OpTest):
    '''Test Adamax operator with default attributes
    '''

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
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'InfNorm': inf_norm,
            'TimeStep': np.array([time_step]).astype("int32"),
            'LearningRate': np.array([learning_rate]).astype("float32")
        }

        moment_out = beta_1 * moment + (1 - beta_1) * grad
        inf_norm_out = np.maximum(beta_2 * inf_norm + epsilon, np.abs(grad))
        lr_t = (learning_rate / (1 - beta_1**time_step))
        param_out = param - lr_t * np.divide(moment_out, inf_norm_out)

        self.outputs = {
            'ParamOut': param_out,
            'MomentOut': moment_out,
            'InfNormOut': inf_norm_out
        }

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
