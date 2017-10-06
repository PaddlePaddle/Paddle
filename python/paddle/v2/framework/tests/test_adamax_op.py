import unittest
import numpy as np
from op_test import OpTest


class TestAdamaxOp1(OpTest):
    def setUp(self):
        '''Test Adamax Operator with supplied attributes
        '''
        self.op_type = "adamax"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The infinity norm is positive
        inf_norm = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.002
        beta_1 = 0.78
        beta_2 = 0.899
        epsilon = 1e-5
        beta_1_pow = beta_1**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'InfNorm': inf_norm,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta_1_pow]).astype("float32")
        }

        self.attrs = {'beta1': beta_1, 'beta2': beta_2, 'epsilon': epsilon}

        moment_out = beta_1 * moment + (1 - beta_1) * grad
        inf_norm_out = np.maximum(beta_2 * inf_norm + epsilon, np.abs(grad))
        beta_1_pow_out = beta_1_pow * beta_1
        lr_t = (learning_rate / (1 - beta_1_pow_out))
        param_out = param - lr_t * np.divide(moment_out, inf_norm_out)

        self.outputs = {
            'ParamOut': param_out,
            'MomentOut': moment_out,
            'InfNormOut': inf_norm_out,
            'Beta1PowOut': beta_1_pow_out
        }

    def test_check_output(self):
        self.check_output()


class TestAdamaxOp2(OpTest):
    '''Test Adamax Operator with default attributes
    '''

    def setUp(self):
        self.op_type = "adamax"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The infinity norm is positive
        inf_norm = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.002
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8
        beta_1_pow = beta_1**8

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'InfNorm': inf_norm,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta_1_pow]).astype("float32")
        }

        moment_out = beta_1 * moment + (1 - beta_1) * grad
        inf_norm_out = np.maximum(beta_2 * inf_norm + epsilon, np.abs(grad))
        beta_1_pow_out = beta_1_pow * beta_1
        lr_t = (learning_rate / (1 - beta_1_pow_out))
        param_out = param - lr_t * np.divide(moment_out, inf_norm_out)

        self.outputs = {
            'ParamOut': param_out,
            'MomentOut': moment_out,
            'InfNormOut': inf_norm_out,
            'Beta1PowOut': beta_1_pow_out
        }

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
