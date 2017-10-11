import unittest
import numpy as np
from op_test import OpTest


class TestAdamOp1(OpTest):
    def setUp(self):
        '''Test Adam Op with supplied attributes
        '''
        np.random.seed(1203)
        self.op_type = "adam"
        param = np.random.uniform(-1, 1, (2, 1)).astype("float32")
        grad = np.random.uniform(-1, 1, (2, 1)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (2, 1)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((2, 1)).astype("float32")

        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32")
        }

        self.attrs = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

        param_out, moment1_out, moment2_out, beta1_pow_out, \
            beta2_pow_out = adam_step(self.inputs, self.attrs)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'Beta1PowOut': beta1_pow_out,
            'Beta2PowOut': beta2_pow_out,
            'ParamOut': param_out
        }

    def test_check_output(self):
        self.check_output()


def adam_step(inputs, attributes):
    '''
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    '''
    param = inputs['Param']
    grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']

    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    epsilon = attributes['epsilon']

    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)
    beta1_pow_out = beta1_pow * beta1
    beta2_pow_out = beta2_pow * beta2
    lr_t = lr * np.sqrt(1 - beta2_pow_out) / (1 - beta1_pow_out)
    param_out = param - lr_t * (moment1_out / (np.sqrt(moment2_out) + epsilon))
    return param_out, moment1_out, moment2_out, beta1_pow_out, beta2_pow_out


if __name__ == "__main__":
    unittest.main()
