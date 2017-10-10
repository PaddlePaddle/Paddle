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
        beta1 = 0.78
        beta2 = 0.899
        epsilon = 1e-5
        beta1_pow = beta1**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'InfNorm': inf_norm,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32")
        }

        self.attrs = {'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon}

        param_out, moment_out, inf_norm_out, beta1_pow_out = adamax_step(
            self.inputs, self.attrs)

        self.outputs = {
            'ParamOut': param_out,
            'MomentOut': moment_out,
            'InfNormOut': inf_norm_out,
            'Beta1PowOut': beta1_pow_out
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
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        beta1_pow = beta1**8

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'InfNorm': inf_norm,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32")
        }

        attrs = {'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon}
        param_out, moment_out, inf_norm_out, beta1_pow_out = adamax_step(
            self.inputs, attrs)

        self.outputs = {
            'ParamOut': param_out,
            'MomentOut': moment_out,
            'InfNormOut': inf_norm_out,
            'Beta1PowOut': beta1_pow_out
        }

    def test_check_output(self):
        self.check_output()


class TestAdamaxOpMultipleSteps(OpTest):
    def setUp(self):
        '''Test Adamax Operator with supplied attributes
        '''
        self.op_type = "adamax"
        self.num_steps = 10

        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The infinity norm is positive
        inf_norm = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.002
        beta1 = 0.8
        beta2 = 0.99
        epsilon = 1e-5
        beta1_pow = 1

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'InfNorm': inf_norm,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32")
        }

        self.attrs = {'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon}

        param_out, moment_out, inf_norm_out, beta1_pow_out = adamax_step(
            self.inputs, self.attrs)

    def test_check_output(self):
        for _ in range(self.num_steps):
            param_out, moment_out, inf_norm_out, beta1_pow_out = adamax_step(
                self.inputs, self.attrs)

            self.outputs = {
                'ParamOut': param_out,
                'MomentOut': moment_out,
                'InfNormOut': inf_norm_out,
                'Beta1PowOut': beta1_pow_out
            }

            # Verify output for this step
            self.check_output()

            # Output of this step becomes input for next step
            self.inputs['Param'] = param_out
            self.inputs['Moment'] = moment_out
            self.inputs['InfNorm'] = inf_norm_out
            self.inputs['Beta1Pow'] = beta1_pow_out

            # Randomize gradient for next step
            self.inputs['Grad'] = np.random.uniform(
                -1, 1, (102, 105)).astype("float32")


def adamax_step(inputs, attributes):
    '''
    Simulate one step of the adamax optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment, inf_norm and
    beta1 power accumulator
    '''
    param = inputs['Param']
    grad = inputs['Grad']
    moment = inputs['Moment']
    inf_norm = inputs['InfNorm']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']

    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    epsilon = attributes['epsilon']

    moment_out = beta1 * moment + (1 - beta1) * grad
    inf_norm_out = np.maximum(beta2 * inf_norm + epsilon, np.abs(grad))
    beta1_pow_out = beta1_pow * beta1
    lr_t = (lr / (1 - beta1_pow_out))
    param_out = param - lr_t * np.divide(moment_out, inf_norm_out)

    return param_out, moment_out, inf_norm_out, beta1_pow_out


if __name__ == "__main__":
    unittest.main()
