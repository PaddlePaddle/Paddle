import unittest
import numpy as np
from op_test import OpTest


class TestAdamOp1(OpTest):
    def setUp(self):
        '''Test Adam Op with supplied attributes
        '''
        self.op_type = "adam"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
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

        param_out, moment1_out, \
            moment2_out = adam_step(self.inputs, self.attrs)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out
        }

    def test_check_output(self):
        self.check_output()


class TestAdamOp2(OpTest):
    def setUp(self):
        '''Test Adam Op with supplied attributes
        '''
        self.op_type = "adam"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

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

        attributes = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

        param_out, moment1_out, \
            moment2_out = adam_step(self.inputs, attributes)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out
        }

    def test_check_output(self):
        self.check_output()


class TestAdamOpMultipleSteps(OpTest):
    def setUp(self):
        '''Test Adam Operator with supplied attributes
        '''
        self.op_type = "adam"
        self.num_steps = 10

        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

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

    def test_check_output(self):
        for _ in range(self.num_steps):
            param_out, moment1_out, \
                moment2_out = adam_step(self.inputs, self.attrs)

            self.outputs = {
                'Moment1Out': moment1_out,
                'Moment2Out': moment2_out,
                'ParamOut': param_out
            }

            # Verify output for this step
            self.check_output()

            # Output of this step becomes input for next step
            self.inputs['Param'] = param_out
            self.inputs['Moment1'] = moment1_out
            self.inputs['Moment2'] = moment2_out

            # Update powers of Beta1 and Beta2 for next time step
            self.inputs['Beta1Pow'] *= self.attrs['beta1']
            self.inputs['Beta2Pow'] *= self.attrs['beta1']

            # Randomize gradient for next step
            self.inputs['Grad'] = np.random.uniform(
                -1, 1, (102, 105)).astype("float32")


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
    lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
    param_out = param - lr_t * (moment1_out / (np.sqrt(moment2_out) + epsilon))
    return param_out, moment1_out, moment2_out


if __name__ == "__main__":
    unittest.main()
