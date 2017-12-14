import unittest
import numpy as np
from op_test import OpTest


class TestDecayedAdagradOp1(OpTest):
    ''' Test DecayedAdagrad operator with explicit attributes
    '''

    def setUp(self):
        self.op_type = "decayed_adagrad"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")
        lr = 0.01
        decay = 0.80
        epsilon = 1e-8

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'LearningRate': np.array([lr]).astype("float32")
        }

        self.attrs = {'decay': decay, 'epsilon': epsilon}

        moment_out = decay * moment + (1 - decay) * grad * grad
        param_out = param - lr * grad / (np.sqrt(moment_out) + epsilon)

        self.outputs = {'ParamOut': param_out, 'MomentOut': moment_out}

    def test_check_output(self):
        self.check_output()


class TestDecayedAdagradOp2(OpTest):
    ''' Test DecayedAdagrad operator with default attributes
    '''

    def setUp(self):
        self.op_type = "decayed_adagrad"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")
        lr = 0.01
        decay = 0.95
        epsilon = 1e-6

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'LearningRate': np.array([lr]).astype("float32")
        }

        self.attrs = {'decay': decay, 'epsilon': epsilon}

        moment_out = decay * moment + (1 - decay) * grad * grad
        param_out = param - lr * grad / (np.sqrt(moment_out) + epsilon)

        self.outputs = {'ParamOut': param_out, 'MomentOut': moment_out}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
