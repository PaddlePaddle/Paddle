import unittest
import numpy as np
from op_test import OpTest


class TestAdagradOp1(OpTest):
    ''' Test Adagrad operator with explicit attributes
    '''

    def setUp(self):
        self.op_type = "adagrad"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")
        lr = 0.01
        epsilon = 1e-8

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'LearningRate': np.array([lr]).astype("float32")
        }

        self.attrs = {'epsilon': epsilon}

        moment_out = moment + grad * grad
        param_out = param - lr * grad / (np.sqrt(moment_out) + epsilon)

        self.outputs = {'ParamOut': param_out, 'MomentOut': moment_out}

    def test_check_output(self):
        self.check_output()


class TestAdagradOp2(OpTest):
    ''' Test Adagrad operator with default attributes
    '''

    def setUp(self):
        self.op_type = "adagrad"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")
        lr = 0.01
        epsilon = 1e-6

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'LearningRate': np.array([lr]).astype("float32")
        }

        self.attrs = {'epsilon': epsilon}

        moment_out = moment + grad * grad
        param_out = param - lr * grad / (np.sqrt(moment_out) + epsilon)

        self.outputs = {'ParamOut': param_out, 'MomentOut': moment_out}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
