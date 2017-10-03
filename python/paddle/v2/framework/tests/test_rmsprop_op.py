import unittest
import numpy as np
from op_test import OpTest


class TestRmspropOp(OpTest):
    def setUp(self):
        self.op_type = "rmsprop"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")
        learning_rate = np.array([0.01]).astype("float32")

        epsilon = 1e-6
        decay_rate = 0.9

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'LearningRate': learning_rate
        }

        self.attrs = {'epsilon': epsilon, 'decayRate': decay_rate}

        moment_out = decay_rate * moment + (1 - decay_rate) * grad * grad
        param_out = param - learning_rate * grad / (np.sqrt(moment_out) +
                                                    epsilon)

        self.outputs = {'ParamOut': param_out, 'MomentOut': moment_out}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
