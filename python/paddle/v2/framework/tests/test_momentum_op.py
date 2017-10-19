import unittest
import numpy as np
from op_test import OpTest


class TestMomentumOp(OpTest):
    def setUp(self):
        self.op_type = "momentum"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        velocity = np.zeros((123, 321)).astype("float32")
        learning_rate = np.array([0.001]).astype("float32")
        mu = 0.0001

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Velocity': velocity,
            'LearningRate': learning_rate
        }

        self.attrs = {'mu': mu}

        velocity_out = mu * velocity + grad
        param_out = param - learning_rate * velocity_out

        self.outputs = {'ParamOut': param_out, 'VelocityOut': velocity_out}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
