import unittest
import numpy as np
from op_test import OpTest


class TestProximalAdagradOp(OpTest):
    def setUp(self):
        self.op_type = "proximal_adagrad"
        w = np.random.random((102, 105)).astype("float32")
        m = np.random.random((102, 105)).astype("float32")
        g = np.random.random((102, 105)).astype("float32")
        lr = np.array([0.1]).astype("float32")
        l1 = 0.1
        l2 = 0.2

        self.inputs = {'Param': w, 'Grad': g, 'Moment': m, 'LearningRate': lr}
        self.attrs = {'l1': l1, 'l2': l2}
        param_out = 0.0

        moment_out = m + g * g
        prox_param = w - lr * g / np.sqrt(moment_out)
        if l1 > 0.0:
            x = np.abs(prox_param) - lr * l1
            x[x < 0] = 0
            param_out = np.sign(prox_param) * (x / (1.0 + lr * l2))
        else:
            param_out = prox_param / (1.0 + lr * l2)

        self.outputs = {'ParamOut': param_out, 'MomentOut': moment_out}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
