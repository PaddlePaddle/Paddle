import unittest
from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op
from paddle.v2.framework.op import Operator
import numpy as np


def huber_loss_forward(val, delta):
    abs_val = abs(val)
    if abs_val <= delta:
        return 0.5 * val * val
    else:
        return delta * (abs_val - 0.5 * delta)


class TestHuberLossOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = 'huber_loss'
        samples_num = 64
        delta = 1.0
        self.inputs = {
            'X': np.random.uniform(0, 1., (samples_num, 1)).astype('float32'),
            'Y': np.random.uniform(0, 1., (samples_num, 1)).astype('float32'),
        }
        residual = self.inputs['Y'] - self.inputs['X']
        loss = np.vectorize(huber_loss_forward)(residual, delta)
        self.attrs = {'delta': delta}
        self.outputs = {
            'residual': residual,
            'Out': loss.reshape((samples_num, 1))
        }


class TestHuberLossGradOp(GradientChecker):
    def test_huber_loss(self):
        samples_num = 10
        delta = 1.0
        inputs = {
            'X': np.random.uniform(-1, 1, (samples_num, 1)).astype('float32'),
            'Y': np.random.uniform(-1, 1, (samples_num, 1)).astype('float32')
        }
        op = Operator(
            "huber_loss",
            X='X',
            Y='Y',
            residual='residual',
            delta=delta,
            Out='Out')
        self.compare_grad(op, inputs, no_grad_set=set(['residual']))
        self.check_grad(op, inputs, set(["X", "Y"]), "Out")


if __name__ == '__main__':
    unittest.main()
