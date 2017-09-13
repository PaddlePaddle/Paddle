import unittest
from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op
from paddle.v2.framework.op import Operator
import numpy as np


def modified_huber_loss_forward(val):
    if val < -1:
        return -4 * val
    elif val < 1:
        return (1 - val) * (1 - val)
    else:
        return 0


class TestModifiedHuberLossOp_f0(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = 'modified_huber_loss'
        samples_num = 32
        self.inputs = {
            'X': np.random.uniform(-1, 1., (samples_num, 1)).astype('float32'),
            'Y': np.random.choice([0, 1], samples_num).reshape((samples_num, 1))
        }
        product_res = self.inputs['X'] * (2 * self.inputs['Y'] - 1)
        loss = np.vectorize(modified_huber_loss_forward)(product_res)

        self.outputs = {
            'IntermediateVal': product_res,
            'Out': loss.reshape((samples_num, 1))
        }


class TestModifiedHuberLossGradOp(GradientChecker):
    def test_modified_huber_loss_b0(self):
        samples_num = 10
        inputs = {
            'X': np.random.uniform(-1, 1, (samples_num, 1)).astype('float32'),
            'Y': np.random.choice([0, 1], samples_num).reshape((samples_num, 1))
        }
        op = Operator(
            "modified_huber_loss",
            X='X',
            Y='Y',
            IntermediateVal='IntermediateVal',
            Out='Out')
        self.compare_grad(op, inputs, no_grad_set=set(['IntermediateVal', 'Y']))
        self.check_grad(op, inputs, set(["X"]), "Out")


if __name__ == '__main__':
    unittest.main()
