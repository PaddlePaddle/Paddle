import unittest
from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op
import functools
import numpy as np
from paddle.v2.framework.op import Operator


def smooth_l1_loss_forward(val, sigma2):
    abs_val = abs(val)
    if abs_val < 1.0 / sigma2:
        return 0.5 * val * val * sigma2
    else:
        return abs_val - 0.5 / sigma2


class TestSmoothL1LossOp_f0(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "smooth_l1_loss"
        dims = (32, 64)
        self.inputs = {
            'X': np.random.random(dims).astype("float32"),
            'Y': np.random.random(dims).astype("float32")
        }
        sigma = 3.0
        self.attrs = {'sigma': sigma}
        sigma2 = sigma * sigma
        diff = self.inputs['X'] - self.inputs['Y']
        loss = np.vectorize(smooth_l1_loss_forward)(diff, sigma2).sum(1)
        loss = loss.reshape((dims[0], 1))
        self.outputs = {'diff': diff, 'Out': loss}


class TestSmoothL1LossOp_f1(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "smooth_l1_loss"
        dims = (32, 64)
        self.inputs = {
            'X': np.random.random(dims).astype("float32"),
            'Y': np.random.random(dims).astype("float32"),
            'InsideWeight': np.random.random(dims).astype("float32"),
            'OutsideWeight': np.random.random(dims).astype("float32")
        }
        sigma = 3.0
        self.attrs = {'sigma': sigma}
        sigma2 = sigma * sigma
        diff = self.inputs['X'] - self.inputs['Y']
        diff = diff * self.inputs['InsideWeight']
        loss = np.vectorize(smooth_l1_loss_forward)(diff, sigma2)
        loss = loss * self.inputs['OutsideWeight']
        loss = loss.sum(1).reshape((dims[0], 1))
        self.outputs = {'diff': diff, 'Out': loss}


class SmoothL1LossGradOpTest(GradientChecker):
    def test_smooth_l1_loss_b0(self):
        dims = (5, 7)
        X = np.random.random(dims).astype("float32")
        Y = np.random.random(dims).astype("float32")
        InsideWeight = np.random.random(dims).astype("float32")
        OutsideWeight = np.random.random(dims).astype("float32")
        inputs = {
            'X': X,
            'Y': Y,
            'InsideWeight': InsideWeight,
            'OutsideWeight': OutsideWeight
        }
        op = Operator(
            "smooth_l1_loss",
            X='X',
            Y='Y',
            InsideWeight='InsideWeight',
            OutsideWeight='OutsideWeight',
            diff="diff",
            Out="Out",
            sigma=3.0)
        self.compare_grad(
            op, inputs, no_grad_set=set(['InsideWeight', 'OutsideWeight']))
        self.check_grad(
            op, inputs, set(["X", "Y"]), "Out", max_relative_error=0.08)

    def test_smooth_l1_loss_b1(self):
        dims = (5, 7)
        X = np.random.random(dims).astype("float32")
        Y = np.random.random(dims).astype("float32")
        inputs = {'X': X, 'Y': Y}
        op = Operator(
            "smooth_l1_loss",
            X='X',
            Y='Y',
            InsideWeight='InsideWeight',
            OutsideWeight='OutsideWeight',
            diff="diff",
            Out="Out",
            sigma=3.0)
        self.compare_grad(
            op, inputs, no_grad_set=set(['InsideWeight', 'OutsideWeight']))
        self.check_grad(op, inputs, set(["X", "Y"]), "Out")


if __name__ == '__main__':
    unittest.main()
