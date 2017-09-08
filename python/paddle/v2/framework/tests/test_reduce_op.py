import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta
from paddle.v2.framework.op import Operator


class TestSumOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': -2}
        out = self.inputs['X'].sum(axis=self.attrs['dim'])
        self.outputs = {'Out': out}


class TestSumGradOp(GradientChecker):
    def test_normal(self):
        op = Operator("reduce_sum", X="X", Out="Out", dim=-2)
        # use small size to decrease the error of numerical calculation
        inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.check_grad(op, inputs, set(["X"]), "Out")

    def test_1d_tensor(self):
        op = Operator("reduce_sum", X="X", Out="Out", dim=0)
        # use small size to decrease the error of numerical calculation
        inputs = {'X': np.random.random(10).astype("float32")}
        self.check_grad(op, inputs, set(["X"]), "Out")


class TestKeepdimSumOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': -2}
        out = self.inputs['X'].sum(axis=self.attrs['dim'], keepdims=True)
        self.outputs = {'Out': out}


class TestMeanOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "reduce_mean"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': -1}
        out = self.inputs['X'].mean(axis=self.attrs['dim'])
        self.outputs = {'Out': out}


class TestMeanGradOp(GradientChecker):
    def test_normal(self):
        op = Operator("reduce_mean", X="X", Out="Out", dim=-2)
        # use small size to decrease the error of numerical calculation
        inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.check_grad(op, inputs, set(["X"]), "Out")

    def test_1d_tensor(self):
        op = Operator("reduce_mean", X="X", Out="Out", dim=0)
        # use small size to decrease the error of numerical calculation
        inputs = {'X': np.random.random(10).astype("float32")}
        self.check_grad(op, inputs, set(["X"]), "Out")


class TestMaxOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "reduce_max"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': -1}
        out = self.inputs['X'].max(axis=self.attrs['dim'])
        self.outputs = {'Out': out}


class TestMinOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "reduce_max"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': -2}
        out = self.inputs['X'].min(axis=self.attrs['dim'])
        self.outputs = {'Out': out}


if __name__ == '__main__':
    unittest.main()
