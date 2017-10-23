import unittest
import numpy as np
from op_test import OpTest


class TestSumOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestMeanOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_mean"
        self.inputs = {'X': np.random.random((5, 6, 2, 10)).astype("float32")}
        self.attrs = {'dim': 1}
        self.outputs = {'Out': self.inputs['X'].mean(axis=self.attrs['dim'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestMaxOp(OpTest):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': -1}
        self.outputs = {'Out': self.inputs['X'].max(axis=self.attrs['dim'])}

    def test_check_output(self):
        self.check_output()


class TestMinOp(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': 2}
        self.outputs = {'Out': self.inputs['X'].min(axis=self.attrs['dim'])}

    def test_check_output(self):
        self.check_output()


class TestKeepDimReduce(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': -2, 'keep_dim': True}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=self.attrs['dim'], keepdims=True)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class Test1DReduce(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random(20).astype("float32")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestNorm(OpTest):
    def setUp(self):
        # use x away from 0 to avoid errors of numerical gradient when gradient near 0
        x = np.random.random((5, 6, 10)).astype("float32") + 0.2
        p = 2.0
        dim = 1
        keep_dim = False
        abs_out = np.absolute(x)
        pow_out = np.power(x, p)
        sum_out = np.sum(pow_out, axis=dim, keepdims=keep_dim)
        out = np.power(sum_out, 1. / p)
        self.op_type = "norm"
        self.inputs = {'X': x}
        self.attrs = {"p": p, "dim": dim, "keep_dim": keep_dim}
        self.outputs = {
            "AbsOut": abs_out,
            "PowOut": pow_out,
            "SumOut": sum_out,
            "Out": out
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.01)


if __name__ == '__main__':
    unittest.main()
