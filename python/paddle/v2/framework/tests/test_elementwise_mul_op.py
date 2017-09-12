import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta
import random
from paddle.v2.framework.op import Operator


class TestElementwiseMulOp_Matrix(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "elementwise_mul"
        self.inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'Y': np.random.random((32, 84)).astype("float32"),
        }
        #self.attrs = {'axis': 0, 'broadcast': 0}
        self.outputs = {'Out': np.multiply(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMulOp_Vector(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "elementwise_mul"
        self.inputs = {
            'X': np.random.random((32, )).astype("float32"),
            'Y': np.random.random((32, )).astype("float32")
        }
        #self.attrs = {'axis': 0, 'broadcast': 0}
        self.outputs = {'Out': np.multiply(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMulOp_broadcast_0(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "elementwise_mul"

        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float32),
            'Y': np.random.rand(2).astype(np.float32)
        }

        self.attrs = {'axis': 0, 'broadcast': 1}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(2, 1, 1)
        }


class TestElementwiseMulOp_broadcast_1(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "elementwise_mul"

        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float32),
            'Y': np.random.rand(3).astype(np.float32)
        }

        self.attrs = {'axis': 1, 'broadcast': 1}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 3, 1)
        }


class TestElementwiseMulOp_broadcast_2(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "elementwise_mul"

        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float32),
            'Y': np.random.rand(4).astype(np.float32)
        }

        self.attrs = {'broadcast': 1}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 1, 4)
        }


class TestElementwiseMulOp_broadcast_3(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "elementwise_mul"

        self.inputs = {
            'X': np.random.rand(2, 3, 4, 5).astype(np.float32),
            'Y': np.random.rand(3, 4).astype(np.float32)
        }

        self.attrs = {'axis': 1, 'broadcast': 1}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 3, 4, 1)
        }


class ElemMulGradOpTest_Matrix(GradientChecker):
    def setUp(self):
        self.op = Operator(
            "elementwise_mul", X="X", Y="Y", Out="Out", axis=0, broadcast=0)

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [13, 17]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [13, 17]).astype("float32")
        }
        #self.attrs = {'axis': 0, 'broadcast': 0}

    def test_mul(self):
        """ Warning
        CPU gradient check error!
        'X': np.random.random((32,84)).astype("float32"),
        'Y': np.random.random((32,84)).astype("float32")
        """
        self.compare_grad(self.op, self.inputs)
        self.check_grad(
            self.op, self.inputs, ["X", "Y"], "Out", max_relative_error=0.1)

    def test_ignore_x(self):
        self.check_grad(
            self.op,
            self.inputs, ["Y"],
            "Out",
            no_grad_set={"X"},
            max_relative_error=0.1)

    def test_ignore_y(self):
        self.check_grad(
            self.op,
            self.inputs, ["X"],
            "Out",
            no_grad_set={"Y"},
            max_relative_error=0.1)


class ElemMulGradOpTest_broadcast_3(GradientChecker):
    def setUp(self):
        self.op = Operator(
            "elementwise_mul", X="X", Y="Y", Out="Out", axis=1, broadcast=1)

        self.inputs = {
            'X': np.random.rand(2, 3, 4, 5).astype(np.float32),
            'Y': np.random.rand(3, 4).astype(np.float32)
        }
        self.attrs = {'axis': 1, 'broadcast': 1}

    def test_mul(self):
        self.compare_grad(self.op, self.inputs)
        self.check_grad(
            self.op, self.inputs, ["X", "Y"], "Out", max_relative_error=0.1)

    def test_ignore_x(self):
        self.check_grad(
            self.op,
            self.inputs, ["Y"],
            "Out",
            no_grad_set={"X"},
            max_relative_error=0.1)

    def test_ignore_y(self):
        self.check_grad(
            self.op,
            self.inputs, ["X"],
            "Out",
            no_grad_set={"Y"},
            max_relative_error=0.1)


if __name__ == '__main__':
    unittest.main()
