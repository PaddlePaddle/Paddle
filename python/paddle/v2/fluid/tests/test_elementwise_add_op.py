import unittest
import numpy as np
from op_test import OpTest


class TestElementwiseOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [13, 17]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [13, 17]).astype("float32")
        }
        self.outputs = {'Out': np.add(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.005)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y'))


class TestElementwiseAddOp_Vector(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.inputs = {
            'X': np.random.random((32, )).astype("float32"),
            'Y': np.random.random((32, )).astype("float32")
        }
        self.outputs = {'Out': np.add(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseAddOp_broadcast_0(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float32),
            'Y': np.random.rand(2).astype(np.float32)
        }

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out': self.inputs['X'] + self.inputs['Y'].reshape(2, 1, 1)
        }


class TestElementwiseAddOp_broadcast_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float32),
            'Y': np.random.rand(3).astype(np.float32)
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] + self.inputs['Y'].reshape(1, 3, 1)
        }


class TestElementwiseAddOp_broadcast_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float32),
            'Y': np.random.rand(4).astype(np.float32)
        }

        self.outputs = {
            'Out': self.inputs['X'] + self.inputs['Y'].reshape(1, 1, 4)
        }


class TestElementwiseAddOp_broadcast_3(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.inputs = {
            'X': np.random.rand(2, 3, 4, 5).astype(np.float32),
            'Y': np.random.rand(3, 4).astype(np.float32)
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] + self.inputs['Y'].reshape(1, 3, 4, 1)
        }


class TestElementwiseAddOp_rowwise_add_0(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float32),
            'Y': np.random.rand(3, 4).astype(np.float32)
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] + self.inputs['Y'].reshape(1, 3, 4)
        }


class TestElementwiseAddOp_rowwise_add_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.inputs = {
            'X': np.random.rand(2, 1).astype(np.float32),
            'Y': np.random.rand(1).astype(np.float32)
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] + self.inputs['Y'].reshape(1, 1)
        }


if __name__ == '__main__':
    unittest.main()
