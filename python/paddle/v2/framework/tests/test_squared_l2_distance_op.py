import unittest
import numpy as np
from op_test import OpTest


class TestSquaredL2DistanceOp_f0(OpTest):
    def setUp(self):
        self.op_type = "squared_l2_distance"
        self.inputs = {
            'X': np.random.uniform(0.1, 0.6, (2, 3)).astype("float32"),
            'Y': np.random.uniform(0.1, 0.6, (2, 3)).astype("float32")
        }
        sub_res = self.inputs['X'] - self.inputs['Y']
        output = sub_res * sub_res
        self.outputs = {
            'sub_result': sub_res,
            'Out': np.expand_dims(output.sum(1), 1)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestSquaredL2DistanceOp_f1(OpTest):
    def setUp(self):
        self.op_type = "squared_l2_distance"
        self.inputs = {
            'X': np.random.uniform(0.1, 0.6, (2, 3)).astype("float32"),
            'Y': np.random.uniform(0.1, 0.6, (1, 3)).astype("float32")
        }
        sub_res = self.inputs['X'] - self.inputs['Y']
        output = sub_res * sub_res
        self.outputs = {
            'sub_result': sub_res,
            'Out': np.expand_dims(output.sum(1), 1)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestSquaredL2DistanceOp_f2(OpTest):
    def setUp(self):
        self.op_type = "squared_l2_distance"
        self.inputs = {
            'X': np.random.uniform(0.1, 0.6, (2, 3, 4)).astype("float32"),
            'Y': np.random.uniform(0.1, 0.6, (1, 3, 4)).astype("float32")
        }
        sub_res = self.inputs['X'] - self.inputs['Y']
        sub_res = sub_res.reshape((2, 3 * 4))
        output = sub_res * sub_res
        self.outputs = {
            'sub_result': sub_res,
            'Out': np.expand_dims(output.sum(1), 1)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')


if __name__ == "__main__":
    unittest.main()
