import unittest
from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op
import numpy as np


class TestSquaredL2DistanceOp_f0(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = 'squared_l2_distance'
        self.inputs = {
            'X': np.random.uniform(0.1, 1., (32, 64)).astype('float32'),
            'Y': np.random.uniform(0.1, 1., (32, 64)).astype('float32')
        }
        sub_res = self.inputs['X'] - self.inputs['Y']
        output = sub_res * sub_res
        self.outputs = {
            'sub_result': sub_res,
            'Out': np.expand_dims(output.sum(1), 1)
        }


class TestSquaredL2DistanceOp_f1(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = 'squared_l2_distance'
        self.inputs = {
            'X': np.random.uniform(0.1, 1., (32, 64)).astype('float32'),
            'Y': np.random.uniform(0.1, 1., (1, 64)).astype('float32')
        }
        sub_res = self.inputs['X'] - self.inputs['Y']
        output = sub_res * sub_res
        self.outputs = {
            'sub_result': sub_res,
            'Out': np.expand_dims(output.sum(1), 1)
        }


class TestSquaredL2DistanceOp_f2(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = 'squared_l2_distance'
        self.inputs = {
            'X': np.random.uniform(0.1, 1., (32, 64, 128)).astype('float32'),
            'Y': np.random.uniform(0.1, 1., (1, 64, 128)).astype('float32')
        }
        sub_res = self.inputs['X'] - self.inputs['Y']
        sub_res = sub_res.reshape((32, 64 * 128))
        output = sub_res * sub_res
        self.outputs = {
            'sub_result': sub_res,
            'Out': np.expand_dims(output.sum(1), 1)
        }


class TestSquaredL2DistanceGradOp(GradientChecker):
    def test_squared_l2_distance_b0(self):
        op = create_op("squared_l2_distance")
        inputs = {
            'X': np.random.uniform(0.1, .6, (2, 3)).astype('float32'),
            'Y': np.random.uniform(0.1, .6, (2, 3)).astype('float32')
        }
        self.compare_grad(op, inputs)
        self.check_grad(op, inputs, set(["X", "Y"]), "Out")

    def test_squared_l2_distance_b1(self):
        op = create_op("squared_l2_distance")
        inputs = {
            'X': np.random.uniform(0.1, .6, (2, 3)).astype('float32'),
            'Y': np.random.uniform(0.1, .6, (1, 3)).astype('float32')
        }
        self.compare_grad(op, inputs)
        self.check_grad(op, inputs, set(["X", "Y"]), "Out")

    def test_squared_l2_distance_b2(self):
        op = create_op("squared_l2_distance")
        inputs = {
            'X': np.random.uniform(0.1, .6, (2, 3, 4)).astype('float32'),
            'Y': np.random.uniform(0.1, .6, (1, 3, 4)).astype('float32')
        }
        self.compare_grad(op, inputs)
        self.check_grad(op, inputs, set(["X", "Y"]), "Out")


if __name__ == '__main__':
    unittest.main()
