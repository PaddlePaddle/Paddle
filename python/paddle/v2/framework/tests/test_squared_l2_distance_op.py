import unittest
from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op
import numpy as np


class TestSquaredL2DistanceOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = 'squared_l2_distance'
        self.inputs = {
            'X': np.random.uniform(0.1, 1., (2, 3)).astype('float32'),
            'Y': np.random.uniform(0.1, 1., (2, 3)).astype('float32')
        }
        subRes = self.inputs['X'] - self.inputs['Y']
        output = subRes * subRes
        self.outputs = {
            'sub_result': subRes,
            'Out': np.expand_dims(output.sum(1), 1)
        }


if __name__ == '__main__':
    unittest.main()
