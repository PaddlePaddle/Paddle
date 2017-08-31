import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestConcatOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "concat"
        self.inputs = {
            'X': [
                np.random.random((16, 32)).astype("float32"), np.random.random(
                    (8, 32)).astype("float32"), np.random.random(
                        (2, 32)).astype("float32")
            ]
        }
        self.attrs = {'axis': 0}
        self.outpus = {
            'Out': np.concatenate(
                (self.inputs['X'][0], self.inputs['X'][1], self.inputs['X'][2]),
                axis=0)
        }


if __name__ == '__main__':
    unittest.main()
