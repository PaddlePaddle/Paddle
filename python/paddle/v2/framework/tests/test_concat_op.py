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
                np.random.random((32, 16)).astype("float32"), np.random.random(
                    (32, 32)).astype("float32"), np.random.random(
                        (32, 64)).astype("float32")
            ]
        }
        self.attrs = {'axis': 0}
        self.outpus = {
            'Out': np.concatenate((self.inputs[0], self.inputs[1],
                                   self.inputs[2]))
        }


if __name__ == '__main__':
    unittest.main()
