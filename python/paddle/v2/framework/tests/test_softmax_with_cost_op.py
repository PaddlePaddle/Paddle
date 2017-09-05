import unittest

import numpy as np

from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestSoftmaxWithLossOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        pass


class SoftmaxWithLossGradOpTest(GradientChecker):
    def test_softmax(self):
        pass


if __name__ == '__main__':
    unittest.main()
