import unittest

import numpy
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator

from op_test_util import OpTestMeta


class TestGatherOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "gather"
        self.inputs = {
            'X': numpy.random.random((10, 20)).astype("float32"),
            'Index': numpy.array([1, 3, 5]).astype("int")
        }
        self.outputs = {'Y': self.input['X'][self.input['Index']]}


if __name__ == "__main__":
    unittest.main()
