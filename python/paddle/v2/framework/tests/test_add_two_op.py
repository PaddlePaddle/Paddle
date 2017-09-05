import unittest

import numpy
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator

# from op_test_util import OpTestMeta

from op_test import OpTest


class TestAddOp(OpTest):
    def setUp(self):
        self.type = "add_two"

    self.inputs = {
        'X': numpy.random.random((102, 105)).astype("float32"),
        'Y': numpy.random.random((102, 105)).astype("float32")
    }
    self.outputs = {'Out': self.inputs['X'] + self.inputs['Y']}

    def test_check_output(self):
        self.check_output(core.CPUPlace())
        self.check_output(core.GPUPlace(0))


# class TestAddOp(unittest.TestCase):
#     __metaclass__ = OpTestMeta

#     def setUp(self):
#         self.type = "add_two"
#         self.inputs = {
#             'X': numpy.random.random((102, 105)).astype("float32"),
#             'Y': numpy.random.random((102, 105)).astype("float32")
#         }
#         self.outputs = {'Out': self.inputs['X'] + self.inputs['Y']}

if __name__ == '__main__':
    unittest.main()
