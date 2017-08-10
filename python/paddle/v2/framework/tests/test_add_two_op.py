import unittest

import numpy
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator

from op_test_util import OpTestMeta


class TestAddOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "add_two"
        self.inputs = {
            'X': numpy.random.random((102, 105)).astype("float32"),
            'Y': numpy.random.random((102, 105)).astype("float32")
        }
        self.outputs = {'Out': self.inputs['X'] + self.inputs['Y']}


class TestAddGradOp(unittest.TestCase):
    def test_add_grad(self):
        op = Operator('add_two', X="X", Y="Y", Out="Out")
        backward_op = core.Operator.backward(op, set())
        self.assertEqual(backward_op.type(), "add_two_grad")
        expected = '''Op(add_two_grad), inputs:(X, Y, Out, Out@GRAD), outputs:(X@GRAD, Y@GRAD).'''
        self.assertEqual(expected, str(backward_op))


if __name__ == '__main__':
    unittest.main()
