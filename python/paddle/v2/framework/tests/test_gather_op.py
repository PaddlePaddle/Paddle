import unittest
from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op
import numpy
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator


class TestGatherOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "gather"
        xnp = numpy.random.random((10, 20)).astype("float32")
        self.inputs = {
            'X': xnp,
            'Index': numpy.array([1, 3, 5]).astype("int32")
        }
        self.outputs = {'Out': self.inputs['X'][self.inputs['Index']]}


class TestGatherGradOp(GradientChecker):
    def test_gather_grad(self):
        op = create_op("gather")
        xnp = numpy.random.random((10, 20)).astype("float32")
        inputs = {'X': xnp, 'Index': numpy.array([1, 3, 5]).astype("int32")}
        self.check_grad(op, inputs, set("X"), "Out")


if __name__ == "__main__":
    unittest.main()
