import unittest
from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op
import numpy
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator


class TestScatterOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "scatter"
        ref_np = numpy.ones((3, 3)).astype("float32")
        index_np = numpy.array([1, 2]).astype("int32")
        updates_np = numpy.random.random((2, 3)).astype("float32")
        output_np = numpy.copy(ref_np)
        output_np[index_np] += updates_np
        self.inputs = {'Ref': ref_np, 'Index': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}


class TestScatterGradOp(GradientChecker):
    def test_scatter_grad(self):
        op = create_op("scatter")
        # test data setup
        ref_np = numpy.ones((3, 10)).astype("float32")
        index_np = numpy.array([1, 2]).astype("int32")
        updates_np = numpy.random.random((2, 10)).astype("float32")
        output_np = numpy.copy(ref_np)
        output_np[index_np] += updates_np
        inputs = {'Ref': ref_np, 'Index': index_np, 'Updates': updates_np}
        self.check_grad(
            op, inputs, set(["Updates", "Ref"]), "Out", in_place=True)


if __name__ == "__main__":
    unittest.main()
