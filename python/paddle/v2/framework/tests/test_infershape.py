import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
import paddle.v2.framework.proto.framework_pb2 as framework_pb2
import unittest


class TestCompileTimeInferShape(unittest.TestCase):
    def test_all(self):
        mul_op = Operator.desc("mul", X="input1", Y="input2", Out="output")
        print(mul_op)
        # opdesc = framework_pb2.OpDesc
        # return core.Operator.create(opdesc.SerializeToString())


if __name__ == "__main__":
    unittest.main()
