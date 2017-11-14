import paddle.v2.fluid.proto.framework_pb2 as framework_pb2
import unittest


class TestFrameworkProto(unittest.TestCase):
    def test_all(self):
        op_proto = framework_pb2.OpProto()
        ipt0 = op_proto.inputs.add()
        ipt0.name = "a"
        ipt0.comment = "the input of cosine op"
        ipt1 = op_proto.inputs.add()
        ipt1.name = "b"
        ipt1.comment = "the other input of cosine op"
        opt = op_proto.outputs.add()
        opt.name = "output"
        opt.comment = "the output of cosine op"
        op_proto.comment = "cosine op, output = scale*cos(a, b)"
        attr = op_proto.attrs.add()
        attr.name = "scale"
        attr.comment = "scale of cosine op"
        attr.type = framework_pb2.FLOAT
        op_proto.type = "cos"
        self.assertTrue(op_proto.IsInitialized())


if __name__ == "__main__":
    unittest.main()
