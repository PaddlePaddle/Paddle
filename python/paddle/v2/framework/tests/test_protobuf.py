import paddle.v2.framework.proto.op_proto_pb2
import paddle.v2.framework.proto.attr_type_pb2
import unittest


class TestFrameworkProto(unittest.TestCase):
    def test_all(self):
        op_proto_lib = paddle.v2.framework.proto.op_proto_pb2
        attr_type_lib = paddle.v2.framework.proto.attr_type_pb2
        op_proto = op_proto_lib.OpProto()
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
        attr.type = attr_type_lib.FLOAT
        op_proto.type = "cos"
        self.assertTrue(op_proto.IsInitialized())


if __name__ == "__main__":
    unittest.main()
