import paddle.v2.framework.core as core
import paddle.v2.framework.proto.op_proto_pb2 as op_proto_pb2


def get_all_op_protos():
    protostrs = core.get_all_op_protos()
    ret_values = []
    for pbstr in protostrs:
        op_proto = op_proto_pb2.OpProto()
        op_proto.ParseFromString(pbstr)
        ret_values.append(op_proto)
    return ret_values
