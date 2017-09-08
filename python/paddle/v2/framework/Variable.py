import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
import paddle.v2.framework.proto.framework_pb2 as framework_pb2


class Variable(object):
    def __init__(self, name, dims=None, data_type=framework_pb2.FP32):
        self.name = name
        self.data_type = data_type

        var_desc = framework_pb2.VarDesc()
        var_desc.name = name
        if isinstance(dims, list):
            var_desc.lod_tensor.data_type = data_type
            var_desc.lod_tensor.lod_level = 0
            var_desc.lod_tensor.dims.extend(dims)
        self.var_desc = core.VarDesc.create(var_desc.SerializeToString())

    def __str__(self):
        return str(self.var_desc)

    def desc(self):
        return self.var_desc


if __name__ == "__main__":
    var = Variable("aa", dims=[1, 2])
    var1 = Variable("aa")
    print(var)
