import paddle.v2.framework.core as core
from paddle.v2.framework.op import OpDescCreationMethod, get_all_op_protos


class Network(object):
    def __init__(self):
        self.net = core.Net()

    def add_op(self, op, **kwargs):
        if len(kwargs) == 0:
            if isinstance(op, Network):
                self.add_op(op.net)
            else:
                self.net.add_op(op)
        else:
            if not isinstance(op, str) and not isinstance(op, unicode):
                raise TypeError("Op should be str/unicode or another operator")
            all_protos = get_all_op_protos()
            if op not in all_protos:
                raise RuntimeError("Op %s has not been registered", op)
            method = OpDescCreationMethod(get_all_op_protos()[op])
            op_desc = method(**kwargs)
            op = self.net.create_and_add_op(op_desc.SerializeToString())
            outs = op.no_intermediate_outputs()
            if len(outs) == 1:
                return outs[0]
            elif len(outs) == 0:
                return None
            else:
                return outs

    def __str__(self):
        return str(self.net)

    def complete_add_op(self):
        self.net.complete_add_op()
