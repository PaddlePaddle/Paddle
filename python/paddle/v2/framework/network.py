import paddle.v2.framework.core as core
from paddle.v2.framework.op import OpDescCreationMethod, get_all_op_protos


class Network(object):
    def __init__(self):
        self.net = core.Net()

    def __getattr__(self, name):
        op_protos = get_all_op_protos()
        if name in op_protos:
            method = OpDescCreationMethod(get_all_op_protos()[name])

            def __impl__(*args, **kwargs):
                op_desc = method(*args, **kwargs)
                op = self.net.create_and_add_op(op_desc.SerializeToString())
                outs = op.no_intermediate_outputs()
                if len(outs) == 1:
                    return outs[0]
                elif len(outs) == 0:
                    return None
                else:
                    return outs

            return __impl__
        else:
            fn = getattr(self.net, name, None)
            if fn is not None:
                return fn
            else:
                raise AttributeError("No such attribute %s" % name)

    def create_and_add_op(self, type, **kwargs):
        return getattr(self, type)(**kwargs)

    def add_op(self, op):
        if isinstance(op, Network):
            self.add_op(op.net)
        else:
            self.net.add_op(op)

    def __str__(self):
        return str(self.net)

    def __len__(self):
        return len(self.net)
