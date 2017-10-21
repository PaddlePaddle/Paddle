from lib_loader import _C_LIB


def create_python_ops_creatation_functions():
    op_protos = paddle.framework.OpRegistry.get_all_op_proto()
    for type_name in op_protos:
        op_proto = op_protos[type_name]
        def __impl__(**kwargs):  # User must use key word args in Paddle API
            inputs = [kwargs.get(ipt.name, "") for ipt in op_proto.inputs]
            outputs = [kwargs.get(opt.name, "") for opt in op_proto.outputs]
            attrs = [cast_to_op_attr(attr, kwargs.get(attr.name, None)) for attr in op_proto.attrs]
            opdesc = (input, outputs, type_name, attrs)
            return CreateOp(opdesc)
        __impl__.__doc__ = create_doc_string(op_proto)
        globals()[type_name] = __impl__


def CreateOp(op_desc):
    return Op(op_desc)


class Op(object):
    """
    Operator is a python wrapper for Operator in c++ end.
    """
    def __init__(self, op_desc):
        self.__op_def__ = op_desc
        self.__handle__ = _C_LIB.OpRegistry.CreateOp(opdesc)
