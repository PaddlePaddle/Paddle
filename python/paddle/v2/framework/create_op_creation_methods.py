import paddle.v2.framework.core as core
import paddle.v2.framework.proto.op_proto_pb2 as op_proto_pb2
import paddle.v2.framework.proto.op_desc_pb2 as op_desc_pb2
import paddle.v2.framework.proto.attr_type_pb2 as attr_type_pb2
import cStringIO


def get_all_op_protos():
    """
    Get all registered op proto from Paddle C++
    :return: list of OpProto
    """
    protostrs = core.get_all_op_protos()
    ret_values = []
    for pbstr in protostrs:
        op_proto = op_proto_pb2.OpProto.FromString(str(pbstr))
        ret_values.append(op_proto)
    return ret_values


class OpDescCreationMethod(object):
    """
    A Functor object to convert user input(use key word args) to OpDesc based on
    OpProto.
    
    :param op_proto: The OpProto object.
    :type op_proto: op_proto_pb2.OpProto
    """

    def __init__(self, op_proto):
        if not isinstance(op_proto, op_proto_pb2.OpProto):
            raise TypeError("Argument should be OpProto")
        self.__op_proto__ = op_proto

    def __call__(self, *args, **kwargs):
        """
        Convert user input to OpDesc. Only key-word args are supported. 
        :return: OpDesc based on user input
        :rtype: op_desc_pb2.OpDesc
        """
        if len(args) != 0:
            raise ValueError("Only keyword arguments is supported by Paddle")
        op_desc = op_desc_pb2.OpDesc()

        # Inputs
        ipts, ipt_format, _ = OpDescCreationMethod.extract_input_or_output(
            "input", kwargs, self.__op_proto__.inputs)
        op_desc.inputs.extend(ipts)
        if ipt_format is not None:
            op_desc.attrs.extend([ipt_format])

        # Outputs
        outs, out_format, tmp_index = OpDescCreationMethod.extract_input_or_output(
            "output", kwargs, self.__op_proto__.outputs)
        op_desc.outputs.extend(outs)
        if out_format is not None:
            op_desc.attrs.extend([out_format])
        if len(tmp_index) != 0:
            tmp_index_attr = op_desc.attrs.add()
            tmp_index_attr.type = attr_type_pb2.INTS
            tmp_index_attr.name = "temporary_index"
            tmp_index_attr.ints.extend(tmp_index)

        # Types
        op_desc.type = self.__op_proto__.type

        # Attrs
        for attr in self.__op_proto__.attrs:
            if attr.generated:
                continue
            user_defined_attr = kwargs.get(attr.name, None)
            if user_defined_attr is not None:
                new_attr = op_desc.attrs.add()
                new_attr.name = attr.name
                new_attr.type = attr.type
                if attr.type == attr_type_pb2.INT:
                    new_attr.i = user_defined_attr
                elif attr.type == attr_type_pb2.FLOAT:
                    new_attr.f = user_defined_attr
                elif attr.type == attr_type_pb2.STRING:
                    new_attr.s = user_defined_attr
                elif attr.type == attr_type_pb2.INTS:
                    new_attr.ints.extend(user_defined_attr)
                elif attr.type == attr_type_pb2.FLOATS:
                    new_attr.floats.extend(user_defined_attr)
                elif attr.type == attr_type_pb2.STRINGS:
                    new_attr.strings.extend(user_defined_attr)
                else:
                    raise NotImplementedError("Not support attribute type " +
                                              attr.type)

        return op_desc

    @staticmethod
    def extract_input_or_output(in_out, kwargs, meta):
        """
        Extract input variable names or output variable names from key-word 
        arguments, which base on VarProtos.
        
        :param in_out: "input" or "output"
        :param kwargs: key-word arguments that user inputted.
        :param meta: a list of VarProto
        :return: The three object will be return. The variable names. The 
        input_format or output_format attribute(None if the input or output is 
        not multiple). The temporary variable index list.
        """
        multiple = OpDescCreationMethod.any_is_true((m.multiple for m in meta))
        tmp_index = []
        retv = []
        if multiple:
            var_format = op_desc_pb2.AttrDesc()
            var_format.type = attr_type_pb2.INTS
            var_format.name = "%s_format" % in_out
            var_format.ints.append(0)

            for var in meta:
                var_name = var.name

                if var.temporary:
                    var_name = [core.var_names.temp()]
                    tmp_index.append(len(retv))
                else:
                    var_name = kwargs.get(var_name, [])
                if not isinstance(var_name, list):
                    var_name = [var_name]
                retv.extend(var_name)
                var_format.ints.append(len(var_name) + var_format.ints[-1])
            return retv, var_format, tmp_index
        else:
            for var in meta:
                if var.temporary:
                    retv.append(kwargs.get(var.name, core.var_names.temp()))
                    tmp_index.append(len(retv))
                else:
                    retv.append(kwargs.get(var.name, core.var_names.empty()))
            return retv, None, tmp_index

    @staticmethod
    def any_is_true(generator):
        """
        Reduce a bool array to one. If any of them is True, then return True.
        """
        for flag in generator:
            if flag:
                return True
        return False


def get_docstring_from_op_proto(op_proto):
    """
    Generate docstring from a OpProto
    :param op_proto: a OpProto instance.
    :type op_proto: op_proto_pb2.OpProto
    :return: docstring
    """
    if not isinstance(op_proto, op_proto_pb2.OpProto):
        raise TypeError("Input must be OpProto")
    f = cStringIO.StringIO()
    f.write(op_proto.comment)
    f.write("\n")

    def __append_param__(name, comment, type):
        # Maybe replace the following line with template engine is better.
        f.write(":param ")
        f.write(name)
        f.write(": ")
        f.write(comment)
        f.write("\n")
        f.write(":type ")
        f.write(name)
        f.write(": ")
        f.write(type)
        f.write("\n")

    for ipt in op_proto.inputs:
        __append_param__(ipt.name, ipt.comment, "list | basestr"
                         if ipt.multiple else "basestr")

    temp_var_prefix = \
        "This is a temporary variable. It does not have to set by user. "
    for opt in op_proto.outputs:
        __append_param__(opt.name, opt.comment if not opt.temporary else
                         temp_var_prefix + opt.comment, "list | basestr"
                         if opt.multiple else "basestr")

    for attr in op_proto.attrs:
        attr_type = None
        if attr.type == attr_type_pb2.INT:
            attr_type = "int"
        elif attr.type == attr_type_pb2.FLOAT:
            attr_type = "float"
        elif attr.type == attr_type_pb2.STRING:
            attr_type = "basestr"
        elif attr.type == attr_type_pb2.INTS:
            attr_type = "list of int"
        elif attr.type == attr_type_pb2.FLOATS:
            attr_type = "list of float"
        elif attr.type == attr_type_pb2.STRINGS:
            attr_type = "list of basestr"

        if attr_type is None:
            raise RuntimeError("Not supported attribute type " + attr.type)

        __append_param__(attr.name, attr.comment, attr_type)

    return f.getvalue()


def create_op_creation_method(op_proto):
    """
    Generate op creation method for an OpProto
    """
    method = OpDescCreationMethod(op_proto)

    def __impl__(*args, **kwargs):
        opdesc = method(*args, **kwargs)
        return core.Operator.create(opdesc.SerializeToString())

    __impl__.__doc__ = get_docstring_from_op_proto(op_proto)
    __impl__.all_input_args = [var.name for var in op_proto.inputs]
    __impl__.all_output_args = [var.name for var in op_proto.outputs]
    __impl__.all_attr_args = [attr.name for attr in op_proto.attrs]
    __impl__.all_not_temp_output_args = [
        var.name for var in op_proto.outputs if not var.temporary
    ]

    return __impl__


class OpCreationsHolder(object):
    """
    A object will holds all op creation methods.
    
    Use `op_creations.xxx_op` to access them.
    """
    pass


op_creations = OpCreationsHolder()


def __bootstrap__():
    """
    Bootstrap function for this module. It will dynamic create all op creation
    methods in runtime.
    """
    for op_proto in get_all_op_protos():
        func = create_op_creation_method(op_proto)
        func.__name__ = str(op_proto.type)
        setattr(op_creations, func.__name__, func)


__bootstrap__()
