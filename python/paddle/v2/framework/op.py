import paddle.v2.framework.core as core
import paddle.v2.framework.proto.attr_type_pb2 as attr_type_pb2
import paddle.v2.framework.proto.op_desc_pb2 as op_desc_pb2
import paddle.v2.framework.proto.op_proto_pb2 as op_proto_pb2


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


def create_op_creation_method(op_proto):
    """
    Generate op creation method for an OpProto
    """
    method = OpDescCreationMethod(op_proto)

    def __impl__(*args, **kwargs):
        opdesc = method(*args, **kwargs)
        return core.Operator.create(opdesc.SerializeToString())

    return {
        'method': __impl__,
        'name': op_proto.type,
        'all_inputs': [var.name for var in op_proto.inputs],
        'all_outputs': [var.name for var in op_proto.outputs],
        'all_attrs': [attr.name for attr in op_proto.attrs],
        'all_no_temp_outputs':
        [var.name for var in op_proto.outputs if not var.temporary]
    }


class OperatorFactory(object):
    def __init__(self):
        self.op_methods = dict()
        for op_proto in get_all_op_protos():
            method = create_op_creation_method(op_proto)
            self.op_methods[method['name']] = method

    def __call__(self, *args, **kwargs):
        if 'type' in kwargs:
            if len(args) != 0:
                raise ValueError("All Paddle argument should be key-word "
                                 "argument except type")
            t = kwargs.pop('type')
        else:
            if len(args) != 1:
                raise ValueError("All Paddle argument should be key-word "
                                 "argument except type")
            t = args[0]

        return self.get_op_creation_info(t)['method'](**kwargs)

    def get_op_creation_info(self, t):
        if t not in self.op_methods:
            raise ValueError("operator %s is not registered", t)
        return self.op_methods.get(t)

    def get_op_input_names(self, type):
        return self.get_op_creation_info(type)['all_inputs']

    def get_op_output_names(self, type):
        return self.get_op_creation_info(type)['all_outputs']

    def get_op_attr_names(self, type):
        return self.get_op_creation_info(type)['all_attrs']

    def get_op_no_temp_output_names(self, type):
        return self.get_op_creation_info(type)['all_no_temp_outputs']


Operator = OperatorFactory()  # Default global factory
