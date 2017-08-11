import paddle.v2.framework.core as core
import paddle.v2.framework.proto.framework_pb2 as framework_pb2


def get_all_op_protos():
    """
    Get all registered op proto from Paddle C++
    :return: list of OpProto
    """
    protostrs = core.get_all_op_protos()
    ret_values = []
    for pbstr in protostrs:
        op_proto = framework_pb2.OpProto.FromString(str(pbstr))
        ret_values.append(op_proto)
    return ret_values


def is_str(s):
    return isinstance(s, str) or isinstance(s, unicode)


class OpDescCreationMethod(object):
    """
    A Functor object to convert user input(use key word args) to OpDesc based on
    OpProto.
    
    :param op_proto: The OpProto object.
    :type op_proto: op_proto_pb2.OpProto
    """

    def __init__(self, op_proto):
        if not isinstance(op_proto, framework_pb2.OpProto):
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
        op_desc = framework_pb2.OpDesc()

        for input_parameter in self.__op_proto__.inputs:
            input_arguments = kwargs.get(input_parameter.name, [])
            if is_str(input_arguments):
                input_arguments = [input_arguments]

            if not input_parameter.duplicable and len(input_arguments) > 1:
                raise ValueError("Input %s only accepts one input, but give %d"
                                 % (input_parameter.name, len(input_arguments)))

            ipt = op_desc.inputs.add()
            ipt.parameter = input_parameter.name
            ipt.arguments.extend(input_arguments)

        for output_parameter in self.__op_proto__.outputs:
            output_arguments = kwargs.get(output_parameter.name, [])
            if is_str(output_arguments):
                output_arguments = [output_arguments]

            if not output_parameter.duplicable and len(output_arguments) > 1:
                raise ValueError(
                    "Output %s only accepts one output, but give %d" %
                    (output_parameter.name, len(output_arguments)))

            out = op_desc.outputs.add()
            out.parameter = output_parameter.name
            out.arguments.extend(output_arguments)

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
                if attr.type == framework_pb2.INT:
                    new_attr.i = user_defined_attr
                elif attr.type == framework_pb2.FLOAT:
                    new_attr.f = user_defined_attr
                elif attr.type == framework_pb2.STRING:
                    new_attr.s = user_defined_attr
                elif attr.type == framework_pb2.INTS:
                    new_attr.ints.extend(user_defined_attr)
                elif attr.type == framework_pb2.FLOATS:
                    new_attr.floats.extend(user_defined_attr)
                elif attr.type == framework_pb2.STRINGS:
                    new_attr.strings.extend(user_defined_attr)
                else:
                    raise NotImplementedError("Not support attribute type " +
                                              attr.type)

        return op_desc

    @staticmethod
    def any_is_true(generator):
        """
        Reduce a bool array to one. If any of them is True, then return True.
        """
        for flag in generator:
            if flag:
                return True
        return False


class OpInfo(object):
    def __init__(self, name, method, inputs, outputs, attrs):
        self.name = name
        self.method = method
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = attrs


def create_op_creation_method(op_proto):
    """
    Generate op creation method for an OpProto
    """
    method = OpDescCreationMethod(op_proto)

    def __impl__(*args, **kwargs):
        opdesc = method(*args, **kwargs)
        return core.Operator.create(opdesc.SerializeToString())

    return OpInfo(
        method=__impl__,
        name=op_proto.type,
        inputs=[var.name for var in op_proto.inputs],
        outputs=[var.name for var in op_proto.outputs],
        attrs=[attr.name for attr in op_proto.attrs])


class OperatorFactory(object):
    def __init__(self):
        self.op_methods = dict()
        for op_proto in get_all_op_protos():
            method = create_op_creation_method(op_proto)
            self.op_methods[method.name] = method

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

        return self.get_op_info(t).method(**kwargs)

    def types(self):
        return self.op_methods.keys()

    def get_op_info(self, t):
        if t not in self.op_methods:
            raise ValueError("operator %s is not registered", t)
        return self.op_methods.get(t)

    def get_op_input_names(self, type):
        return self.get_op_info(type).inputs

    def get_op_output_names(self, type):
        return self.get_op_info(type).outputs

    def get_op_attr_names(self, type):
        return self.get_op_info(type).attrs


Operator = OperatorFactory()  # Default global factory
