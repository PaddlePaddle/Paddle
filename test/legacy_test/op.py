#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from paddle.base import core
from paddle.base.proto import framework_pb2


# NOTE: this is added to support creating a Scalar message
# from a python number
def make_scalar_proto(value):
    s = framework_pb2.Scalar()
    if isinstance(value, bool):
        s.type = framework_pb2.Scalar.Type.BOOLEAN
        s.b = value
    elif isinstance(value, int):
        s.type = framework_pb2.Scalar.Type.LONG
        s.i = value
    elif isinstance(value, float):
        s.type = framework_pb2.Scalar.Type.FLOAT64
        s.r = value
    elif isinstance(value, complex):
        s.type = framework_pb2.Scalar.Type.COMPLEX128
        complex_value = framework_pb2.Complex()
        complex_value.r = value.real
        complex_value.i = value.imag
        s.c.CopyFrom(complex_value)
    return s


def get_all_op_protos():
    """
    Get all registered op proto from PaddlePaddle C++ end.
    :return: A list of registered OpProto.
    """
    protostrs = core.get_all_op_protos()
    ret_values = []
    for pbstr in protostrs:
        op_proto = framework_pb2.OpProto.FromString(bytes(pbstr))
        ret_values.append(op_proto)
    return ret_values


def is_str(s):
    return isinstance(s, str)


class OpDescCreationMethod:
    """
    Convert the user's input(only keyword arguments are supported) to OpDesc
    based on the OpProto.

    :param op_proto: The OpProto object.
    :type op_proto: op_proto_pb2.OpProto
    """

    def __init__(self, op_proto):
        if not isinstance(op_proto, framework_pb2.OpProto):
            raise TypeError(
                "Type of op_proto should be OpProto in PaddlePaddle."
            )
        self.__op_proto__ = op_proto
        self.__extra_attrs__ = core.get_op_extra_attrs(op_proto.type)

    def __call__(self, *args, **kwargs):
        """
        Convert user's input to OpDesc. Only keyword arguments are supported.
        :return: The OpDesc based on user input.
        :rtype: op_desc_pb2.OpDesc
        """
        if len(args) != 0:
            raise ValueError("Only keyword arguments are supported.")
        op_desc = framework_pb2.OpDesc()
        for input_parameter in self.__op_proto__.inputs:
            input_arguments = kwargs.get(input_parameter.name, [])
            if is_str(input_arguments):
                input_arguments = [input_arguments]

            if not input_parameter.duplicable and len(input_arguments) > 1:
                raise ValueError(
                    f"Input {input_parameter.name} expects only one input, but {len(input_arguments)} are given."
                )

            ipt = op_desc.inputs.add()
            ipt.parameter = input_parameter.name
            ipt.arguments.extend(input_arguments)

        for output_parameter in self.__op_proto__.outputs:
            output_arguments = kwargs.get(output_parameter.name, [])
            if is_str(output_arguments):
                output_arguments = [output_arguments]

            if not output_parameter.duplicable and len(output_arguments) > 1:
                raise ValueError(
                    f"Output {output_parameter.name} expects only one output, but {len(output_arguments)} are given."
                )

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
                if isinstance(user_defined_attr, np.ndarray):
                    user_defined_attr = user_defined_attr.tolist()
                if attr.type == framework_pb2.INT:
                    new_attr.i = user_defined_attr
                elif attr.type == framework_pb2.FLOAT:
                    new_attr.f = user_defined_attr
                elif attr.type == framework_pb2.LONG:
                    new_attr.l = user_defined_attr
                elif attr.type == framework_pb2.STRING:
                    new_attr.s = user_defined_attr
                elif attr.type == framework_pb2.BOOLEAN:
                    new_attr.b = user_defined_attr
                elif attr.type == framework_pb2.INTS:
                    new_attr.ints.extend(user_defined_attr)
                elif attr.type == framework_pb2.FLOATS:
                    new_attr.floats.extend(user_defined_attr)
                elif attr.type == framework_pb2.STRINGS:
                    new_attr.strings.extend(user_defined_attr)
                elif attr.type == framework_pb2.BOOLEANS:
                    new_attr.bools.extend(user_defined_attr)
                elif attr.type == framework_pb2.LONGS:
                    new_attr.longs.extend(user_defined_attr)
                elif attr.type == framework_pb2.FLOAT64:
                    new_attr.float64 = user_defined_attr
                elif attr.type == framework_pb2.FLOAT64S:
                    new_attr.float64s.extend(user_defined_attr)
                # the code below manipulates protobuf directly
                elif attr.type == framework_pb2.SCALAR:
                    scalar = make_scalar_proto(user_defined_attr)
                    new_attr.scalar.CopyFrom(scalar)
                elif attr.type == framework_pb2.SCALARS:
                    scalars = [
                        make_scalar_proto(item) for item in user_defined_attr
                    ]
                    for item in scalars:
                        new_attr.scalars.MergeFrom(item)
                else:
                    raise NotImplementedError(
                        f"A not supported attribute type: {attr.type}."
                    )
        for attr_name, default_val in self.__extra_attrs__.items():
            user_defined_attr = kwargs.get(attr_name, None)
            if user_defined_attr is not None:
                attr_type = int(
                    core.get_attribute_type(op_desc.type, attr_name)
                )
                new_attr = op_desc.attrs.add()
                new_attr.name = attr_name
                new_attr.type = attr_type
                if isinstance(user_defined_attr, np.ndarray):
                    user_defined_attr = user_defined_attr.tolist()
                if attr_type == framework_pb2.INT:
                    new_attr.i = user_defined_attr
                elif attr_type == framework_pb2.FLOAT:
                    new_attr.f = user_defined_attr
                elif attr_type == framework_pb2.LONG:
                    new_attr.l = user_defined_attr
                elif attr_type == framework_pb2.STRING:
                    new_attr.s = user_defined_attr
                elif attr_type == framework_pb2.BOOLEAN:
                    new_attr.b = user_defined_attr
                elif attr_type == framework_pb2.INTS:
                    new_attr.ints.extend(user_defined_attr)
                elif attr_type == framework_pb2.FLOATS:
                    new_attr.floats.extend(user_defined_attr)
                elif attr_type == framework_pb2.STRINGS:
                    new_attr.strings.extend(user_defined_attr)
                elif attr_type == framework_pb2.BOOLEANS:
                    new_attr.bools.extend(user_defined_attr)
                elif attr_type == framework_pb2.LONGS:
                    new_attr.longs.extend(user_defined_attr)
                elif attr.type == framework_pb2.FLOAT64:
                    new_attr.float64 = user_defined_attr
                elif attr.type == framework_pb2.FLOAT64S:
                    new_attr.float64s.extend(user_defined_attr)
                # the code below manipulates protobuf directly
                elif attr.type == framework_pb2.SCALAR:
                    scalar = make_scalar_proto(user_defined_attr)
                    new_attr.scalar.CopyFrom(scalar)
                elif attr.type == framework_pb2.SCALARS:
                    scalars = [
                        make_scalar_proto(item) for item in user_defined_attr
                    ]
                    for item in scalars:
                        new_attr.scalars.MergeFrom(item)
                else:
                    raise NotImplementedError(
                        f"A not supported attribute type: {attr_type}."
                    )

        return op_desc

    @staticmethod
    def any_is_true(generator):
        """
        Reduce a boolean array to a single boolean parameter. If any element in
        the array is True, this function will return True, otherwise False.
        """
        for flag in generator:
            if flag:
                return True
        return False


class OpInfo:
    def __init__(self, name, method, inputs, outputs, attrs, extra_attrs):
        self.name = name
        self.method = method
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = attrs
        self.extra_attrs = extra_attrs


def create_op_creation_method(op_proto):
    """
    Generate op creation method for an OpProto.
    """
    method = OpDescCreationMethod(op_proto)

    def __impl__(*args, **kwargs):
        opdesc = method(*args, **kwargs)
        return core.Operator.create(opdesc.SerializeToString())

    extra_attrs_map = core.get_op_extra_attrs(op_proto.type)

    return OpInfo(
        method=__impl__,
        name=op_proto.type,
        inputs=[(var.name, var.duplicable) for var in op_proto.inputs],
        outputs=[(var.name, var.duplicable) for var in op_proto.outputs],
        attrs=[attr.name for attr in op_proto.attrs],
        extra_attrs=list(extra_attrs_map.keys()),
    )


class OperatorFactory:
    def __init__(self):
        self.op_methods = {}

        for op_proto in get_all_op_protos():
            method = create_op_creation_method(op_proto)
            self.op_methods[method.name] = method

    def __call__(self, *args, **kwargs):
        if "type" in kwargs:
            if len(args) != 0:
                raise ValueError(
                    'Except the argument "type",'
                    'all of the other arguments should be keyword arguments.'
                )
            t = kwargs.pop("type")
        else:
            if len(args) != 1:
                raise ValueError(
                    'Except the argument "type",'
                    'all of the other arguments should be keyword arguments.'
                )
            t = args[0]

        return self.get_op_info(t).method(**kwargs)

    def types(self):
        return list(self.op_methods.keys())

    def get_op_info(self, t):
        if t not in self.op_methods:
            raise ValueError(f"The operator: {t} is not registered.")
        return self.op_methods.get(t)

    def get_op_input_names(self, type):
        return [x[0] for x in self.get_op_info(type).inputs]

    def get_op_inputs(self, type):
        return self.get_op_info(type).inputs

    def get_op_output_names(self, type):
        return [x[0] for x in self.get_op_info(type).outputs]

    def get_op_outputs(self, type):
        return self.get_op_info(type).outputs

    def get_op_attr_names(self, type):
        return self.get_op_info(type).attrs

    def get_op_extra_attr_names(self, type):
        return self.get_op_info(type).extra_attrs


Operator = OperatorFactory()  # The default global factory
