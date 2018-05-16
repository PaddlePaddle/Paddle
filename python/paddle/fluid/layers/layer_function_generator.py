#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import cStringIO
import functools
import warnings

from ..proto import framework_pb2
from ..framework import OpProtoHolder, Variable
from ..layer_helper import LayerHelper

__all__ = [
    'deprecated',
    'generate_layer_fn',
    'autodoc',
]


def _convert_(name):
    """
    Formatting.

    Args:
       name: The name/alias

    This function takes in a name and converts it to a standard format of
    group1_group2. Where as per the regular expression, group1 can have
    alphabets and numbers and group2 has capital alphabets.

    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _generate_doc_string_(op_proto):
    """
    Generate docstring by OpProto

    Args:
        op_proto (framework_pb2.OpProto): a protobuf message typed OpProto

    Returns:
        str: the document string
    """

    def _type_to_str_(tp):
        return framework_pb2.AttrType.Name(tp)

    if not isinstance(op_proto, framework_pb2.OpProto):
        raise TypeError("OpProto should be `framework_pb2.OpProto`")

    buf = cStringIO.StringIO()
    buf.write(op_proto.comment)
    buf.write('\nArgs:\n')
    for each_input in op_proto.inputs:
        line_begin = '    {0}: '.format(_convert_(each_input.name))
        buf.write(line_begin)
        buf.write(each_input.comment)
        buf.write('\n')
        buf.write(' ' * len(line_begin))
        buf.write('Duplicable: ')
        buf.write(str(each_input.duplicable))
        buf.write('  Optional: ')
        buf.write(str(each_input.dispensable))
        buf.write('\n')

    for each_attr in op_proto.attrs:
        buf.write('    ')
        buf.write(each_attr.name)
        buf.write(' (')
        buf.write(_type_to_str_(each_attr.type))
        buf.write('): ')
        buf.write(each_attr.comment)
        buf.write('\n')

    if len(op_proto.outputs) != 0:
        buf.write('\nReturns:\n')
        buf.write('    ')
        for each_opt in op_proto.outputs:
            if not each_opt.intermediate:
                break
        buf.write(each_opt.comment)

    return buf.getvalue()


def generate_layer_fn(op_type):
    """Register the Python layer for an Operator.

    Args:
       op_type: The name of the operator to be created.

    This function takes in the operator type (sigmoid, mean , average etc) and
    creates the operator functionality.

    """
    op_proto = OpProtoHolder.instance().get_op_proto(op_type)
    not_intermediate_outputs = \
        filter(lambda output: not output.intermediate, op_proto.outputs)
    intermediate_outputs = \
        filter(lambda output: output.intermediate, op_proto.outputs)

    if len(not_intermediate_outputs) != 1:
        raise ValueError("Only one non intermediate output operator can be",
                         "automatically generated. {0}".format(op_type))

    if not_intermediate_outputs[0].duplicable:
        raise ValueError(
            "Only non duplicable op can be automatically generated.")

    for output in intermediate_outputs:
        if output.duplicable:
            raise ValueError("The op can be automatically generated only when ",
                             "all intermediate ops are not duplicable.")

    o_name = not_intermediate_outputs[0].name
    intermediate_output_names = [output.name for output in intermediate_outputs]

    def infer_and_check_dtype(op_proto, *args, **kwargs):
        """
        This function performs the sanity check for dtype and
        instance type.
        """
        dtype = None
        for ipt in op_proto.inputs:
            name = _convert_(ipt.name)
            val = kwargs.pop(name, [])
            if not isinstance(val, list) and not isinstance(val, tuple):
                val = [val]
            if len(val) == 0:
                val = [args[0]]
                args = args[1:]

            for each in val:
                if not isinstance(each, Variable):
                    raise ValueError("input of {0} must be variable".format(
                        op_type))

                if dtype is None:
                    dtype = each.dtype
                elif dtype != each.dtype:
                    raise ValueError(
                        "operator {0} must input same dtype. {1} vs {2}".format(
                            op_type, dtype, each.dtype))

        return dtype

    def func(*args, **kwargs):
        helper = LayerHelper(op_type, **kwargs)

        dtype = infer_and_check_dtype(op_proto, *args, **kwargs)

        inputs = dict()
        for ipt in op_proto.inputs:
            name = _convert_(ipt.name)
            val = kwargs.pop(name, [])
            if not isinstance(val, list) and not isinstance(val, tuple):
                val = [val]
            if len(val) == 0 and len(args) != 0:
                val = args[0]
                args = args[1:]
            inputs[ipt.name] = val

        outputs = dict()
        out = kwargs.pop(_convert_(o_name), [])
        if out:
            out_var = out[0] if (isinstance(out, list) or
                                 isinstance(out, tuple)) else out
        else:
            out_var = helper.create_tmp_variable(dtype=dtype)
        outputs[o_name] = [out_var]
        for name in intermediate_output_names:
            outputs[name] = [helper.create_tmp_variable(dtype=dtype)]
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=kwargs)
        return helper.append_activation(out_var)

    func.__name__ = op_type
    func.__doc__ = _generate_doc_string_(op_proto)
    return func


def deprecated(func_or_class):
    """
    Deprecated warning decorator. It will result a warning message.
    Should be used before class or function, member function
    """

    @functools.wraps(func)
    def func_wrapper(*args, **kwargs):
        """
        Wrap func with deprecated warning
        """
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return func_wrapper


def autodoc(comment=""):
    def __impl__(func):
        func.__doc__ = _generate_doc_string_(OpProtoHolder.instance(
        ).get_op_proto(func.__name__)) + comment
        return func

    return __impl__
