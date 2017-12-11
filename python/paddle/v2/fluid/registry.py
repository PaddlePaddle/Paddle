import warnings
import functools
import inspect

from framework import OpProtoHolder, Variable, Program, Operator
from paddle.v2.fluid.layer_helper import LayerHelper, unique_name


def layer_registry(op_type):
    """
    Register an Python layer for an Operator

    Args:
       op_type: The name of the operator to be created

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
                         "automatically generated")

    if not_intermediate_outputs[0].duplicable:
        raise ValueError(
            "Only non duplicable op can be automatically generated")

    for output in intermediate_outputs:
        if output.duplicable:
            raise ValueError("The op can be automatically generated only when ",
                             "all intermediate ops are not duplicable")

    o_name = not_intermediate_outputs[0].name
    intermediate_output_names = [output.name for output in intermediate_outputs]

    def infer_and_check_dtype(op_proto, **kwargs):
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

    def func(**kwargs):
        helper = LayerHelper(op_type, **kwargs)

        dtype = infer_and_check_dtype(op_proto, **kwargs)

        inputs = dict()
        for ipt in op_proto.inputs:
            name = _convert_(ipt.name)
            val = kwargs.pop(name, [])
            if not isinstance(val, list) and not isinstance(val, tuple):
                val = [val]
            inputs[ipt.name] = val

        outputs = dict()
        out = helper.create_tmp_variable(dtype=dtype)
        outputs[o_name] = [out]
        for name in intermediate_output_names:
            outputs[name] = [helper.create_tmp_variable(dtype=dtype)]
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=kwargs)
        return helper.append_activation(out)

    func.__name__ = op_type
    globals()[op_type] = func
    func.__doc__ = _generate_doc_string_(op_proto)
    global __all__
    __all__.append(op_type)


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
        warnings.simplefilter('always', DeprecationWarning)  #turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  #reset filter
        return func(*args, **kwargs)

    return func_wrapper
