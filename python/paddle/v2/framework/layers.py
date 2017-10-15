from paddle.v2.framework.layer_helper import LayerHelper
import paddle.v2.framework.core as core
from paddle.v2.framework.framework import OpProtoHolder, Variable
import re

__all__ = ['fc_layer', 'data_layer', 'cross_entropy']


def fc_layer(input,
             size,
             param_attr=None,
             bias_attr=True,
             name=None,
             act=None,
             num_flatten_dims=1,
             program=None):
    # create helper
    helper = LayerHelper('fc', **locals())

    dtype = helper.input_dtype()

    # mul
    mul_results = []
    for input_var, param_attr in helper.iter_inputs_and_params():
        input_shape = input_var.shape
        param_shape = list(input_shape[num_flatten_dims:]) + [size]
        w = helper.create_parameter(
            attr=param_attr, shape=param_shape, dtype=dtype)
        tmp = helper.create_tmp_variable(dtype)
        helper.append_op(
            type="mul",
            inputs={
                "X": input_var,
                "Y": w,
            },
            outputs={"Out": tmp},
            attrs={'x_num_col_dims': num_flatten_dims})
        mul_results.append(tmp)

    # sum
    if len(mul_results) == 1:
        pre_bias = mul_results[0]
    else:
        pre_bias = helper.create_tmp_variable(dtype)
        helper.append_op(
            type="sum", inputs={"X": mul_results}, outputs={"Out": pre_bias})
    # add bias
    pre_activation = helper.append_bias_op(pre_bias)
    # add activation
    return helper.append_activation(pre_activation)


def data_layer(name,
               shape,
               data_type='float32',
               type=core.VarDesc.VarType.LOD_TENSOR,
               program=None):
    helper = LayerHelper('data', **locals())
    shape = [-1] + shape  # append batch size as -1
    return helper.create_global_variable(
        name=name, shape=shape, dtype=data_type, type=type)


def _convert_(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _create_op_func_(op_type):
    op_proto = OpProtoHolder.instance().get_op_proto(op_type)
    if len(op_proto.outputs) != 1:
        raise ValueError(
            "Only one output operator can be automatically generated")

    if op_proto.outputs[0].duplicable:
        raise ValueError(
            "Only not duplicable op can be automatically generated")

    o_name = op_proto.outputs[0].name

    def func(**kwargs):
        helper = LayerHelper(op_type, **locals())
        inputs = dict()
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
                    dtype = each.data_type
                elif dtype != each.data_type:
                    raise ValueError(
                        "operator {0} must input same dtype".format(op_type))
            inputs[ipt.name] = val

        out = helper.create_tmp_variable(dtype=dtype)
        helper.append_op(
            type=op_type, inputs=inputs, outputs={o_name: [out]}, attrs=kwargs)
        return out

    func.__name__ = op_type
    globals()[op_type] = func
    global __all__
    __all__.append(op_type)


_create_op_func_('mean')


def cross_entropy(input, label, program=None, **kwargs):
    helper = LayerHelper('cross_entropy', **locals())
    out = helper.create_tmp_variable(dtype=input.data_type)
    helper.append_op(
        type='cross_entropy',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out]},
        attrs=kwargs)
    return out
