from paddle.v2.framework.layer_helper import LayerHelper
import paddle.v2.framework.core as core
from paddle.v2.framework.framework import OpProtoHolder, Variable
import re

__all__ = ['fc', 'data', 'cross_entropy', 'conv2d']


def fc(input,
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


def data(name,
         shape,
         data_type='float32',
         type=core.VarDesc.VarType.LOD_TENSOR,
         append_batch_size=True,
         program=None):
    helper = LayerHelper('data', **locals())
    if append_batch_size:
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
        helper = LayerHelper(op_type, **kwargs)
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
_create_op_func_('mul')
_create_op_func_('pool2d')


def cross_entropy(input, label, **kwargs):
    helper = LayerHelper('cross_entropy', **kwargs)
    out = helper.create_tmp_variable(dtype=input.data_type)
    helper.append_op(
        type='cross_entropy',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out]},
        attrs=kwargs)
    return out


def square_error_cost(input, label, **kwargs):
    helper = LayerHelper('square_error_cost', **kwargs)
    minus_out = helper.create_tmp_variable(dtype=input.data_type)
    helper.append_op(
        type='elementwise_sub',
        inputs={'X': [input],
                'Y': [label]},
        outputs={'Out': [minus_out]})

    square_out = helper.create_tmp_variable(dtype=input.data_type)
    helper.append_op(
        type='pow',
        inputs={'X': [minus_out]},
        outputs={'Y': [square_out]},
        attrs={'factor': 2.0})
    return square_out


def conv2d(input,
           num_filters,
           name=None,
           filter_size=[1, 1],
           act=None,
           groups=None,
           stride=[1, 1],
           padding=None,
           bias_attr=None,
           param_attr=None,
           program=None):
    helper = LayerHelper('conv2d', **locals())
    dtype = helper.input_dtype()

    num_channels = input.shape[1]
    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups is not 0:
            raise ValueError("num_channels must be divisible by groups.")
        num_filter_channels = num_channels / groups

    input_shape = input.shape
    filter_shape = [num_filters, num_filter_channels] + filter_size
    filter = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype)
    pre_bias = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='conv2d',
        inputs={
            'Input': input,
            'Filter': filter,
        },
        outputs={"Output": pre_bias},
        attrs={'strides': stride,
               'paddings': padding,
               'groups': groups})

    pre_act = helper.append_bias_op(pre_bias)

    return helper.append_activation(pre_act)
