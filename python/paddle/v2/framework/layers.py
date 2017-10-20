from paddle.v2.framework.layer_helper import LayerHelper
import paddle.v2.framework.core as core
from paddle.v2.framework.framework import OpProtoHolder, Variable
import re

__all__ = ['fc', 'data', 'cross_entropy', 'conv2d', 'pool2d']


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
            attrs={
                'x_num_col_dims': num_flatten_dims,
                'y_num_col_dims': len(input_shape) - num_flatten_dims
            })
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
        outputs={'Y': [squhare_out]},
        attrs={'factor': 2.0})
    return square_out


def accuracy(input, label, **kwargs):
    helper = LayerHelper("accuracy", **kwargs)
    out_dtype = kwargs.get("out_dtype", "float32")
    acc_out = helper.create_tmp_variable(dtype=out_dtype)
    helper.append_op(
        type="accuracy", inputs={"Inference": [input],
                                 "Label": [label]})
    return acc_out


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

    if isinstance(filter_size, int):
        filter_size = [filter_size, filter_size]
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(padding, int):
        padding = [padding, padding]

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


def pool2d(input,
           pool_size,
           pool_type,
           pool_stride=[1, 1],
           pool_padding=[0, 0],
           global_pooling=False,
           program=None):
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))
    if isinstance(pool_size, int):
        pool_size = [pool_size, pool_size]
    if isinstance(pool_stride, int):
        pool_stride = [pool_stride, pool_stride]
    if isinstance(pool_padding, int):
        pool_padding = [pool_padding, pool_padding]

    helper = LayerHelper('conv2d', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_tmp_variable(dtype)

    helper.append_op(
        type="pool2d",
        inputs={"X": input},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "global_pooling": global_pooling,
            "strides": pool_stride,
            "paddings": pool_padding
        })

    return pool_out
