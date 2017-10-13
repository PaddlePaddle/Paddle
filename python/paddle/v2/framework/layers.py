from paddle.v2.framework.layer_helper import LayerHelper
import paddle.v2.framework.core as core


def fc_layer(input,
             size,
             param_attr=None,
             bias_attr=True,
             name=None,
             act=None,
             num_flatten_dims=1,
             program=None):
    # create helper
    helper = LayerHelper(locals())

    dtype = helper.input_dtype

    # mul
    mul_results = []
    for input_var, param_attr in helper.iter_inputs_and_params():
        input_shape = input_var.shape
        param_shape = input_shape[num_flatten_dims:] + [size]
        w = helper.create_parameter(param_attr, param_shape, dtype)
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
        pre_bias = mul_results
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
    helper = LayerHelper(locals())
    shape = [-1] + shape  # append batch size as -1
    return helper.create_global_variable(
        name=name, shape=shape, dtype=data_type, type=type)


def cross_entropy(input, label, program=None, **kwargs):
    helper = LayerHelper(locals())
    out = helper.create_tmp_variable(dtype=input.data_type)
    helper.append_op(
        type='cross_entropy',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Out': [out]},
        attrs=kwargs)
    return out
