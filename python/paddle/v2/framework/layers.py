from paddle.v2.framework.layer_helper import LayerHelper


def fc_layer(input,
             size,
             param_attr=None,
             bias_attr=False,
             name=None,
             act=None,
             num_flatten_dims=1):
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
