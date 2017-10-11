from paddle.v2.framework.graph import g_program, Variable
import paddle.v2.framework.core as core
import copy
import itertools


def unique_name(prefix):
    uid = core.unique_integer()  # unique during whole process.
    return "_".join([prefix, str(uid)])


def fc_layer(input,
             size,
             param_attr=None,
             bias_attr=False,
             name=None,
             act=None,
             num_flatten_dims=1):
    # get name
    if name is None:
        name = unique_name("fc")

    # check size is a integer
    if isinstance(size, int):
        raise TypeError("size of fc layer must be integer")

    # FC accept multiple input
    if isinstance(input, Variable):
        input = [input]

    input_type_error = TypeError(
        "input of fc should be a variable, or a sequence of variable")
    if not isinstance(input, list) and not isinstance(input, tuple):
        raise input_type_error
    else:
        dtype = None
        for each_var in input:
            if not isinstance(each_var, Variable):
                raise input_type_error
            if dtype is None:
                dtype = each_var.data_type
            elif dtype != each_var.data_type:
                raise ValueError("input of fc layer should be same data type")

    # Set default value of param_attr
    if param_attr is None:
        param_attr = {
            "name": None,
            "init_attr": {
                "type": 'uniform_random',
                'min': -1.0,
                'max': 1.0
            }
        }

    if isinstance(param_attr, dict):
        param_attr = [param_attr]

    if not isinstance(param_attr, list) and not isinstance(param_attr, tuple):
        raise TypeError(
            "param attribute of fc should be a dict, or a sequence of dict")

    if len(param_attr) == 1 and len(input) > 1:
        tmp = [None] * len(input)
        for i in xrange(len(tmp)):
            tmp[i] = copy.deepcopy(param_attr)
        param_attr = tmp

    # mul
    mul_results = []
    for ipt, attr in itertools.izip(input, param_attr):
        if attr['name'] is None:
            param_name = unique_name(".".join([name, "w"]))
            input_shape = ipt.shape
            param_shape = input_shape[num_flatten_dims:] + [size]
            w = g_program.global_block().create_parameter(
                name=param_name,
                shape=param_shape,
                dtype=dtype,
                initialize_attr=attr['init_attr'])

            tmp = g_program.current_block().create_var(
                name=unique_name(".".join([name, 'tmp'])), dtype=dtype)

            g_program.current_block().append_op(
                type="mul",
                inputs={"X": [ipt],
                        "Y": [w]},
                outputs={"Out": [tmp]},
                attrs={'x_num_col_dims': num_flatten_dims})
            mul_results.append(tmp)

    # sum mul result
    if len(mul_results) == 1:
        pre_bias = mul_results[0]
    else:
        tmp = g_program.current_block().create_var(
            name=unique_name(".".join([name, 'tmp'])), dtype=dtype)

        g_program.current_block().append_op(
            type="sum", inputs={"X": mul_results}, outputs={"Out": tmp})
        pre_bias = tmp

    if bias_attr is None:
        bias_attr = {
            'name': None,
            'init_attr': {
                'type': 'fill',
                'shape': [size],
                'value': 0.0
            }
        }

    if bias_attr:
        # add bias
        tmp = g_program.current_block().create_var(
            name=unique_name(".".join([name, 'tmp'])), dtype=dtype)

        if bias_attr['name'] is None:
            bias_name = unique_name(".".join([name, "b"]))
        bias_shape = [size]
        b = g_program.global_block().create_parameter(
            name=bias_name,
            shape=bias_shape,
            dtype=dtype,
            initialize_attr=bias_attr['init_attr'])

        g_program.current_block().append_op(
            "elementwise_add",
            inputs={"X": [pre_bias],
                    "Y": [b]},
            outputs={"Out": [tmp]})

        pre_activation = tmp
    else:
        pre_activation = pre_bias

    if act is not None:
        act_type = act['type']
        act_attr = act.get('attr', None)

        tmp = g_program.current_block().create_var(
            name=unique_name(".".join([name, 'tmp'])), dtype=dtype)

        g_program.current_block().append_op(
            type=act_type,
            inputs={"X": pre_activation},
            outputs={"Out": tmp},
            attrs=act_attr)
        out = tmp
    else:
        out = pre_activation

    return out
