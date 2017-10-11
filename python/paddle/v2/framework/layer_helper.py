from paddle.v2.framework.graph import Variable, g_program
import paddle.v2.framework.core as core
import copy
import itertools


def unique_name(prefix):
    uid = core.unique_integer()  # unique during whole process.
    return "_".join([prefix, str(uid)])


class LayerHelper(object):
    def __init__(self, layer_type, **kwargs):
        self.kwargs = kwargs
        self.layer_type = layer_type
        name = self.kwargs.get('name', None)
        if name is None:
            self.kwargs['name'] = unique_name(self.layer_type)

    @property
    def name(self):
        return self.kwargs['name']

    @property
    def program(self):
        return self.kwargs.get('program', g_program)

    def append_op(self, *args, **kwargs):
        return self.program.current_block().append_op(*args, **kwargs)

    @property
    def multiple_input(self):
        inputs = self.kwargs.get('input', [])
        type_error = TypeError(
            "Input of {0} layer should be Variable or sequence of Variable".
            format(self.layer_type))
        if isinstance(inputs, Variable):
            inputs = [inputs]
        elif not isinstance(inputs, list) and not isinstance(inputs, tuple):
            raise type_error
        else:
            for each in inputs:
                if not isinstance(each, Variable):
                    raise type_error
        return inputs

    @property
    def input(self):
        inputs = self.multiple_input
        if len(inputs) != 1:
            raise "{0} layer only takes one input".format(self.layer_type)
        return inputs[0]

    @property
    def param_attr(self):
        return self.kwargs.get('param_attr', {
            'name': None,
            'init_attr': {
                'type': 'uniform_random',
                'min': -1.0,
                'max': 1.0
            }
        })

    def bias_attr(self, size):
        bias_attr = self.kwargs.get('bias_attr', False)
        if bias_attr is None:
            bias_attr = {
                'name': None,
                'init_attr': {
                    'type': 'fill',
                    'value': 0.0,
                    'shape': [size]
                }
            }
        return bias_attr

    def multiple_param_attr(self, length):
        param_attr = self.param_attr
        if isinstance(param_attr, dict):
            param_attr = [param_attr]

        if len(param_attr) != 1 and len(param_attr) != length:
            raise ValueError("parameter number mismatch")
        elif len(param_attr) == 1:
            tmp = [None] * length
            for i in xrange(length):
                tmp[i] = copy.deepcopy(param_attr)
            param_attr = tmp

        return param_attr

    def iter_inputs_and_params(self):
        inputs = self.multiple_input
        param_attrs = self.multiple_param_attr(len(inputs))
        for ipt, param_attr in itertools.izip(inputs, param_attrs):
            yield ipt, param_attr

    @property
    def input_dtype(self):
        inputs = self.multiple_input
        dtype = None
        for each in inputs:
            if dtype is None:
                dtype = each.data_type
            elif dtype != each.data_type:
                raise ValueError("Data Type mismatch")

    def create_parameter(self, attr, shape, dtype, suffix='w'):
        if attr['name'] is None:
            attr['name'] = unique_name(".".join([self.name, suffix]))
        return self.program.global_block().create_parameter(
            name=attr['name'],
            dtype=dtype,
            shape=shape,
            initialize_attr=attr['init_attr'])

    def create_tmp_variable(self, dtype):
        return self.program.current_block().create_var(
            name=unique_name(".".join([self.name, 'tmp'])), dtype=dtype)

    def append_bias_op(self, input_var):
        bias_attr = self.bias_attr(self.kwargs['size'])
        if not bias_attr:
            return input_var
        b = self.create_parameter(
            bias_attr,
            shape=[self.kwargs['size']],
            dtype=input_var.data_type,
            suffix='b')
        tmp = self.create_tmp_variable(dtype=input_var.data_type)
        self.append_op(
            type='elementwise_add',
            inputs={'X': [input_var],
                    'Y': [b]},
            outputs={'Out': [tmp]})
        return tmp

    def append_activation(self, input_var):
        act = self.kwargs.get('act', None)
        if act is None:
            return input_var

        tmp = self.create_tmp_variable(dtype=input_var.data_type)
        act_type = act.pop('type')
        self.append_op(
            type=act_type,
            inputs={"X": [input_var]},
            outputs={"Y": [tmp]},
            attrs=act)
        return tmp
