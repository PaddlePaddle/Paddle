import copy
import itertools

from paddle.v2.framework.framework import Variable, g_main_program, \
    g_startup_program, unique_name, Program
from paddle.v2.framework.initializer import ConstantInitializer, \
    UniformInitializer, XavierInitializer


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
    def main_program(self):
        prog = self.kwargs.get('main_program', None)
        if prog is None:
            return g_main_program
        else:
            return prog

    @property
    def startup_program(self):
        prog = self.kwargs.get('startup_program', None)
        if prog is None:
            return g_startup_program
        else:
            return prog

    def append_op(self, *args, **kwargs):
        return self.main_program.current_block().append_op(*args, **kwargs)

    def multiple_input(self, input_param_name='input'):
        inputs = self.kwargs.get(input_param_name, [])
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

    def input(self, input_param_name='input'):
        inputs = self.multiple_input(input_param_name)
        if len(inputs) != 1:
            raise "{0} layer only takes one input".format(self.layer_type)
        return inputs[0]

    @property
    def param_attr(self):
        default = {'name': None, 'initializer': XavierInitializer()}
        actual = self.kwargs.get('param_attr', None)
        if actual is None:
            actual = default
        for default_field in default.keys():
            if default_field not in actual:
                actual[default_field] = default[default_field]
        return actual

    @property
    def bias_attr(self):
        default = {'name': None, 'initializer': XavierInitializer()}
        bias_attr = self.kwargs.get('bias_attr', None)
        if bias_attr is None:
            bias_attr = default

        if isinstance(bias_attr, dict):
            for default_field in default.keys():
                if default_field not in bias_attr:
                    bias_attr[default_field] = default[default_field]
        return bias_attr

    def multiple_param_attr(self, length):
        param_attr = self.param_attr
        if isinstance(param_attr, dict):
            param_attr = [param_attr]

        if len(param_attr) != 1 and len(param_attr) != length:
            raise ValueError("parameter number mismatch")
        elif len(param_attr) == 1 and length != 1:
            tmp = [None] * length
            for i in xrange(length):
                tmp[i] = copy.deepcopy(param_attr[0])
            param_attr = tmp
        return param_attr

    def iter_inputs_and_params(self, input_param_name='input'):
        inputs = self.multiple_input(input_param_name)
        param_attrs = self.multiple_param_attr(len(inputs))
        for ipt, param_attr in itertools.izip(inputs, param_attrs):
            yield ipt, param_attr

    def input_dtype(self, input_param_name='input'):
        inputs = self.multiple_input(input_param_name)
        dtype = None
        for each in inputs:
            if dtype is None:
                dtype = each.data_type
            elif dtype != each.data_type:
                raise ValueError("Data Type mismatch")
        return dtype

    def create_parameter(self, attr, shape, dtype, suffix='w',
                         initializer=None):
        # Deepcopy the attr so that parameters can be shared in program
        attr_copy = copy.deepcopy(attr)
        if initializer is not None:
            attr_copy['initializer'] = initializer
        if attr_copy['name'] is None:
            attr_copy['name'] = unique_name(".".join([self.name, suffix]))
        self.startup_program.global_block().create_parameter(
            dtype=dtype, shape=shape, **attr_copy)
        return self.main_program.global_block().create_parameter(
            name=attr_copy['name'], dtype=dtype, shape=shape)

    def create_tmp_variable(self, dtype):
        return self.main_program.current_block().create_var(
            name=unique_name(".".join([self.name, 'tmp'])),
            dtype=dtype,
            persistable=False)

    def create_variable(self, *args, **kwargs):
        return self.main_program.current_block().create_var(*args, **kwargs)

    def create_global_variable(self, persistable=False, *args, **kwargs):
        return self.main_program.global_block().create_var(
            *args, persistable=persistable, **kwargs)

    def set_variable_initializer(self, var, initializer):
        assert isinstance(var, Variable)
        self.startup_program.global_block().create_var(
            name=var.name,
            type=var.type,
            dtype=var.data_type,
            shape=var.shape,
            persistable=True,
            initializer=initializer)

    def append_bias_op(self, input_var, num_flatten_dims=None):
        """
        Append bias operator and return its output. If the user does not set 
        bias_attr, append_bias_op will return input_var
         
        :param input_var: the input variable. The len(input_var.shape) is larger
        or equal than 2.
        :param num_flatten_dims: The input tensor will be flatten as a matrix 
        when adding bias.
        `matrix.shape = product(input_var.shape[0:num_flatten_dims]), product(
                input_var.shape[num_flatten_dims:])`
        """
        if num_flatten_dims is None:
            num_flatten_dims = self.kwargs.get('num_flatten_dims', None)
            if num_flatten_dims is None:
                num_flatten_dims = 1

        size = list(input_var.shape[num_flatten_dims:])
        bias_attr = self.bias_attr
        if not bias_attr:
            return input_var

        b = self.create_parameter(
            attr=bias_attr, shape=size, dtype=input_var.data_type, suffix='b')
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
        if isinstance(act, basestring):
            act = {'type': act}
        tmp = self.create_tmp_variable(dtype=input_var.data_type)
        act_type = act.pop('type')
        self.append_op(
            type=act_type,
            inputs={"X": [input_var]},
            outputs={"Y": [tmp]},
            attrs=act)
        return tmp
