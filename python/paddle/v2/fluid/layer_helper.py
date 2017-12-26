import copy
import itertools

from framework import Variable, Parameter, default_main_program, default_startup_program, \
    unique_name, dtype_is_floating
from paddle.v2.fluid.initializer import Constant, Xavier
from param_attr import ParamAttr


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
        return default_main_program()

    @property
    def startup_program(self):
        return default_startup_program()

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
        return ParamAttr.to_attr(self.kwargs.get('param_attr', None))

    @property
    def bias_attr(self):
        return ParamAttr.to_attr(self.kwargs.get('bias_attr', None))

    def multiple_param_attr(self, length):
        param_attr = self.param_attr
        if isinstance(param_attr, ParamAttr):
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
                dtype = each.dtype
            elif dtype != each.dtype:
                raise ValueError("Data Type mismatch")
        return dtype

    def create_parameter(self,
                         attr,
                         shape,
                         dtype,
                         is_bias=False,
                         default_initializer=None):
        # Deepcopy the attr so that parameters can be shared in program
        assert isinstance(attr, ParamAttr)
        suffix = 'b' if is_bias else 'w'

        if default_initializer is None:
            if is_bias:
                attr.set_default_bias_initializer()
            else:
                attr.set_default_param_initializer()
        else:
            attr.set_default_initializer(default_initializer)
        if attr.name is None:
            attr.name = unique_name(".".join([self.name, suffix]))

        self.startup_program.global_block().create_parameter(
            dtype=dtype, shape=shape, **attr.to_kwargs(with_initializer=True))
        return self.main_program.global_block().create_parameter(
            dtype=dtype, shape=shape, **attr.to_kwargs())

    def get_parameter(self, name):
        param = self.main_program.global_block().var(name)
        if not isinstance(param, Parameter):
            raise ValueError("no Parameter name %s found" % name)
        return param

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
            dtype=var.dtype,
            shape=var.shape,
            persistable=True,
            initializer=initializer)

    def append_bias_op(self, input_var, dim_start=1, dim_end=None):
        """
        Append bias operator and return its output. If the user does not set
        bias_attr, append_bias_op will return input_var

        :param input_var: the input variable. The len(input_var.shape) is
        larger or equal than 2.
        :bias_initializer: an instance of a subclass of Initializer used to
        initialize the bias
        :param dim_start:
        :param dim_end: the shape of the bias will be
        input_var.shape[dim_start:dim_end]. The bias is broadcasted to other
        dimensions and added to input_var to get the output
        """
        size = list(input_var.shape[dim_start:dim_end])
        bias_attr = self.bias_attr
        if not bias_attr:
            return input_var

        b = self.create_parameter(
            attr=bias_attr, shape=size, dtype=input_var.dtype, is_bias=True)
        tmp = self.create_tmp_variable(dtype=input_var.dtype)
        self.append_op(
            type='elementwise_add',
            inputs={'X': [input_var],
                    'Y': [b]},
            outputs={'Out': [tmp]},
            attrs={'axis': dim_start})
        return tmp

    def append_activation(self, input_var):
        act = self.kwargs.get('act', None)
        if act is None:
            return input_var
        if isinstance(act, basestring):
            act = {'type': act}
        tmp = self.create_tmp_variable(dtype=input_var.dtype)
        act_type = act.pop('type')
        self.append_op(
            type=act_type,
            inputs={"X": [input_var]},
            outputs={"Y": [tmp]},
            attrs=act)
        return tmp

    def _get_default_initializer(self, dtype):
        if dtype is None or dtype_is_floating(dtype) is True:
            return Xavier()
        else:
            # For integer and boolean types, initialize with all zeros
            return Constant()

    def is_instance(self, param_name, cls):
        param = self.kwargs.get(param_name, None)
        if not isinstance(param, cls):
            raise TypeError("The input {0} parameter of method {1} must be {2}",
                            param_name, self.layer_type, cls.__name__)
