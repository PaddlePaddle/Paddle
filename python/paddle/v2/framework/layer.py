import itertools

from paddle.v2.framework.activation import Activation
from paddle.v2.framework.graph import g_program, Variable
from paddle.v2.framework.unique_name import unique_name

__all__ = []


def export(export_name):
    if isinstance(export_name, basestring):
        export_name = [export_name]

    def __wrap__(cls):
        def func(*args, **kwargs):
            layer = cls(*args, **kwargs)
            return layer()

        global __all__
        for name in export_name:
            globals()[name] = func
            __all__.append(name)

    return __wrap__


class Layer(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self):
        raise NotImplementedError()

    def program(self):
        return self.kwargs.get('program', g_program)

    def multiple_input(self, input_name='input'):
        ipt = self.kwargs.get(input_name, None)
        if ipt is None:
            return []
        elif isinstance(ipt, Variable):
            return [ipt]
        elif isinstance(ipt, list) or isinstance(ipt, tuple):
            for each in ipt:
                if not isinstance(each, Variable):
                    raise ValueError("each input must be Variable")
            return ipt

    @property
    def class_name(self):
        return self.__class__.__name__

    def input(self, input_name='input'):
        inputs = self.multiple_input(input_name)
        if len(inputs) != 1:
            raise ValueError("Input {0} should be only one for {2}, not {3}",
                             input_name, self.class_name, len(inputs))
        return inputs[0]

    def param_attr(self, param_attr_name='param_attr'):
        return self.kwargs.get(param_attr_name, {
            'name': None,
            'init_attr': {
                'type': 'uniform_random'
            }
        })

    def iter_multiple_input_and_param(self,
                                      input_name='input',
                                      param_attr_name='param_attr'):
        inputs = self.multiple_input(input_name)
        attrs = self.param_attr(param_attr_name)
        if isinstance(attrs, dict):
            attrs = [attrs] * len(inputs)
        elif isinstance(attrs, list) or isinstance(attrs, tuple):
            if len(inputs) != len(attrs):
                raise ValueError(
                    "Size of input {0} and parameter attribute {1} are not "
                    "same in layer {2}.".format(input_name, param_attr_name,
                                                self.class_name))
        else:
            raise ValueError("Unsupported parameter attribute type for layer "
                             "{0}".format(self.class_name))

        for ipt, attr in itertools.izip(inputs, attrs):
            yield ipt, attr

    def create_parameter(self, shape, attr):
        if attr.name is None:
            attr.name = unique_name("_".join([self.name, 'w']))

        block = self.program().global_block()
        dtype = attr.pop('dtype', None)
        if dtype is None:
            raise ValueError("Must specific dtype for parameter")
        return block.create_parameter(
            name=attr.name,
            dtype=dtype,
            shape=shape,
            initialize_attr=attr['init_attr'],
            optimize_attr=attr['optimize_attr'])

    @property
    def name(self):
        name = self.kwargs.get('name', None)
        if name is None:
            name = unique_name(self.class_name)
            self.kwargs['name'] = name
        return name

    @property
    def size(self):
        sz = self.kwargs['size']
        if not isinstance(sz, int):
            raise ValueError("layer %s size should be an integer".format(
                self.name))
        return sz

    def create_var(self, *args, **kwargs):
        return self.program().current_block().create_var(*args, **kwargs)

    def create_tmp_var(self, dtype='float32'):
        tmp_var_name = unique_name("_".join([self.name, 'tmp']))
        return self.create_var(name=tmp_var_name, dtype=dtype)

    def infer_dtype(self, *args):
        dtype = None
        for each in args:
            assert isinstance(each, Variable)
            if dtype is None:
                dtype = each.data_type
            elif dtype != each.data_type:
                raise ValueError("{0} input contain many kinds of data type".
                                 format(self.name))
        return dtype

    def append_op(self, *args, **kwargs):
        self.program().current_block().append_op(*args, **kwargs)

    def append_activation(self, input):
        act = self.kwargs.get('act', None)
        if act is None:
            return input
        elif not isinstance(act, Activation):
            raise TypeError("act should be ActivationBase or None")
        else:
            assert isinstance(act, Activation)
            out = self.create_tmp_var(dtype=self.infer_dtype(input))
            self.append_op(
                type=act.type,
                attrs=act.attr,
                inputs={'X': [input]},
                outputs={'Out': [out]})
            return out


@export("data_layer")
class DataLayer(Layer):
    def __init__(self, dtype, shape, **kwargs):
        super(DataLayer, self).__init__(**kwargs)
        self.dtype = dtype
        self.shape = [-1] + list(shape)

    @property
    def name(self):
        return self.kwargs['name']

    def __call__(self):
        global_block = self.program().global_block()
        return global_block.create_var(
            name=self.name, dtype=self.dtype, shape=self.shape)


@export("fc_layer")
class FCLayer(Layer):
    def __init__(self, num_flatten_dims=1, **kwargs):
        super(FCLayer, self).__init__(**kwargs)
        self.num_flatten_dims = num_flatten_dims

    def __call__(self, *args, **kwargs):
        mul_results = []
        for ipt, attr in self.iter_multiple_input_and_param():
            assert isinstance(ipt, Variable)
            shape = ipt.shape
            param_shape = shape[self.num_flatten_dims:] + [self.size]
            if 'dtype' not in attr:
                attr['dtype'] = self.infer_dtype(ipt)
            w = self.create_parameter(param_shape, attr)
            tmp_out = self.create_tmp_var(dtype=self.infer_dtype(ipt, w))
            self.append_op(
                type='mul',
                inputs={'X': [ipt],
                        'Y': [w]},
                outputs={"Out": [tmp_out]},
                attrs={'x_num_col_dims': self.num_flatten_dims})
            mul_results.append(tmp_out)

        if len(mul_results) == 0:
            raise ValueError("FC take one input at least")
        elif len(mul_results) == 1:
            sum_result = mul_results[0]
        else:
            sum_result = self.create_tmp_var(
                dtype=self.infer_dtype(*mul_results))
            self.append_op(
                type='sum',
                inputs={'X': mul_results},
                outputs={'Out': [sum_result]})

        return self.append_activation(sum_result)
