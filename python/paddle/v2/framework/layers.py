from paddle.v2.framework.layer_helper import LayerHelper, unique_name
import paddle.v2.framework.core as core
from paddle.v2.framework.framework import OpProtoHolder, Variable, Program
import re

__all__ = [
    'fc', 'data', 'cross_entropy', 'conv2d', 'pool2d', 'embedding', 'concat',
    'StaticRNN'
]


def fc(input,
       size,
       param_attr=None,
       bias_attr=True,
       name=None,
       act=None,
       num_flatten_dims=1,
       program=None,
       init_program=None):
    # create helper
    helper = LayerHelper('fc', **locals())

    dtype = helper.input_dtype()

    # mul
    mul_results = []
    for input_var, param_attr in helper.iter_inputs_and_params():
        input_shape = input_var.shape
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[num_flatten_dims:], 1)
        ] + [size]

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
            attrs={'x_num_col_dims': num_flatten_dims,
                   'y_num_col_dims': 1})
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


def embedding(input,
              size,
              data_type='float32',
              param_attr=None,
              program=None,
              init_program=None):
    helper = LayerHelper('embedding', **locals())
    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=data_type)
    tmp = helper.create_tmp_variable(data_type)
    helper.append_op(
        type='lookup_table',
        inputs={'Ids': input,
                'W': w},
        outputs={'Out': tmp})
    return tmp


def data(name,
         shape,
         data_type='float32',
         type=core.VarDesc.VarType.LOD_TENSOR,
         append_batch_size=True,
         program=None,
         init_program=None):
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
    not_intermediate_outputs = \
        filter(lambda output: not output.intermediate, op_proto.outputs)
    intermediate_outputs = \
        filter(lambda output: output.intermediate, op_proto.outputs)

    if len(not_intermediate_outputs) != 1:
        raise ValueError(
            "Only one not intermediate output operator can be automatically generated"
        )

    if not_intermediate_outputs[0].duplicable:
        raise ValueError(
            "Only not duplicable op can be automatically generated")

    for output in intermediate_outputs:
        if output.duplicable:
            raise ValueError(
                "Only when all intermediate ops are not duplicable, "
                "this op can be automatically generated")

    o_name = not_intermediate_outputs[0].name
    intermediate_output_names = [output.name for output in intermediate_outputs]

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

        outputs = dict()
        out = helper.create_tmp_variable(dtype=dtype)
        outputs[o_name] = [out]
        for name in intermediate_output_names:
            outputs[name] = [helper.create_tmp_variable(dtype=dtype)]
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=kwargs)
        return out

    func.__name__ = op_type
    globals()[op_type] = func
    global __all__
    __all__.append(op_type)


_create_op_func_('mean')
_create_op_func_('mul')
_create_op_func_('dropout')
_create_op_func_('reshape')


def concat(input, axis, program=None, init_program=None):
    helper = LayerHelper('concat', **locals())
    if not isinstance(input, list) and not isinstance(input, tuple):
        input = [input]
    out = helper.create_tmp_variable(dtype=input[0].data_type)
    helper.append_op(
        type='concat',
        inputs={'X': input},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    return out


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
           program=None,
           init_program=None):
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
           program=None,
           init_program=None):
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
            "poolingType": pool_type,
            "ksize": pool_size,
            "globalPooling": global_pooling,
            "strides": pool_stride,
            "paddings": pool_padding
        })

    return pool_out


def batch_norm(input,
               act=None,
               is_test=False,
               momentum=0.9,
               epsilon=1e05,
               param_attr=None,
               bias_attr=None,
               data_layout='NCHW',
               program=None,
               init_program=None):
    helper = LayerHelper('batch_norm', **locals())
    dtype = helper.input_dtype()

    input_shape = input.shape
    if data_layout == 'NCHW':
        channel_num = input_shape[1]
    else:
        if data_layout == 'NHWC':
            channel_num = input_shape[-1]
        else:
            raise ValueError("unsupported data layout:" + data_layout)

    def get_init_attr(value):
        if not isinstance(value, float):
            raise ValueError("attr value should be a float")
        return {'type': 'fill_constant', 'value': value}

    def prepend_init_op(var, init_attr):
        assert isinstance(var, Variable)
        op_type = init_attr['type']
        init_attr['shape'] = var.shape
        init_attr['data_type'] = int(var.data_type)
        op = var.block.prepend_op(
            type=op_type, inputs=None, outputs={'Out': [var]}, attrs=init_attr)
        return op

    def create_persistable_var(dtype, shape, init_attr=None):
        name = unique_name(".".join([helper.name, "xxxx"]))
        var = init_program.global_block().create_var(
            dtype=dtype, shape=shape, name=name, persistable=True)
        if 'init_attr' is not None:
            prepend_init_op(var, init_attr)
        return program.global_block().create_var(
            name=name, dtype=dtype, shape=shape, persistable=True)

    param_shape = [channel_num]

    # create parameter
    scale = helper.create_parameter(
        attr=helper.param_attr, shape=param_shape, dtype=dtype)
    bias = helper.create_parameter(
        attr=helper.param_attr, shape=param_shape, dtype=dtype)

    # create input
    mean = create_persistable_var(dtype, param_shape, get_init_attr(0.0))
    variance = create_persistable_var(dtype, param_shape, get_init_attr(1.0))

    # create output
    # mean and mean_out share the same memory
    mean_out = mean
    # variance and variance out share the same memory
    variance_out = variance
    saved_mean = helper.create_tmp_variable(dtype)
    saved_variance = helper.create_tmp_variable(dtype)

    batch_norm_out = helper.create_tmp_variable(dtype)

    helper.append_op(
        type="batch_norm",
        inputs={
            "X": input,
            "Scale": scale,
            "Bias": bias,
            "Mean": mean,
            "Variance": variance
        },
        outputs={
            "Y": batch_norm_out,
            "MeanOut": mean_out,
            "VarianceOut": variance_out,
            "SavedMean": saved_mean,
            "SavedVariance": saved_variance
        },
        attrs={"momentum": momentum,
               "epsilon": epsilon,
               "is_test": is_test})

    return helper.append_activation(batch_norm_out)


class BlockGuard(object):
    """
    BlockGuard used to create sub-block in program by using Python `with` 
    keyword.
    """

    def __init__(self, program):
        if not isinstance(program, Program):
            raise TypeError("BlockGuard takes a program")
        self.program = program

    def __enter__(self):
        self.program.create_block()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.program.rollback()
        if exc_type is not None:
            return False  # re-raise exception
        return True


class StaticRNNGuard(BlockGuard):
    def __init__(self, rnn):
        if not isinstance(rnn, StaticRNN):
            raise TypeError("StaticRNNGuard takes an StaticRNN")
        super(StaticRNNGuard, self).__init__(rnn.helper.program)
        self.rnn = rnn

    def __enter__(self):
        self.rnn.status = StaticRNN.IN_RNN_BLOCK
        return super(StaticRNNGuard, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rnn.status = StaticRNN.AFTER_RNN_BLOCK
        self.rnn.complete_rnn_op()
        return super(StaticRNNGuard, self).__exit__(exc_type, exc_val, exc_tb)


class StaticRNNMemoryLink(object):
    """
    :param init: the initial variable for Memory
    :type init: Variable
    :param pre_mem: the memory variable in previous time step
    :type pre_mem: Variable
    :param mem: the memory variable in current time step
    :type mem: Variable
    """

    def __init__(self, init, pre_mem, mem=None):
        self.init = init
        self.pre_mem = pre_mem
        self.mem = mem


class StaticRNN(object):
    BEFORE_RNN_BLOCK = 0
    IN_RNN_BLOCK = 1
    AFTER_RNN_BLOCK = 2

    def __init__(self, name=None, program=None):
        self.helper = LayerHelper("static_rnn", name=name, program=program)
        self.memories = {}  # memory map, from pre_mem.name --> MemoryLink
        self.inputs = []  # input variable list in current block
        self.outputs = []  # output variable list in parent block
        self.status = StaticRNN.BEFORE_RNN_BLOCK  # status flag.
        # sequence length, since it is a static RNN, sequence length are fixed.
        self.seq_len = None

    def step(self):
        return StaticRNNGuard(self)

    def _assert_in_rnn_block_(self, method):
        if self.status != StaticRNN.IN_RNN_BLOCK:
            raise ValueError("You must invoke {0} in rnn block".format(method))

    def memory(self, init=None, shape=None, dtype=None, init_value=0):
        self._assert_in_rnn_block_('memory')
        if init is None:
            if shape is None or dtype is None:
                raise ValueError(
                    "if init is None, memory at least need shape and dtype")
            parent_block = self.parent_block()
            var_name = unique_name("@".join([self.helper.name, "memory_boot"]))
            boot_var = parent_block.create_var(
                name=var_name, shape=shape, dtype=dtype, persistable=False)

            parent_block.append_op(
                type="fill_constant",
                inputs={},
                outputs={'Out': [boot_var]},
                attrs={
                    'value': init_value,
                    'shape': boot_var.shape,
                    'data_type': boot_var.data_type
                })

            return self.memory(init=boot_var)
        else:
            pre_mem = self.helper.create_variable(
                name=unique_name("@".join([self.helper.name, "mem"])),
                dtype=init.data_type,
                shape=init.shape)
            self.memories[pre_mem.name] = StaticRNNMemoryLink(
                init=init, pre_mem=pre_mem)
            return pre_mem

    def step_input(self, x):
        self._assert_in_rnn_block_('step_input')
        if not isinstance(x, Variable):
            raise TypeError("step input takes a Variable")
        if self.seq_len is None:
            self.seq_len = x.shape[1]
        elif self.seq_len != x.shape[1]:
            raise ValueError("Static RNN only take fix seq_len input")

        ipt = self.helper.create_variable(
            name=x.name,
            dtype=x.data_type,
            shape=[-1] + list(x.shape[2:]),
            type=x.type)
        self.inputs.append(ipt)
        return ipt

    def step_output(self, o):
        self._assert_in_rnn_block_('step_output')
        if not isinstance(o, Variable):
            raise TypeError("step output takes a Variable")

        out_var = self.parent_block().create_var(
            name=o.name,
            shape=[-1, self.seq_len] + list(o.shape[1:]),
            dtype=o.data_type)

        self.outputs.append(out_var)

    def output(self, *outputs):
        for each in outputs:
            self.step_output(each)

    def update_memory(self, mem, var):
        if not isinstance(mem, Variable) or not isinstance(var, Variable):
            raise TypeError("update memory should take variables")
        self.memories[mem.name].mem = var

    def parent_block(self):
        prog = self.helper.program
        parent_idx = prog.current_block().parent_idx
        assert parent_idx >= 0
        parent_block = prog.block(parent_idx)
        return parent_block

    def __call__(self, *args, **kwargs):
        if self.status != StaticRNN.AFTER_RNN_BLOCK:
            raise ValueError("RNN output can only be retrieved after rnn block")
        if len(self.outputs) == 0:
            raise ValueError("RNN has no output")
        elif len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return self.outputs

    def complete_rnn_op(self):
        # TODO(yuyang18): Create RNN Op here.
        # Implement this method after RNN op complete.
        pass
