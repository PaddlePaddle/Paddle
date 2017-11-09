import paddle.v2.framework.core as core
from paddle.v2.framework.framework import OpProtoHolder, Variable, Program, \
    Operator
from paddle.v2.framework.initializer import ConstantInitializer, \
    NormalInitializer
from paddle.v2.framework.layer_helper import LayerHelper, unique_name
import re

__all__ = [
    'fc', 'data', 'cross_entropy', 'conv2d', 'pool2d', 'embedding', 'concat',
    'StaticRNN', 'cast', 'sequence_conv', 'sequence_pool', 'sums', 'cos_sim',
    'batch_norm', 'accuracy'
]


def fc(input,
       size,
       param_attr=None,
       bias_attr=True,
       name=None,
       act=None,
       num_flatten_dims=1,
       main_program=None,
       startup_program=None):
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
              is_sparse=False,
              param_attr=None,
              main_program=None,
              startup_program=None):
    helper = LayerHelper('embedding', **locals())
    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=data_type)
    tmp = helper.create_tmp_variable(data_type)
    helper.append_op(
        type='lookup_table',
        inputs={'Ids': input,
                'W': w},
        outputs={'Out': tmp},
        attrs={'is_sparse': is_sparse})
    return tmp


def data(name,
         shape,
         data_type='float32',
         type=core.VarDesc.VarType.LOD_TENSOR,
         append_batch_size=True,
         main_program=None,
         startup_program=None,
         stop_gradient=True):
    helper = LayerHelper('data', **locals())
    shape = list(shape)
    for i in xrange(len(shape)):
        if shape[i] is None:
            shape[i] = -1
            append_batch_size = False
        elif shape[i] < 0:
            append_batch_size = False

    if append_batch_size:
        shape = [-1] + shape  # append batch size as -1

    return helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=data_type,
        type=type,
        stop_gradient=stop_gradient)


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

    def infer_and_check_data_type(op_proto, **kwargs):
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

        return dtype

    def func(**kwargs):
        helper = LayerHelper(op_type, **kwargs)

        dtype = infer_and_check_data_type(op_proto, **kwargs)

        inputs = dict()
        for ipt in op_proto.inputs:
            name = _convert_(ipt.name)
            val = kwargs.pop(name, [])
            if not isinstance(val, list) and not isinstance(val, tuple):
                val = [val]
            inputs[ipt.name] = val

        outputs = dict()
        out = helper.create_tmp_variable(dtype=dtype)
        outputs[o_name] = [out]
        for name in intermediate_output_names:
            outputs[name] = [helper.create_tmp_variable(dtype=dtype)]
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=kwargs)
        return helper.append_activation(out)

    func.__name__ = op_type
    globals()[op_type] = func
    global __all__
    __all__.append(op_type)


_create_op_func_('mean')
_create_op_func_('mul')
_create_op_func_('elementwise_add')
_create_op_func_('dropout')
_create_op_func_('reshape')
_create_op_func_('elementwise_add')
_create_op_func_('sigmoid')
_create_op_func_('scale')
_create_op_func_('reshape')
_create_op_func_('transpose')


def fill_constant(data_type, shape, value=None, program=None):
    helper = LayerHelper('fill_constant', **locals())
    out = helper.create_tmp_variable(dtype=data_type)
    helper.append_op(
        type='fill_constant',
        outputs={'Out': [out]},
        attrs={'data_type': data_type,
               'shape': shape,
               'value': value})
    return out


def cast(x, data_type, main_program=None):
    helper = LayerHelper('cast', **locals())
    out = helper.create_tmp_variable(dtype=data_type)
    helper.append_op(
        type='cast',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'in_data_type': x.data_type,
               'out_data_type': out.data_type})
    return out


def concat(input, axis, main_program=None, startup_program=None):
    helper = LayerHelper('concat', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='concat',
        inputs={'X': input},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    return out


def sums(input, main_program=None, startup_program=None):
    helper = LayerHelper('sum', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(type='sum', inputs={'X': input}, outputs={'Out': out})
    return out


def cos_sim(X, Y, **kwargs):
    helper = LayerHelper('cos_sim', **kwargs)
    out = helper.create_tmp_variable(dtype=X.data_type)
    xnorm = helper.create_tmp_variable(dtype=X.data_type)
    ynorm = helper.create_tmp_variable(dtype=X.data_type)
    helper.append_op(
        type='cos_sim',
        inputs={'X': [X],
                'Y': [Y]},
        outputs={'Out': [out],
                 'XNorm': [xnorm],
                 'YNorm': [ynorm]})
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
        type='square', inputs={'X': [minus_out]}, outputs={'Y': [square_out]})
    return square_out


def accuracy(input, label, k=1, **kwargs):
    helper = LayerHelper("accuracy", **kwargs)
    topk_out = helper.create_tmp_variable(dtype=input.data_type)
    topk_indices = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="top_k",
        inputs={"X": [input]},
        outputs={"Out": [topk_out],
                 "Indices": [topk_indices]},
        attrs={"k": k})
    acc_out_dtype = kwargs.get("out_dtype", "float32")
    acc_out = helper.create_tmp_variable(dtype=acc_out_dtype)
    helper.append_op(
        type="accuracy",
        inputs={
            "Out": [topk_out],
            "Indices": [topk_indices],
            "Label": [label]
        },
        outputs={"Accuracy": [acc_out]})
    return acc_out


def sequence_conv(input,
                  num_filters,
                  filter_size=3,
                  filter_stride=1,
                  act=None,
                  padding=None,
                  bias_attr=None,
                  param_attr=None,
                  main_program=None,
                  startup_program=None):
    # FIXME(dzh) : want to unify the argument of python layer
    # function. So we ignore some unecessary attributes.
    # such as, padding_trainable, context_start.

    helper = LayerHelper('sequence_conv', **locals())
    dtype = helper.input_dtype()

    filter_shape = [filter_size * input.shape[1], num_filters]
    filter = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype)
    pre_bias = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='sequence_conv',
        inputs={
            'X': [input],
            'Filter': [filter],
        },
        outputs={"Out": pre_bias},
        attrs={
            'contextStride': filter_stride,
            'contextStart': -int(filter_size / 2),
            'contextLength': filter_size
        })
    pre_act = helper.append_bias_op(pre_bias)
    return helper.append_activation(pre_act)


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
           main_program=None,
           startup_program=None):
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

    std = (2.0 / (filter_size[0]**2 * num_channels))**0.5
    filter = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        initializer=NormalInitializer(0.0, std, 0))
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

    pre_act = helper.append_bias_op(pre_bias, 1)

    return helper.append_activation(pre_act)


def sequence_pool(input, pool_type, **kwargs):
    helper = LayerHelper('sequence_pool', input=input, **kwargs)
    dtype = helper.input_dtype()
    pool_out = helper.create_tmp_variable(dtype)
    max_index = helper.create_tmp_variable(dtype)

    helper.append_op(
        type="sequence_pool",
        inputs={"X": input},
        outputs={"Out": pool_out,
                 "MaxIndex": max_index},
        attrs={"pooltype": pool_type.upper()})

    return pool_out


def pool2d(input,
           pool_size,
           pool_type,
           pool_stride=[1, 1],
           pool_padding=[0, 0],
           global_pooling=False,
           main_program=None,
           startup_program=None):
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

    helper = LayerHelper('pool2d', **locals())
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


def batch_norm(input,
               act=None,
               is_test=False,
               momentum=0.9,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               data_layout='NCHW',
               main_program=None,
               startup_program=None):
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

    param_shape = [channel_num]

    # create parameter
    scale = helper.create_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype=dtype,
        initializer=ConstantInitializer(1.0))
    bias = helper.create_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype=dtype,
        initializer=ConstantInitializer(0.0))

    mean = helper.create_global_variable(
        dtype=input.data_type, shape=param_shape, persistable=True)
    helper.set_variable_initializer(
        var=mean, initializer=ConstantInitializer(0.0))

    variance = helper.create_global_variable(
        dtype=input.data_type, shape=param_shape, persistable=True)
    helper.set_variable_initializer(
        var=variance, initializer=ConstantInitializer(1.0))

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

    def __init__(self, main_program):
        if not isinstance(main_program, Program):
            raise TypeError("BlockGuard takes a program")
        self.main_program = main_program

    def __enter__(self):
        self.main_program.create_block()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.main_program.rollback()
        if exc_type is not None:
            return False  # re-raise exception
        return True


class StaticRNNGuard(BlockGuard):
    def __init__(self, rnn):
        if not isinstance(rnn, StaticRNN):
            raise TypeError("StaticRNNGuard takes an StaticRNN")
        super(StaticRNNGuard, self).__init__(rnn.helper.main_program)
        self.rnn = rnn

    def __enter__(self):
        self.rnn.status = StaticRNN.IN_RNN_BLOCK
        return super(StaticRNNGuard, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
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

    def __init__(self, name=None, main_program=None):
        self.helper = LayerHelper(
            "static_rnn", name=name, main_program=main_program)
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

    def memory(self,
               init=None,
               shape=None,
               batch_ref=None,
               init_value=0.0,
               init_batch_dim_idx=0,
               ref_batch_dim_idx=1):
        '''
        :param init: boot memory, if not set, a shape, batch_ref must be provided
        :param shape: shape of the boot memory
        :param batch_ref: batch size reference variable
        :param init_value: the init value of boot memory
        :param init_batch_dim_idx: the index of batch size in init's dimension
        :param ref_batch_dim_idx: the index of batch size in batch_ref's dimension
        :return: boot memory
        '''
        self._assert_in_rnn_block_('memory')
        if init is None:
            if shape is None or batch_ref is None:
                raise ValueError(
                    "if init is None, memory at least need shape and batch_ref")
            parent_block = self.parent_block()
            var_name = unique_name("@".join([self.helper.name, "memory_boot"]))
            boot_var = parent_block.create_var(
                name=var_name,
                shape=shape,
                dtype=batch_ref.data_type,
                persistable=False)

            parent_block.append_op(
                type="fill_constant_batch_size_like",
                inputs={'Input': [batch_ref]},
                outputs={'Out': [boot_var]},
                attrs={
                    'value': init_value,
                    'shape': boot_var.shape,
                    'data_type': boot_var.data_type,
                    'input_dim_idx': ref_batch_dim_idx,
                    'output_dim_idx': init_batch_dim_idx
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
            self.seq_len = x.shape[0]
        elif self.seq_len != x.shape[0]:
            raise ValueError("Static RNN only take fix seq_len input")

        ipt = self.helper.create_variable(
            name=x.name,
            dtype=x.data_type,
            shape=list(x.shape[1:]),
            type=x.type)
        self.inputs.append(ipt)
        return ipt

    def step_output(self, o):
        self._assert_in_rnn_block_('step_output')
        if not isinstance(o, Variable):
            raise TypeError("step output takes a Variable")

        tmp_o = self.helper.create_tmp_variable(dtype=o.data_type)
        self.helper.append_op(
            type='rnn_memory_helper',
            inputs={'X': [o]},
            outputs={'Out': tmp_o},
            attrs={'data_type': o.data_type})

        out_var = self.parent_block().create_var(
            name=tmp_o.name,
            shape=[self.seq_len] + list(tmp_o.shape),
            dtype=tmp_o.data_type)

        self.outputs.append(out_var)

    def output(self, *outputs):
        for each in outputs:
            self.step_output(each)

    def update_memory(self, mem, var):
        if not isinstance(mem, Variable) or not isinstance(var, Variable):
            raise TypeError("update memory should take variables")
        self.memories[mem.name].mem = var

    def parent_block(self):
        prog = self.helper.main_program
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
        main_program = self.helper.main_program
        rnn_block = main_program.current_block()
        parent_block = self.parent_block()

        local_inputs = set()

        for op in rnn_block.ops:
            assert isinstance(op, Operator)
            for oname in op.output_names:
                for out_var_name in op.output(oname):
                    local_inputs.add(out_var_name)

        for var in self.inputs:
            local_inputs.add(var.name)
        for m in self.memories:
            local_inputs.add(m)

        params = list()
        for op in rnn_block.ops:
            assert isinstance(op, Operator)
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    if in_var_name not in local_inputs:
                        params.append(in_var_name)

        parameters = [parent_block.var(name) for name in params]

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)

        inlinks = [parent_block.var(i.name) for i in self.inputs]
        outlinks = self.outputs

        boot_memories = []
        pre_memories = []
        memories = []
        for _, mem in self.memories.iteritems():
            boot_memories.append(mem.init)
            pre_memories.append(mem.pre_mem.name)
            mem_var = rnn_block.var(mem.mem.name)
            assert isinstance(mem_var, Variable)
            new_mem = self.helper.create_tmp_variable(dtype=mem_var.data_type)

            rnn_block.append_op(
                type='rnn_memory_helper',
                inputs={'X': [mem_var]},
                outputs={'Out': [new_mem]},
                attrs={'data_type': mem_var.data_type})

            memories.append(new_mem.name)

        parent_block.append_op(
            type='recurrent',
            inputs={
                'inputs': inlinks,
                'initial_states': boot_memories,
                'parameters': parameters
            },
            outputs={'outputs': outlinks,
                     'step_scopes': [step_scope]},
            attrs={
                'ex_states': pre_memories,
                'states': memories,
                'step_block': rnn_block
            })


def lstm(x,
         c_pre_init,
         hidden_dim,
         forget_bias=None,
         main_program=None,
         startup_program=None):
    helper = LayerHelper('lstm_unit', **locals())
    rnn = StaticRNN()
    with rnn.step():
        c_pre = rnn.memory(init=c_pre_init)
        x_t = rnn.step_input(x)

        before_fc = concat(
            input=[x_t, c_pre],
            axis=1,
            main_program=main_program,
            startup_program=startup_program)
        after_fc = fc(input=before_fc,
                      size=hidden_dim * 4,
                      main_program=main_program,
                      startup_program=startup_program)

        data_type = x.data_type
        c = helper.create_tmp_variable(data_type)
        h = helper.create_tmp_variable(data_type)

        helper.append_op(
            type='lstm_unit',
            inputs={"X": after_fc,
                    "C_prev": c_pre},
            outputs={"C": c,
                     "H": h},
            attrs={"forget_bias": forget_bias})

        rnn.update_memory(c_pre, c)
        rnn.output(h)

    return rnn()


def lod_rank_table(x, level=0, main_program=None):
    helper = LayerHelper("lod_rank_table", **locals())
    table = helper.create_variable(
        type=core.VarDesc.VarType.LOD_RANK_TABLE,
        name=unique_name("lod_rank_table"))
    helper.append_op(
        type='lod_rank_table',
        inputs={'X': x},
        outputs={'Out': table},
        attrs={'level': level})
    return table


def lod_tensor_to_array(x, table, main_program=None):
    helper = LayerHelper("lod_tensor_to_array", **locals())
    array = helper.create_variable(
        name=unique_name("lod_tensor_to_array"),
        type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
        dtype=x.data_type)
    helper.append_op(
        type='lod_tensor_to_array',
        inputs={'X': x,
                'RankTable': table},
        outputs={'Out': array})
    return array


def array_to_lod_tensor(x, table, main_program=None):
    helper = LayerHelper("array_to_lod_tensor", **locals())
    tmp = helper.create_tmp_variable(dtype=x.data_type)
    helper.append_op(
        type="array_to_lod_tensor",
        inputs={'X': x,
                'RankTable': table},
        outputs={'Out': tmp})
    return tmp


def fill_constant(shape, dtype, value, main_program=None):
    helper = LayerHelper("ones", **locals())
    out = helper.create_tmp_variable(dtype=dtype)
    helper.append_op(
        type='fill_constant',
        inputs={},
        outputs={'Out': [out]},
        attrs={
            'shape': shape,
            'data_type': out.data_type,
            'value': float(value)
        })
    out.stop_gradient = True
    return out


def ones(shape, dtype, main_program=None):
    return fill_constant(value=1.0, **locals())


def zeros(shape, dtype, main_program=None):
    return fill_constant(value=0.0, **locals())


def increment(x, value=1.0, main_program=None):
    helper = LayerHelper("increment", **locals())
    out = helper.create_tmp_variable(dtype=x.data_type)
    helper.append_op(
        type='increment',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'step': value})
    return out


def array_write(x, i, array=None, main_program=None):
    helper = LayerHelper('array_write', **locals())
    if array is None:
        array = helper.create_variable(
            name="{0}.out".format(helper.name),
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
            dtype=x.data_type)
    helper.append_op(
        type='write_to_array',
        inputs={'X': [x],
                'I': [i]},
        outputs={'Out': [array]})
    return array


def array_read(array, i, main_program=None):
    helper = LayerHelper('array_read', **locals())
    if not isinstance(
            array,
            Variable) or array.type != core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        raise TypeError("array should be tensor array vairable")
    out = helper.create_tmp_variable(dtype=array.data_type)
    helper.append_op(
        type='read_from_array',
        inputs={'X': [array],
                'I': [i]},
        outputs={'Out': [out]})
    return out


def shrink_memory(x, i, table, main_program=None):
    helper = LayerHelper('shrink_memory', **locals())
    out = helper.create_tmp_variable(dtype=x.data_type)
    helper.append_op(
        type='shrink_rnn_memory',
        inputs={'X': [x],
                'I': [i],
                'RankTable': [table]},
        outputs={'Out': [out]},
        attrs={})
    return out
