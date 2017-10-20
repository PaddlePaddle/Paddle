from paddle.v2.framework.framework import Variable, OpProtoHolder, g_program, \
    Program
import paddle.v2.framework.core as core
import copy
import itertools


def unique_name(prefix):
    uid = core.unique_integer()  # unique during whole process.
    return "_".join([prefix, str(uid)])


class BlockGuard(object):
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


class RNNGuard(BlockGuard):
    def __init__(self, rnn):
        if not isinstance(rnn, StaticRNNHelper):
            raise TypeError("RNNGuard takes an RNNHelper")
        super(RNNGuard, self).__init__(rnn.helper.program)
        self.rnn = rnn

    def __enter__(self):
        self.rnn.status = StaticRNNHelper.IN_RNN_BLOCK
        return super(RNNGuard, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rnn.status = StaticRNNHelper.AFTER_RNN_BLOCK
        return super(RNNGuard, self).__exit__(exc_type, exc_val, exc_tb)


class RNNMemoryLink(object):
    def __init__(self, init, pre_mem, mem=None):
        self.init = init
        self.pre_mem = pre_mem
        self.mem = None


class StaticRNNHelper(object):
    BEFORE_RNN_BLOCK = 0
    IN_RNN_BLOCK = 1
    AFTER_RNN_BLOCK = 2

    def __init__(self, name=None, program=None):
        self.helper = LayerHelper(**locals())
        self.memories = {}
        self.inputs = []
        self.outputs = []
        self.status = StaticRNNHelper.BEFORE_RNN_BLOCK
        self.seq_len = None

    def step(self):
        return RNNGuard(self)

    def _assert_in_rnn_block_(self, method):
        if not self.status != StaticRNNHelper.IN_RNN_BLOCK:
            raise ValueError("You must invoke {0} in rnn block".format(method))

    def memory(self, init=None, shape=None, dtype=None, value=0):
        self._assert_in_rnn_block_('memory')
        if init is None:
            if shape is None or dtype is None:
                raise ValueError(
                    "if init is None, memory at least need shape and dtype")

            prog = self.helper.program
            parent_idx = prog.current_block().parent_idx
            assert parent_idx >= 0
            parent_block = prog.block(parent_idx)
            var_name = unique_name("@".join([self.helper.name, "memory_boot"]))
            boot_var = parent_block.create_var(
                name=var_name, shape=shape, dtype=dtype, persistable=False)

            parent_block.append_op(
                type="fill_constant",
                inputs={},
                outputs={'Out': [boot_var]},
                attrs={
                    'value': value,
                    'shape': boot_var.shape,
                    'data_type': boot_var.data_type
                })

            return self.memory(init=boot_var)
        else:
            pre_mem = self.helper.create_variable(
                name=unique_name("@".join([self.helper.name, "mem"])),
                dtype=init.data_type,
                shape=init.shape)
            self.memories[pre_mem.name] = RNNMemoryLink(
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
            shape=[-1] + x.shape[2:],
            type=x.type)
        self.inputs.append(ipt)
        return ipt

    def step_output(self, o):
        self._assert_in_rnn_block_('step_output')
        if not isinstance(o, Variable):
            raise TypeError("step output takes a Variable")


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
        prog = self.kwargs.get('program', None)
        if prog is None:
            return g_program
        else:
            return prog

    def append_op(self, *args, **kwargs):
        return self.program.current_block().append_op(*args, **kwargs)

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
        default = {
            'name': None,
            'init_attr': {
                'type': 'uniform_random',
                'min': -1.0,
                'max': 1.0
            }
        }
        actual = self.kwargs.get('param_attr', None)
        return actual if actual is not None else default

    def bias_attr(self, shape, dtype):
        bias_attr = self.kwargs.get('bias_attr', None)
        if bias_attr is True:
            bias_attr = {
                'name': None,
                'init_attr': {
                    'type': 'fill_constant',
                    'value': 0.0,
                    'shape': shape,
                    'dataType': dtype
                }
            }
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

    def create_variable(self, *args, **kwargs):
        return self.program.current_block().create_var(*args, **kwargs)

    def create_global_variable(self, *args, **kwargs):
        return self.program.global_block().create_var(*args, **kwargs)

    def append_bias_op(self, input_var):
        size = list(input_var.shape[1:])
        bias_attr = self.bias_attr(size, dtype=input_var.data_type)
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
