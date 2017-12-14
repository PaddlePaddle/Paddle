from ..layer_helper import LayerHelper, unique_name
from ..framework import Program, Variable, Operator
from .. import core
from tensor import assign, fill_constant
import contextlib

__all__ = [
    'split_lod_tensor', 'merge_lod_tensor', 'BlockGuard', 'StaticRNNGuard',
    'StaticRNNMemoryLink', 'WhileGuard', 'While', 'lod_rank_table',
    'max_sequence_len', 'topk', 'lod_tensor_to_array', 'array_to_lod_tensor',
    'increment', 'array_write', 'create_array', 'less_than', 'array_read',
    'shrink_memory', 'array_length', 'IfElse', 'DynamicRNN', 'ConditionalBlock',
    'StaticRNN'
]


def split_lod_tensor(input,
                     mask,
                     level=0,
                     main_program=None,
                     startup_program=None):
    helper = LayerHelper('split_lod_tensor', **locals())
    out_true = helper.create_tmp_variable(dtype=input.dtype)
    out_false = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='split_lod_tensor',
        inputs={
            'X': input,
            'Mask': mask,
        },
        outputs={'OutTrue': out_true,
                 'OutFalse': out_false},
        attrs={'level': level})
    return out_true, out_false


def merge_lod_tensor(in_true,
                     in_false,
                     x,
                     mask,
                     level=0,
                     main_program=None,
                     startup_program=None):
    helper = LayerHelper('merge_lod_tensor', **locals())
    out = helper.create_tmp_variable(dtype=in_true.dtype)
    helper.append_op(
        type='merge_lod_tensor',
        inputs={'X': x,
                'Mask': mask,
                'InTrue': in_true,
                'InFalse': in_false},
        outputs={'Out': out},
        attrs={'level': level})
    return out


class BlockGuard(object):
    """
    BlockGuard class.

    BlockGuard class is used to create a sub-block in a program by
    using the Python `with` keyword.
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
    """
    StaticRNNGuard class.

    StaticRNNGuard class is used to create a StaticRNN block in a program.
    """

    def __init__(self, rnn):
        if not isinstance(rnn, StaticRNN):
            raise TypeError("StaticRNNGuard takes a StaticRNN")
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
    StaticRNNMemoryLink class.

    Args:
        init: the initial variable for Memory
        init: Variable
        pre_mem: the memory variable in previous time step
        pre_mem: Variable
        mem: the memory variable in current time step
        mem: Variable

    StaticRNNMemoryLink class is used to create a link between two
    memory cells of a StaticRNN.
    """

    def __init__(self, init, pre_mem, mem=None):
        self.init = init
        self.pre_mem = pre_mem
        self.mem = mem


class StaticRNN(object):
    """
    StaticRNN class.

    StaticRNN class is used to create a StaticRNN. The RNN will have its
    own parameters like inputs, outputs, memories, status and length.
    """
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
        """
        Args:
            init: boot memory, if not set, a shape, batch_ref must be provided
            shape: shape of the boot memory
            batch_ref: batch size reference variable
            init_value: the init value of boot memory
            init_batch_dim_idx: the index of batch size in init's dimension
            ref_batch_dim_idx: the index of batch size in batch_ref's dimension
        """
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
                dtype=batch_ref.dtype,
                persistable=False)

            parent_block.append_op(
                type="fill_constant_batch_size_like",
                inputs={'Input': [batch_ref]},
                outputs={'Out': [boot_var]},
                attrs={
                    'value': init_value,
                    'shape': boot_var.shape,
                    'dtype': boot_var.dtype,
                    'input_dim_idx': ref_batch_dim_idx,
                    'output_dim_idx': init_batch_dim_idx
                })

            return self.memory(init=boot_var)
        else:
            pre_mem = self.helper.create_variable(
                name=unique_name("@".join([self.helper.name, "mem"])),
                dtype=init.dtype,
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
            name=x.name, dtype=x.dtype, shape=list(x.shape[1:]), type=x.type)
        self.inputs.append(ipt)
        return ipt

    def step_output(self, o):
        self._assert_in_rnn_block_('step_output')
        if not isinstance(o, Variable):
            raise TypeError("step output takes a Variable")

        tmp_o = self.helper.create_tmp_variable(dtype=o.dtype)
        self.helper.append_op(
            type='rnn_memory_helper',
            inputs={'X': [o]},
            outputs={'Out': tmp_o},
            attrs={'dtype': o.dtype})

        out_var = self.parent_block().create_var(
            name=tmp_o.name,
            shape=[self.seq_len] + list(tmp_o.shape),
            dtype=tmp_o.dtype)

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
            new_mem = self.helper.create_tmp_variable(dtype=mem_var.dtype)

            rnn_block.append_op(
                type='rnn_memory_helper',
                inputs={'X': [mem_var]},
                outputs={'Out': [new_mem]},
                attrs={'dtype': mem_var.dtype})

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
                'sub_block': rnn_block
            })


class WhileGuard(BlockGuard):
    def __init__(self, while_op):
        if not isinstance(while_op, While):
            raise TypeError("WhileGuard takes a while op")
        super(WhileGuard, self).__init__(while_op.helper.main_program)
        self.while_op = while_op

    def __enter__(self):
        self.while_op.status = While.IN_WHILE_BLOCK
        return super(WhileGuard, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.while_op.status = While.AFTER_WHILE_BLOCK
        self.while_op.complete()
        return super(WhileGuard, self).__exit__(exc_type, exc_val, exc_tb)


class While(object):
    BEFORE_WHILE_BLOCK = 0
    IN_WHILE_BLOCK = 1
    AFTER_WHILE_BLOCK = 2

    def __init__(self, cond, name=None, main_program=None):
        self.helper = LayerHelper("while", name=name, main_program=main_program)
        self.status = While.BEFORE_WHILE_BLOCK
        if not isinstance(cond, Variable):
            raise TypeError("condition should be a variable")
        assert isinstance(cond, Variable)
        if cond.dtype != core.DataType.BOOL:
            raise TypeError("condition should be a bool variable")
        if reduce(lambda a, b: a * b, cond.shape, 1) != 1:
            raise TypeError("condition should be a bool scalar")
        self.cond_var = cond

    def block(self):
        return WhileGuard(self)

    def complete(self):
        main_program = self.helper.main_program
        while_block = main_program.current_block()
        parent_block = main_program.block(main_program.current_block()
                                          .parent_idx)

        inner_outputs = {self.cond_var.name}
        x_name_list = set()
        for op in while_block.ops:
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    if in_var_name not in inner_outputs:
                        x_name_list.add(in_var_name)

            for oname in op.output_names:
                for out_var_name in op.output(oname):
                    inner_outputs.add(out_var_name)

        out_vars = []
        for inner_out_name in inner_outputs:
            if inner_out_name in parent_block.vars:
                out_vars.append(parent_block.var(inner_out_name))

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)

        parent_block.append_op(
            type='while',
            inputs={
                'X': [parent_block.var(x_name) for x_name in x_name_list],
                'Condition': [self.cond_var]
            },
            outputs={'Out': out_vars,
                     'StepScopes': [step_scope]},
            attrs={'sub_block': while_block})


def lod_rank_table(x, level=0, main_program=None):
    """
    This function creates an operator for creating a LOD_RANK_TABLE
    using the input x.
    """
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


def max_sequence_len(rank_table, main_program=None):
    """
    This function creates an operator to calculate the length of
    max seqence through input rank_table(should be a lod_rank_table)
    """
    helper = LayerHelper("max_seqence_len", **locals())
    res = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="max_sequence_len",
        inputs={"RankTable": rank_table},
        outputs={"Out": res})
    return res


def topk(input, k, main_program=None, startup_program=None):
    helper = LayerHelper('topk', **locals())
    topk_out = helper.create_tmp_variable(dtype=input.data_type)
    topk_indices = helper.create_tmp_variable(dtype='int64')
    helper.append_op(
        type='top_k',
        inputs={'X': [input]},
        outputs={'Out': [topk_out],
                 'Indices': [topk_indices]},
        attrs={'k': k})
    return topk_out, topk_indices


def lod_tensor_to_array(x, table, main_program=None):
    """
    This function creates an operator to convert an LOD_Tensor to
    an array.
    """
    helper = LayerHelper("lod_tensor_to_array", **locals())
    array = helper.create_variable(
        name=unique_name("lod_tensor_to_array"),
        type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
        dtype=x.dtype)
    helper.append_op(
        type='lod_tensor_to_array',
        inputs={'X': x,
                'RankTable': table},
        outputs={'Out': array})
    return array


def array_to_lod_tensor(x, table, main_program=None, startup_program=None):
    """
    This function creates an operator to convert an array to a
    LOD_Tensor.
    """
    helper = LayerHelper("array_to_lod_tensor", **locals())
    tmp = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type="array_to_lod_tensor",
        inputs={'X': x,
                'RankTable': table},
        outputs={'Out': tmp})
    return tmp


def increment(x,
              value=1.0,
              in_place=True,
              main_program=None,
              startup_program=None):
    """
    This function creates an operator to increment each value in the input
    `x` by an amount: `value` as mentioned in the input parameter. This
    operation is performed in-place by default.
    """
    helper = LayerHelper("increment", **locals())
    if not in_place:
        out = helper.create_tmp_variable(dtype=x.dtype)
    else:
        out = x
    helper.append_op(
        type='increment',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'step': float(value)})
    return out


def array_write(x, i, array=None, main_program=None, startup_program=None):
    """
    This function creates an operator to write the data out as a
    LOD_TENSOR_ARRAY.
    """
    helper = LayerHelper('array_write', **locals())
    if array is None:
        array = helper.create_variable(
            name="{0}.out".format(helper.name),
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
            dtype=x.dtype)
    helper.append_op(
        type='write_to_array',
        inputs={'X': [x],
                'I': [i]},
        outputs={'Out': [array]})
    return array


def create_array(dtype, main_program=None):
    helper = LayerHelper("array", **locals())
    return helper.create_variable(
        name="{0}.out".format(helper.name),
        type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
        dtype=dtype)


def less_than(x, y, cond=None, main_program=None, **ignored):
    helper = LayerHelper("less_than", **locals())
    if cond is None:
        cond = helper.create_tmp_variable(dtype='bool')
        cond.stop_gradient = True

    helper.append_op(
        type='less_than', inputs={'X': [x],
                                  'Y': [y]}, outputs={'Out': [cond]})
    return cond


def array_read(array, i, main_program=None, startup_program=None):
    """
    This function creates an operator to read the data in as a
    LOD_TENSOR_ARRAY.
    """
    helper = LayerHelper('array_read', **locals())
    if not isinstance(
            array,
            Variable) or array.type != core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        raise TypeError("array should be tensor array vairable")
    out = helper.create_tmp_variable(dtype=array.dtype)
    helper.append_op(
        type='read_from_array',
        inputs={'X': [array],
                'I': [i]},
        outputs={'Out': [out]})
    return out


def shrink_memory(x, i, table, main_program=None, startup_program=None):
    """
    This function creates an operator to shrink_rnn_memory using the RankTable
    as mentioned in the input parameter.
    """
    helper = LayerHelper('shrink_memory', **locals())
    out = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type='shrink_rnn_memory',
        inputs={'X': [x],
                'I': [i],
                'RankTable': [table]},
        outputs={'Out': [out]},
        attrs={})
    return out


def array_length(array, main_program=None):
    """
    This function creates an operator to find the length of the
    LOD_TENSOR_ARRAY.
    """
    helper = LayerHelper('array_length', **locals())
    tmp = helper.create_tmp_variable(dtype='int64')
    tmp.stop_gradient = True
    helper.append_op(
        type='lod_array_length', inputs={'X': [array]}, outputs={'Out': [tmp]})
    return tmp


class ConditionalBlockGuard(BlockGuard):
    def __init__(self, block):
        if not isinstance(block, ConditionalBlock):
            raise TypeError("block should be conditional block")
        super(ConditionalBlockGuard, self).__init__(block.helper.main_program)
        self.block = block

    def __enter__(self):
        return super(ConditionalBlockGuard, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.block.complete()
        return super(ConditionalBlockGuard, self).__exit__(exc_type, exc_val,
                                                           exc_tb)


class ConditionalBlock(object):
    def __init__(self,
                 inputs,
                 name=None,
                 main_program=None,
                 startup_program=None):
        for each_input in inputs:
            if not isinstance(each_input, Variable):
                raise TypeError("Each input should be variable")
        self.inputs = inputs
        self.helper = LayerHelper(
            'conditional_block',
            name=name,
            main_program=main_program,
            startup_program=startup_program)

    def block(self):
        return ConditionalBlockGuard(self)

    def complete(self):
        inside_block = self.helper.main_program.current_block()
        parent_block = self.helper.main_program.block(inside_block.parent_idx)

        intermediate = set()
        params = set()

        for each_op in inside_block.ops:
            assert isinstance(each_op, Operator)
            for iname in each_op.input_names:
                for in_var_name in each_op.input(iname):
                    if in_var_name not in intermediate:
                        params.add(in_var_name)

            for oname in each_op.output_names:
                for out_var_name in each_op.output(oname):
                    intermediate.add(out_var_name)
        input_set = set([ipt.name for ipt in self.inputs])

        param_list = [
            parent_block.var(each_name) for each_name in params
            if each_name not in input_set
        ]

        out_list = [
            parent_block.var(var_name) for var_name in parent_block.vars
            if var_name not in intermediate
        ]

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)
        parent_block.append_op(
            type='conditional_block',
            inputs={
                'X': self.inputs,
                'Params': param_list,
            },
            outputs={'Out': out_list,
                     'Scope': [step_scope]},
            attrs={'sub_block': inside_block})


class IfElseBlockGuard(object):
    def __init__(self, is_true, ifelse):
        if not isinstance(ifelse, IfElse):
            raise TypeError("ifelse must be an instance of IfElse class")

        if ifelse.status != IfElse.OUT_IF_ELSE_BLOCKS:
            raise ValueError("You cannot invoke IfElse.block() inside a block")

        self.is_true = is_true
        self.ie = ifelse
        if is_true:
            self.cond_block = ifelse.conditional_true_block
        else:
            self.cond_block = ifelse.conditional_false_block

        if not isinstance(self.cond_block, ConditionalBlock):
            raise TypeError("Unexpected situation")

        self.cond_block = self.cond_block.block()

    def __enter__(self):
        self.ie.status = IfElse.IN_IF_ELSE_TRUE_BLOCKS if self.is_true else IfElse.IN_IF_ELSE_FALSE_BLOCKS
        self.cond_block.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.cond_block.__exit__(exc_type, exc_val, exc_tb):
            # re-raise inside exception
            return False
        if len(self.ie.output_table[1 if self.is_true else 0]) == 0:
            raise ValueError("Must set output inside block")
        self.ie.status = IfElse.OUT_IF_ELSE_BLOCKS


class IfElse(object):
    OUT_IF_ELSE_BLOCKS = 0
    IN_IF_ELSE_TRUE_BLOCKS = 1
    IN_IF_ELSE_FALSE_BLOCKS = 2

    def __init__(self, cond, name=None, main_program=None,
                 startup_program=None):
        if not isinstance(cond, Variable):
            raise TypeError("cond must be a Variable")
        self.helper = LayerHelper(
            'ifelse',
            name=name,
            main_program=main_program,
            startup_program=startup_program)
        self.cond = cond
        self.input_table = {}
        self.status = IfElse.OUT_IF_ELSE_BLOCKS
        self.conditional_true_block = ConditionalBlock(inputs=[self.cond])
        self.conditional_false_block = ConditionalBlock(inputs=[self.cond])
        self.output_table = ([], [])  # (true_outs, false_outs)

    def input(self, x):
        if self.status == IfElse.OUT_IF_ELSE_BLOCKS:
            raise ValueError("input must in true/false blocks")
        if id(x) not in self.input_table:
            parent_block = self.parent_block()
            out_true = parent_block.create_var(
                name=unique_name('ifelse_input' + self.helper.name),
                dtype=x.dtype)

            out_false = parent_block.create_var(
                name=unique_name('ifelse_input' + self.helper.name),
                dtype=x.dtype)
            parent_block.append_op(
                type='split_lod_tensor',
                inputs={
                    'X': x,
                    'Mask': self.cond,
                },
                outputs={'OutTrue': out_true,
                         'OutFalse': out_false},
                attrs={'level': 0})
            self.input_table[id(x)] = (out_true, out_false)
        else:
            out_true, out_false = self.input_table[id(x)]

        if self.status == IfElse.IN_IF_ELSE_TRUE_BLOCKS:
            return out_true
        else:
            return out_false

    def parent_block(self):
        current_block = self.helper.main_program.current_block()
        return self.helper.main_program.block(current_block.parent_idx)

    def true_block(self):
        return IfElseBlockGuard(True, self)

    def false_block(self):
        return IfElseBlockGuard(False, self)

    def output(self, *outs):
        if self.status == self.OUT_IF_ELSE_BLOCKS:
            raise ValueError("output can only be invoked in the sub-block")

        out_table = self.output_table[1 if self.status ==
                                      self.IN_IF_ELSE_TRUE_BLOCKS else 0]
        parent_block = self.parent_block()
        for each_out in outs:
            if not isinstance(each_out, Variable):
                raise TypeError("Each output should be a variable")
            # create outside tensor
            outside_out = parent_block.create_var(
                name=unique_name("_".join([self.helper.name, 'output'])),
                dtype=each_out.dtype)
            out_table.append(outside_out)

            # assign local var to outside
            assign(
                input=each_out,
                output=outside_out,
                main_program=self.helper.main_program,
                startup_program=self.helper.startup_program)

    def __call__(self):
        if self.status != self.OUT_IF_ELSE_BLOCKS:
            raise ValueError("IfElse::__call__ must be out of sub-block")
        false_len, true_len = map(len, self.output_table)
        if false_len == 0 and true_len == 0:
            raise ValueError("Must invoke true_block/false_block before "
                             "__call__")
        elif false_len != true_len and false_len != 0 and true_len != 0:
            raise ValueError("The output side must be same")
        elif false_len == 0 or true_len == 0:
            return self.output_table[0 if false_len != 0 else 1]

        # else none of false_len/true_len is zero
        # merge together
        rlist = []
        for false_var, true_var in zip(*self.output_table):
            rlist.append(
                merge_lod_tensor(
                    in_true=true_var,
                    in_false=false_var,
                    mask=self.cond,
                    x=self.cond,
                    level=0,
                    main_program=self.helper.main_program,
                    startup_program=self.helper.startup_program))
        return rlist


class DynamicRNN(object):
    BEFORE_RNN = 0
    IN_RNN = 1
    AFTER_RNN = 2

    def __init__(self, name=None, main_program=None, startup_program=None):
        self.helper = LayerHelper(
            'dynamic_rnn',
            name=name,
            main_program=main_program,
            startup_program=startup_program)
        self.status = DynamicRNN.BEFORE_RNN
        self.lod_rank_table = None
        self.max_seq_len = None
        self.step_idx = None
        self.zero_idx = fill_constant(shape=[1], value=0, dtype='int64')
        self.mem_dict = dict()
        self.output_array = []
        self.outputs = []
        self.cond = self.helper.create_tmp_variable(dtype='bool')
        self.cond.stop_gradient = False
        self.while_op = While(self.cond)
        self.input_array = []
        self.mem_link = []

    def step_input(self, x):
        self._assert_in_rnn_block_("step_input")
        if not isinstance(x, Variable):
            raise TypeError(
                "step_input() can only take a Variable as its input")
        parent_block = self._parent_block_()
        if self.lod_rank_table is None:
            self.lod_rank_table = parent_block.create_var(
                name=unique_name('lod_rank_table'),
                type=core.VarDesc.VarType.LOD_RANK_TABLE)
            self.lod_rank_table.stop_gradient = True
            parent_block.append_op(
                type='lod_rank_table',
                inputs={"X": x},
                outputs={"Out": self.lod_rank_table})
            self.max_seq_len = parent_block.create_var(
                name=unique_name('dynamic_rnn_max_seq_len'), dtype='int64')
            self.max_seq_len.stop_gradient = False
            parent_block.append_op(
                type='max_sequence_len',
                inputs={'RankTable': self.lod_rank_table},
                outputs={"Out": self.max_seq_len})
            self.cond.stop_gradient = True
            parent_block.append_op(
                type='less_than',
                inputs={'X': self.step_idx,
                        'Y': self.max_seq_len},
                outputs={'Out': self.cond})

        input_array = parent_block.create_var(
            name=unique_name('dynamic_rnn_input_array'),
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
            dtype=x.dtype)
        self.input_array.append((input_array, x.dtype))
        parent_block.append_op(
            type='lod_tensor_to_array',
            inputs={'X': x,
                    'RankTable': self.lod_rank_table},
            outputs={'Out': input_array})
        return array_read(
            array=input_array, i=self.step_idx, **self.helper.to_kwargs)

    @contextlib.contextmanager
    def block(self):
        if self.status != DynamicRNN.BEFORE_RNN:
            raise ValueError("rnn.block() can only be invoke once")
        self.step_idx = fill_constant(shape=[1], dtype='int64', value=0)
        self.step_idx.stop_gradient = False
        self.status = DynamicRNN.IN_RNN
        with self.while_op.block():
            yield
            increment(
                x=self.step_idx,
                value=1.0,
                in_place=True,
                **self.helper.to_kwargs)

            for new_mem, mem_array in self.mem_link:
                array_write(
                    x=new_mem,
                    i=self.step_idx,
                    array=mem_array,
                    **self.helper.to_kwargs)

            less_than(
                x=self.step_idx,
                y=self.max_seq_len,
                cond=self.cond,
                **self.helper.to_kwargs)

        self.status = DynamicRNN.AFTER_RNN
        for each_array in self.output_array:
            self.outputs.append(
                array_to_lod_tensor(
                    x=each_array,
                    table=self.lod_rank_table,
                    **self.helper.to_kwargs))

    def __call__(self, *args, **kwargs):
        if self.status != DynamicRNN.AFTER_RNN:
            raise ValueError(
                "Dynamic RNN outputs can only be retrieved after rnn block")
        if len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return self.outputs

    def memory(self, init=None, shape=None, value=0.0, dtype='float32'):
        self._assert_in_rnn_block_('memory')
        if init is not None:
            if not isinstance(init, Variable):
                raise TypeError(
                    "The input arg `init` of memory() must be a Variable")
            parent_block = self._parent_block_()
            mem_array = parent_block.create_var(
                name=unique_name('dynamic_rnn_mem_array'),
                type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
                dtype=init.dtype)
            parent_block.append_op(
                type='write_to_array',
                inputs={'X': init,
                        'I': self.zero_idx},
                outputs={'Out': mem_array})
            retv = array_read(
                array=mem_array, i=self.step_idx, **self.helper.to_kwargs)
            retv = shrink_memory(
                x=retv,
                i=self.step_idx,
                table=self.lod_rank_table,
                **self.helper.to_kwargs)
            self.mem_dict[retv.name] = mem_array
            return retv
        else:
            if len(self.input_array) == 0:
                raise ValueError(
                    "step_input should be invoked before memory(shape=..., value=...)"
                )
            parent_block = self._parent_block_()
            init = parent_block.create_var(
                name=unique_name('mem_init'), dtype=dtype)
            arr, dtype = self.input_array[0]
            in0 = parent_block.create_var(name=unique_name('in0'), dtype=dtype)
            parent_block.append_op(
                type='read_from_array',
                inputs={'X': [arr],
                        'I': [self.zero_idx]},
                outputs={'Out': [in0]})
            parent_block.append_op(
                type='fill_constant_batch_size_like',
                inputs={'Input': [in0]},
                outputs={'Out': [init]},
                attrs={
                    'shape': [-1] + shape,
                    'value': float(value),
                    'dtype': init.dtype
                })
            return self.memory(init=init)

    def update_memory(self, ex_mem, new_mem):
        self._assert_in_rnn_block_('update_memory')
        if not isinstance(ex_mem, Variable):
            raise TypeError("The input arg `ex_mem` of update_memory() must "
                            "be a Variable")
        if not isinstance(new_mem, Variable):
            raise TypeError("The input arg `new_mem` of update_memory() must "
                            "be a Variable")

        mem_array = self.mem_dict.get(ex_mem.name, None)
        if mem_array is None:
            raise ValueError("Please invoke memory before update_memory")
        if self.lod_rank_table is None:
            raise ValueError("Please invoke step_input before update_memory")

        self.mem_link.append((new_mem, mem_array))

    def output(self, *outputs):
        self._assert_in_rnn_block_('output')
        parent_block = self._parent_block_()
        for each in outputs:
            outside_array = parent_block.create_var(
                name=unique_name("_".join(
                    [self.helper.name, "output_array", each.name])),
                type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
                dtype=each.dtype)
            array_write(x=each, i=self.step_idx, array=outside_array)
            self.output_array.append(outside_array)

    def _parent_block_(self):
        prog = self.helper.main_program
        parent_idx = prog.current_block().parent_idx
        assert parent_idx >= 0
        parent_block = prog.block(parent_idx)

        return parent_block

    def _assert_in_rnn_block_(self, method):
        if self.status != DynamicRNN.IN_RNN:
            raise ValueError("{0} can only be invoked inside rnn block.".format(
                method))
