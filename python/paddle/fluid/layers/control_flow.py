# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from ..wrapped_decorator import signature_safe_contextmanager

from .layer_function_generator import autodoc, templatedoc
from .tensor import assign, fill_constant
from .. import core
from ..framework import Program, Variable, Operator
from ..layer_helper import LayerHelper, unique_name
from ..initializer import force_init_on_cpu
from .nn import logical_and, logical_not, logical_or
import numpy
import warnings
import six
from functools import reduce

__all__ = [
    'While', 'Switch', 'increment', 'array_write', 'create_array', 'less_than',
    'less_equal', 'greater_than', 'greater_equal', 'equal', 'not_equal',
    'array_read', 'array_length', 'IfElse', 'DynamicRNN', 'StaticRNN',
    'reorder_lod_tensor_by_rank', 'Print', 'is_empty'
]


def split_lod_tensor(input, mask, level=0):
    """
    This function takes in an input that contains the complete lod information,
    and takes in a mask which is used to mask certain parts of the input.
    The output is the true branch and the false branch with the mask applied to
    the input at a certain level in the tensor. Mainly used in IfElse to split
    data into two parts.

    Args:
        input(tuple|list|None): The input tensor that contains complete
                                lod information needed to construct the output.
        mask(list): A bool column vector which masks the input.
        level(int): The specific lod level to split.

    Returns:
        tuple(Variable, Variable):
        The true branch of tensor as per the mask applied to input.

        The false branch of tensor as per the mask applied to input.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          x = fluid.layers.data(name='x', shape=[1])
          x.persistable = True

          y = fluid.layers.data(name='y', shape=[1])
          y.persistable = True

          out_true, out_false = fluid.layers.split_lod_tensor(
                input=x, mask=y, level=level)

    """
    helper = LayerHelper('split_lod_tensor', **locals())
    out_true = helper.create_variable_for_type_inference(dtype=input.dtype)
    out_false = helper.create_variable_for_type_inference(dtype=input.dtype)
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


def merge_lod_tensor(in_true, in_false, x, mask, level=0):
    """
    **merge_lod_tensor**

    This function takes in an input :math:`x`, the True branch, the False
    branch and a binary :math:`mask`. Using this information, this function
    merges the True and False branches of the tensor into a single tensor as
    output at a certain lod level indicated by :math:`level`. Used in IfElse
    to merge the output if True block and False Block.

    Args:
        in_true(tuple|list|None): The True branch to be merged.
        in_false(tuple|list|None): The False branch to be merged.
        x(tuple|list|None): The input tensor that contains complete
                            lod information needed to construct the output.
        mask(list): A bool column vector which masks the input.
        level(int): The specific lod level to merge.

    Returns:
        Variable: The merged output tensor.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          x = layers.data(
                      name='x', shape=[1], dtype='float32', stop_gradient=False)
          y = layers.data(
                name='y', shape=[1], dtype='bool', stop_gradient=False)

          level = 0

          out_true, out_false = layers.split_lod_tensor(
                input=x, mask=y, level=level)
          out = layers.merge_lod_tensor(
                in_true=out_true, in_false=out_false, mask=y, x=x, level=level)
    """
    helper = LayerHelper('merge_lod_tensor', **locals())
    out = helper.create_variable_for_type_inference(dtype=in_true.dtype)
    helper.append_op(
        type='merge_lod_tensor',
        inputs={'X': x,
                'Mask': mask,
                'InTrue': in_true,
                'InFalse': in_false},
        outputs={'Out': out},
        attrs={'level': level})
    return out


def Print(input,
          first_n=-1,
          message=None,
          summarize=20,
          print_tensor_name=True,
          print_tensor_type=True,
          print_tensor_shape=True,
          print_tensor_lod=True,
          print_phase='both'):
    '''
    **Print operator**

    This creates a print op that will print when a tensor is accessed.

    Wraps the tensor passed in so that whenever that a tensor is accessed,
    the message `message` is printed, along with the current value of the
    tensor `t`.

    Args:
        input (Variable): A Tensor to print.
        summarize (int): Print this number of elements in the tensor, will print
                all if left is negative.
        message (str): A string message to print as a prefix.
        first_n (int): Only log `first_n` number of times.
        print_tensor_name (bool): Print the tensor name.
        print_tensor_type (bool): Print the tensor type.
        print_tensor_shape (bool): Print the tensor shape.
        print_tensor_lod (bool): Print the tensor lod.
        print_phase (str): Which phase to displace, including 'forward',
                'backward' and 'both'. If set to 'backward' or 'both', will
                print the gradients of input tensor.

    Returns:
        Variable: Output tensor.

    NOTES:
        The input and output are two different variables, and in the
        following process, you should use the output variable but not the input,
        otherwise, the print layer doesn't have backward.

    Examples:
        .. code-block:: python
           
           import paddle.fluid as fluid
           
           input = fluid.layers.fill_constant(shape=[10,2], value=3, dtype='int64')
           input = fluid.layers.Print(input, message="The content of input layer:")
           
           main_program = fluid.default_main_program()
           exe = fluid.Executor(fluid.CPUPlace())
           exe.run(main_program)

    Output at runtime:
        .. code-block:: bash 
           
           1564546375   The content of input layer:     The place is:CPUPlace
           Tensor[fill_constant_0.tmp_0]
               shape: [10,2,]
               dtype: x
               data: 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3, 
               
           # The information of dtype at runtime may vary in different environments.
           # Eg: 
           #    If the dtype='int64' of Tensor y, the corresponding c++ type is int64_t.
           #    The dtype of output is "x" ("x" is typeid(int64_t).name()) with MacOS and gcc4.8.2
    '''
    helper = LayerHelper('print' + "_" + input.name, **locals())
    output = helper.create_variable_for_type_inference(input.dtype)
    helper.append_op(
        type='print',
        inputs={'In': input},
        outputs={'Out': output},
        attrs={
            'first_n': first_n,
            'summarize': summarize,
            'message': message or "",
            'print_tensor_name': print_tensor_name,
            'print_tensor_type': print_tensor_type,
            'print_tensor_shape': print_tensor_shape,
            'print_tensor_lod': print_tensor_lod,
            'print_phase': print_phase.upper()
        })
    return output


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
        self.main_program._create_block()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.main_program._rollback()
        if exc_type is not None:
            return False  # re-raise exception
        return True


class BlockGuardWithCompletion(BlockGuard):
    """
    BlockGuardWithCompletion class.

    BlockGuardWithCompletion class is used to create an op with a block in a program.
    """

    def __init__(self, rnn):
        if not isinstance(rnn, StaticRNN):
            raise TypeError("BlockGuardWithCompletion takes a StaticRNN")
        super(BlockGuardWithCompletion, self).__init__(rnn.helper.main_program)
        self.rnn = rnn

    def __enter__(self):
        self.rnn.status = StaticRNN.IN_RNN_BLOCK
        return super(BlockGuardWithCompletion, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.rnn.status = StaticRNN.AFTER_RNN_BLOCK
        self.rnn._complete_op()
        return super(BlockGuardWithCompletion, self).__exit__(exc_type, exc_val,
                                                              exc_tb)


class StaticRNNMemoryLink(object):
    """
    StaticRNNMemoryLink class.

    StaticRNNMemoryLink class is used to create a link between two
    memory cells of a StaticRNN.


    NOTE: This is a internal data structure of a very low-level API.
    Please use StaticRNN instead.

    Args:
        init(Variable): the initial variable for Memory.
        pre_mem(Variable): the memory variable in previous time step.
        mem(Variable): the memory variable in current time step.
    """

    def __init__(self, init, pre_mem, mem=None):
        self.init = init
        self.pre_mem = pre_mem
        self.mem = mem


class StaticRNN(object):
    """
    StaticRNN class.

    The StaticRNN can process a batch of sequence data. The first dimension of inputs
    represents sequence length, the length of each input sequence must be equal.
    StaticRNN will unfold sequence into time steps, user needs to define how to process
    each time step during the :code:`with` step.

    Args:
        name (str, optional): Please refer to :ref:`api_guide_Name`, Default None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            vocab_size, hidden_size=10000, 200
            x = fluid.data(name="x", shape=[None, 1, 1], dtype='int64')
            # create word sequence
            x_emb = layers.embedding(
                input=x,
                size=[vocab_size, hidden_size],
                dtype='float32',
                is_sparse=False)
            # transform batch size to dim 1
            x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

            rnn = fluid.layers.StaticRNN()
            with rnn.step():
                # mark created x_emb as input, each step process a word
                word = rnn.step_input(x_emb)
                # create prev memory parameter, batch size comes from word
                prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
                # use hidden to update prev
                rnn.update_memory(prev, hidden)
                # mark hidden as output 
                rnn.step_output(hidden)
            # get StaticrNN final output
            result = rnn()

    """
    BEFORE_RNN_BLOCK = 0
    IN_RNN_BLOCK = 1
    AFTER_RNN_BLOCK = 2

    def __init__(self, name=None):
        self.helper = LayerHelper("static_rnn", name=name)
        self.memories = {}  # memory map, from pre_mem.name --> MemoryLink
        self.inputs = []  # input variable list in current block
        self.outputs = []  # output variable list in parent block
        self.status = StaticRNN.BEFORE_RNN_BLOCK  # status flag.
        # sequence length, since it is a static RNN, sequence length are fixed.
        self.seq_len = None

    def step(self):
        """
        Define operators in each step. step is used in :code:`with` block, OP in :code:`with` block
        will be executed sequence_len times (sequence_len is the length of input)
        """
        return BlockGuardWithCompletion(self)

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
        Create a memory variable for static rnn.
        If the :code:`init` is not None, :code:`memory` will be initialized by
        this Variable. If the :code:`init` is None, :code:`shape` and :code:`batch_ref`
        must be set, and this function will create a new variable with shape and batch_ref
        to initialize :code:`init` Variable.

        Args:
            init(Variable, optional): Tensor used to init memory. If it is not set,
                :code:`shape` and :code:`batch_ref` must be provided.
                Default: None.
            shape(list|tuple): When :code:`init` is None use this arg to initialize memory shape.
            NOTE the shape does not contain batch_size. Default: None.
            batch_ref(Variable, optional): When :code:`init` is None, memory's batch size will
            be set as batch_ref's ref_batch_dim_idx value. Default: None.
            init_value(float, optional): When :code:`init` is None, used to init memory's value. Default: 0.0.
            init_batch_dim_idx(int, optional): the batch_size axis of the :code:`init` Variable. Default: 0.
            ref_batch_dim_idx(int, optional): the batch_size axis of the :code:`batch_ref` Variable. Default: 1.

        Returns:
            Variable: The memory variable.

        Examples 1:
            .. code-block:: python

            	import paddle.fluid as fluid
            	import paddle.fluid.layers as layers

            	vocab_size, hidden_size=10000, 200
            	x = fluid.data(name="x", shape=[None, 1, 1], dtype='int64')
            	# create word sequence
            	x_emb = layers.embedding(
                	input=x,
                	size=[vocab_size, hidden_size],
                	dtype='float32',
                	is_sparse=False)
            	# transform batch size to dim 1
            	x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

            	rnn = fluid.layers.StaticRNN()
            	with rnn.step():
                	# mark created x_emb as input, each step process a word
                	word = rnn.step_input(x_emb)
                	# create prev memory parameter, batch size comes from word
                	prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                	hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
                	# use hidden to update prev
                	rnn.update_memory(prev, hidden)


        Examples 2:
            .. code-block:: python

            	import paddle.fluid as fluid
            	import paddle.fluid.layers as layers
            	vocab_size, hidden_size=10000, 200
            	x = fluid.data(name="x", shape=[None, 1, 1], dtype='int64')
            	# create word sequence
            	x_emb = layers.embedding(
                	input=x,
                	size=[vocab_size, hidden_size],
                	dtype='float32',
                	is_sparse=False)
            	# transform batch size to dim 1
            	x_emb = layers.transpose(x_emb, perm=[1, 0, 2])
            	boot_memory = fluid.layers.data(name='boot', shape=[hidden_size], dtype='float32', lod_level=1)
            	rnn = fluid.layers.StaticRNN()
            	with rnn.step():
            		# mark created x_emb as input, each step process a word
            		word = rnn.step_input(x_emb)
            		# init memory
            		prev = rnn.memory(init=boot_memory)
            		hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
            		# update hidden with prev
            		rnn.update_memory(prev, hidden)

        """
        self._assert_in_rnn_block_('memory')
        if init is None:
            if shape is None or batch_ref is None:
                raise ValueError(
                    "if init is None, memory at least need shape and batch_ref")
            parent_block = self._parent_block()
            var_name = unique_name.generate_with_ignorable_key("@".join(
                [self.helper.name, "memory_boot"]))
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
                name=unique_name.generate_with_ignorable_key("@".join(
                    [self.helper.name, "mem"])),
                dtype=init.dtype,
                shape=init.shape)
            self.memories[pre_mem.name] = StaticRNNMemoryLink(
                init=init, pre_mem=pre_mem)
            return pre_mem

    def step_input(self, x):
        """
        Mark a sequence as a StaticRNN input.

        Args:
            x(Variable): The input sequence, the shape of x
                should be [seq_len, ...].

        Returns:
            Variable: The current time step data in the input sequence.

        Examples:
            .. code-block:: python

            	import paddle.fluid as fluid
            	import paddle.fluid.layers as layers

            	vocab_size, hidden_size=10000, 200
            	x = fluid.data(name="x", shape=[None, 1, 1], dtype='int64')
            	# create word sequence
            	x_emb = layers.embedding(
                	input=x,
                	size=[vocab_size, hidden_size],
                	dtype='float32',
                	is_sparse=False)
            	# transform batch size to dim 1
            	x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

            	rnn = fluid.layers.StaticRNN()
            	with rnn.step():
                	# mark created x_emb as input, each step process a word
                	word = rnn.step_input(x_emb)
                	# create prev memory parameter, batch size comes from word
                	prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                	hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
                	# use hidden to update prev
                	rnn.update_memory(prev, hidden)

        """
        self._assert_in_rnn_block_('step_input')
        if not isinstance(x, Variable):
            raise TypeError("step input takes a Variable")
        if self.seq_len is None:
            self.seq_len = x.shape[0]
        elif x.shape[0] != -1 and self.seq_len != x.shape[0]:
            raise ValueError("Static RNN only take fix seq_len input")

        ipt = self.helper.create_variable(
            name=x.name, dtype=x.dtype, shape=list(x.shape[1:]), type=x.type)
        self.inputs.append(ipt)
        return ipt

    def step_output(self, o):
        """
        Mark a sequence as a StaticRNN output.

        Args:
            o(Variable): The output sequence.

        Returns:
            None.

        Examples:
            .. code-block:: python

            	import paddle.fluid as fluid
            	import paddle.fluid.layers as layers

            	vocab_size, hidden_size=10000, 200
            	x = fluid.data(name="x", shape=[None, 1, 1], dtype='int64')
            	# create word sequence
            	x_emb = layers.embedding(
                	input=x,
                	size=[vocab_size, hidden_size],
               		dtype='float32',
                	is_sparse=False)
            	# transform batch size to dim 1
            	x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

            	rnn = fluid.layers.StaticRNN()
            	with rnn.step():
                	# mark created x_emb as input, each step process a word
               		word = rnn.step_input(x_emb)
                	# create prev memory parameter, batch size comes from word
                	prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                	hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
                	# use hidden to update prev
                	rnn.update_memory(prev, hidden)
                	rnn.step_output(hidden)

            	result = rnn()

        """
        self._assert_in_rnn_block_('step_output')
        if not isinstance(o, Variable):
            raise TypeError("step output takes a Variable")

        tmp_o = self.helper.create_variable_for_type_inference(dtype=o.dtype)
        self.helper.append_op(
            type='rnn_memory_helper',
            inputs={'X': [o]},
            outputs={'Out': tmp_o},
            attrs={'dtype': o.dtype})

        out_var = self._parent_block().create_var(
            name=tmp_o.name,
            shape=[self.seq_len] + list(tmp_o.shape),
            dtype=tmp_o.dtype)

        self.outputs.append(out_var)

    def output(self, *outputs):
        """
        Mark the StaticRNN output variables.

        Args:
            outputs: The output Tensor, can mark multiple variables as output

        Returns:
            None

        Examples:
            .. code-block:: python

            	import paddle.fluid as fluid
            	import paddle.fluid.layers as layers

            	vocab_size, hidden_size=10000, 200
            	x = fluid.data(name="x", shape=[None, 1, 1], dtype='int64')
            	# create word sequence
            	x_emb = layers.embedding(
                	input=x,
                	size=[vocab_size, hidden_size],
                	dtype='float32',
                	is_sparse=False)
            	# transform batch size to dim 1
            	x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

            	rnn = fluid.layers.StaticRNN()
            	with rnn.step():
                	# mark created x_emb as input, each step process a word
                	word = rnn.step_input(x_emb)
                	# create prev memory parameter, batch size comes from word
                	prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                	hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
                	# use hidden to update prev
                	rnn.update_memory(prev, hidden)
                	# mark each step's hidden and word as output
                	rnn.output(hidden, word)

            	result = rnn()
        """
        for each in outputs:
            self.step_output(each)

    def update_memory(self, mem, var):
        """
        Update the memory from :code:`mem` to :code:`var`.

        Args:
            mem(Variable): the memory variable.
            var(Variable): the plain variable generated in RNN block, used to update memory.
                           var and mem should hava same dims and data type.

        Returns:
            None

        """
        if not isinstance(mem, Variable) or not isinstance(var, Variable):
            raise TypeError("update memory should take variables")
        self.memories[mem.name].mem = var

    def _parent_block(self):
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

    def _complete_op(self):
        main_program = self.helper.main_program
        rnn_block = main_program.current_block()
        parent_block = self._parent_block()

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

        # NOTE(zcd): the params have two categories of variables.
        #   - the variables that are the out of StaticRnn.
        #   - the variables that are the parameters of some layers, for example, conv2d.
        params = list()
        for op in rnn_block.ops:
            assert isinstance(op, Operator)
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    if in_var_name not in local_inputs:
                        params.append(in_var_name)

        parameters = [parent_block.var(name) for name in set(params)]

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)

        inlinks = [parent_block.var(i.name) for i in self.inputs]
        outlinks = self.outputs

        # NOTE(zcd): the states maybe empty in some case.
        boot_memories = []
        pre_memories = []
        memories = []
        for _, mem in six.iteritems(self.memories):
            boot_memories.append(mem.init)
            pre_memories.append(mem.pre_mem.name)
            assert mem.mem is not None, "%s should be updated in every step." % (
                mem.init.name)
            mem_var = rnn_block.var(mem.mem.name)
            assert isinstance(mem_var, Variable)
            new_mem = self.helper.create_variable_for_type_inference(
                dtype=mem_var.dtype)
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
                'has_states': len(pre_memories) > 0,
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
        self.while_op._complete()
        return super(WhileGuard, self).__exit__(exc_type, exc_val, exc_tb)


class While(object):
    """
    while loop control flow. Repeat while body until cond is False.

    Args:
        cond(Variable): A Tensor whose data type is bool controlling whether to continue looping.
        is_test(bool, optional): A flag indicating whether execution is in test phase. Default value is None.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Examples:
          .. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np

            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)           # loop counter

            loop_len = fluid.layers.fill_constant(shape=[1],dtype='int64', value=10)    # loop length

            cond = fluid.layers.less_than(x=i, y=loop_len)              
            while_op = fluid.layers.While(cond=cond)
            with while_op.block():  
                i = fluid.layers.increment(x=i, value=1, in_place=True)
                fluid.layers.less_than(x=i, y=loop_len, cond=cond)      

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            res = exe.run(fluid.default_main_program(), feed={}, fetch_list=[i])
            print(res) # [array([10])]           
    """

    BEFORE_WHILE_BLOCK = 0
    IN_WHILE_BLOCK = 1
    AFTER_WHILE_BLOCK = 2

    def __init__(self, cond, is_test=False, name=None):
        self.helper = LayerHelper("while", name=name)
        self.status = While.BEFORE_WHILE_BLOCK
        if not isinstance(cond, Variable):
            raise TypeError("condition should be a variable")
        assert isinstance(cond, Variable)
        if cond.dtype != core.VarDesc.VarType.BOOL:
            raise TypeError("condition should be a boolean variable")
        if reduce(lambda a, b: a * b, cond.shape, 1) != 1:
            raise TypeError(
                "condition expected shape as [], but given shape as {0}.".
                format(list(cond.shape)))
        self.cond_var = cond
        self.is_test = is_test

    def block(self):
        return WhileGuard(self)

    def _complete(self):
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
            inner_var = parent_block._find_var_recursive(inner_out_name)
            if inner_var:
                out_vars.append(inner_var)

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)

        parent_block.append_op(
            type='while',
            inputs={
                'X': [
                    parent_block._var_recursive(x_name)
                    for x_name in x_name_list
                ],
                'Condition': [self.cond_var]
            },
            outputs={'Out': out_vars,
                     'StepScopes': [step_scope]},
            attrs={'sub_block': while_block,
                   "is_test": self.is_test})


def lod_rank_table(x, level=0):
    """
    LoD Rank Table Operator. Given an input variable **x** and a level number
    of LoD, this layer creates a LodRankTable object. A LoDRankTable object
    contains a list of bi-element tuples. Each tuple consists of an index and
    a length, both of which are int type. Refering to specified level of LoD,
    the index is the sequence index number and the length representes the
    sequence length. Please note that the list is ranked in descending order by
    the length. The following is an example:

        .. code-block:: text

            x is a LoDTensor:
                x.lod = [[2,                1],
                         [5,             1, 1]]
                x.data = [a, b, c, d, e, f, g]

            1. set level to 0:
                Create lod rank table:
                    lod_rank_table_obj = lod_rank_table(x, level=0)

                Get:
                    lod_rank_table_obj.items() = [(0, 2), (1, 1)]

            2. set level to 1:
                Create lod rank table:
                    lod_rank_table_obj = lod_rank_table(x, level=1)

                Get:
                    lod_rank_table_obj.items() = [(0, 5), (1, 1), (2, 1)]

    Args:
        x (Variable): Input variable, a LoDTensor based which to create the lod
            rank table.
        level (int): Specify the LoD level, on which to create the lod rank
            table.

    Returns:
        Variable: The created LoDRankTable object.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[10],
                                  dtype='float32', lod_level=1)
            out = layers.lod_rank_table(x=x, level=0)
    """
    helper = LayerHelper("lod_rank_table", **locals())
    table = helper.create_variable(
        type=core.VarDesc.VarType.LOD_RANK_TABLE,
        name=unique_name.generate("lod_rank_table"))
    helper.append_op(
        type='lod_rank_table',
        inputs={'X': x},
        outputs={'Out': table},
        attrs={'level': level})
    return table


@templatedoc()
def max_sequence_len(rank_table):
    """
    ${comment}

    >>> import paddle.fluid as fluid
    >>> x = fluid.layers.data(name='x', shape=[10], dtype='float32',
    >>>                       lod_level=1)
    >>> rank_table = layers.lod_rank_table(x=x, level=0)
    >>> max_seq_len = layers.max_sequence_len(rank_table)

    Args:
        rank_table(${rank_table_type}): ${rank_table_comment}.

    Returns:
        ${out_comment}.
    """
    helper = LayerHelper("max_seqence_len", **locals())
    res = helper.create_variable_for_type_inference(dtype="int64")
    helper.append_op(
        type="max_sequence_len",
        inputs={"RankTable": rank_table},
        outputs={"Out": res})
    return res


def lod_tensor_to_array(x, table):
    """
    Convert a LoDTensor to a LoDTensorArray.

    This function split a LoDTesnor to a LoDTensorArray according to its LoD
    information. LoDTensorArray is an alias of C++ std::vector<LoDTensor> in
    PaddlePaddle. The generated LoDTensorArray of this function can be further read
    or written by `read_from_array()` and `write_to_array()` operators. However,
    this function is generally an internal component of PaddlePaddle `DynamicRNN`.
    Users should not use it directly.

    Args:
        x (Variable|list): The LoDTensor to be converted to a LoDTensorArray.
        table (ParamAttr|list): The variable that stores the level of lod
                                which is ordered by sequence length in
                                descending order. It is generally generated
                                by `layers.lod_rank_table()` API.

    Returns:
        Variable: The LoDTensorArray that has been converted from the input tensor.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          x = fluid.layers.data(name='x', shape=[10])
          table = fluid.layers.lod_rank_table(x, level=0)
          array = fluid.layers.lod_tensor_to_array(x, table)
    """
    helper = LayerHelper("lod_tensor_to_array", **locals())
    array = helper.create_variable(
        name=unique_name.generate("lod_tensor_to_array"),
        type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
        dtype=x.dtype)
    helper.append_op(
        type='lod_tensor_to_array',
        inputs={'X': x,
                'RankTable': table},
        outputs={'Out': array})
    return array


def array_to_lod_tensor(x, table):
    """Convert a LoD_Tensor_Aarry to an LoDTensor.

    Args:
        x (Variable|list): The lod tensor array to be converted to a tensor.
        table (ParamAttr|list): The variable that stores the level of lod
                                which is ordered by sequence length in
                                descending order.

    Returns:
        Variable: The variable of type tensor that has been converted
                  from an array.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          x = fluid.layers.data(name='x', shape=[10])
          table = fluid.layers.lod_rank_table(x, level=0)
          array = fluid.layers.lod_tensor_to_array(x, table)
          lod_tensor = fluid.layers.array_to_lod_tensor(array, table)
    """
    helper = LayerHelper("array_to_lod_tensor", **locals())
    tmp = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="array_to_lod_tensor",
        inputs={'X': x,
                'RankTable': table},
        outputs={'Out': tmp})
    return tmp


def increment(x, value=1.0, in_place=True):
    """
    This function performs an operation that increments the value in the
    input :math:`x` by an amount: :math:`value` as mentioned in the input
    parameter. This operation is performed in-place by default. Notice that
    the number of elements in :math:`x` must be equal to 1.

    Args:
        x (Variable|list): The tensor that has the input values.
        value (float): The amount by which the values should be incremented.
        in_place (bool): If the increment should be performed in-place.

    Returns:
        Variable: The elementwise-incremented object.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='data', shape=[1], dtype='float32',
                                   append_batch_size=False)
          data = fluid.layers.increment(x=data, value=3.0, in_place=True)
    """
    helper = LayerHelper("increment", **locals())
    if not in_place:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = x
    helper.append_op(
        type='increment',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'step': float(value)})
    return out


def array_write(x, i, array=None):
    """
    This function writes the given input variable to the specified position
    indicating by the arrary index to an output LOD_TENSOR_ARRAY. If the
    output LOD_TENSOR_ARRAY is not given(None), a new one will be created and
    returned.

    Args:
        x (Variable|list): The input tensor from which the data will be read.
        i (Variable|list): The index of the output LOD_TENSOR_ARRAY, pointing to
                           the position to which the input tensor will be
                           written.
        array (Variable|list): The output LOD_TENSOR_ARRAY to which the input
                               tensor will be written. If this parameter is
                               NONE, a new LOD_TENSOR_ARRAY will be created and
                               returned.

    Returns:
        Variable: The output LOD_TENSOR_ARRAY where the input tensor is written.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          tmp = fluid.layers.zeros(shape=[10], dtype='int32')
          i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
          arr = fluid.layers.array_write(tmp, i=i)
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


def create_array(dtype):
    """
    **Create LoDTensorArray**

    This function creates an array of LOD_TENSOR_ARRAY . It is mainly used to
    implement RNN with array_write, array_read and While.

    Args:
        dtype (int|float): The data type of the elements in the lod_tensor_array.

    Returns:
        Variable: The lod_tensor_array variable storing the elements of data type.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.create_array(dtype='float32')

    """
    helper = LayerHelper("array", **locals())
    return helper.create_variable(
        name="{0}.out".format(helper.name),
        type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
        dtype=dtype)


@templatedoc()
def less_than(x, y, force_cpu=None, cond=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}.
        y(${y_type}): ${y_comment}.
        force_cpu(${force_cpu_type}): ${force_cpu_comment}.
        cond(Variable|None): Optional output variable to store the result of *less_than*

    Returns:
        ${out_comment}.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          label = fluid.layers.data(name='y', shape=[1], dtype='int64')
          limit = fluid.layers.fill_constant(shape=[1], dtype='int64', value=5)
          cond = fluid.layers.less_than(x=label, y=limit)
    """
    helper = LayerHelper("less_than", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    attrs = dict()
    if force_cpu is not None:
        attrs['force_cpu'] = force_cpu
    elif force_init_on_cpu():
        attrs['force_cpu'] = force_init_on_cpu()

    helper.append_op(
        type='less_than',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [cond]},
        attrs=attrs)
    return cond


@templatedoc()
def less_equal(x, y, cond=None):
    """
    This OP returns the truth value of :math:`x <= y` elementwise, which is equivalent function to the overloaded operator `<=`.

    Args:
        x(Variable): First input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64. 
        y(Variable): Second input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        cond(Variable, optional): If is :attr:`None`, the op will create a variable as output tensor, the input shape and data type of \
            this tensor is the same as input :attr:`x`. If is not :attr:`None`, the op will set the variable as output tensor, the input shape \
            and data type of this tensor should be the same as input :attr:`x`. Default value is :attr:`None`.

    Returns:
        Variable, the output data type is bool.: The tensor variable storing the output, the output shape is the same as input :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          label = fluid.layers.assign(np.array([1, 3], dtype='int32'))
          limit = fluid.layers.assign(np.array([1, 2], dtype='int32'))
          out = fluid.layers.less_equal(x=label, y=limit) #out=[True, False]
          out1 = label<= limit #out1=[True, False]

    """
    helper = LayerHelper("less_equal", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    attrs = dict()
    if force_init_on_cpu():
        attrs['force_cpu'] = force_init_on_cpu()

    helper.append_op(
        type='less_equal',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [cond]},
        attrs=attrs)
    return cond


@templatedoc()
def greater_than(x, y, cond=None):
    """
    This OP returns the truth value of :math:`x > y` elementwise, which is equivalent function to the overloaded operator `>`.

    Args:
        x(Variable): First input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64. 
        y(Variable): Second input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        cond(Variable, optional): If is :attr:`None`, the op will create a variable as output tensor, the shape and data type of this \
            tensor is the same as input :attr:`x` . If is not :attr:`None`, the op will set the variable as output tensor, the shape and data type \
            of this tensor should be the same as input :attr:`x` . Default value is :attr:`None`.

    Returns:
        Variable, the output data type is bool.: The tensor variable storing the output, the output shape is the same as input :attr:`x` .

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          label = fluid.layers.assign(np.array([2, 3], dtype='int32'))
          limit = fluid.layers.assign(np.array([3, 2], dtype='int32'))
          out = fluid.layers.greater_than(x=label, y=limit) #out=[False, True]
          out1 = label > limit #out1=[False, True]
    """
    helper = LayerHelper("greater_than", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    attrs = dict()
    if force_init_on_cpu():
        attrs['force_cpu'] = force_init_on_cpu()

    helper.append_op(
        type='greater_than',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [cond]},
        attrs=attrs)
    return cond


@templatedoc()
def greater_equal(x, y, cond=None):
    """
    This OP returns the truth value of :math:`x >= y` elementwise, which is equivalent function to the overloaded operator `>=`.

    Args:
        x(Variable): First input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64. 
        y(Variable): Second input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        cond(Variable, optional): If is :attr:`None` , the op will create a variable as output tensor, the shape and data type of this \
            tensor is the same as input :attr:`x`. If is not :attr:`None` , the op will set the variable as output tensor, the shape and data \
            type of this tensor is the same as input :attr:`x`. Default value is :attr:`None`.

    Returns:
        Variable, the output data type is bool.: The tensor variable storing the output, the output shape is the same as input :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np

          label = fluid.layers.assign(np.array([2, 2], dtype='int32'))
          limit = fluid.layers.assign(np.array([2, 3], dtype='int32'))
          out = fluid.layers.greater_equal(x=label, y=limit) #out=[True, False]
          out_1 = label >= limit #out1=[True, False]

    """
    helper = LayerHelper("greater_equal", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    attrs = dict()
    if force_init_on_cpu():
        attrs['force_cpu'] = force_init_on_cpu()

    helper.append_op(
        type='greater_equal',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [cond]},
        attrs=attrs)
    return cond


def equal(x, y, cond=None):
    """
    This layer returns the truth value of :math:`x == y` elementwise.

    Args:
        x(Variable): First operand of *equal*
        y(Variable): Second operand of *equal*
        cond(Variable|None): Optional output variable to store the result of *equal*

    Returns:
        Variable: The tensor variable storing the output of *equal*.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          label = fluid.layers.data(name="label", shape=[3,10,32,32], dtype="float32")
          limit = fluid.layers.data(name="limit", shape=[3,10,32,32], dtype="float32")
          less = fluid.layers.equal(x=label, y=limit)
    """
    helper = LayerHelper("equal", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    helper.append_op(
        type='equal', inputs={'X': [x],
                              'Y': [y]}, outputs={'Out': [cond]})
    return cond


def not_equal(x, y, cond=None):
    """
    This OP returns the truth value of :math:`x != y` elementwise, which is equivalent function to the overloaded operator `!=`.

    Args:
        x(Variable): First input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64. 
        y(Variable): Second input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        cond(Variable, optional): If is :attr:`None`, the op will create a variable as output tensor, the shape and data type of this \
             tensor is the same as input :attr:`x`. If is not :attr:`None`, the op will set the variable as output tensor, the shape and data \
             type of this tensor should be the same as input :attr:`x`. Default value is :attr:`None`.

    Returns:
        Variable, the output data type is bool.: The tensor variable storing the output, the output shape is the same as input :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          
          label = fluid.layers.data(name='label', shape=[1], dtype='int64')
          limit = fluid.layers.fill_constant(shape=[1], value=1, dtype='int64')
          out = fluid.layers.not_equal(x=label, y=limit)
    """
    helper = LayerHelper("not_equal", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    helper.append_op(
        type='not_equal', inputs={'X': [x],
                                  'Y': [y]}, outputs={'Out': [cond]})
    return cond


def array_read(array, i):
    """
    This function performs the operation to read the data in as an
    LOD_TENSOR_ARRAY.

    .. code-block:: text

        Given:

        array = [0.6, 0.1, 0.3, 0.1]

        And:

        i = 2

        Then:

        output = 0.3

    Args:
        array (Variable|list): The input tensor that store data to be read.
        i (Variable|list): The index of the data to be read from input array.

    Returns:
        Variable: The tensor type variable that has the data written to it.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          array = fluid.layers.create_array(dtype='float32')
          i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
          item = fluid.layers.array_read(array, i)
    """
    helper = LayerHelper('array_read', **locals())
    if not isinstance(
            array,
            Variable) or array.type != core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        raise TypeError("array should be tensor array vairable")
    out = helper.create_variable_for_type_inference(dtype=array.dtype)
    helper.append_op(
        type='read_from_array',
        inputs={'X': [array],
                'I': [i]},
        outputs={'Out': [out]})
    return out


def shrink_memory(x, i, table):
    """
    This function creates an operator to shrink rnn memory using the RankTable
    as mentioned in the input parameter.

    NOTE: This API is very low-level API. It is used by DynamicRNN only.

    Since the Dynamic RNN uses no-padding way to implement RNN. The sequence
    will be sorted by order, and the length of valid memory will be shrink after
    each time step.

    Args:
        x(Variable): The memory object in the previous time step.
        i(Variable): The step count variable. A int scalar as LoDTensor.
        table(Variable): The RNNRankTable object.

    Returns:
        the memory variable after shrink.

    Examples:

        Since this API is very low level API. The example is not provided.
        Please reference the implementation of class DynamicRNN for detail
        usage.
    """
    helper = LayerHelper('shrink_memory', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='shrink_rnn_memory',
        inputs={'X': [x],
                'I': [i],
                'RankTable': [table]},
        outputs={'Out': [out]},
        attrs={})
    return out


def array_length(array):
    """
    **Get the Length of Input LoDTensorArray**

    This function performs the operation to find the length of the input
    LOD_TENSOR_ARRAY.

    Related API: array_read, array_write, While.

    Args:
        array (LOD_TENSOR_ARRAY): The input array that will be used
                                  to compute the length.

    Returns:
        Variable: The length of the input LoDTensorArray.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          tmp = fluid.layers.zeros(shape=[10], dtype='int32')
          i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
          arr = fluid.layers.array_write(tmp, i=i)
          arr_len = fluid.layers.array_length(arr)

    """
    helper = LayerHelper('array_length', **locals())
    tmp = helper.create_variable_for_type_inference(dtype='int64')
    tmp.stop_gradient = True
    helper.append_op(
        type='lod_array_length', inputs={'X': [array]}, outputs={'Out': [tmp]})
    return tmp


class ConditionalBlockGuard(BlockGuard):
    """
    ConditionalBlockGuard is derived from BlockGuard. It is dedicated for
    holding a ConditionalBlock, and helping users entering and exiting the
    ConditionalBlock via Python's 'with' keyword. However, ConditionalBlockGuard
    is generally an internal component of IfElse, users should not use it directly.
    """

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
    '''
    **ConditionalBlock**

    ConditionalBlock is an operator that bind a block to a specific condition,
    if the condition matches, the corresponding block will be executed.

    Args:
        inputs (Variable): bool conditions.
        is_scalar_condition (bool): whether the branch is controled by a scalar.
        name(str): name of this ConditionalBlock.

    Examples:
        .. code-block:: python

             import paddle.fluid as fluid
             cond = layers.less_than(x=label, y=limit)
             true_image, false_image = layers.split_lod_tensor(
                 input=image, mask=cond)
             true_cond = layers.ConditionalBlock([true_image])

             with true_cond.block():
                 ...
             with false_cond.block():
                 ...
    '''

    def __init__(self, inputs, is_scalar_condition=False, name=None):
        for each_input in inputs:
            if not isinstance(each_input, Variable):
                raise TypeError("Each input should be variable")
        self.inputs = inputs
        self.is_scalar_condition = is_scalar_condition
        self.helper = LayerHelper('conditional_block', name=name)

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
            parent_block._var_recursive(each_name) for each_name in params
            if each_name not in input_set
        ]

        out_list = []
        for inner_out_name in intermediate:
            inner_var = parent_block._find_var_recursive(inner_out_name)
            if inner_var:
                out_list.append(inner_var)

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)
        parent_block.append_op(
            type='conditional_block',
            inputs={
                'Cond': self.inputs,
                'Input': param_list,
            },
            outputs={'Out': out_list,
                     'Scope': [step_scope]},
            attrs={
                'sub_block': inside_block,
                'is_scalar_condition': self.is_scalar_condition
            })


class Switch(object):
    """

    This class is used to implement Switch branch control function. 
    Switch branch contains several case branches and one default branch. 
    Switch control flow checks whether the case branch conditions are satisfied in turn, 
    and only executes the statement after the first case branch that satisfies the conditions. 
    If there is no case branch that satisfies the condition, 
    only the statement following the default branch is executed.

    Member Functions:
        case(cond): The case branch of Switch whose parameter cond is a scalar Variable of bool type. Only if the cond of the current case branch is True and the cond of the previous case branch is False, the statement after the case branch will be executed, and the statement after the case branch will not be executed.
        
        default(): The default branch of Switch. When cond of all case branches is False, the statement after default branch is executed.

    Case and default functions can only be used inside the scope of Switch, as shown below:

    .. code-block:: python
        
        '''
        with fluid.layers.Switch() as switch:
            with switch.case(cond1):
                i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=1)
            with switch.case(cond2):
                i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=2)
            with switch.default():
                i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
        '''

    Args:
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid

            lr = fluid.layers.create_global_var(
                shape=[1],
                value=0.0,
                dtype='float32',
                persistable=True,
                name="learning_rate")
            zero_var = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=0.0)
            one_var = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=1.0)
            two_var = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=2.0)

            global_step = fluid.layers.autoincreased_step_counter(counter_name='@LR_DECAY_COUNTER@', begin=0, step=1)

            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(global_step == zero_var):
                    fluid.layers.assign(input=one_var, output=lr)
                with switch.default():
                    fluid.layers.assign(input=two_var, output=lr)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            res = exe.run(fluid.default_main_program(), feed={}, fetch_list=[lr])
            print(res) # [array([1.], dtype=float32)]
    """

    def __init__(self, name=None):
        self.helper = LayerHelper('switch', name=name)
        self.inside_scope = False
        self.pre_not_conditions = []

    def case(self, condition):
        if not self.inside_scope:
            raise ValueError("case should be called inside with")

        if len(self.pre_not_conditions) == 0:
            cond_block = ConditionalBlock([condition], is_scalar_condition=True)
            not_cond = logical_not(x=condition)
            self.pre_not_conditions.append(not_cond)
        else:
            pre_cond_num = len(self.pre_not_conditions)
            pre_not_cond = self.pre_not_conditions[pre_cond_num - 1]
            new_not_cond = logical_and(
                x=pre_not_cond, y=logical_not(x=condition))
            self.pre_not_conditions.append(new_not_cond)
            cond_block = ConditionalBlock(
                [logical_and(
                    x=pre_not_cond, y=condition)],
                is_scalar_condition=True)

        return ConditionalBlockGuard(cond_block)

    def default(self):
        pre_cond_num = len(self.pre_not_conditions)
        if pre_cond_num == 0:
            raise ValueError("there should be at least one condition")
        cond_block = ConditionalBlock(
            [self.pre_not_conditions[pre_cond_num - 1]],
            is_scalar_condition=True)
        return ConditionalBlockGuard(cond_block)

    def __enter__(self):
        """
        set flag that now is inside switch.block {}
        :return:
        """
        self.inside_scope = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.inside_scope = False
        if exc_type is not None:
            return False  # re-raise exception

        return True


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
    """
    This class is used to implement IfElse branch control function. IfElse contains two blocks, true_block and false_block. IfElse will put data satisfying True or False conditions into different blocks to run.

    Cond is a 2-D Tensor with shape [N, 1] and data type bool, representing the execution conditions of the corresponding part of the input data.

    IfElse OP is different from other OPs in usage, which may cause some users confusion. Here is a simple example to illustrate this OP.

    .. code-block:: python
        
        # The following code completes the function: subtract 10 from the data greater than 0 in x, add 10 to the data less than 0 in x, and sum all the data.
        import numpy as np
        import paddle.fluid as fluid

        x = fluid.layers.data(name='x', shape=[4, 1], dtype='float32', append_batch_size=False)
        y = fluid.layers.data(name='y', shape=[4, 1], dtype='float32', append_batch_size=False)

        x_d = np.array([[3], [1], [-2], [-3]]).astype(np.float32)
        y_d = np.zeros((4, 1)).astype(np.float32)
        
        # Compare the size of x, y pairs of elements, output cond, cond is shape [4, 1], data type bool 2-D tensor.
        # Based on the input data x_d, y_d, it can be inferred that the data in cond are [[true], [true], [false], [false]].
        cond = fluid.layers.greater_than(x, y)
        # Unlike other common OPs, ie below returned by the OP is an IfElse OP object
        ie = fluid.layers.IfElse(cond)

        with ie.true_block():
            # In this block, according to cond condition, the data corresponding to true dimension in X is obtained and subtracted by 10.
            out_1 = ie.input(x)
            out_1 = out_1 - 10
            ie.output(out_1)
        with ie.false_block():
            # In this block, according to cond condition, get the data of the corresponding condition in X as false dimension, and add 10
            out_1 = ie.input(x)
            out_1 = out_1 + 10
            ie.output(out_1)

        # According to cond condition, the data processed in the two blocks are merged. The output here is output, the type is List, and the element type in List is Variable.
        output = ie() #  [array([[-7.], [-9.], [ 8.], [ 7.]], dtype=float32)] 

        # Get the first Variable in the output List and add all elements.
        out = fluid.layers.reduce_sum(output[0])

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        res = exe.run(fluid.default_main_program(), feed={"x":x_d, "y":y_d}, fetch_list=[out])
        print res
        # [array([-1.], dtype=float32)] 

    Args:
        cond (Variable): cond is a 2-D Tensor with shape [N, 1] and data type bool, representing the corresponding execution conditions of N input data. The data type is bool.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Unlike other common OPs, the OP call returns an IfElse OP object (e.g. ie in the example), which branches the input data by calling the internal functions of the object ``true_block ()``, ``false_block ()``, ``input ()``, ``output ()``, and integrates the data processed by different branches as the overall output by calling the internal ``call ()`` function. The output type is a list, and the type of each element in the list is Variable.

    Internal Functions:
        The block is constructed by calling the ``with ie. true_block()`` function in the object, and the computational logic under condition true is put into the block. If no corresponding block is constructed, the input data in the corresponding conditional dimension is unchanged.
 
        The block is constructed by calling the ``with ie. false_block()`` function in the object, and the computational logic under condition false is put into the block. If no corresponding block is constructed, the input data in the corresponding conditional dimension is unchanged.

        ``Out = ie. input (x)`` will take out the data of the corresponding conditional dimension in X and put it into out, supporting the internal processing of multiple inputs in block.

        ``ie. output (out)`` writes the result to the output of the corresponding condition.

        There is a ``call ()`` function inside the object, that is, by calling ``output = ie ()``, all the outputs inside the block of False are fused as the whole output, the output type is a list, and the type of each element in the list is Variable.

    """
    OUT_IF_ELSE_BLOCKS = 0
    IN_IF_ELSE_TRUE_BLOCKS = 1
    IN_IF_ELSE_FALSE_BLOCKS = 2

    def __init__(self, cond, name=None):
        if not isinstance(cond, Variable):
            raise TypeError("cond must be a Variable")
        self.helper = LayerHelper('ifelse', name=name)
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
            parent_block = self._parent_block()
            out_true = parent_block.create_var(
                name=unique_name.generate_with_ignorable_key('ifelse_input' +
                                                             self.helper.name),
                dtype=x.dtype)

            out_false = parent_block.create_var(
                name=unique_name.generate_with_ignorable_key('ifelse_input' +
                                                             self.helper.name),
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

    def _parent_block(self):
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
        parent_block = self._parent_block()
        for each_out in outs:
            if not isinstance(each_out, Variable):
                raise TypeError("Each output should be a variable")
            # create outside tensor
            outside_out = parent_block.create_var(
                name=unique_name.generate_with_ignorable_key("_".join(
                    [self.helper.name, 'output'])),
                dtype=each_out.dtype)
            out_table.append(outside_out)

            # assign local var to outside
            assign(input=each_out, output=outside_out)

    def __call__(self):
        if self.status != self.OUT_IF_ELSE_BLOCKS:
            raise ValueError("IfElse::__call__ must be out of sub-block")
        false_len, true_len = list(map(len, self.output_table))
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
                    level=0))
        return rlist


class DynamicRNN(object):
    """
    The dynamic RNN can process a batch of sequence data. The length of each
    sample sequence can be different. This API automatically process them in
    batch.

    The input lod must be set. Please reference to `lod_tensor`.

    The dynamic RNN will unfold sequence into timesteps. Users need to define
    how to process each time step during the :code:`with` block.

    The `memory` is used staging data cross time step. The initial value of
    memory can be zero or another variable.

    The dynamic RNN can mark multiple variables as its output. Use `drnn()` to
    get the output sequence.

    NOTES:
        Currently it is not supported that setting is_sparse to True of any 
        layers within DynamicRNN.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid

          sentence = fluid.layers.data(name='sentence', shape=[1], dtype='int64', lod_level=1)
          embedding = fluid.layers.embedding(input=sentence, size=[65536, 32], is_sparse=True)
    
          drnn = fluid.layers.DynamicRNN()
          with drnn.block():
              word = drnn.step_input(embedding)
              prev = drnn.memory(shape=[200])
              hidden = fluid.layers.fc(input=[word, prev], size=200, act='relu')
              drnn.update_memory(prev, hidden)  # set prev to hidden
              drnn.output(hidden)

          # Get the last time step of rnn. It is the encoding result.
          rnn_output = drnn()
          last = fluid.layers.sequence_last_step(rnn_output)
    """
    BEFORE_RNN = 0
    IN_RNN = 1
    AFTER_RNN = 2

    def __init__(self, name=None):
        self.helper = LayerHelper('dynamic_rnn', name=name)
        self.status = DynamicRNN.BEFORE_RNN
        self.lod_rank_table = None
        self.max_seq_len = None
        self.step_idx = None
        self.zero_idx = None
        self.mem_dict = dict()
        self.output_array = []
        self.outputs = []
        self.cond = self.helper.create_variable_for_type_inference(dtype='bool')
        self.cond.stop_gradient = False
        self.while_op = While(self.cond)
        self.input_array = []
        self.mem_link = []

    def step_input(self, x, level=0):
        """
        Mark a sequence as a dynamic RNN input.

        Args:
            x (Variable): The input sequence which should have lod information.
            level (int): The level of lod used to split steps. Default: 0.

        Returns:
            The current timestep in the input sequence.
        """
        self._assert_in_rnn_block_("step_input")
        if not isinstance(x, Variable):
            raise TypeError(
                "step_input() can only take a Variable as its input.")
        parent_block = self._parent_block_()
        if self.lod_rank_table is None:
            self.lod_rank_table = parent_block.create_var(
                name=unique_name.generate('lod_rank_table'),
                type=core.VarDesc.VarType.LOD_RANK_TABLE)
            self.lod_rank_table.stop_gradient = True
            parent_block.append_op(
                type='lod_rank_table',
                inputs={"X": x},
                outputs={"Out": self.lod_rank_table},
                attrs={"level": level})
            self.max_seq_len = parent_block.create_var(
                name=unique_name.generate('dynamic_rnn_max_seq_len'),
                dtype='int64')
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
                outputs={'Out': self.cond},
                attrs={'force_cpu': True})

        input_array = parent_block.create_var(
            name=unique_name.generate('dynamic_rnn_input_array'),
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
            dtype=x.dtype)
        self.input_array.append((input_array, x.dtype))
        parent_block.append_op(
            type='lod_tensor_to_array',
            inputs={'X': x,
                    'RankTable': self.lod_rank_table},
            outputs={'Out': input_array})
        return array_read(array=input_array, i=self.step_idx)

    def static_input(self, x):
        """
        Mark a variable as a RNN input. The input will not be scattered into
        time steps. It is optional.

        Args:
            x (Variable): The input variable.

        Returns:
            The input variable that can access in RNN.

        Examples:
            .. code-block:: python

              import paddle.fluid as fluid

              sentence = fluid.layers.data(name='sentence', dtype='float32', shape=[32], lod_level=1)
              encoder_proj = fluid.layers.data(name='encoder_proj', dtype='float32', shape=[32], lod_level=1)
              decoder_boot = fluid.layers.data(name='boot', dtype='float32', shape=[10], lod_level=1)

              drnn = fluid.layers.DynamicRNN()
              with drnn.block():
                  current_word = drnn.step_input(sentence)
                  encoder_word = drnn.static_input(encoder_proj)
                  hidden_mem = drnn.memory(init=decoder_boot, need_reorder=True)
                  fc_1 = fluid.layers.fc(input=encoder_word, size=30, bias_attr=False)
                  fc_2 = fluid.layers.fc(input=current_word, size=30, bias_attr=False)
                  decoder_inputs = fc_1 + fc_2
                  h, _, _ = fluid.layers.gru_unit(input=decoder_inputs, hidden=hidden_mem, size=30)
                  drnn.update_memory(hidden_mem, h)
                  out = fluid.layers.fc(input=h, size=10, bias_attr=True, act='softmax') 
                  drnn.output(out)

              rnn_output = drnn()
        """
        self._assert_in_rnn_block_("static_input")
        if not isinstance(x, Variable):
            raise TypeError(
                "static_input() can only take a Variable as its input")
        if self.lod_rank_table is None:
            raise RuntimeError(
                "static_input() must be called after step_input().")
        parent_block = self._parent_block_()
        x_reordered = parent_block.create_var(
            name=unique_name.generate("dynamic_rnn_static_input_reordered"),
            type=core.VarDesc.VarType.LOD_TENSOR,
            dtype=x.dtype)
        parent_block.append_op(
            type='reorder_lod_tensor_by_rank',
            inputs={'X': [x],
                    'RankTable': [self.lod_rank_table]},
            outputs={'Out': [x_reordered]})
        return shrink_memory(x_reordered, self.step_idx, self.lod_rank_table)

    @signature_safe_contextmanager
    def block(self):
        """
        The block for user to define operators in RNN.
        """
        if self.status != DynamicRNN.BEFORE_RNN:
            raise ValueError("rnn.block() can only be invoke once")
        self.step_idx = fill_constant(
            shape=[1], dtype='int64', value=0, force_cpu=True)
        self.step_idx.stop_gradient = False
        self.status = DynamicRNN.IN_RNN
        with self.while_op.block():
            yield
            increment(x=self.step_idx, value=1.0, in_place=True)

            for new_mem, mem_array in self.mem_link:
                array_write(x=new_mem, i=self.step_idx, array=mem_array)

            less_than(
                x=self.step_idx,
                y=self.max_seq_len,
                force_cpu=True,
                cond=self.cond)

        self.status = DynamicRNN.AFTER_RNN
        for each_array in self.output_array:
            self.outputs.append(
                array_to_lod_tensor(
                    x=each_array, table=self.lod_rank_table))

    def __call__(self, *args, **kwargs):
        """
        Get the output of RNN. This API should only be invoked after RNN.block()
        """
        if self.status != DynamicRNN.AFTER_RNN:
            raise ValueError(("Output of the dynamic RNN can only be visited "
                              "outside the rnn block."))
        if len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return self.outputs

    def memory(self,
               init=None,
               shape=None,
               value=0.0,
               need_reorder=False,
               dtype='float32'):
        """
        Create a memory variable for dynamic rnn.

        If the :code:`init` is not None, :code:`memory` will be initialized by
        this variable. The :code:`need_reorder` is used to reorder the memory as
        the input variable. It should be set to true when the initialized memory
        depends on the input sample.

        Examples:
            .. code-block:: python

              import paddle.fluid as fluid

              sentence = fluid.layers.data(name='sentence', shape=[32], dtype='float32', lod_level=1)
              boot_memory = fluid.layers.data(name='boot', shape=[10], dtype='float32', lod_level=1)
              
              drnn = fluid.layers.DynamicRNN()
              with drnn.block():
                  word = drnn.step_input(sentence)
                  memory = drnn.memory(init=boot_memory, need_reorder=True)
                  hidden = fluid.layers.fc(input=[word, memory], size=10, act='tanh')
                  drnn.update_memory(ex_mem=memory, new_mem=hidden)
                  drnn.output(hidden)

              rnn_output = drnn()


        Otherwise, if :code:`shape`, :code:`value`, :code:`dtype` are set, the
        :code:`memory` will be initialized by this :code:`value`.

        Examples:
            .. code-block:: python

              import paddle.fluid as fluid

              sentence = fluid.layers.data(name='sentence', dtype='float32', shape=[32], lod_level=1)
              
              drnn = fluid.layers.DynamicRNN()
              with drnn.block():
                  word = drnn.step_input(sentence)
                  memory = drnn.memory(shape=[10], dtype='float32', value=0)
                  hidden = fluid.layers.fc(input=[word, memory], size=10, act='tanh')
                  drnn.update_memory(ex_mem=memory, new_mem=hidden)
                  drnn.output(hidden)

              rnn_output = drnn()


        Args:
            init(Variable|None): The initialized variable.
            shape(list|tuple): The memory shape. The shape does not contain batch_size.
            value(float): the initalized value.
            need_reorder(bool): True if the initialized memory depends on the input sample.
            dtype(str|numpy.dtype): The data type of the initialized memory.

        Returns:
            The memory variable.
        """
        self._assert_in_rnn_block_('memory')
        self._init_zero_idx_()
        if init is not None:
            if not isinstance(init, Variable):
                raise TypeError(
                    "The input arg `init` of memory() must be a Variable")
            parent_block = self._parent_block_()
            init_tensor = init
            if need_reorder == True:
                if self.lod_rank_table is None:
                    raise ValueError(
                        'If set need_reorder to True, make sure step_input be '
                        'invoked before '
                        'memory(init=init, need_reordered=True, ...).')
                init_reordered = parent_block.create_var(
                    name=unique_name.generate('dynamic_rnn_mem_init_reordered'),
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    dtype=init.dtype)
                parent_block.append_op(
                    type='reorder_lod_tensor_by_rank',
                    inputs={
                        'X': [init_tensor],
                        'RankTable': [self.lod_rank_table]
                    },
                    outputs={'Out': [init_reordered]})
                init_tensor = init_reordered
            mem_array = parent_block.create_var(
                name=unique_name.generate('dynamic_rnn_mem_array'),
                type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
                dtype=init.dtype)
            parent_block.append_op(
                type='write_to_array',
                inputs={'X': init_tensor,
                        'I': self.zero_idx},
                outputs={'Out': mem_array})
            retv = array_read(array=mem_array, i=self.step_idx)
            retv = shrink_memory(
                x=retv, i=self.step_idx, table=self.lod_rank_table)
            self.mem_dict[retv.name] = mem_array
            return retv
        else:
            if len(self.input_array) == 0:
                raise ValueError(
                    "step_input should be invoked before memory(shape=..., value=...)"
                )
            parent_block = self._parent_block_()
            init = parent_block.create_var(
                name=unique_name.generate('mem_init'), dtype=dtype)
            arr, dtype = self.input_array[0]
            in0 = parent_block.create_var(
                name=unique_name.generate('in0'), dtype=dtype)
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
        """
        Update the memory from ex_mem to new_mem. NOTE that the shape and data
        type of :code:`ex_mem` and :code:`new_mem` must be same.
        
        Args:
            ex_mem(Variable): the memory variable.
            new_mem(Variable): the plain variable generated in RNN block.

        Returns:
            None
        """
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
        """
        Mark the RNN output variables.

        Args:
            outputs: The output variables.

        Returns:
            None
        """
        self._assert_in_rnn_block_('output')
        parent_block = self._parent_block_()
        for each in outputs:
            outside_array = parent_block.create_var(
                name=unique_name.generate_with_ignorable_key("_".join(
                    [self.helper.name, "output_array", each.name])),
                type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
                dtype=each.dtype)
            array_write(x=each, i=self.step_idx, array=outside_array)
            self.output_array.append(outside_array)

    def _init_zero_idx_(self):
        if self.zero_idx is None:
            parent_block = self._parent_block_()
            self.zero_idx = parent_block.create_var(
                name=unique_name.generate('zero_idx'), dtype='int64')
            parent_block.append_op(
                type='fill_constant',
                inputs={},
                outputs={'Out': [self.zero_idx]},
                attrs={
                    'shape': [1],
                    'dtype': self.zero_idx.dtype,
                    'value': float(0),
                    'force_cpu': True
                })

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


@templatedoc()
def reorder_lod_tensor_by_rank(x, rank_table):
    """
    ${comment}

    Args:
    
        x(${x_type}): ${x_comment}
        rank_table(${rank_table_type}): ${rank_table_type}
    
    Returns:
        out(${out_type}): ${out_comment} 

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data_desc = (['input', [9], 0], ['ref', [5], 1])
          data = fluid.layers.data(name=data_desc[0][0], shape=data_desc[0][1])
          rank_data = fluid.layers.data(name=data_desc[1][0], shape=data_desc[1][1])
          table = fluid.layers.control_flow.lod_rank_table(rank_data)
          new_data = fluid.layers.reorder_lod_tensor_by_rank(
                           x=data, rank_table=table)

    """
    helper = LayerHelper('reorder_lod_tensor_by_rank', **locals())
    helper.is_instance('x', Variable)
    helper.is_instance('rank_table', Variable)

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='reorder_lod_tensor_by_rank',
        inputs={'X': [x],
                'RankTable': [rank_table]},
        outputs={'Out': [out]})
    return out


def is_empty(x, cond=None):
    """
    Test whether a Variable is empty.

    Args:
        x (Variable): The Variable to be tested.
        cond (Variable|None): Output parameter. Returns the test result
                              of given 'x'. Default: None

    Returns:
        Variable: A bool scalar. True if 'x' is an empty Variable.

    Raises:
        TypeError: If input cond is not a variable, or cond's dtype is
                   not bool.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          input = fluid.layers.data(name="input", shape=[4, 32, 32], dtype="float32")
          res = fluid.layers.is_empty(x=input)
          # or:
          # fluid.layers.is_empty(x=input, cond=res)

    """
    helper = LayerHelper("is_empty", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True
    elif not isinstance(cond, Variable):
        raise TypeError("cond takes a variable")
    elif cond.dtype != 'bool':
        raise TypeError("The data type of cond must be bool")

    helper.append_op(
        type='is_empty', inputs={'X': [x]}, outputs={'Out': [cond]})
    return cond
