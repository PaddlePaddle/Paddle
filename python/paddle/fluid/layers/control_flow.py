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

from ..wrapped_decorator import signature_safe_contextmanager

from .layer_function_generator import templatedoc
from .. import core
from ..framework import (
    Program,
    Variable,
    Operator,
    static_only,
    in_dygraph_mode,
)
from ..layer_helper import LayerHelper, unique_name
from ...utils import (
    assert_same_structure,
    map_structure,
    hold_mutable_vars,
    copy_mutable_vars,
    is_sequence,
    pack_sequence_as,
    flatten,
    to_sequence,
)
import numpy
import warnings
from functools import reduce, partial
from ..data_feeder import (
    convert_dtype,
    check_variable_and_dtype,
    check_type,
    check_dtype,
)
from ..backward import _infer_var_data_type_shape_
import paddle
from paddle import _C_ops, _legacy_C_ops

__all__ = [
    'Switch',
    'StaticRNN',
    'while_loop',
]


def select_output(input, outputs, mask):
    """
    **select_output**
    This API takes in one input and multiple outputs and an integer mask. It
    selects the output specified by the mask and copy the input to selected
    output. It is useful in control flow.

    Args:
        input(Variable): The input variable
        outputs(tuple|list): The output variables
        mask(Variable): A tensor containing 1 integer number selecting which
            output to be copied with input

    Returns:
        Variable: The outputs variables
    """
    helper = LayerHelper('select_output', **locals())
    check_type(input, 'input', (Variable), 'select_output')
    check_variable_and_dtype(mask, 'mask', ['int32'], 'select_output')
    check_type(outputs, 'outputs', (list, tuple), 'select_output')

    helper.append_op(
        type='select_output',
        inputs={'X': input, 'Mask': mask},
        outputs={'Out': outputs},
    )
    return outputs


def _select_input_infer_shape(first_shape, second_shape):
    """
    This function infer the output shape by following algorithm:
    1. if the dims is different, raise a error.
    2. compare axis one by one:
        if a == b: we set axis to a
        if a != b: we set axis to -1
    for compatibility, non declarative mode, we just return second_shape.
    """
    if len(first_shape) != len(second_shape):
        warnings.warn(
            f"the input shapes of select_input should have the same rank, but get {first_shape}, {second_shape}"
        )
        return second_shape
    out_shape = list(
        map(lambda a, b: a if a == b else -1, first_shape, second_shape)
    )
    return out_shape


def select_input(inputs, mask):
    """
    **select_input**

    This API takes in multiple inputs and uses an integer mask to select one
    input to output. It is useful in control flow.

    Args:
        inputs(tuple|list): The input variables
        mask(Variable): A tensor containing 1 integer number selecting which
            input to output

    Returns:
        Variable: The selected input variable
    """
    helper = LayerHelper('select_input', **locals())
    check_type(inputs, 'inputs', (list, tuple), 'select_input')
    check_variable_and_dtype(mask, 'mask', ['int32'], 'select_input')

    # Select input should expand the shape. If it is - 1 and valid number, use - 1 first. If the dim is different, an error will be reported directly
    # assert inputs[0].dtype == inputs[1].dtype, f"Expect the inputs should have the same dtype, but get {inputs[0].dtype} and {inputs[1].dtype}"

    output_shape = _select_input_infer_shape(inputs[0].shape, inputs[1].shape)
    output_dtype = inputs[1].dtype
    output_type = inputs[1].type

    out = helper.create_variable(
        dtype=output_dtype, shape=output_shape, type=output_type
    )
    helper.append_op(
        type='select_input',
        inputs={'X': inputs, 'Mask': mask},
        outputs={'Out': out},
    )
    return out


# (TODO: Mine) There exists dependency. It will be removed later.
class BlockGuard:
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


# (TODO: Mine) There exists dependency. It will be removed later.
class BlockGuardWithCompletion(BlockGuard):
    """
    BlockGuardWithCompletion class.

    BlockGuardWithCompletion class is used to create an op with a block in a program.
    """

    def __init__(self, rnn):
        if not isinstance(rnn, StaticRNN):
            raise TypeError("BlockGuardWithCompletion takes a StaticRNN")
        super().__init__(rnn.helper.main_program)
        self.rnn = rnn

    def __enter__(self):
        self.rnn.status = StaticRNN.IN_RNN_BLOCK
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.rnn.status = StaticRNN.AFTER_RNN_BLOCK
        self.rnn._complete_op()
        return super().__exit__(exc_type, exc_val, exc_tb)


class StaticRNNMemoryLink:
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


class StaticRNN:
    """
    :api_attr: Static Graph

    StaticRNN class.

    The StaticRNN can process a batch of sequence data. The first dimension of inputs
    represents sequence length, the length of each input sequence must be equal.
    StaticRNN will unfold sequence into time steps, user needs to define how to process
    each time step during the :code:`with` step.

    Args:
        name (str, optional): Please refer to :ref:`api_guide_Name`, Default None.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            vocab_size, hidden_size=10000, 200
            paddle.enable_static()
            x = paddle.static.data(name="x", shape=[None, 1, 1], dtype='int64')
            # create word sequence
            x_emb = layers.embedding(
                input=x,
                size=[vocab_size, hidden_size],
                dtype='float32',
                is_sparse=False)
            # transform batch size to dim 1
            x_emb = paddle.transpose(x_emb, perm=[1, 0, 2])

            rnn = fluid.layers.StaticRNN()
            with rnn.step():
                # mark created x_emb as input, each step process a word
                word = rnn.step_input(x_emb)
                # create prev memory parameter, batch size comes from word
                prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                hidden = paddle.static.nn.fc(x=[word, prev], size=hidden_size, activation='relu')
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
        check_type(name, "name", (str, type(None)), "fluid.layers.StaticRNN")
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

    def memory(
        self,
        init=None,
        shape=None,
        batch_ref=None,
        init_value=0.0,
        init_batch_dim_idx=0,
        ref_batch_dim_idx=1,
    ):
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

                import paddle
                import paddle.fluid as fluid
                import paddle.fluid.layers as layers

                vocab_size, hidden_size=10000, 200
                paddle.enable_static()
                x = paddle.static.data(name="x", shape=[None, 1, 1], dtype='int64')
                # create word sequence
                x_emb = layers.embedding(
                        input=x,
                        size=[vocab_size, hidden_size],
                        dtype='float32',
                        is_sparse=False)
                # transform batch size to dim 1
                x_emb = paddle.transpose(x_emb, perm=[1, 0, 2])

                rnn = fluid.layers.StaticRNN()
                with rnn.step():
                        # mark created x_emb as input, each step process a word
                        word = rnn.step_input(x_emb)
                        # create prev memory parameter, batch size comes from word
                        prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                        hidden = paddle.static.nn.fc(x=[word, prev], size=hidden_size, activation='relu')
                        # use hidden to update prev
                        rnn.update_memory(prev, hidden)


        Examples 2:
            .. code-block:: python

                import paddle
                import paddle.fluid as fluid
                import paddle.fluid.layers as layers
                vocab_size, hidden_size=10000, 200
                paddle.enable_static()
                x = paddle.static.data(name="x", shape=[None, 1, 1], dtype='int64')
                # create word sequence
                x_emb = layers.embedding(
                        input=x,
                        size=[vocab_size, hidden_size],
                        dtype='float32',
                        is_sparse=False)
                # transform batch size to dim 1
                x_emb = paddle.transpose(x_emb, perm=[1, 0, 2])
                boot_memory = paddle.static.data(name='boot', shape=[-1, hidden_size], dtype='float32', lod_level=1)
                rnn = fluid.layers.StaticRNN()
                with rnn.step():
                        # mark created x_emb as input, each step process a word
                        word = rnn.step_input(x_emb)
                        # init memory
                        prev = rnn.memory(init=boot_memory)
                        hidden = paddle.static.nn.fc(x=[word, prev], size=hidden_size, activation='relu')
                        # update hidden with prev
                        rnn.update_memory(prev, hidden)

        """
        self._assert_in_rnn_block_('memory')
        check_type(
            init,
            "init",
            (Variable, type(None)),
            "fluid.layers.StaticRNN.memory",
        )
        check_type(
            shape,
            "shape",
            (list, tuple, type(None)),
            "fluid.layers.StaticRNN.memory",
        )
        check_type(
            batch_ref,
            "batch_ref",
            (Variable, type(None)),
            "fluid.layers.StaticRNN.memory",
        )
        if init is None:
            if shape is None or batch_ref is None:
                raise ValueError(
                    "if init is None, memory at least need shape and batch_ref"
                )
            parent_block = self._parent_block()
            var_name = unique_name.generate_with_ignorable_key(
                "@".join([self.helper.name, "memory_boot"])
            )
            boot_var = parent_block.create_var(
                name=var_name,
                shape=shape,
                dtype=batch_ref.dtype,
                persistable=False,
            )

            parent_block.append_op(
                type="fill_constant_batch_size_like",
                inputs={'Input': [batch_ref]},
                outputs={'Out': [boot_var]},
                attrs={
                    'value': init_value,
                    'shape': boot_var.shape,
                    'dtype': boot_var.dtype,
                    'input_dim_idx': ref_batch_dim_idx,
                    'output_dim_idx': init_batch_dim_idx,
                },
            )

            return self.memory(init=boot_var)
        else:
            pre_mem = self.helper.create_variable(
                name=unique_name.generate_with_ignorable_key(
                    "@".join([self.helper.name, "mem"])
                ),
                dtype=init.dtype,
                shape=init.shape,
            )
            self.memories[pre_mem.name] = StaticRNNMemoryLink(
                init=init, pre_mem=pre_mem
            )
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

                import paddle
                import paddle.fluid as fluid
                import paddle.fluid.layers as layers

                vocab_size, hidden_size=10000, 200
                paddle.enable_static()
                x = paddle.static.data(name="x", shape=[None, 1, 1], dtype='int64')
                # create word sequence
                x_emb = layers.embedding(
                        input=x,
                        size=[vocab_size, hidden_size],
                        dtype='float32',
                        is_sparse=False)
                # transform batch size to dim 1
                x_emb = paddle.transpose(x_emb, perm=[1, 0, 2])

                rnn = fluid.layers.StaticRNN()
                with rnn.step():
                        # mark created x_emb as input, each step process a word
                        word = rnn.step_input(x_emb)
                        # create prev memory parameter, batch size comes from word
                        prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                        hidden = paddle.static.nn.fc(x=[word, prev], size=hidden_size, activation='relu')
                        # use hidden to update prev
                        rnn.update_memory(prev, hidden)

        """
        self._assert_in_rnn_block_('step_input')
        check_type(x, "x", Variable, "fluid.layers.StaticRNN.step_input")
        if self.seq_len is None:
            self.seq_len = x.shape[0]
        elif x.shape[0] != -1 and self.seq_len != x.shape[0]:
            raise ValueError("Static RNN only take fix seq_len input")

        ipt = self.helper.create_variable(
            name=x.name, dtype=x.dtype, shape=list(x.shape[1:]), type=x.type
        )
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

                import paddle
                import paddle.fluid as fluid
                import paddle.fluid.layers as layers

                vocab_size, hidden_size=10000, 200
                paddle.enable_static()
                x = paddle.static.data(name="x", shape=[None, 1, 1], dtype='int64')
                # create word sequence
                x_emb = layers.embedding(
                        input=x,
                        size=[vocab_size, hidden_size],
                        dtype='float32',
                        is_sparse=False)
                # transform batch size to dim 1
                x_emb = paddle.transpose(x_emb, perm=[1, 0, 2])

                rnn = fluid.layers.StaticRNN()
                with rnn.step():
                        # mark created x_emb as input, each step process a word
                        word = rnn.step_input(x_emb)
                        # create prev memory parameter, batch size comes from word
                        prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                        hidden = paddle.static.nn.fc(x=[word, prev], size=hidden_size, activation='relu')
                        # use hidden to update prev
                        rnn.update_memory(prev, hidden)
                        rnn.step_output(hidden)

                result = rnn()

        """
        self._assert_in_rnn_block_('step_output')
        check_type(o, "o", Variable, "fluid.layers.StaticRNN.step_output")

        tmp_o = self.helper.create_variable_for_type_inference(dtype=o.dtype)
        self.helper.append_op(
            type='rnn_memory_helper',
            inputs={'X': [o]},
            outputs={'Out': tmp_o},
            attrs={'dtype': o.dtype},
        )

        out_var = self._parent_block().create_var(
            name=tmp_o.name,
            shape=[self.seq_len] + list(tmp_o.shape),
            dtype=tmp_o.dtype,
        )

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

                import paddle
                import paddle.fluid as fluid
                import paddle.fluid.layers as layers

                vocab_size, hidden_size=10000, 200
                paddle.enable_static()
                x = paddle.static.data(name="x", shape=[None, 1, 1], dtype='int64')
                # create word sequence
                x_emb = layers.embedding(
                        input=x,
                        size=[vocab_size, hidden_size],
                        dtype='float32',
                        is_sparse=False)
                # transform batch size to dim 1
                x_emb = paddle.transpose(x_emb, perm=[1, 0, 2])

                rnn = fluid.layers.StaticRNN()
                with rnn.step():
                        # mark created x_emb as input, each step process a word
                        word = rnn.step_input(x_emb)
                        # create prev memory parameter, batch size comes from word
                        prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                        hidden = paddle.static.nn.fc(x=[word, prev], size=hidden_size, activation='relu')
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
                           var and mem should have same dims and data type.

        Returns:
            None

        """
        check_type(mem, "mem", Variable, "fluid.layers.StaticRNN.update_memory")
        check_type(var, "var", Variable, "fluid.layers.StaticRNN.update_memory")
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

        parameters = [
            parent_block._find_var_recursive(name) for name in set(params)
        ]

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES
        )

        inlinks = [parent_block.var(i.name) for i in self.inputs]
        outlinks = self.outputs

        # NOTE(zcd): the states maybe empty in some case.
        boot_memories = []
        pre_memories = []
        memories = []
        for _, mem in self.memories.items():
            boot_memories.append(mem.init)
            pre_memories.append(mem.pre_mem.name)
            assert (
                mem.mem is not None
            ), "%s should be updated in every step." % (mem.init.name)
            mem_var = rnn_block.var(mem.mem.name)
            assert isinstance(mem_var, Variable)
            new_mem = self.helper.create_variable_for_type_inference(
                dtype=mem_var.dtype
            )
            rnn_block.append_op(
                type='rnn_memory_helper',
                inputs={'X': [mem_var]},
                outputs={'Out': [new_mem]},
                attrs={'dtype': mem_var.dtype},
            )

            memories.append(new_mem.name)

        parent_block.append_op(
            type='recurrent',
            inputs={
                'inputs': inlinks,
                'initial_states': boot_memories,
                'parameters': parameters,
            },
            outputs={'outputs': outlinks, 'step_scopes': [step_scope]},
            attrs={
                'has_states': len(pre_memories) > 0,
                'ex_states': pre_memories,
                'states': memories,
                'sub_block': rnn_block,
            },
        )


# (TODO: Mine) There exists dependency. It will be removed later.
class WhileGuard(BlockGuard):
    def __init__(self, while_op):
        if not isinstance(while_op, While):
            raise TypeError("WhileGuard takes a while op")
        super().__init__(while_op.helper.main_program)
        self.while_op = while_op

    def __enter__(self):
        self.while_op.status = While.IN_WHILE_BLOCK
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.while_op.status = While.AFTER_WHILE_BLOCK
        self.while_op._complete()
        return super().__exit__(exc_type, exc_val, exc_tb)


# (TODO: Mine) There exists dependency. It will be removed later.
def get_inputs_outputs_in_block(
    current_block, inner_inputs, inner_outputs, helper
):
    """
    Find inputs and outputs in current control flow block.
    :param current_block: Current control flow block.
    :param inner_inputs: Input var name of ops in current block.
    :param inner_outputs: Output var name of ops in current block.
    :return: inner_inputs, inner_outputs
    """

    def is_ignore_vars(op, var_name):
        # NOTE(dev): There are some persistable var created in some non-standard API
        # such as "contrib.layers.shuffle_batch". It create a "Seed" used both in
        # Input and Output. This var shall not be considered as a loop_var in
        # control_flow.
        IGNORE_VAR_NAMES = {"shuffle_batch": ["shuffle_batch_seed"]}
        if op.type in IGNORE_VAR_NAMES:
            var_names = IGNORE_VAR_NAMES[op.type]
            for name in var_names:
                if name in var_name:
                    return True
        return False

    # Step1: update inner_inputs and inner_outputs
    # NOTE: Here assumes that all variables are input or output of Ops,
    # but some variables are created without appendding a real op.
    # For example, in `arr = create_array(dtype)`, `arr` is not a output of a op.
    for op in current_block.ops:
        assert isinstance(op, Operator)
        for iname in op.input_names:
            for in_var_name in op.input(iname):
                if in_var_name not in inner_outputs and not is_ignore_vars(
                    op, in_var_name
                ):
                    inner_inputs.add(in_var_name)

        for oname in op.output_names:
            for out_var_name in op.output(oname):
                inner_outputs.add(out_var_name)

    # Step2: Remove LOD_TENSOR_ARRAY created in current control flow block.
    remove_inner_inputs = set()
    parent_block = helper.main_program.block(current_block.parent_idx)

    for in_var_name in inner_inputs:
        parent_block_var = parent_block._find_var_recursive(in_var_name)
        current_block_var = None
        if current_block.has_var(in_var_name):
            current_block_var = current_block.var(in_var_name)
        if (
            not parent_block_var
            and current_block_var
            and current_block_var.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY
        ):
            remove_inner_inputs.add(in_var_name)

    inner_inputs = inner_inputs - remove_inner_inputs

    return inner_inputs, inner_outputs


# (TODO: Mine) There exists dependency. It will be removed later.
class While:
    """
    :api_attr: Static Graph

    while loop control flow. Repeat while body until cond is False.

    Note:
        A new OP :ref:`api_fluid_layers_while_loop` is highly recommended instead of ``While`` if the shape of parameter ``cond`` is [1].
        OP :ref:`api_fluid_layers_while_loop` is easier to use and is called with less code but does the same thing as ``While`` .

    Notice:
        Local variables created in ``While`` are similar to that created in while of C++, and cannot be referenced externally.
        As a result, they cannot be obtained through ``fetch_list`` of ``Executor``. If you would like to access the variable
        out of ``while`` , PaddlePaddle provides ``assign`` API to assign local variables to external. Please refer to example
        code 2 or refer to `issue#22724 <https://github.com/PaddlePaddle/Paddle/issues/22724>`_.

    Args:
        cond(Variable): A Tensor whose data type is bool controlling whether to continue looping.
        is_test(bool, optional): A flag indicating whether execution is in test phase. Default value is False.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Examples 1:
          .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            import numpy as np

            i = paddle.full(shape=[1], dtype='int64', fill_value=0)           # loop counter

            loop_len = paddle.full(shape=[1],dtype='int64', fill_value=10)    # loop length

            cond = paddle.less_than(x=i, y=loop_len)
            while_op = fluid.layers.While(cond=cond)
            with while_op.block():
                i = paddle.increment(x=i, value=1)
                paddle.assign(paddle.less_than(x=i, y=loop_len), cond)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            res = exe.run(fluid.default_main_program(), feed={}, fetch_list=[i])
            print(res) # [array([10])]


    Examples 2:
          .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            paddle.enable_static()
            i = paddle.full(shape=[1], dtype='int64', fill_value=0)
            loop_len = paddle.full(shape=[1], dtype='int64', fill_value=10)
            one = paddle.full(shape=[1], dtype='float32', fill_value=1)
            data = paddle.static.data(name='data', shape=[1], dtype='float32')
            sums = paddle.full(shape=[1], dtype='float32', fill_value=0)  # Define the variable to be obtained ouside of While, which name should be different from the variable inside the While to be obtained

            cond = paddle.less_than(x=i, y=loop_len)
            while_op = fluid.layers.While(cond=cond)
            with while_op.block():
                sums_tensor = paddle.add(x=data, y=data)
                fluid.layers.assign(sums_tensor, sums)  # Update the value of sums_tensor defined in While to the sums which defined outside of While through layers.assign
                i = paddle.increment(x=i, value=1)
                data = paddle.add(x=data, y=one)
                paddle.assign(paddle.less_than(x=i, y=loop_len), cond)

            feed_data = np.ones(1).astype('float32')
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            res = exe.run(fluid.default_main_program(), feed={'data': feed_data}, fetch_list=sums)
            print(res[0])  # [2.]    # Because the data in While does not update the value outside the While, the value of sums is [2.] after the loop
    """

    BEFORE_WHILE_BLOCK = 0
    IN_WHILE_BLOCK = 1
    AFTER_WHILE_BLOCK = 2

    def __init__(self, cond, is_test=False, name=None):
        self.helper = LayerHelper("while", name=name)
        self.status = While.BEFORE_WHILE_BLOCK
        check_variable_and_dtype(cond, 'cond', ['bool'], 'fluid.layers.While')
        if reduce(lambda a, b: a * b, cond.shape, 1) != 1:
            raise TypeError(
                "condition expected shape as [1], but given shape as {0}.".format(
                    list(cond.shape)
                )
            )
        self.cond_var = cond
        self.is_test = is_test

    def block(self):
        return WhileGuard(self)

    def _complete(self):
        main_program = self.helper.main_program
        while_block = main_program.current_block()
        parent_block = main_program.block(
            main_program.current_block().parent_idx
        )

        inner_outputs = {self.cond_var.name}
        x_name_list = set()
        x_name_list, inner_outputs = get_inputs_outputs_in_block(
            while_block, x_name_list, inner_outputs, self.helper
        )

        out_vars = []
        for inner_out_name in inner_outputs:
            inner_var = parent_block._find_var_recursive(inner_out_name)
            if inner_var:
                out_vars.append(inner_var)

        x_name_list |= set(map(lambda x: x.name, out_vars))
        # NOTE(dev): cond_var has been contained in Input('Condition'), so
        # we remove it from Input('X')
        x_name_list -= {self.cond_var.name}

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES
        )

        parent_block.append_op(
            type='while',
            inputs={
                'X': [
                    parent_block._var_recursive(x_name)
                    for x_name in x_name_list
                ],
                'Condition': [self.cond_var],
            },
            outputs={'Out': out_vars, 'StepScopes': [step_scope]},
            attrs={'sub_block': while_block, "is_test": self.is_test},
        )


support_ret_buildin_type = (bool, float, int)


# (TODO: Mine) There exists dependency. It will be removed later.
def assign_skip_lod_tensor_array(input, output):
    """
    Assign input to output, but skip the process of copying LoDTensorArray unless it's created in while_block.
    """

    def has_shape_diff(x_var, y_var):
        if len(x_var.shape) != len(y_var.shape):
            return True
        for x_dim, y_dim in zip(x_var.shape, y_var.shape):
            if x_dim != y_dim and -1 not in [x_dim, y_dim]:
                return True
        return False

    if not isinstance(input, (Variable, core.eager.Tensor)):
        if isinstance(output, Variable) and isinstance(
            input, support_ret_buildin_type
        ):
            paddle.assign(input, output)
        else:
            output = input
        return

    if input.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        main_program = input.block.program
        parent_block = main_program.block(
            main_program.current_block().parent_idx
        )
        if parent_block and not parent_block._find_var_recursive(input.name):
            paddle.assign(input, output)
    else:
        if (
            isinstance(output, Variable)
            and isinstance(input, Variable)
            and has_shape_diff(input, output)
        ):
            warnings.warn(
                "In dy2static mode, we attemp to assign a variable with shape {} into a variable with shape{}, which is not always right.".format(
                    input.shape, output.shape
                )
            )
        paddle.assign(input, output)


# (TODO: Mine) There exists dependency (jit.dy2static.convert_operators). It will be removed later.
def while_loop(cond, body, loop_vars, is_test=False, name=None):
    """
    :api_attr: Static Graph

    while_loop is one of the control flows. Repeats while_loop `body` until `cond` returns False.

    Notice:
        Local variables defined in ``body`` cannot be obtained through ``fetch_list`` of ``Executor`` , variables should
        be defined outside ``body`` and placed in ``loop_vars`` for looping, then these variables can be fetched by ``fetch_list`` .

    Args:
        cond(Callable): A callable returning a boolean tensor controlling whether to continue looping. And ``cond`` takes
            as many arguments as ``loop_vars`` .
        body(Callable): A callable returning a tuple or list of tensors or LoDTensorArrays of the same arity
            (length and structure) and types as ``loops_vars`` . And ``body`` takes as many arguments as ``loop_vars`` .
        loop_vars(list|tuple): A list or tuple of tensors or LoDTensorArrays that is passed to both ``cond`` and ``body`` .
        is_test(bool, optional): A flag indicating whether execution is in test phase. Default value is False.
        name(str, optional): Normally there is no need for users to set this property. For more information, please
            refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        A list or tuple of Tensors or LoDTensorArrays which returned by ``body`` .

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()

            def cond(i, ten):
                return i < ten

            def body(i, ten):
                i = i + 1
                return [i, ten]

            main_program = paddle.static.default_main_program()
            startup_program = paddle.static.default_startup_program()
            with paddle.static.program_guard(main_program, startup_program):
                i = paddle.full(shape=[1], fill_value=0, dtype='int64')     # loop counter
                ten = paddle.full(shape=[1], fill_value=10, dtype='int64')  # loop length
                i, ten = paddle.static.nn.while_loop(cond, body, [i, ten])

                exe = paddle.static.Executor(paddle.CPUPlace())
                res = exe.run(main_program, feed={}, fetch_list=[i])
                print(res) # [array([10])]
    """
    helper = LayerHelper('while_loop', **locals())

    if not callable(cond):
        raise TypeError("cond in while_loop should be callable")
    if not callable(body):
        raise TypeError("body in while_loop should be callable")
    check_type(loop_vars, 'loop_vars', (list, tuple), 'fluid.layers.while_loop')
    if len(loop_vars) == 0:
        raise ValueError("loop_vars in while_loop should not be empty")

    pre_cond = cond(*loop_vars)

    if reduce(lambda a, b: a * b, pre_cond.shape, 1) != 1:
        raise TypeError(
            "the shape of the variable returned by cond should be [1],"
            "but given shape as {0}.".format(list(pre_cond.shape))
        )

    if in_dygraph_mode():
        now_cond = pre_cond.item()
        while now_cond:
            output_vars = body(*loop_vars)
            if not isinstance(output_vars, (list, tuple)):
                output_vars = [output_vars]
            if len(output_vars) != len(loop_vars):
                raise ValueError(
                    "body in while_loop should return the same arity "
                    "(length and structure) and types as loop_vars"
                )
            now_cond = cond(*output_vars).item()
            map_structure(assign_skip_lod_tensor_array, output_vars, loop_vars)
        return loop_vars
    else:
        check_variable_and_dtype(
            pre_cond,
            'var of cond returned',
            ['bool'],
            'fluid.layers.while_loop',
        )
        while_loop_block = While(pre_cond, is_test, name)
        has_mutable_vars_in_loop = hold_mutable_vars(loop_vars)
        with while_loop_block.block():
            # If a variable with mutable type is included in loop_vars, like `dict/list`,
            # modifying it in the body function will cause origin variable to be modified
            # synchronously. This will raise an assignment error out of while block.
            # Here we make a copy of the mutable vars to avoid this problem.
            if has_mutable_vars_in_loop:
                new_loop_vars = copy_mutable_vars(loop_vars)
                output_vars = body(*new_loop_vars)
            else:
                output_vars = body(*loop_vars)
            if not isinstance(output_vars, (list, tuple)):
                output_vars = [output_vars]
            try:
                loop_vars = _deal_with_undefined_var(output_vars, loop_vars)
                assert_same_structure(output_vars, loop_vars, check_types=False)
            except ValueError as e:
                raise ValueError(
                    "body in while_loop should return the same arity "
                    "(length and structure) as loop_vars: {0}".format(e)
                )
            now_cond = cond(*output_vars)
            map_structure(assign_skip_lod_tensor_array, output_vars, loop_vars)
            paddle.assign(now_cond, pre_cond)
        return loop_vars


# (TODO: Mine) There exists dependency. It will be removed later.
def _deal_with_undefined_var(output_vars, loop_vars):
    """Deal with undefined var cases, We create undefined variable based on the results of body().
    In Dy2Static, we use undefined var to represent the var created in control flow. This function
    expand the loop_vars and replace original loop_vars.
    1. UndefinedVar = Variable      # create a variable
    2. UndefinedVar = None          # create a undefined var with RETURN_NO_VALUE_MAGIC_NUM
    3. UndefinedVar = List(int)     # create a list of variable
    4. UndefinedVar = value         # create a variable
    """
    from paddle.jit.dy2static.utils import (
        UndefinedVar,
        create_undefined_variable,
    )

    def create_var_like(o_var):
        if (
            isinstance(o_var, (Variable,) + support_ret_buildin_type)
            or o_var is None
        ):
            return create_undefined_variable()
        if is_sequence(o_var):
            """
            Create a complex container class inside the body of while, including Python list and python Dict
            """
            return map_structure(lambda x: create_undefined_variable(), o_var)

    if len(output_vars) != len(loop_vars):
        raise ValueError("The length of loop_vars should be the same.")

    results = []
    for o_var, l_var in zip(output_vars, loop_vars):
        if isinstance(l_var, UndefinedVar) or l_var is None:
            results.append(create_var_like(o_var))
        else:
            results.append(l_var)
    return results


class ConditionalBlockGuard(BlockGuard):
    """
    ConditionalBlockGuard is derived from BlockGuard. It is dedicated for
    holding a ConditionalBlock, and helping users entering and exiting the
    ConditionalBlock via Python's 'with' keyword. However, ConditionalBlockGuard
    is generally an internal component of IfElse, users should not use it directly.
    """

    def __init__(self, block):
        check_type(block, "block", ConditionalBlock, "ConditionalBlockGuard")
        super().__init__(block.helper.main_program)
        self.block = block

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.block.complete()
        return super().__exit__(exc_type, exc_val, exc_tb)


class ConditionalBlock:
    '''
    **ConditionalBlock**

    ConditionalBlock is an operator that bind a block to a specific condition,
    if the condition matches, the corresponding block will be executed.

    Args:
        inputs (Variable): bool conditions.
        is_scalar_condition (bool): whether the branch is controlled by a scalar.
        name(str): name of this ConditionalBlock.

    Examples:
        .. code-block:: python

             import paddle
             import paddle.fluid as fluid
             cond = paddle.less_than(x=label, y=limit)
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
            check_type(each_input, "input", Variable, "ConditionalBlock")
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
        params, intermediate = get_inputs_outputs_in_block(
            inside_block, params, intermediate, helper=self.helper
        )

        # Todo(liym27) Here assume that all params are in recursive parent block
        # but when minimize() called in control flow, some params may be in
        # conditional grad block
        param_list = [
            parent_block._var_recursive(each_name) for each_name in params
        ]

        out_list = []
        for inner_out_name in intermediate:
            inner_var = parent_block._find_var_recursive(inner_out_name)
            if inner_var:
                out_list.append(inner_var)

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES
        )
        conditional_block_op = parent_block.append_op(
            type='conditional_block',
            inputs={
                'Cond': self.inputs,
                'Input': param_list,
            },
            outputs={'Out': out_list, 'Scope': [step_scope]},
            attrs={
                'sub_block': inside_block,
                'is_scalar_condition': self.is_scalar_condition,
            },
        )

        if self.need_append_conditional_block_grad(inside_block):
            self.append_conditional_block_grad(
                parent_block, inside_block, conditional_block_op
            )

    def need_append_conditional_block_grad(self, inside_block):
        grad_sub_block_idx = inside_block.backward_block_idx
        inside_block_idx = inside_block.idx

        # if inside_block have grad_block and grad_block is not itself,
        # we will append conditional block grad.
        return (
            grad_sub_block_idx != -1 and grad_sub_block_idx != inside_block_idx
        )

    def append_conditional_block_grad(
        self, parent_block, inside_block, conditional_block_op
    ):
        '''
        Append op `conditional_block_grad` manually.
        When `optimizer.minimize/append_backward` is called in Paddle control flow,
        grad ops will be appended before appending op `conditional_block` so that
        op `conditional_block_grad` can't be appended when calling
        `optimizer.minimize/append_backward`. After appending op `conditional_block`,
        `conditional_block_grad` is appended manually.

        Args:
            parent_block (Block): The block that `conditional_block_op` blongs to.
            inside_block (Block): The sub block of `conditional_block_op`.
            conditional_block_op (Operator): The forward op conditional_block.
        '''

        grad_sub_block_idx = inside_block.backward_block_idx
        grad_sub_block = self.helper.main_program.block(grad_sub_block_idx)

        intermediate = set()
        params = set()

        for each_op in grad_sub_block.ops:
            assert isinstance(each_op, Operator)
            for iname in each_op.input_names:
                for in_var_name in each_op.input(iname):
                    if in_var_name not in intermediate:
                        params.add(in_var_name)

            for oname in each_op.output_names:
                for out_var_name in each_op.output(oname):
                    intermediate.add(out_var_name)

        param_list = []
        for inner_input_name in params:
            inner_var = parent_block._find_var_recursive(inner_input_name)
            if inner_var:
                param_list.append(inner_var.name)

        grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
            conditional_block_op.desc, set(), [grad_sub_block.desc]
        )

        # append op_desc in grad_op_descs to target_block
        op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
        backward = core.op_proto_and_checker_maker.OpRole.Backward
        new_op_desc = parent_block.desc.append_op()
        new_op_desc.copy_from(grad_op_desc[0])
        new_op_desc._set_attr(op_role_attr_name, backward)
        # set input and output manually
        new_op_desc.set_input('Input', param_list)
        new_op_desc.set_output(
            'Input@GRAD', [param + "@GRAD" for param in param_list]
        )

        new_vars = set()
        for grad_var_name in new_op_desc.output_arg_names():
            if (
                grad_sub_block.desc.has_var_recursive(grad_var_name.encode())
                or grad_var_name == core.empty_var_name()
            ):
                continue
            grad_sub_block.desc.var(grad_var_name.encode())
            new_vars.add(grad_var_name)
            if grad_var_name not in op_grad_to_var:
                continue

        # infer_shape and infer_type
        new_op_desc.infer_var_type(grad_sub_block.desc)
        new_op_desc.infer_shape(grad_sub_block.desc)

        for arg in new_op_desc.output_arg_names():
            if arg in new_vars:
                _infer_var_data_type_shape_(arg, grad_sub_block)

        self.helper.main_program._sync_with_cpp()


def _to_sequence_except_dict(x):
    """
    In this function, dict is not viewed as sequence.
    """
    if isinstance(x, dict):
        return [x]
    return to_sequence(x)


def _is_sequence_except_dict(x):
    """
    In this function, dict is not viewed as sequence.
    """
    if isinstance(x, dict):
        return False
    return is_sequence(x)


def expand_undefined_var(nest1, nest2, names):
    """TODO: make this function recursively.
    nest1: Var1, (UndefinedVar, [1,2,3])
    nest2: Var2, ([1,2,3,4], UndefinedVar)
    In this case, we should not expand recursively.
    """
    from paddle.jit.dy2static.utils import UndefinedVar
    from paddle.jit.dy2static.return_transformer import (
        RETURN_VALUE_PREFIX,
    )

    def pack_undefined_var_as(seq):
        return pack_sequence_as(
            seq, [UndefinedVar("padding") for i in flatten(seq)]
        )

    def map_fn(n1, n2, name, order):
        if not name.startswith(RETURN_VALUE_PREFIX) and (
            isinstance(n1, UndefinedVar) or n1 is None
        ):
            if n1 is None and n2 is not None:
                if order == 0:
                    warnings.warn(
                        "In cond : Var '{}' or part of it is set differently in ifelse branchs, "
                        "<{}, {}> in true branch and <{}, {}> in false branch. Set var to "
                        "'None' in ifelse block might lead to error.".format(
                            name, type(n1), n1, type(n2), n2
                        )
                    )
                else:
                    warnings.warn(
                        "In cond : Var '{}' or part of it is set differently in ifelse branchs, "
                        "<{}, {}> in true branch and <{}, {}> in false branch. Set var to "
                        "'None' in ifelse block might lead to error.".format(
                            name, type(n2), n2, type(n1), n1
                        )
                    )
            return pack_undefined_var_as(n2)
        return n1

    nest1_out = list(
        map(
            map_fn,
            _to_sequence_except_dict(nest1),
            _to_sequence_except_dict(nest2),
            _to_sequence_except_dict(names),
            [0 for i in _to_sequence_except_dict(names)],
        )
    )
    nest2_out = list(
        map(
            map_fn,
            _to_sequence_except_dict(nest2),
            _to_sequence_except_dict(nest1),
            _to_sequence_except_dict(names),
            [1 for i in _to_sequence_except_dict(names)],
        )
    )
    if not _is_sequence_except_dict(nest1):
        nest1_out = nest1_out[0]
    if not _is_sequence_except_dict(nest2):
        nest2_out = nest2_out[0]
    return nest1_out, nest2_out


# TODO: It will be deleted later.
class Switch:
    """
    :api_attr: Static Graph

    This class is used to implement Switch branch control function.
    Switch branch contains several case branches and one default branch.
    Switch control flow checks whether the case branch conditions are satisfied in turn,
    and only executes the statement after the first case branch that satisfies the conditions.
    If there is no case branch that satisfies the condition,
    only the statement following the default branch is executed.

    Note:
        A new OP :ref:`api_fluid_layers_case` is highly recommended instead of ``Switch`` if the shape of parameter ``cond`` is [1].
        OP :ref:`api_fluid_layers_case` is easier to use and is called with less code but does the same thing as ``Switch`` .

    Member Functions:
        case(condition): The case branch of Switch whose parameter cond is a scalar Variable of bool type. Only if the cond of the current case branch is True and the cond of the previous case branch is False, the statement after the case branch will be executed, and the statement after the case branch will not be executed.

        default(): The default branch of Switch. When cond of all case branches is False, the statement after default branch is executed.

    Case and default functions can only be used inside the scope of Switch, as shown below:

    .. code-block:: python

        '''
        import paddle
        import paddle.fluid as fluid
        with fluid.layers.Switch() as switch:
            with switch.case(cond1):
                i = paddle.full(shape=[1], dtype='int64', fill_value=1)
            with switch.case(cond2):
                i = paddle.full(shape=[1], dtype='int64', fill_value=2)
            with switch.default():
                i = paddle.full(shape=[1], dtype='int64', fill_value=0)
        '''

    Args:
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            lr = paddle.static.create_global_var(
                shape=[1],
                value=0.0,
                dtype='float32',
                persistable=True,
                name="learning_rate")
            zero_var = paddle.full(
                shape=[1], dtype='float32', fill_value=0.0)
            one_var = paddle.full(
                shape=[1], dtype='float32', fill_value=1.0)
            two_var = paddle.full(
                shape=[1], dtype='float32', fill_value=2.0)

            global_step = fluid.layers.autoincreased_step_counter(counter_name='@LR_DECAY_COUNTER@', begin=0, step=1)

            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(global_step == zero_var):
                    paddle.assign(input=one_var, output=lr)
                with switch.default():
                    paddle.assign(input=two_var, output=lr)

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

        check_variable_and_dtype(
            condition,
            'condition',
            ['bool'],
            'the member function case of fluid.layers.Switch',
        )

        if len(self.pre_not_conditions) == 0:
            cond_block = ConditionalBlock([condition], is_scalar_condition=True)
            not_cond = paddle.logical_not(x=condition)
            self.pre_not_conditions.append(not_cond)
        else:
            pre_cond_num = len(self.pre_not_conditions)
            pre_not_cond = self.pre_not_conditions[pre_cond_num - 1]
            new_not_cond = paddle.logical_and(
                x=pre_not_cond, y=paddle.logical_not(x=condition)
            )
            self.pre_not_conditions.append(new_not_cond)
            cond_block = ConditionalBlock(
                [paddle.logical_and(x=pre_not_cond, y=condition)],
                is_scalar_condition=True,
            )

        return ConditionalBlockGuard(cond_block)

    def default(self):
        pre_cond_num = len(self.pre_not_conditions)
        if pre_cond_num == 0:
            raise ValueError("there should be at least one condition")
        cond_block = ConditionalBlock(
            [self.pre_not_conditions[pre_cond_num - 1]],
            is_scalar_condition=True,
        )
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
