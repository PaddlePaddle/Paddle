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
]


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
