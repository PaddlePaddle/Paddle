#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import warnings
from functools import cached_property, partial, reduce
from typing import Any

import paddle
from paddle import _C_ops
from paddle.base import core
from paddle.base.backward import _infer_var_data_type_shape_
from paddle.base.framework import (
    Operator,
    Program,
    Variable,
    in_pir_mode,
    static_only,
)
from paddle.base.libpaddle.pir import (
    build_assert_op,
    build_if_op,
    build_while_op,
    cf_yield,
)
from paddle.common_ops_import import (
    LayerHelper,
    check_type,
    check_variable_and_dtype,
    convert_dtype,
    in_dygraph_mode,
)
from paddle.framework import use_pir_api
from paddle.pir.core import _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE
from paddle.utils import (
    assert_same_structure,
    copy_mutable_vars,
    flatten,
    hold_mutable_vars,
    is_sequence,
    map_structure,
    pack_sequence_as,
    to_sequence,
)


def Assert(cond, data=None, summarize=20, name=None):
    '''
    This API creates an op that asserts the given condition is true. If the
    condition is false, prints the tensors in data. ``summarize`` specifies the
    number of the elements in the tensors to print.

    Args:
        cond (Tensor): The boolean condition tensor whose numel should be 1.
        data (list|tuple, optional): list or tuple of tensors to print when
            condition is not true. If it's ``None``, no tensor will be printed.
            The default value is ``None``.
        summarize (int, optional): Number of elements in the tensor to be
            printed. If its value is -1, then all elements in the tensor will
            be printed. The default value is 20.
        name (str, optional): The default value is ``None`` . Normally users
            don't have to set this parameter. For more information, please
            refer to :ref:`api_guide_Name` .

    Returns:
        Operator: the created operation.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.static.nn.control_flow import Assert

            >>> paddle.enable_static()
            >>> x = paddle.full([2, 3], 2.0, 'float32')
            >>> condition = paddle.max(x) < 1.0 # False
            >>> Assert(condition, [x], 10, "example_assert_layer")

            >>> exe = paddle.static.Executor()
            >>> try:
            ...     exe.run(paddle.static.default_main_program())
            ...     # Print x and throws ValueError
            ...     # Example printed message for x:
            ...     #
            ...     # Variable: fill_constant_0.tmp_0
            ...     #   - lod: {}
            ...     #   - place: CPUPlace()
            ...     #   - shape: [2, 3]
            ...     #   - layout: NCHW
            ...     #   - dtype: float
            ...     #   - data: [2 2 2 2 2 2]
            ... except ValueError as e:
            ...     print("Assert Exception Example")

    '''
    check_variable_and_dtype(
        cond, "cond", ["bool"], "static.nn.control_flow.Assert"
    )
    check_type(
        data, "data", (list, tuple, type(None)), "static.nn.control_flow.Assert"
    )
    check_type(summarize, "summarize", int, "static.nn.control_flow.Assert")
    check_type(name, "name", (str, type(None)), "static.nn.control_flow.Assert")

    if in_pir_mode():
        input_data = [] if data is None else list(data)
        assert_op = build_assert_op(cond, input_data, summarize)
        return

    layer_name = name if name else ('assert_' + cond.name)
    helper = LayerHelper(layer_name, **locals())

    op = helper.append_op(
        type="assert",
        inputs={"Cond": cond, "Data": [] if data is None else list(data)},
        attrs={"summarize": summarize},
    )

    return op


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


class WhileGuard(BlockGuard):
    def __init__(self, while_op):
        if not isinstance(while_op, While):
            raise TypeError("WhileGuard takes a while op")
        if not in_pir_mode():
            super().__init__(while_op.helper.main_program)
        self.while_op = while_op

    def __enter__(self):
        if in_pir_mode():
            self.block = build_while_op(self.while_op.cond_var, []).body()
            return self.block.__enter__()
        self.while_op.status = While.IN_WHILE_BLOCK
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if in_pir_mode():
            cf_yield([self.while_op.cond_var])
            return self.block.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is not None:
            return False
        self.while_op.status = While.AFTER_WHILE_BLOCK
        self.while_op._complete()
        return super().__exit__(exc_type, exc_val, exc_tb)


class If:
    '''
    **If**

    If is an operator that bind two blocks (true_block and false_block) to a specific condition,
    According to the condition, the corresponding block will be executed.

    Args:
        cond (Value): A value whose data type is bool controlling which block is executed.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.static.nn.control_flow import ConditionalBlock

            >>> label = paddle.rand([1])
            >>> limit = paddle.ones([1]) * 0.5
            >>> cond = paddle.less_than(x=label, y=limit)
            >>> if_op = If(cond)
            >>> with if_op.true_block():
            ...     pass
            >>> with if_op.false_block():
            ...     pass
    '''

    def __init__(self, cond):
        if not isinstance(cond, list):
            check_variable_and_dtype(cond, 'cond', ['bool'], 'static.nn.If')
            if reduce(lambda a, b: a * b, cond.shape, 1) != 1:
                raise TypeError(
                    f"condition expected shape as [1], but given shape as {list(cond.shape)}."
                )
        self.if_op = build_if_op(cond)
        self.cond_var = self.if_op.cond()

    def true_block(self):
        return self.if_op.true_block()

    def false_block(self):
        return self.if_op.false_block()


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

            >>> import paddle
            >>> from paddle.static.nn.control_flow import ConditionalBlock

            >>> label = paddle.rand([1])
            >>> limit = paddle.ones([1]) * 0.5
            >>> cond = paddle.less_than(x=label, y=limit)
            >>> image = paddle.ones([1])

            >>> true_image = image[cond]
            >>> true_cond = ConditionalBlock([true_image])

            >>> with true_cond.block():
            ...     pass
            >>> with false_cond.block():
            ...     pass
    '''

    def __init__(self, inputs, is_scalar_condition=False, name=None):
        self.inputs = inputs
        if in_pir_mode():
            if is_scalar_condition and len(inputs) != 1:
                raise TypeError(
                    "For ConditionalBlock Api,  Only support one input while is_scalar_condition is True"
                )
            return
        else:
            for each_input in inputs:
                check_type(each_input, "input", Variable, "ConditionalBlock")

        self.is_scalar_condition = is_scalar_condition
        self.helper = LayerHelper('conditional_block', name=name)

    def block(self):
        if in_pir_mode():
            return If(self.inputs).true_block()
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
            parent_block (Block): The block that `conditional_block_op` belongs to.
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
    # but some variables are created without appending a real op.
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

    # Step2: Remove DENSE_TENSOR_ARRAY created in current control flow block.
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
            and current_block_var.type
            == core.VarDesc.VarType.DENSE_TENSOR_ARRAY
        ):
            remove_inner_inputs.add(in_var_name)

    inner_inputs = inner_inputs - remove_inner_inputs

    return inner_inputs, inner_outputs


class While:
    """
    :api_attr: Static Graph

    while loop control flow. Repeat while body until cond is False.

    Note:
        A new OP :ref:`api_paddle_static_nn_while_loop` is highly recommended instead of ``While`` if the shape of parameter ``cond`` is [1].
        OP :ref:`api_paddle_static_nn_while_loop` is easier to use and is called with less code but does the same thing as ``While`` .

    Notice:
        Local variables created in ``While`` are similar to that created in while of C++, and cannot be referenced externally.
        As a result, they cannot be obtained through ``fetch_list`` of ``Executor``. If you would like to access the variable
        out of ``while`` , PaddlePaddle provides ``assign`` API to assign local variables to external. Please refer to example
        code 2 or refer to `issue#22724 <https://github.com/PaddlePaddle/Paddle/issues/22724>`_.

    Args:
        cond(Variable): A Tensor whose data type is bool controlling whether to continue looping.
        is_test(bool, optional): A flag indicating whether execution is in test phase. Default value is False.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python
            :name: example-1

            >>> import paddle
            >>> import numpy as np

            >>> paddle.enable_static()

            >>> i = paddle.full(shape=[1], dtype='int64', fill_value=0)           # loop counter

            >>> loop_len = paddle.full(shape=[1],dtype='int64', fill_value=10)    # loop length

            >>> cond = paddle.less_than(x=i, y=loop_len)
            >>> while_op = paddle.static.nn.control_flow.While(cond=cond)
            >>> with while_op.block():
            ...     i = paddle.increment(x=i, value=1)
            ...     paddle.assign(paddle.less_than(x=i, y=loop_len), output=cond)

            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> exe.run(paddle.static.default_startup_program())

            >>> res = exe.run(paddle.static.default_main_program(), feed={}, fetch_list=[i])
            >>> print(res)
            [array([10], dtype=int64)]

        .. code-block:: python
            :name: example-2

            >>> import paddle
            >>> import numpy as np

            >>> paddle.enable_static()

            >>> i = paddle.full(shape=[1], dtype='int64', fill_value=0)
            >>> loop_len = paddle.full(shape=[1], dtype='int64', fill_value=10)
            >>> one = paddle.full(shape=[1], dtype='float32', fill_value=1)
            >>> data = paddle.static.data(name='data', shape=[1], dtype='float32')
            >>> sums = paddle.full(shape=[1], dtype='float32', fill_value=0)  # Define the variable to be obtained outside of While, which name should be different from the variable inside the While to be obtained

            >>> cond = paddle.less_than(x=i, y=loop_len)
            >>> while_op = paddle.static.nn.control_flow.While(cond=cond)
            >>> with while_op.block():
            ...     sums_tensor = paddle.add(x=data, y=data)
            ...     paddle.assign(sums_tensor, sums)  # Update the value of sums_tensor defined in While to the sums which defined outside of While through layers.assign
            ...     i = paddle.increment(x=i, value=1)
            ...     data = paddle.add(x=data, y=one)
            ...     paddle.assign(paddle.less_than(x=i, y=loop_len), output=cond)

            >>> feed_data = np.ones(1).astype('float32')
            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> exe.run(paddle.static.default_startup_program())
            >>> res = exe.run(paddle.static.default_main_program(), feed={'data': feed_data}, fetch_list=sums)
            >>> print(res[0]) # Because the data in While does not update the value outside the While, the value of sums is [2.] after the loop
            [2.]
    """

    BEFORE_WHILE_BLOCK = 0
    IN_WHILE_BLOCK = 1
    AFTER_WHILE_BLOCK = 2

    def __init__(self, cond, is_test=False, name=None):
        self.cond_var = cond
        check_variable_and_dtype(cond, 'cond', ['bool'], 'static.nn.While')
        if reduce(lambda a, b: a * b, cond.shape, 1) != 1:
            raise TypeError(
                f"condition expected shape as [1], but given shape as {list(cond.shape)}."
            )
        if in_pir_mode():
            return
        self.status = While.BEFORE_WHILE_BLOCK
        self.helper = LayerHelper("while", name=name)
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

        x_name_list |= {x.name for x in out_vars}
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


def assign_skip_lod_tensor_array(input, output):
    """
    Assign input to output, but skip the process of copying DenseTensorArray unless it's created in while_block.
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

    if input.type == core.VarDesc.VarType.DENSE_TENSOR_ARRAY:
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
                f"In dy2static mode, we attempt to assign a variable with shape {input.shape} into a variable with shape{output.shape}, which is not always right."
            )
        # NOTE(dev): Avoid assign if input is output in Variable level which means
        # input is not generated in While sub block and modified by in-place and only
        # belong to inplace ops in constructing program process, because in-place pass
        # is only available in Graph level.
        with paddle.base.framework._stride_in_no_check_dy2st_diff():
            paddle.assign(input, output)


def create_fake_value_for_undefined_var(while_op, value):
    # Create a fake value for create WhileOp, and set its type and stop_gradient as next_var
    stop_gradient = value.stop_gradient
    fake_value = paddle.full(shape=[], dtype=value.dtype, fill_value=0)
    fake_value_op = fake_value.get_defining_op()
    fake_value_op.move_before(while_op.as_operation())

    fake_value.set_type(value.type())
    fake_value.stop_gradient = stop_gradient
    while_op.add_extra_input(fake_value)

    block_arg = while_op.body().add_arg(value.type())
    block_arg.stop_gradient = stop_gradient
    return fake_value, block_arg


class LoopVar:
    def __init__(self, curr_var, next_var=None, block_arg=None):
        self.curr_var = curr_var
        self.next_var = next_var
        self.block_arg = block_arg
        self._is_fake = False

    @property
    def is_variable_curr_var(self):
        return isinstance(self.curr_var, paddle.pir.Value)

    @property
    def is_undefined_curr_var(self):
        return isinstance(
            self.curr_var, paddle.jit.dy2static.utils.UndefinedVar
        )

    @property
    def is_variable_next_var(self):
        return isinstance(self.next_var, paddle.pir.Value)

    @property
    def is_fake(self):
        return self._is_fake

    def bind_block_arg(self, block_arg):
        self.block_arg = block_arg

    def bind_next_var(self, next_var):
        self.next_var = next_var

    def infer_type_with_next_var(self, next_var, while_op):
        assert self.is_undefined_curr_var

        def create_loop_var_like(while_op, next_var):
            fake_value, block_arg = create_fake_value_for_undefined_var(
                while_op, next_var
            )
            loop_var = LoopVar(fake_value, next_var, block_arg)
            loop_var._is_fake = True
            return loop_var

        if isinstance(next_var, paddle.pir.Value):
            return create_loop_var_like(while_op, next_var)
        if is_sequence(next_var):
            return map_structure(
                lambda var: self.infer_type_with_next_var(var, while_op),
                next_var,
            )
        return LoopVar(self.curr_var, next_var, self.block_arg)

    def __repr__(self):
        return f"LoopVar(curr_var={self.curr_var}, next_var={self.next_var}, block_arg={self.block_arg})"


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
        body(Callable): A callable returning a tuple or list of tensors or DenseTensorArrays of the same arity
            (length and structure) and types as ``loops_vars`` . And ``body`` takes as many arguments as ``loop_vars`` .
        loop_vars(list|tuple): A list or tuple of tensors or DenseTensorArrays that is passed to both ``cond`` and ``body`` .
        is_test(bool, optional): A flag indicating whether execution is in test phase. Default value is False.
        name(str, optional): Normally there is no need for users to set this property. For more information, please
            refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        A list or tuple of Tensors or DenseTensorArrays which returned by ``body`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> def cond(i, ten):
            ...     return i < ten

            >>> def body(i, ten):
            ...     i = i + 1
            ...     return [i, ten]

            >>> main_program = paddle.static.default_main_program()
            >>> startup_program = paddle.static.default_startup_program()
            >>> with paddle.static.program_guard(main_program, startup_program):
            ...     i = paddle.full(shape=[1], fill_value=0, dtype='int64')     # loop counter
            ...     ten = paddle.full(shape=[1], fill_value=10, dtype='int64')  # loop length
            ...     i, ten = paddle.static.nn.while_loop(cond, body, [i, ten])

            ...     exe = paddle.static.Executor(paddle.CPUPlace())
            ...     res = exe.run(main_program, feed={}, fetch_list=[i])
            ...     print(res)
            [array([10], dtype=int64)]
    """
    if not callable(cond):
        raise TypeError("cond in while_loop should be callable")
    if not callable(body):
        raise TypeError("body in while_loop should be callable")
    check_type(loop_vars, 'loop_vars', (list, tuple), 'static.nn.while_loop')
    if len(loop_vars) == 0:
        raise ValueError("loop_vars in while_loop should not be empty")

    pre_cond = cond(*loop_vars)

    check_variable_and_dtype(
        pre_cond, 'var of cond returned', ['bool'], 'static.nn.while_loop'
    )
    if reduce(lambda a, b: a * b, pre_cond.shape, 1) != 1:
        raise TypeError(
            "the shape of the variable returned by cond should be [1],"
            f"but given shape as {list(pre_cond.shape)}."
        )

    if in_pir_mode():

        def cast_value_in_amp(loop_var):
            amp_attrs = core._get_amp_attrs()
            amp_level = amp_attrs._amp_level
            apply_amp_level_list = [
                core.AmpLevel.O1,
                core.AmpLevel.O2,
            ]
            if amp_level not in apply_amp_level_list:
                return
            if not loop_var.is_variable_curr_var:
                return
            if loop_var.curr_var.dtype != loop_var.next_var.dtype:
                cast_out_var = paddle.cast(
                    loop_var.next_var, loop_var.curr_var.dtype
                )
                loop_var.next_var = cast_out_var

        loop_vars: Any = map_structure(LoopVar, loop_vars)
        variable_loop_vars = [
            loop_var
            for loop_var in flatten(loop_vars)
            if loop_var.is_variable_curr_var
        ]
        while_op = build_while_op(
            pre_cond, [var.curr_var for var in variable_loop_vars]
        )
        with while_op.body() as cur_block:
            assert len(cur_block.args()) == len(variable_loop_vars)
            for loop_var, arg in zip(variable_loop_vars, cur_block.args()):
                loop_var.bind_block_arg(arg._clone())

            # For non-variable inputs, we use the original value directly.
            args = map_structure(
                lambda var: (
                    var.block_arg if var.is_variable_curr_var else var.curr_var
                ),
                loop_vars,
            )
            next_vars = body(*args)

            if not isinstance(next_vars, (list, tuple)):
                next_vars = [next_vars]

            def infer_loop_vars_type_with_next_vars(loop_vars, next_vars):
                def infer_loop_var_type_with_next_var(loop_var, next_var):
                    if is_sequence(loop_var):
                        return map_structure(
                            infer_loop_var_type_with_next_var,
                            loop_var,
                            next_var,
                        )
                    if loop_var.is_undefined_curr_var:
                        new_loop_var = loop_var.infer_type_with_next_var(
                            next_var, while_op
                        )
                    else:
                        loop_var.bind_next_var(next_var)
                        new_loop_var = loop_var
                    return new_loop_var

                new_loop_vars = []
                for next_var, loop_var in zip(next_vars, loop_vars):
                    new_loop_vars.append(
                        infer_loop_var_type_with_next_var(loop_var, next_var)
                    )
                return new_loop_vars

            try:
                assert_same_structure(
                    loop_vars,
                    next_vars,
                    check_types=False,
                    skip_if=lambda x: (
                        isinstance(x, LoopVar)
                        and isinstance(
                            x.curr_var, paddle.jit.dy2static.utils.UndefinedVar
                        )
                    )
                    or (isinstance(x, paddle.jit.dy2static.utils.UndefinedVar)),
                )
            except ValueError as e:
                raise ValueError(
                    "body in while_loop should return the same arity "
                    f"(length and structure) as loop_vars: {e}"
                )

            loop_vars = infer_loop_vars_type_with_next_vars(
                loop_vars, next_vars
            )

            from paddle.jit.dy2static.convert_operators import (
                to_static_variable,
            )

            def check_next_var(loop_var):
                if not loop_var.is_variable_curr_var:
                    return
                if not isinstance(
                    loop_var.next_var, paddle.pir.Value
                ) and not isinstance(loop_var.next_var, (bool, float, int)):
                    raise ValueError(
                        "The loop var in the while op is variable, but the corresponding yielded var is not variable, and it is not a constant of type bool, int, or float."
                    )
                loop_var.next_var = to_static_variable(loop_var.next_var)

            paddle.utils.map_structure(check_next_var, loop_vars)

            next_cond = cond(
                *map_structure(lambda var: var.next_var, loop_vars)
            )
            next_cond.stop_gradient = True

            # Filter out the constants from next_vars, we only pass the variables (Value) into cf_yield.
            # And pass the original fake value directly to constants position.
            map_structure(cast_value_in_amp, loop_vars)
            # Move all Fake Value to the end of next_vars
            variable_loop_vars = list(
                filter(
                    lambda var: var.is_variable_curr_var and not var.is_fake,
                    flatten(loop_vars),
                ),
            ) + list(
                filter(
                    lambda var: var.is_variable_curr_var and var.is_fake,
                    flatten(loop_vars),
                ),
            )
            cf_yield([next_cond, *(var.next_var for var in variable_loop_vars)])

        # Restore the outputs by variable and constants
        optimized_results = while_op.optimize_update()
        assert len(optimized_results) == len(variable_loop_vars)
        for loop_var, result in zip(variable_loop_vars, optimized_results):
            loop_var.next_var = result

        # Prune unused fake values
        for loop_var in flatten(loop_vars):
            if loop_var.is_fake and loop_var.curr_var.use_empty():
                fake_value_def_op = loop_var.curr_var.get_defining_op()
                fake_value_def_op.get_parent_block().remove_op(
                    fake_value_def_op
                )

        return map_structure(
            lambda var: var.next_var,
            loop_vars,
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
                f"(length and structure) as loop_vars: {e}"
            )
        now_cond = cond(*output_vars)
        map_structure(assign_skip_lod_tensor_array, output_vars, loop_vars)
        paddle.assign(now_cond, pre_cond)
    return loop_vars


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
            isinstance(o_var, (Variable, *support_ret_buildin_type))
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


def _error_message(what, arg_name, op_name, right_value, error_value):
    error_message = (
        f"{what} of '{arg_name}' in {op_name} must be "
        f"{right_value}, but received: {error_value}."
    )

    return error_message


def case(pred_fn_pairs, default=None, name=None):
    '''
    :api_attr: Static Graph

    This operator works like an if-elif-elif-else chain.

    Args:
        pred_fn_pairs(list|tuple): A list or tuple of (pred, fn) pairs. ``pred`` is a boolean Tensor whose numel should be 1 (shape [] or shape [1]), ``fn`` is a callable. All callables return the same structure of Tensors.
        default(callable, optional): Callable that returns a structure of Tensors.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor|list(Tensor): Tensors returned by the callable from the first pair whose pred is True,
        or Tensors returned by ``default`` if no pred in ``pred_fn_pairs`` is True and ``default`` is not None,
        or Tensors returned by the last callable in ``pred_fn_pairs``  if no pred in ``pred_fn_pairs`` is True and ``default`` is None.

    Raises:
        TypeError: If the type of ``pred_fn_pairs`` is not list or tuple.
        TypeError: If the type of elements in ``pred_fn_pairs`` is not tuple.
        TypeError: If the size of tuples in ``pred_fn_pairs`` is not 2.
        TypeError: If the first element of 2-tuple in ``pred_fn_pairs`` is not a Tensor.
        TypeError: If the second element of 2-tuple in ``pred_fn_pairs`` is not callable.
        TypeError: If ``default`` is not None but it is not callable.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> def fn_1():
            ...     return paddle.full(shape=[1, 2], dtype='float32', fill_value=1)

            >>> def fn_2():
            ...     return paddle.full(shape=[2, 2], dtype='int32', fill_value=2)

            >>> def fn_3():
            ...     return paddle.full(shape=[3], dtype='int32', fill_value=3)

            >>> main_program = paddle.static.default_startup_program()
            >>> startup_program = paddle.static.default_main_program()

            >>> with paddle.static.program_guard(main_program, startup_program):
            ...     x = paddle.full(shape=[1], dtype='float32', fill_value=0.3)
            ...     y = paddle.full(shape=[1], dtype='float32', fill_value=0.1)
            ...     z = paddle.full(shape=[1], dtype='float32', fill_value=0.2)

            ...     pred_1 = paddle.less_than(z, x)  # true: 0.2 < 0.3
            ...     pred_2 = paddle.less_than(x, y)  # false: 0.3 < 0.1
            ...     pred_3 = paddle.equal(x, y)      # false: 0.3 == 0.1

            ...     # Call fn_1 because pred_1 is True
            ...     out_1 = paddle.static.nn.case(
            ...         pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3)

            ...     # Argument default is None and no pred in pred_fn_pairs is True. fn_3 will be called.
            ...     # because fn_3 is the last callable in pred_fn_pairs.
            ...     out_2 = paddle.static.nn.case(pred_fn_pairs=[(pred_2, fn_2), (pred_3, fn_3)])

            ...     exe = paddle.static.Executor(paddle.CPUPlace())
            ...     res_1, res_2 = exe.run(main_program, fetch_list=[out_1, out_2])
            ...     print(res_1, res_2)
            [[1. 1.]] [3 3 3]
    '''

    def _case_check_args(pred_fn_pairs, default):
        '''
        Check arguments pred_fn_pairs and default. Return canonical pre_fn_pairs and default.
        '''
        check_type(pred_fn_pairs, 'pred_fn_pairs', (list, tuple), 'case')

        for pred_fn in pred_fn_pairs:
            if not isinstance(pred_fn, tuple):
                raise TypeError(
                    _error_message(
                        "The elements' type",
                        "pred_fn_pairs",
                        "case",
                        tuple,
                        type(pred_fn),
                    )
                )
            if len(pred_fn) != 2:
                raise TypeError(
                    _error_message(
                        "The tuple's size",
                        "pred_fn_pairs",
                        "case",
                        "2",
                        str(len(pred_fn)) + "-tuple",
                    )
                )
            pred, fn = pred_fn

            check_variable_and_dtype(
                pred, 'pred', ['bool'], 'paddle.static.nn.case'
            )

            if not callable(fn):
                raise TypeError(
                    "The fn of pred_fn_pairs in Op(case) must" " be callable."
                )

        if default is None:
            default_index = len(pred_fn_pairs) - 1  # pick the last one
            default = pred_fn_pairs[default_index][1]
            pred_fn_pairs = pred_fn_pairs[:default_index]
        elif not callable(default):
            raise TypeError("The default in Op(case) must be callable.")

        return pred_fn_pairs, default

    pred_fn_pairs, default = _case_check_args(pred_fn_pairs, default)

    false_fn = default
    for pred, true_fn in reversed(pred_fn_pairs):
        false_fn = partial(cond, pred=pred, true_fn=true_fn, false_fn=false_fn)

    final_fn = false_fn

    return final_fn()


def switch_case(branch_index, branch_fns, default=None, name=None):
    '''
    :api_attr: Static Graph

    This operator is like a C++ switch/case statement.

    Args:
        branch_index(Tensor): A Tensor whose numel should be 1 (shape [] or shape [1]) to specify which branch to execute. The data type is ``int32``, ``int64`` or ``uint8``.
        branch_fns(dict|list|tuple): If it's a list or tuple, the elements in it could be pairs of (int, callable) or simple callables whose actual index will be used as the index of callable. If it's a dict, its key is a python integer and the value is a callable. All callables return the same structure of Tensors.
        default(callable, optional): Callable that returns a structure of Tensors.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor|list(Tensor): Tensors returned by the callable specified by ``branch_index`` in ``branch_fns``,
        or Tensors returned by ``default`` if ``default`` is not None and no index matches in ``branch_fns``,
        or Tensors returned by the callable with the max index in ``branch_fns`` if ``default`` is None and no index matches in ``branch_fns``.

    Raises:
        TypeError: If the type of ``branch_index`` is not Tensor.
        TypeError: If the data type of ``branch_index`` is not ``int32``, ``int64`` or ``uint8``.
        TypeError: If the type of ``branch_fns`` is not dict, list or tuple.
        TypeError: If the elements of ``branch_fns`` is not 2-tuple.
        TypeError: If the first element of 2-tuple in ``branch_fns`` is not integer.
        ValueError: If the first element of 2-tuple in ``branch_fns`` is not unique.
        TypeError: If the second element of 2-tuple in ``branch_fns`` is not callable.
        TypeError: If ``default`` is not None but it is not callable.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> def fn_1():
            ...    return paddle.full(shape=[1, 2], dtype='float32', fill_value=1)

            >>> def fn_2():
            ...    return paddle.full(shape=[2, 2], dtype='int32', fill_value=2)

            >>> def fn_3():
            ...    return paddle.full(shape=[3], dtype='int32', fill_value=3)

            >>> startup_program = paddle.static.default_startup_program()
            >>> main_program = paddle.static.default_main_program()
            >>> with paddle.static.program_guard(main_program, startup_program):
            ...    index_1 = paddle.full(shape=[1], dtype='int32', fill_value=1)
            ...    index_2 = paddle.full(shape=[1], dtype='int32', fill_value=2)
            ...
            ...    out_1 = paddle.static.nn.switch_case(
            ...        branch_index=index_1,
            ...        branch_fns={1: fn_1, 2: fn_2},
            ...        default=fn_3)
            ...
            ...    out_2 = paddle.static.nn.switch_case(
            ...        branch_index=index_2,
            ...        branch_fns=[(1, fn_1), (2, fn_2)],
            ...        default=fn_3)
            ...
            ...    # Argument default is None and no index matches. fn_3 will be called because of the max index 7.
            ...    out_3 = paddle.static.nn.switch_case(
            ...        branch_index=index_2,
            ...        branch_fns=[(0, fn_1), (4, fn_2), (7, fn_3)])
            ...
            ...    exe = paddle.static.Executor(paddle.CPUPlace())
            ...    res_1, res_2, res_3 = exe.run(main_program, fetch_list=[out_1, out_2, out_3])
            ...    # Variable: fill_constant_1.tmp_0
            ...    #   - message: The content of input layer:
            ...    #   - lod: {}
            ...    #   - place: Place(cpu)
            ...    #   - shape: [2, 3]
            ...    #   - layout: NCHW
            ...    #   - dtype: int64
            ...    #   - data: [3 3 3 3 3 3]

            >>> print(res_1)
            [[1. 1.]]

            >>> print(res_2)
            [[2 2]
             [2 2]]

            >>> print(res_3)
            [3 3 3]
    '''
    helper = LayerHelper('switch_case', **locals())

    def _check_args(branch_index, branch_fns, default):
        check_variable_and_dtype(
            branch_index,
            'branch_index',
            ['uint8', 'int32', 'int64'],
            'static.nn.switch_case',
        )

        if convert_dtype(branch_index.dtype) != "int64":
            branch_index = paddle.cast(branch_index, "int64")

        check_type(branch_fns, 'branch_fns', (list, tuple, dict), 'switch_case')

        branch_fns = (
            branch_fns.items() if isinstance(branch_fns, dict) else branch_fns
        )

        branch_fns = (
            list(enumerate(branch_fns))
            if all(callable(fn) for fn in branch_fns)
            else branch_fns
        )

        keys_of_fns = []
        for index_fn_pair in branch_fns:
            if not isinstance(index_fn_pair, tuple):
                raise TypeError(
                    _error_message(
                        "The elements' type",
                        "branch_fns",
                        "switch_case",
                        tuple,
                        type(branch_fns),
                    )
                )

            if len(index_fn_pair) != 2:
                raise TypeError(
                    _error_message(
                        "The tuple's size",
                        "branch_fns",
                        "switch_case",
                        "2",
                        str(len(index_fn_pair)) + "-tuple",
                    )
                )

            key, fn = index_fn_pair

            if not isinstance(key, int):
                raise TypeError(
                    _error_message(
                        "The key's type",
                        "branch_fns",
                        "switch_case",
                        int,
                        type(key),
                    )
                )

            if key in keys_of_fns:
                raise ValueError(
                    f"The key in 'branch_fns' must be unique, but '{key}' appears more than once."
                )
            else:
                keys_of_fns.append(key)

            if not callable(fn):
                raise TypeError(
                    _error_message(
                        f"The type of function for key {key}",
                        "branch_fns",
                        "switch_case",
                        "callable",
                        type(fn),
                    )
                )

        if default is None:
            default = sorted(branch_fns)[-1][1]
            branch_fns = sorted(branch_fns)[:-1]
        elif not callable(default):
            raise TypeError("The default in Op(case) must be callable.")

        pred_fn_pairs = []
        for index, fn in branch_fns:
            new_index = paddle.full(shape=[1], dtype="int64", fill_value=index)
            pred = paddle.equal(branch_index, new_index)
            pred_fn_pairs.append((pred, fn))

        return pred_fn_pairs, default

    pred_fn_pairs, default = _check_args(branch_index, branch_fns, default)
    false_fn = default
    for pred, true_fn in pred_fn_pairs:
        false_fn = partial(cond, pred=pred, true_fn=true_fn, false_fn=false_fn)

    final_fn = false_fn
    return final_fn()


def get_indices_by_discriminator(container, *discriminators):
    buckets = [[] for _ in range(len(discriminators) + 1)]
    for idx, item in enumerate(container):
        for i, cond in enumerate(discriminators):
            if cond(item):
                buckets[i].append(idx)
                break
        else:
            buckets[-1].append(idx)
    return buckets


def select_by_indices(container, *index_groups):
    buckets = [[] for _ in range(len(index_groups))]
    for idx, item in enumerate(container):
        for i, indices in enumerate(index_groups):
            if idx in indices:
                buckets[i].append(item)
                break
    return buckets


def create_container_by_items_and_indices(*items_indices_pairs):
    total_length = reduce(
        lambda acc, pair: acc + len(pair[0]), items_indices_pairs, 0
    )
    container = [None for _ in range(total_length)]
    for partial_container, indices in items_indices_pairs:
        assert len(partial_container) == len(indices)
        for idx, item in zip(indices, partial_container):
            container[idx] = item
    return container


def run_with_block(fn, block):
    def new_fn(*args, **kwargs):
        with block():
            return fn(*args, **kwargs)

    return new_fn


class OutputSelector:
    def __init__(
        self, if_op, flattened_true_output, flattened_false_output, names
    ):
        self.if_op = if_op
        self.true_output = flattened_true_output
        self.false_output = flattened_false_output
        self.names = names
        self.num_output = len(flattened_true_output)
        assert len(flattened_false_output) == self.num_output
        assert len(names) == self.num_output

    @cached_property
    def unified_output(self):
        unified_true_output = []
        unified_false_output = []
        for true_out, false_out, name in zip(
            self.true_output, self.false_output, self.names
        ):
            (
                true_out,
                false_out,
            ) = OutputSelector.constant_to_variable_promotion(
                [
                    (true_out, self.if_op.true_block),
                    (false_out, self.if_op.false_block),
                ],
                name,
            )
            (true_out, false_out) = OutputSelector.precision_promotion(
                [
                    (true_out, self.if_op.true_block),
                    (false_out, self.if_op.false_block),
                ],
                name,
            )
            unified_true_output.append(true_out)
            unified_false_output.append(false_out)
        return unified_true_output, unified_false_output

    @property
    def unified_true_output(self):
        return self.unified_output[0]

    @property
    def unified_false_output(self):
        return self.unified_output[1]

    @property
    def variable_indices(self):
        true_variable_indices, _ = get_indices_by_discriminator(
            self.unified_true_output,
            lambda x: isinstance(x, paddle.pir.Value),
        )
        false_variable_indices, _ = get_indices_by_discriminator(
            self.unified_false_output,
            lambda x: isinstance(x, paddle.pir.Value),
        )
        assert (
            true_variable_indices == false_variable_indices
        ), "true_variable_indices and false_variable_indices should be same"
        return true_variable_indices

    @property
    def constant_indices(self):
        return [
            i
            for i in range(len(self.true_output))
            if i not in self.variable_indices
        ]

    def get_variable_outputs(self):
        (variable_true_output,) = select_by_indices(
            self.unified_true_output,
            self.variable_indices,
        )
        (variable_false_output,) = select_by_indices(
            self.unified_false_output,
            self.variable_indices,
        )
        return variable_true_output, variable_false_output

    def restore_outputs_by_variable_results(self, variable_results):
        (constant_output,) = select_by_indices(
            self.unified_true_output,
            self.constant_indices,
        )

        restored_output = create_container_by_items_and_indices(
            (variable_results, self.variable_indices),
            (constant_output, self.constant_indices),
        )
        return restored_output

    @staticmethod
    def constant_to_variable_promotion(out_with_blocks, name):
        from paddle.jit.dy2static.convert_operators import to_static_variable
        from paddle.jit.dy2static.utils import UndefinedVar

        promotion_builtin_types = (bool, int, float)
        outs, _ = zip(*out_with_blocks)

        def all_has_same_value(outs):
            if len(outs) <= 1:
                return True
            return all(out == outs[0] for out in outs[1:])

        def all_has_same_type(outs):
            if len(outs) <= 1:
                return True
            return all(type(out) is type(outs[0]) for out in outs[1:])

        def get_first_value_dtype(outs):
            for out in outs:
                if isinstance(out, paddle.pir.Value):
                    return out.dtype
            return None

        if all(isinstance(out, paddle.pir.Value) for out in outs):
            return outs

        if all(out is None for out in outs):
            return outs

        if all(
            isinstance(out, promotion_builtin_types) for out in outs
        ) and all_has_same_type(outs):
            if all_has_same_value(outs):
                return outs
            else:
                warnings.warn(
                    f"Return results from different branches in cond has same type: {type(outs[0])}, "
                    f"but has different value: true value is '{outs[0]}' and false value is '{outs[1]}', "
                    "so we will promote the constant to variable."
                )
                return [
                    # TODO(SigureMo): Should we use the same dtype for all the constants?
                    # e.g. in true branch var is 3, else branch var is 2, then the dtype should be float64.
                    run_with_block(to_static_variable, block)(out, dtype=None)
                    for out, block in out_with_blocks
                ]

        if any(isinstance(out, paddle.pir.Value) for out in outs) and all(
            isinstance(out, (paddle.pir.Value, *promotion_builtin_types))
            for out in outs
        ):
            warnings.warn(
                "Return results from different branches in cond are not same type: "
                + f"false_var returned by false_fn is '{type(outs[1])}' and true_var of true_fn is "
                + f"'{type(outs[0])}'"
            )
            return [
                run_with_block(to_static_variable, block)(
                    out, dtype=get_first_value_dtype(outs)
                )
                for out, block in out_with_blocks
            ]

        if any(isinstance(out, UndefinedVar) for out in outs):
            warnings.warn(
                f"Return results has maybe unbound local variable `{name}`, please ensure do not use `{name}`"
                + "after cond."
            )
            return [UndefinedVar(name) for _ in out_with_blocks]

        raise TypeError(
            "Unsupported return type of true_fn and false_fn in cond: false_var "
            f"returned `{name}` by false_fn is `{outs[0]}` and true_var of true_fn is `{outs[1]}`"
        )

    @staticmethod
    def precision_promotion(out_with_blocks, name):
        # Only support promotion from fp16 to fp32 in AMP mode
        outs, _ = zip(*out_with_blocks)

        amp_attrs = core._get_amp_attrs()
        amp_level = amp_attrs._amp_level
        apply_amp_level_list = [
            core.AmpLevel.O1,
            core.AmpLevel.O2,
        ]
        if amp_level not in apply_amp_level_list:
            return outs

        def all_has_same_dtype(outs):
            if len(outs) <= 1:
                return True
            return all(out.dtype == outs[0].dtype for out in outs[1:])

        def promote_precision(out_with_blocks):
            def get_expected_precision(out_with_blocks):
                if len(outs) <= 1:
                    return outs[0].dtype
                # now only support fp16 to fp32
                if any(
                    out.dtype == paddle.float16 for out, _ in out_with_blocks
                ) and any(
                    out.dtype == paddle.float32 for out, _ in out_with_blocks
                ):
                    return paddle.float32
                else:
                    return out_with_blocks[0][0].dtype

            new_outs = []
            expected_dtype = get_expected_precision(out_with_blocks)
            for out, block in out_with_blocks:
                if expected_dtype != out.dtype:
                    out = run_with_block(paddle.cast, block)(
                        out, _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE[expected_dtype]
                    )
                new_outs.append(out)
            return new_outs

        if all(
            isinstance(out, paddle.pir.Value) for out in outs
        ) and not all_has_same_dtype(outs):
            warnings.warn(
                f"Return results from different branches in cond has different dtype: true value dtype is '{outs[0].dtype}' and false value dtype is '{outs[1].dtype}', "
                "so we will promote the lower precision to the higher one."
            )
            outs = promote_precision(out_with_blocks)
            return outs

        return outs


def cond(pred, true_fn=None, false_fn=None, name=None, return_names=None):
    """
    This API returns ``true_fn()`` if the predicate ``pred`` is true else
    ``false_fn()`` . Users could also set ``true_fn`` or ``false_fn`` to
    ``None`` if do nothing and this API will treat the callable simply returns
    ``None`` in this case.

    ``true_fn`` and ``false_fn`` should return same nest structure of tensors
    or both return ``None`` if user doesn't like to return anything. A nest
    structure of tensors in PaddlePaddle is tensor(s), or tuple of tensors, or
    list of tensors.

    Note:
        1. The tuples or lists returned by ``true_fn`` and ``false_fn`` must have
        the same shape because of dataflow model of PaddlePaddle while the
        tensors in the tuples or the lists can have different shapes.

        2. This API could be used under both static graph mode or dygraph mode. If it
        is in dygraph mode, the API only runs one branch based on condition.

        3. If it is in static graph mode, any tensors or operations created outside
        or inside of ``true_fn`` and ``false_fn`` will be in net building
        regardless of which branch is selected at runtime. This has frequently
        surprised users who expected a lazy semantics.

        Examples:
            .. code-block:: python
                :name: code-example-1

                >>> import paddle

                >>> a = paddle.zeros((1, 1))
                >>> b = paddle.zeros((1, 1))
                >>> c = a * b
                >>> out = paddle.static.nn.cond(a < b, lambda: a + c, lambda: b * b)

        No matter whether ``a < b`` , ``c = a * b`` will be in net building and
        run. ``a + c`` and ``b * b`` will be in net building, but only one
        branch will be executed during runtime.

    Args:
        pred(Tensor): A boolean tensor whose numel should be 1 (shape []
            or shape [1]). The boolean value determines whether to return the
            result of ``true_fn`` or ``false_fn`` .
        true_fn(callable, optional): A callable to be performed if ``pred`` is
            true. The default value is ``None`` .
        false_fn(callable, optional): A callable to be performed if ``pred`` is
            false. The default value is ``None`` .
        name(str, optional): The default value is ``None`` . Normally users
             don't have to set this parameter. For more information, please
             refer to :ref:`api_guide_Name` .
        return_names(sequence of string, optional): The default value is ``None`` .
             Normally users don't have to set this parameters.  A sequence of strings
             to represents the name of returned vars.  The structure of sequence must
             be same with return values of true_fn and false_fn.

    Returns:
        Tensor|list(Tensor)|tuple(Tensor): returns ``true_fn()`` if the
        predicate ``pred`` is true else ``false_fn()`` .

    Examples:
        .. code-block:: python
            :name: code-example-2

            >>> import paddle

            >>> # pseudocode:
            >>> # if 0.1 < 0.23:
            >>> #     return 1, True
            >>> # else:
            >>> #     return 3, 2

            >>> def true_func():
            ...     return paddle.full(shape=[1, 2],
            ...                        dtype='int32',
            ...                        fill_value=1
            ...         ), paddle.full(shape=[2, 3],
            ...                        dtype='bool',
            ...                        fill_value=True
            ...         )


            >>> def false_func():
            ...     return paddle.full(shape=[3, 4],
            ...                        dtype='float32',
            ...                        fill_value=3
            ...         ), paddle.full(shape=[4, 5],
            ...                        dtype='int64',
            ...                        fill_value=2
            ...         )


            >>> x = paddle.full(shape=[1], dtype='float32', fill_value=0.1)
            >>> y = paddle.full(shape=[1], dtype='float32', fill_value=0.23)
            >>> pred = paddle.less_than(x=x, y=y, name=None)
            >>> a, b = paddle.static.nn.cond(pred, true_func, false_func)

            >>> print(a)
            Tensor(shape=[1, 2], dtype=int32, place=Place(cpu), stop_gradient=True,
                   [[1, 1]])
            >>> print(b)
            Tensor(shape=[2, 3], dtype=bool, place=Place(cpu), stop_gradient=True,
                   [[True, True, True],
                    [True, True, True]])
    """
    if in_dygraph_mode():
        assert isinstance(pred, Variable), "The pred in cond must be Variable"
        assert pred.size == 1, "condition input's numel should be 1"
        pred = pred.item()
        if pred:
            if true_fn is not None:
                if not callable(true_fn):
                    raise TypeError(
                        f"The true_fn in cond must be callable, but received {type(true_fn).__name__}"
                    )
                return true_fn()
        else:
            if false_fn is not None:
                if not callable(false_fn):
                    raise TypeError(
                        f"The false_fn in cond must be callable, but received {type(false_fn).__name__}"
                    )
                return false_fn()
        return None
    true_output = None
    false_output = None
    check_variable_and_dtype(pred, "pred", ['bool'], "paddle.static.nn.cond")
    check_type(name, "name", (str, type(None)), "paddle.static.nn.cond")
    if in_pir_mode():
        if_op = build_if_op(pred)
        if true_fn is not None:
            if not callable(true_fn):
                raise TypeError(
                    f"The true_fn in cond must be callable, but received {type(true_fn).__name__}"
                )
            with if_op.true_block():
                true_output = true_fn()
        if false_fn is not None:
            if not callable(false_fn):
                raise TypeError(
                    f"The false_fn in cond must be callable, but received {type(false_fn).__name__}"
                )
            with if_op.false_block():
                false_output = false_fn()
    else:
        helper = LayerHelper('cond', **locals())
        copy_to_parent_func = lambda var: copy_var_to_parent_block(var, helper)
        if true_fn is not None:
            if not callable(true_fn):
                raise TypeError(
                    f"The true_fn in cond must be callable, but received {type(true_fn).__name__}"
                )
            true_cond_block = ConditionalBlock([pred], is_scalar_condition=True)
            with true_cond_block.block():
                origin_true_output = true_fn()
                if origin_true_output is not None:
                    true_output = map_structure(
                        copy_to_parent_func, origin_true_output
                    )
        if false_fn is not None:
            if not callable(false_fn):
                raise TypeError(
                    f"The false_fn in cond must be callable, but received {type(false_fn).__name__}"
                )
            false_cond_block = ConditionalBlock(
                [paddle.logical_not(pred)], is_scalar_condition=True
            )
            with false_cond_block.block():
                origin_false_output = false_fn()
                if origin_false_output is not None:
                    false_output = map_structure(
                        copy_to_parent_func, origin_false_output
                    )

    if true_output is None and false_output is None:
        return None

    if true_output is None:
        raise ValueError(
            "Incompatible return values of true_fn and false_fn in cond: "
            "true_fn returns None while false_fn returns non-None"
        )
    if false_output is None:
        raise ValueError(
            "Incompatible return values of true_fn and false_fn in cond: "
            "true_fn returns non-None while false_fn returns None"
        )

    # Merge true and false output if they are not None
    if return_names is None:
        is_dy2static = False
        return_names = ["no name"] * len(_to_sequence_except_dict(true_output))
    else:
        """
        dy2static will set the return_names and expand the return values to UndefinedVar.
        """
        is_dy2static = True

        # TODO:  expand_undefined_var will replace None to Undefinedvar(), to fix cases like:
        #       a = None
        #       if condition:
        #           a = 1
        # Because we can not use variable to express 'None'
        true_output, false_output = expand_undefined_var(
            true_output, false_output, return_names
        )

    if len(_to_sequence_except_dict(true_output)) != len(
        _to_sequence_except_dict(false_output)
    ):
        raise ValueError(
            f"true fn returns {len(_to_sequence_except_dict(true_output))} vars, but false fn returns {len(_to_sequence_except_dict(false_output))} vars, which is not equals"
        )
    for true_out, false_out, return_name in zip(
        _to_sequence_except_dict(true_output),
        _to_sequence_except_dict(false_output),
        _to_sequence_except_dict(return_names),
    ):
        try:
            assert_same_structure(true_out, false_out, check_types=False)
        except ValueError as e:
            raise ValueError(
                f"Incompatible return values of `{return_name}` in true_fn and false_fn in cond: {e}"
            )

    def check_ret_none(seq_true, seq_false, seq_names):
        for f_true, f_false, f_name in zip(seq_true, seq_false, seq_names):
            f_true = flatten(f_true)
            f_false = flatten(f_false)
            for idx in range(len(f_true)):
                if (
                    f_true[idx] is None
                    and f_false[idx] is not None
                    or f_false[idx] is None
                    and f_true[idx] is not None
                ):
                    warnings.warn(
                        f"In cond : Var '{f_name}' or part of it is set differently in ifelse branches, "
                        f"<{type(f_true[idx])}, {f_true[idx]}> in true branch and <{type(f_false[idx])}, {f_false[idx]}> in false branch. Set var to "
                        "'None' in ifelse block might lead to error."
                    )

    check_ret_none(
        _to_sequence_except_dict(true_output),
        _to_sequence_except_dict(false_output),
        _to_sequence_except_dict(return_names),
    )

    if is_dy2static and not use_pir_api():
        true_output, false_output = change_none_to_undefinedvar(
            true_output, false_output
        )

    if in_pir_mode():
        flattened_true_output, flattened_false_output = flatten(
            true_output
        ), flatten(false_output)
        flattened_return_names = [
            name
            for seq_out, name in zip(
                _to_sequence_except_dict(true_output),
                _to_sequence_except_dict(return_names),
            )
            for _ in flatten(seq_out)
        ]
        output_selector = OutputSelector(
            if_op,
            flattened_true_output,
            flattened_false_output,
            names=flattened_return_names,
        )
        (
            variable_true_output,
            variable_false_output,
        ) = output_selector.get_variable_outputs()

        with if_op.true_block():
            cf_yield(variable_true_output)
        with if_op.false_block():
            cf_yield(variable_false_output)

        if_op.update_output()
        variable_results = flatten(if_op.results())

        restored_output = output_selector.restore_outputs_by_variable_results(
            variable_results
        )
        return pack_sequence_as(true_output, restored_output)

    mask = paddle.cast(pred, dtype='int32')
    merge_func = (
        lambda name, false_var, true_var: select_input_with_buildin_type(
            [false_var, true_var], mask, name
        )
    )

    def merge_every_var_list(false_vars, true_vars, name):
        return map_structure(partial(merge_func, name), false_vars, true_vars)

    merged_output_fns = list(
        map(
            merge_every_var_list,
            _to_sequence_except_dict(false_output),
            _to_sequence_except_dict(true_output),
            _to_sequence_except_dict(return_names),
        )
    )
    merged_output = map_structure(lambda fn: fn(), merged_output_fns)
    merged_output = pack_sequence_as(false_output, flatten(merged_output))
    return merged_output


def copy_var_to_parent_block(var, layer_helper):
    if not isinstance(var, Variable):
        return var
    prog = layer_helper.main_program
    parent_idx = prog.current_block().parent_idx
    assert (
        parent_idx >= 0
    ), "Got wrong parent block index when assigning var to parent scope in control_flow"
    parent_block = prog.block(parent_idx)

    if (
        var.type == core.VarDesc.VarType.DENSE_TENSOR_ARRAY
        and parent_block._find_var_recursive(var.name)
    ):
        parent_block_var = var
    else:
        parent_block_var = parent_block.create_var(
            dtype=var.dtype, shape=var.shape, type=var.type
        )
        paddle.assign(var, parent_block_var)
    return parent_block_var


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
    out_shape = [a if a == b else -1 for a, b in zip(first_shape, second_shape)]
    return out_shape


def select_input(inputs, mask):
    """
    **select_input**

    This API takes in multiple inputs and uses an integer mask to select one
    input to output. It is useful in control flow.

    Args:
        inputs(tuple|list): The input variables
        mask(Tensor): A tensor containing 1 integer number selecting which
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


def select_input_with_buildin_type(inputs, mask, name):
    from paddle.jit.dy2static.convert_operators import to_static_variable
    from paddle.jit.dy2static.utils import UndefinedVar

    false_var, true_var = inputs

    def start_select_input():
        try:
            return select_input(inputs, mask)
        except Exception as e:
            raise RuntimeError(
                f"Exceptions thrown while doing select_input on {name}:\n{e}"
            )

    if isinstance(false_var, UndefinedVar) and isinstance(
        true_var, UndefinedVar
    ):
        """None -> UndefinedVar, so the real value is a [None, UndefinedVar] or [None, None], we just return None."""
        return lambda: None

    if isinstance(false_var, Variable) and isinstance(true_var, Variable):
        return start_select_input

    elif isinstance(false_var, support_ret_buildin_type) and isinstance(
        false_var, type(true_var)
    ):
        if false_var == true_var:
            return lambda: false_var
        else:
            inputs = [
                to_static_variable(false_var),
                to_static_variable(true_var),
            ]
    # Deal with the situations like this: false_var is int and true_var is Variable
    elif (
        isinstance(false_var, support_ret_buildin_type)
        and isinstance(true_var, Variable)
    ) or (
        isinstance(true_var, support_ret_buildin_type)
        and isinstance(false_var, Variable)
    ):
        inputs = [to_static_variable(false_var), to_static_variable(true_var)]
        warnings.warn(
            "Return results from different branches in cond are not same type: "
            f"false_var returned by false_fn is '{type(false_var)}' and true_var of true_fn is "
            f"'{type(true_var)}'"
        )
    elif (
        isinstance(false_var, UndefinedVar)
        and isinstance(true_var, (Variable, *support_ret_buildin_type))
    ) or (
        isinstance(true_var, UndefinedVar)
        and isinstance(false_var, (Variable, *support_ret_buildin_type))
    ):
        true_var, false_var = to_static_variable(true_var), to_static_variable(
            false_var
        )
        inputs = [false_var, true_var]
    else:
        raise TypeError(
            "Unsupported return type of true_fn and false_fn in cond: false_var "
            f"returned by false_fn is '{type(false_var)}' and true_var of true_fn is '{type(true_var)}'"
        )
    return start_select_input


def _is_sequence_except_dict(x):
    """
    In this function, dict is not viewed as sequence.
    """
    if isinstance(x, dict):
        return False
    return is_sequence(x)


def _to_sequence_except_dict(x):
    """
    In this function, dict is not viewed as sequence.
    """
    if isinstance(x, dict):
        return [x]
    return to_sequence(x)


def expand_undefined_var(nest1, nest2, names):
    """TODO: make this function recursively.
    nest1: Var1, (UndefinedVar, [1,2,3])
    nest2: Var2, ([1,2,3,4], UndefinedVar)
    In this case, we should not expand recursively.
    """
    from paddle.jit.dy2static.transformers.return_transformer import (
        RETURN_VALUE_PREFIX,
    )
    from paddle.jit.dy2static.utils import UndefinedVar

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
                        f"In cond : Var '{name}' or part of it is set differently in ifelse branches, "
                        f"<{type(n1)}, {n1}> in true branch and <{type(n2)}, {n2}> in false branch. Set var to "
                        "'None' in ifelse block might lead to error."
                    )
                else:
                    warnings.warn(
                        f"In cond : Var '{name}' or part of it is set differently in ifelse branches, "
                        f"<{type(n2)}, {n2}> in true branch and <{type(n1)}, {n1}> in false branch. Set var to "
                        "'None' in ifelse block might lead to error."
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


def change_none_to_undefinedvar(nest1, nest2):
    from paddle.jit.dy2static.utils import UndefinedVar

    def map_fn(x):
        if x is None:
            return UndefinedVar("padding")
        return x

    nest1_out = pack_sequence_as(nest1, list(map(map_fn, flatten(nest1))))
    nest2_out = pack_sequence_as(nest2, list(map(map_fn, flatten(nest2))))
    return nest1_out, nest2_out


@static_only
def Print(
    input,
    first_n=-1,
    message=None,
    summarize=20,
    print_tensor_name=True,
    print_tensor_type=True,
    print_tensor_shape=True,
    print_tensor_layout=True,
    print_tensor_lod=True,
    print_phase='both',
):
    '''
    :api_attr: Static Graph

    **Print operator**

    This creates a print op that will print when a tensor is accessed.

    Wraps the tensor passed in so that whenever that a tensor is accessed,
    the message `message` is printed, along with the current value of the
    tensor `t`.

    Args:
        input (Tensor): A Tensor to print.
        first_n (int, optional): Only log `first_n` number of times. Default: -1.
        message (str, optional): A string message to print as a prefix. Default: None.
        summarize (int, optional): Number of elements in the tensor to be print. If
                it's value is -1, then all elements in the tensor will be print.
        print_tensor_name (bool, optional): Print the tensor name. Default: True.
        print_tensor_type (bool, optional): Print the tensor type. Default: True.
        print_tensor_shape (bool, optional): Print the tensor shape. Default: True.
        print_tensor_layout (bool, optional): Print the tensor layout. Default: True.
        print_tensor_lod (bool, optional): Print the tensor lod. Default: True.
        print_phase (str, optional): Which phase to displace, including 'forward',
                'backward' and 'both'. Default: 'both'. If set to 'backward', will
                only print the gradients of input tensor; If set to 'both', will
                both print the input tensor itself and the gradients of input tensor.

    Returns:
        Tensor: Output tensor.

    NOTES:
        The input and output are two different Tensor, and in the
        following process, you should use the output Tensor but not the input,
        otherwise, the print layer doesn't have backward.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.enable_static()

            >>> x = paddle.full(shape=[2, 3], fill_value=3, dtype='int64')
            >>> out = paddle.static.Print(x, message="The content of input layer:")

            >>> main_program = paddle.static.default_main_program()
            >>> exe = paddle.static.Executor(place=paddle.CPUPlace())
            >>> res = exe.run(main_program, fetch_list=[out])
            >>> # doctest: +SKIP('Unable to get output')
            Variable: fill_constant_1.tmp_0
              - message: The content of input layer:
              - lod: {}
              - place: Place(cpu)
              - shape: [2, 3]
              - layout: NCHW
              - dtype: int64
              - data: [3 3 3 3 3 3]
            >>> # doctest: -SKIP
            >>> res
            [array([[3, 3, 3],
                    [3, 3, 3]], dtype=int64)]
    '''
    check_variable_and_dtype(
        input,
        'input',
        [
            'uint16',
            'float16',
            'float32',
            'float64',
            'int32',
            'int64',
            'bool',
            'float8_e4m3fn',
            'float8_e5m2',
        ],
        'paddle.static.Print',
    )
    message = message or ""
    helper = LayerHelper('print', **locals())

    if in_pir_mode():
        return _C_ops.print(
            input,
            first_n,
            message,
            summarize,
            print_tensor_name,
            print_tensor_type,
            print_tensor_shape,
            print_tensor_layout,
            print_tensor_lod,
            print_phase.upper(),
            True,
        )

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
            'print_tensor_layout': print_tensor_layout,
            'print_tensor_lod': print_tensor_lod,
            'print_phase': print_phase.upper(),
        },
    )
    return output


class Switch:
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
            'the member function case of base.layers.Switch',
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
