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

import warnings
from functools import partial, reduce

import paddle
import paddle.fluid.core as core
from paddle.common_ops_import import (
    LayerHelper,
    _non_static_mode,
    check_type,
    check_variable_and_dtype,
    convert_dtype,
)
from paddle.fluid.framework import Operator, Program, Variable

# Temporary solution, it will be deleted later
from paddle.fluid.layers.control_flow import ConditionalBlock, select_input
from paddle.fluid.layers.tensor import assign, cast
from paddle.fluid.layers.utils import (
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

            import paddle
            from paddle.static.nn.control_flow import Assert

            paddle.enable_static()
            x = paddle.full([2, 3], 2.0, 'float32')
            condition = paddle.max(x) < 1.0 # False
            Assert(condition, [x], 10, "example_assert_layer")

            exe = paddle.static.Executor()
            try:
                exe.run(paddle.static.default_main_program())
                # Print x and throws ValueError
                # Example printed message for x:
                #
                # Variable: fill_constant_0.tmp_0
                #   - lod: {}
                #   - place: CPUPlace()
                #   - shape: [2, 3]
                #   - layout: NCHW
                #   - dtype: float
                #   - data: [2 2 2 2 2 2]
            except ValueError as e:
                print("Assert Exception Example")

    '''
    check_variable_and_dtype(
        cond, "cond", ["bool"], "static.nn.control_flow.Assert"
    )
    check_type(
        data, "data", (list, tuple, type(None)), "static.nn.control_flow.Assert"
    )
    check_type(summarize, "summarize", int, "static.nn.control_flow.Assert")
    check_type(name, "name", (str, type(None)), "static.nn.control_flow.Assert")

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

            import paddle
            import numpy as np

            paddle.enable_static()

            i = paddle.full(shape=[1], dtype='int64', fill_value=0)           # loop counter

            loop_len = paddle.full(shape=[1],dtype='int64', fill_value=10)    # loop length

            cond = paddle.less_than(x=i, y=loop_len)
            while_op = paddle.static.nn.control_flow.While(cond=cond)
            with while_op.block():
                i = paddle.increment(x=i, value=1)
                paddle.assign(paddle.less_than(x=i, y=loop_len), output=cond)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())

            res = exe.run(paddle.static.default_main_program(), feed={}, fetch_list=[i])
            print(res) # [array([10])]


    Examples 2:
          .. code-block:: python

            import paddle
            import numpy as np

            paddle.enable_static()

            i = paddle.full(shape=[1], dtype='int64', fill_value=0)
            loop_len = paddle.full(shape=[1], dtype='int64', fill_value=10)
            one = paddle.full(shape=[1], dtype='float32', fill_value=1)
            data = paddle.static.data(name='data', shape=[1], dtype='float32')
            sums = paddle.full(shape=[1], dtype='float32', fill_value=0)  # Define the variable to be obtained ouside of While, which name should be different from the variable inside the While to be obtained

            cond = paddle.less_than(x=i, y=loop_len)
            while_op = paddle.static.nn.control_flow.While(cond=cond)
            with while_op.block():
                sums_tensor = paddle.add(x=data, y=data)
                paddle.assign(sums_tensor, sums)  # Update the value of sums_tensor defined in While to the sums which defined outside of While through layers.assign
                i = paddle.increment(x=i, value=1)
                data = paddle.add(x=data, y=one)
                paddle.assign(paddle.less_than(x=i, y=loop_len), output=cond)

            feed_data = np.ones(1).astype('float32')
            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())
            res = exe.run(paddle.static.default_main_program(), feed={'data': feed_data}, fetch_list=sums)
            print(res[0])  # [2.]    # Because the data in While does not update the value outside the While, the value of sums is [2.] after the loop
    """

    BEFORE_WHILE_BLOCK = 0
    IN_WHILE_BLOCK = 1
    AFTER_WHILE_BLOCK = 2

    def __init__(self, cond, is_test=False, name=None):
        self.helper = LayerHelper("while", name=name)
        self.status = While.BEFORE_WHILE_BLOCK
        check_variable_and_dtype(cond, 'cond', ['bool'], 'static.nn.While')
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

    if not isinstance(input, (Variable, core.VarBase)):
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
            "but given shape as {0}.".format(list(pre_cond.shape))
        )

    if _non_static_mode():
        now_cond = pre_cond.numpy()[0]
        while now_cond:
            output_vars = body(*loop_vars)
            if not isinstance(output_vars, (list, tuple)):
                output_vars = [output_vars]
            if len(output_vars) != len(loop_vars):
                raise ValueError(
                    "body in while_loop should return the same arity "
                    "(length and structure) and types as loop_vars"
                )
            now_cond = cond(*output_vars).numpy()[0]
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
                "(length and structure) as loop_vars: {0}".format(e)
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


def _error_message(what, arg_name, op_name, right_value, error_value):
    error_message = (
        "{what} of '{arg_name}' in {op_name} must be "
        "{right_value}, but received: {error_value}.".format(
            what=what,
            arg_name=arg_name,
            op_name=op_name,
            right_value=right_value,
            error_value=error_value,
        )
    )

    return error_message


def case(pred_fn_pairs, default=None, name=None):
    '''
    :api_attr: Static Graph

    This operator works like an if-elif-elif-else chain.

    Args:
        pred_fn_pairs(list|tuple): A list or tuple of (pred, fn) pairs. ``pred`` is a boolean Tensor with shape [1], ``fn`` is a callable. All callables return the same structure of Tensors.
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

            import paddle

            paddle.enable_static()

            def fn_1():
                return paddle.full(shape=[1, 2], dtype='float32', fill_value=1)

            def fn_2():
                return paddle.full(shape=[2, 2], dtype='int32', fill_value=2)

            def fn_3():
                return paddle.full(shape=[3], dtype='int32', fill_value=3)

            main_program = paddle.static.default_startup_program()
            startup_program = paddle.static.default_main_program()

            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.full(shape=[1], dtype='float32', fill_value=0.3)
                y = paddle.full(shape=[1], dtype='float32', fill_value=0.1)
                z = paddle.full(shape=[1], dtype='float32', fill_value=0.2)

                pred_1 = paddle.less_than(z, x)  # true: 0.2 < 0.3
                pred_2 = paddle.less_than(x, y)  # false: 0.3 < 0.1
                pred_3 = paddle.equal(x, y)      # false: 0.3 == 0.1

                # Call fn_1 because pred_1 is True
                out_1 = paddle.static.nn.case(
                    pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3)

                # Argument default is None and no pred in pred_fn_pairs is True. fn_3 will be called.
                # because fn_3 is the last callable in pred_fn_pairs.
                out_2 = paddle.static.nn.case(pred_fn_pairs=[(pred_2, fn_2), (pred_3, fn_3)])

                exe = paddle.static.Executor(paddle.CPUPlace())
                res_1, res_2 = exe.run(main_program, fetch_list=[out_1, out_2])
                print(res_1)  # [[1. 1.]]
                print(res_2)  # [3 3 3]
    '''
    helper = LayerHelper('case', **locals())

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

            if not isinstance(pred, Variable):
                raise TypeError(
                    _error_message(
                        "The pred's type",
                        "pred_fn_pairs",
                        "case",
                        "boolean Variable",
                        type(pred),
                    )
                )

            if not callable(fn):
                raise TypeError(
                    "The fn for {} of pred_fn_pairs in Op(case) must"
                    " be callable.".format(pred.name)
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
        branch_index(Tensor): A Tensor with shape [1] to specify which branch to execute. The data type is ``int32``, ``int64`` or ``uint8``.
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

            import paddle

            paddle.enable_static()

            def fn_1():
                return paddle.full(shape=[1, 2], dtype='float32', fill_value=1)

            def fn_2():
                return paddle.full(shape=[2, 2], dtype='int32', fill_value=2)

            def fn_3():
                return paddle.full(shape=[3], dtype='int32', fill_value=3)

            main_program = paddle.static.default_startup_program()
            startup_program = paddle.static.default_main_program()
            with paddle.static.program_guard(main_program, startup_program):
                index_1 = paddle.full(shape=[1], dtype='int32', fill_value=1)
                index_2 = paddle.full(shape=[1], dtype='int32', fill_value=2)

                out_1 = paddle.static.nn.switch_case(
                    branch_index=index_1,
                    branch_fns={1: fn_1, 2: fn_2},
                    default=fn_3)

                out_2 = paddle.static.nn.switch_case(
                    branch_index=index_2,
                    branch_fns=[(1, fn_1), (2, fn_2)],
                    default=fn_3)

                # Argument default is None and no index matches. fn；,,_3 will be called because of the max index 7.
                out_3 = paddle.static.nn.switch_case(
                    branch_index=index_2,
                    branch_fns=[(0, fn_1), (4, fn_2), (7, fn_3)])

                exe = paddle.static.Executor(paddle.CPUPlace())
                res_1, res_2, res_3 = exe.run(main_program, fetch_list=[out_1, out_2, out_3])
                print(res_1)  # [[1. 1.]]
                print(res_2)  # [[2 2] [2 2]]
                print(res_3)  # [3 3 3]
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
                    "The key in 'branch_fns' must be unique, but '{}' appears more than once.".format(
                        key
                    )
                )
            else:
                keys_of_fns.append(key)

            if not callable(fn):
                raise TypeError(
                    _error_message(
                        "The type of function for key {}".format(key),
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


def cond(pred, true_fn=None, false_fn=None, name=None, return_names=None):
    """
    This API returns ``true_fn()`` if the predicate ``pred`` is true else
    ``false_fn()`` . Users could also set ``true_fn`` or ``false_fn`` to
    ``None`` if do nothing and this API will treat the callable simply returns
    ``None`` in this case.

    ``true_fn`` and ``false_fn`` should return same nest structure of tensors
    or both return ``None`` if user doens't like to return anything. A nest
    structure of tensors in PaddlePaddle is tensor(s), or tuple of tensors, or
    list of tensors.

    Note:
        1. The tuples or lists returned by ``true_fn`` and ``false_fn`` must have
        the same shape because of dataflow model of PaddlePaddle while the
        tensors in the tuples or the lists can have different shapes.

        2. This API could be used under both static mode or dygraph mode. If it
        is in dygraph mode, the API only runs one branch based on condition.

        3. If it is in static mode, any tensors or operations created outside
        or inside of ``true_fn`` and ``false_fn`` will be in net building
        regardless of which branch is selected at runtime. This has frequently
        surprised users who expected a lazy semantics. For example:

        .. code-block:: python

            import paddle

            a = paddle.zeros((1, 1))
            b = paddle.zeros((1, 1))
            c = a * b
            out = paddle.static.nn.cond(a < b, lambda: a + c, lambda: b * b)

        No matter whether ``a < b`` , ``c = a * b`` will be in net building and
        run. ``a + c`` and ``b * b`` will be in net building, but only one
        branch will be executed during runtime.

    Args:
        pred(Tensor): A boolean tensor whose numel should be 1. The boolean
            value determines whether to return the result of ``true_fn`` or
            ``false_fn`` .
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

            import paddle

            #
            # pseudocode:
            # if 0.1 < 0.23:
            #     return 1, True
            # else:
            #     return 3, 2
            #

            def true_func():
                return paddle.full(shape=[1, 2], dtype='int32',
                                   fill_value=1), paddle.full(shape=[2, 3],
                                                              dtype='bool',
                                                              fill_value=True)


            def false_func():
                return paddle.full(shape=[3, 4], dtype='float32',
                                   fill_value=3), paddle.full(shape=[4, 5],
                                                              dtype='int64',
                                                              fill_value=2)


            x = paddle.full(shape=[1], dtype='float32', fill_value=0.1)
            y = paddle.full(shape=[1], dtype='float32', fill_value=0.23)
            pred = paddle.less_than(x=x, y=y, name=None)
            ret = paddle.static.nn.cond(pred, true_func, false_func)
            # ret is a tuple containing 2 tensors
            # ret[0] = [[1 1]]
            # ret[1] = [[ True  True  True]
            #           [ True  True  True]]

    """
    if _non_static_mode():
        assert isinstance(pred, Variable), "The pred in cond must be Variable"
        assert pred.size == 1, "condition input's numel should be 1"
        pred = pred.numpy()[0]
        if pred:
            if true_fn is not None:
                if not callable(true_fn):
                    raise TypeError(
                        "The true_fn in cond must be callable, but received {}".format(
                            type(true_fn).__name__
                        )
                    )
                return true_fn()
        else:
            if false_fn is not None:
                if not callable(false_fn):
                    raise TypeError(
                        "The false_fn in cond must be callable, but received {}".format(
                            type(false_fn).__name__
                        )
                    )
                return false_fn()
        return None

    check_variable_and_dtype(pred, "pred", ['bool'], "fluid.layers.cond")
    check_type(name, "name", (str, type(None)), "fluid.layers.cond")
    helper = LayerHelper('cond', **locals())
    true_output = None
    false_output = None
    copy_to_parent_func = lambda var: copy_var_to_parent_block(var, helper)
    if true_fn is not None:
        if not callable(true_fn):
            raise TypeError(
                "The true_fn in cond must be callable, but received {}".format(
                    type(true_fn).__name__
                )
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
                "The false_fn in cond must be callable, but received {}".format(
                    type(false_fn).__name__
                )
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
        is_dy2staic = False
        return_names = ["no name"] * len(_to_sequence_except_dict(true_output))
    else:
        """
        dy2static will set the return_names and expand the return values to UndefinedVar.
        """
        is_dy2staic = True

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
            "true fn returns {} vars, but false fn returns {} vars, which is not equals".format(
                len(_to_sequence_except_dict(true_output)),
                len(_to_sequence_except_dict(false_output)),
            )
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
                "Incompatible return values of `{}` in true_fn and false_fn in cond: {}".format(
                    return_name, e
                )
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
                        "In cond : Var '{}' or part of it is set differently in ifelse branchs, "
                        "<{}, {}> in true branch and <{}, {}> in false branch. Set var to "
                        "'None' in ifelse block might lead to error.".format(
                            f_name,
                            type(f_true[idx]),
                            f_true[idx],
                            type(f_false[idx]),
                            f_false[idx],
                        )
                    )

    check_ret_none(
        _to_sequence_except_dict(true_output),
        _to_sequence_except_dict(false_output),
        _to_sequence_except_dict(return_names),
    )

    if is_dy2staic:
        true_output, false_output = change_none_to_undefinedvar(
            true_output, false_output
        )

    mask = cast(pred, dtype='int32')
    merge_func = (
        lambda name, false_var, true_var: select_input_with_buildin_type(
            [false_var, true_var], mask, name
        )
    )

    def merge_every_var_list(false_vars, true_vars, name):
        return map_structure(partial(merge_func, name), false_vars, true_vars)

    merged_output = list(
        map(
            merge_every_var_list,
            _to_sequence_except_dict(false_output),
            _to_sequence_except_dict(true_output),
            _to_sequence_except_dict(return_names),
        )
    )
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
        var.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY
        and parent_block._find_var_recursive(var.name)
    ):
        parent_block_var = var
    else:
        parent_block_var = parent_block.create_var(
            dtype=var.dtype, shape=var.shape, type=var.type
        )
        assign(var, parent_block_var)
    return parent_block_var


def select_input_with_buildin_type(inputs, mask, name):
    from paddle.jit.dy2static.utils import UndefinedVar
    from paddle.jit.dy2static.variable_trans_func import to_static_variable

    false_var, true_var = inputs

    if isinstance(false_var, UndefinedVar) and isinstance(
        true_var, UndefinedVar
    ):
        """None -> UndefinedVar, so the real value is a [None, UndefinedVar] or [None, None], we just return None."""
        return None

    if isinstance(false_var, Variable) and isinstance(true_var, Variable):
        try:
            return select_input(inputs, mask)
        except Exception as e:
            raise RuntimeError(
                f"Exceptions throwed while doing select_input on {name}:\n{e}"
            )

    elif isinstance(false_var, support_ret_buildin_type) and isinstance(
        false_var, type(true_var)
    ):
        if false_var == true_var:
            return false_var
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
            "false_var returned by false_fn is '{}' and true_var of true_fn is "
            "'{}'".format(type(false_var), type(true_var))
        )
    elif (
        isinstance(false_var, UndefinedVar)
        and isinstance(true_var, (Variable,) + support_ret_buildin_type)
    ) or (
        isinstance(true_var, UndefinedVar)
        and isinstance(false_var, (Variable,) + support_ret_buildin_type)
    ):

        def create_var_if_not_undefined_var(a):
            if isinstance(a, UndefinedVar):
                return a
            return to_static_variable(a)

        true_var, false_var = to_static_variable(true_var), to_static_variable(
            false_var
        )
        inputs = [false_var, true_var]
    else:
        raise TypeError(
            "Unsupported return type of true_fn and false_fn in cond: false_var "
            "returned by false_fn is '{}' and true_var of true_fn is '{}'".format(
                type(false_var), type(true_var)
            )
        )
    try:
        return select_input(inputs, mask)
    except Exception as e:
        raise RuntimeError(
            f"Exceptions throwed while doing select_input on {name}:\n{e}"
        )


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
    from paddle.jit.dy2static.return_transformer import RETURN_VALUE_PREFIX
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


def change_none_to_undefinedvar(nest1, nest2):
    from paddle.jit.dy2static.utils import UndefinedVar

    def map_fn(x):
        if x is None:
            return UndefinedVar("padding")
        return x

    nest1_out = pack_sequence_as(nest1, list(map(map_fn, flatten(nest1))))
    nest2_out = pack_sequence_as(nest2, list(map(map_fn, flatten(nest2))))
    return nest1_out, nest2_out
