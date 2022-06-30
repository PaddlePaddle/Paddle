# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.data_feeder import convert_dtype
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import to_static_variable
from paddle.fluid.framework import core, Variable
from paddle.fluid.layers import Assert, Print
from paddle.fluid.layers import array_length, array_read, array_write, create_array
from paddle.fluid.layers import assign, fill_constant, slice, reduce_all, reduce_any
from paddle.fluid.layers import cast, control_flow, logical_and, logical_not, logical_or, nn
from paddle.fluid.layers.control_flow import cond, while_loop, less_than, increment
from paddle.fluid.dygraph.dygraph_to_static.return_transformer import RETURN_NO_VALUE_VAR_NAME
from paddle.fluid.dygraph.dygraph_to_static.utils import UndefinedVar


def convert_while_loop(cond, body, getter, setter):
    """
    A function representation of a Python ``while`` statement.

    Args:
        cond(Callable): A callable object that returns a boolean variable to control whether to execute the loop body. It takes ``loop_vars`` as arguments.
        body(Callable): A callable object that returns a tuple or list of variables with the same arguments ``loops_vars`` as ``cond`` .
        loop_vars(list|tuple): A list or tuple of variables passed to ``cond`` and ``body`` .

    Returns:
        A list or tuple of variables which returned by ``body``.
    """

    # NOTE: It may be slower if cond is very expensive, but usually cond is just O(1).
    # If loop_vars is changed during cond callable, then it causes bug, but current logical_and/logical_not/... doesn't change the loop_vars.
    pred = cond()
    if isinstance(pred, Variable):
        loop_vars = _run_paddle_while(cond, body, getter, setter)
    else:
        loop_vars = _run_py_while(cond, body, getter, setter)

    return loop_vars


def _run_paddle_while(cond, body, getter, setter):
    # NOTE: loop_vars of Paddle op `control_flow.while_loop` must be Paddle Tensors.
    def to_list(x):
        if isinstance(x, (tuple, list)): return x
        return [x]

    # UndefinedVar will become data layer not check.
    loop_vars = [to_static_variable(var) for var in to_list(getter())]
    setter(loop_vars if len(loop_vars) > 1 else
           loop_vars[0])  # change the non-local var to variable
    # variable maybe modified to inner var. change it into
    loop_vars = control_flow.while_loop(cond, body, loop_vars)
    setter(loop_vars if len(loop_vars) > 1 else
           loop_vars[0])  # change the non-local var to variable
    return loop_vars


def _run_py_while(cond, body, getter, setter):
    loop_vars = getter()
    while cond():
        loop_vars = body()
    return loop_vars


def convert_logical_and(x_func, y_func):
    """
    A function representation of a Python ``and`` statement.

    Args:
        x_func(callable): x_func() is the left hand operand of ``and`` operator. x_func() is bool or Tensor.
        y_func(callable): y_func() is the right hand operand of ``and`` operator.  y_func() is bool or Tensor.

    Returns:
        A python bool variable or a bool Tensor.

    NOTE(liym27):
        1) The operands are executed sequentially according to the running logic of Python. So here the arguments
        should be callable.
        2) If the left hand operand is False, the right hand operand should be executed.

        For example:
            a = x > 1 and y < 1
        Transformed code:
            a = paddle.jit.dy2static.convert_logical_and(lambda:x>1, lambda:y<1)

          In `convert_logical_and(lambda:x>1, lambda:y<1)`, `lambda:y<1` must be run after `lambda:x>1`. And
        if `x>1` is False, `y<1` should NOT be run.
    """
    x_value = x_func()
    if not isinstance(x_value, Variable):
        return _run_py_logical_and(lambda: x_value, y_func)

    y_value = y_func()
    if not isinstance(y_value, Variable):
        return _run_py_logical_and(lambda: y_value, lambda: x_value)

    return _run_paddle_logical_and(x_value, y_value)


def _run_paddle_logical_and(x, y):
    x = cast_bool_if_necessary(x)
    y = cast_bool_if_necessary(y)
    return logical_and(x, y)


def _run_py_logical_and(x_func, y_func):
    x_value = x_func()
    assert not isinstance(x_value, Variable)

    # NOTE(liym27):
    #  1. Returns y_func() if x_value is False;
    #  2. If x_value is False, y_func() should not be run.
    return x_value and y_func()


def convert_logical_or(x_func, y_func):
    """
    A function representation of a Python ``or`` statement.

    Args:
        x_func(callable): x_func() is the left hand operand of ``or`` operator. x_func() is bool or Tensor.
        y_func(callable): y_func() is the right hand operand of ``or`` operator.  y_func() is bool or Tensor.

    Returns:
        A python bool variable or a bool Tensor.

    NOTE(liym27):
        1) The operands are executed sequentially according to the running logic of Python. So here the arguments
        should be callable.
        2) If the left hand operand is True, the right hand operand should be executed.

        For example:
            a = x > 1 or y < 1
        Transformed code:
            a = paddle.jit.dy2static.convert_logical_or(lambda:x>1, lambda:y<1)

        In `convert_logical_or(lambda:x>1, lambda:y<1)`, `lambda:y<1` must be run after `lambda:x>1`. And
        if `x>1` is True, `y<1` should NOT be run.
    """
    x_value = x_func()
    if not isinstance(x_value, Variable):
        return _run_py_logical_or(lambda: x_value, y_func)

    y_value = y_func()
    if not isinstance(y_value, Variable):
        return _run_py_logical_or(lambda: y_value, lambda: x_value)

    return _run_paddle_logical_or(x_value, y_value)


def _run_paddle_logical_or(x, y):
    x = cast_bool_if_necessary(x)
    y = cast_bool_if_necessary(y)
    return logical_or(x, y)


def _run_py_logical_or(x_func, y_func):
    x_value = x_func()
    assert not isinstance(x_value, Variable)

    # NOTE(liym27):
    #  1. Returns y_func() if x_value is False;
    #  2. If x_value is True, y_func() should not be run.
    return x_value or y_func()


def convert_logical_not(x):
    """
    A function representation of a Python ``not`` statement.

    Args:
        x(bool|Tensor): Operand of ``not`` operator.

    Returns:
        A python bool variable or a bool Tensor.
    """

    if isinstance(x, Variable):
        return _run_paddle_logical_not(x)
    else:
        return _run_py_logical_not(x)


def _run_paddle_logical_not(x):
    x = cast_bool_if_necessary(x)
    return logical_not(x)


def _run_py_logical_not(x):
    return not x


def convert_ifelse(pred, true_fn, false_fn, get_args, set_args,
                   return_name_ids):
    """
    A function representation of a Python ``if/else`` statement.

    Args:
        pred(bool|Tensor): A boolean Tensor which determines whether to return the result of ``true_fn`` or ``false_fn`` .
        true_fn(callable): A callable to be performed if ``pred`` is true.
        false_fn(callable): A callable to be performed if ``pred`` is false.
        get_args(callable): Get all arguments that needed in true_fn and false_fn.
        set_args(callable): Update arguments that modified in trure_fn and false_fn.

    Returns:
        ``true_fn()`` if the predicate ``pred`` is true else ``false_fn()`` .

    """
    if isinstance(pred, Variable):
        out = _run_paddle_cond(pred, true_fn, false_fn, get_args, set_args,
                               return_name_ids)
    else:
        out = _run_py_ifelse(pred, true_fn, false_fn, get_args, set_args,
                             return_name_ids)

    return out


def _run_paddle_cond(pred, true_fn, false_fn, get_args, set_args,
                     return_name_ids):
    """
    Paddle cond API will evaluate both ture_fn and false_fn codes.
    """
    pred = cast_bool_if_necessary(pred)
    init_args = get_args()

    def new_true_fn():
        set_args(init_args)
        outs = true_fn()
        _check_no_undefined_var(outs, return_name_ids, 'if_body')
        return outs

    def new_false_fn():
        set_args(init_args)
        outs = false_fn()
        _check_no_undefined_var(outs, return_name_ids, 'else_body')
        return outs

    cond_outs = control_flow.cond(pred, new_true_fn, new_false_fn)
    return _recover_args_state(cond_outs, get_args, set_args, return_name_ids)


def _run_py_ifelse(pred, true_fn, false_fn, get_args, set_args,
                   return_name_ids):
    """
    Evaluate python original branch function if-else.
    """
    py_outs = true_fn() if pred else false_fn()
    py_outs = _remove_no_value_return_var(py_outs)
    return _recover_args_state(py_outs, get_args, set_args, return_name_ids)


def _remove_no_value_return_var(out):
    if isinstance(out, tuple) and len(out) > 0:
        processed_out = out
        align_ret = out[0]
        if isinstance(align_ret, tuple):
            for index, item in enumerate(align_ret):
                if isinstance(item, Variable) and (RETURN_NO_VALUE_VAR_NAME
                                                   in item.name):
                    # return None
                    if index == 0:
                        processed_out = (None, ) + out[1:]
                    elif index == 1:
                        processed_out = align_ret[:1] + out[1:]
                    else:
                        processed_out = (align_ret[:index], ) + out[1:]
                    break

        for index, item in enumerate(processed_out):
            if isinstance(item, Variable) and (RETURN_NO_VALUE_VAR_NAME
                                               in item.name):
                processed_out = processed_out[:index]

        if not processed_out:
            return None
        elif len(processed_out) == 1:
            return processed_out[0]
        else:
            return processed_out

    else:
        return out


def _check_no_undefined_var(outs, names, branch_name):
    if names is None: return
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    for var, name in zip(list(outs), names):
        if isinstance(var, UndefinedVar):
            raise ValueError(
                "Required '{}' must be initialized both in if-else branch, but found it not initialized in '{}'."
                .format(name, branch_name))


def _recover_args_state(outs, get_args, set_args, return_name_ids):
    """
    Currently we support variant length of early return statement by padding
    _no_return_value.

    # TODO(dev): We shall consider to evaluate whether should support this for Python if-else?
    """
    # IfExpr's return_name_ids maybe None
    if return_name_ids is None:
        return outs

    init_args = get_args()
    # recover args state
    num_outs = len(return_name_ids)
    num_args = len(init_args)
    assert num_outs <= num_args

    if num_args == 1:
        final_outs = (outs, )
    else:
        outs = (outs, ) if num_outs == 1 else outs
        final_outs = outs + init_args[num_outs:]

    set_args(final_outs)
    return final_outs


def convert_len(var):
    """
    Returns variable(length) from shape ops based on var.type

    Note: In addition to some ast transformations, some block-related
          operations are added in `len` transformation, such as appending
          `shape_op` in var.block.
    """
    if isinstance(var, Variable):
        if var.type in [
                core.VarDesc.VarType.LOD_TENSOR,
                core.VarDesc.VarType.SELECTED_ROWS
        ]:
            # Note: Length of var may be known ahead of time in dygraph,
            # but it probably represents batch size which can be variant.
            # so we return a variable dynamically inferred from var.shape.
            return nn.shape(var)[0]
        elif var.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY:
            return control_flow.array_length(var)
        else:
            raise TypeError(
                'len(var) only supports LoDTensor/LoDTensorArray/SelectedRows, but received %s.'
                % type(var))
    else:
        return len(var)


def convert_zip(*args):
    for i, arg in enumerate(args):
        if isinstance(arg, Variable) and arg.shape[0] == -1:
            raise RuntimeError(
                "Not support zip(tensor, ...) when tensor.shape[0] == -1, "
                "but found args[{}].shape[0] == -1 in 'zip'".format(str(i)))
    return zip(*args)


def convert_shape(x):
    """
    A function representation of the shape of variable.
    """

    def has_negative(list_shape):
        return any([x < 0 for x in list_shape])

    # When `x` is Variable:
    #  (1) if x.shape contains -1, such as [2, -1, 64], returns [2, var, 64],
    #      where var = paddle.shape(x)[1]

    #  (2) if x.shape does not contains -1, return lsit(x.shape) directly

    if isinstance(x, Variable):
        values = list(x.shape)
        if has_negative(values):
            shape_tensor = nn.shape(x)
            for i, v in enumerate(values):
                if v is None or v < 0:
                    values[i] = shape_tensor[i]
        return values
    else:
        return x.shape


def convert_shape_compare(left, *args):
    """
    A function handles comparison difference between Paddle and Python.
    For example, if x and y are Tensors, x.shape == y.shape will return single
    boolean Value (True/False). However, paddle.shape(x) == paddle.shape(y) is
    an element-wise comparison. The difference can cause dy2stat error. So we
    create this function to handle the difference.

    Args:
        left: variable
        *args: compare_op(str), variable, compare_op(str), variable, where
            compare_op means "<", ">", "==", "!=", etc.
    Returns:
        If the variables to compare are NOT Paddle Variables, we will return as
        Python like "a op1 b and b op2 c and ... ".
        If the variables to compare are Paddle Variables, we will do elementwise
        comparsion first and then reduce to a boolean whose numel is 1.
        
    """
    args_len = len(args)
    assert args_len >= 2, "convert_shape_compare needs at least one right compare variable"
    assert args_len % 2 == 0, "Illegal input for convert_shape_compare, *args should be op(str), var, op(str), var ..."
    num_cmp = args_len // 2
    if isinstance(left, Variable):

        def reduce_compare(x, op_str, y):
            element_wise_result = eval("x " + op_str + " y")
            if op_str == "!=":
                return reduce_any(element_wise_result)
            elif op_str == "is" or op_str == "is not" or op_str == "in" or op_str == "not in":
                return element_wise_result
            else:
                return reduce_all(element_wise_result)

        final_result = reduce_compare(left, args[0], args[1])
        for i in range(1, num_cmp):
            cmp_left = args[i * 2 - 1]
            cmp_op = args[i * 2]
            cmp_right = args[i * 2 + 1]
            cur_result = reduce_compare(cmp_left, cmp_op, cmp_right)
            final_result = convert_logical_and(lambda: final_result,
                                               lambda: cur_result)
        return final_result
    else:
        cmp_left = left
        final_result = None
        for i in range(num_cmp):
            cmp_op = args[i * 2]
            cmp_right = args[i * 2 + 1]
            cur_result = eval("cmp_left " + cmp_op + " cmp_right")
            if final_result is None:
                final_result = cur_result
            else:
                final_result = final_result and cur_result

            if final_result is False:
                return False
            cmp_left = cmp_right
        return final_result


def cast_bool_if_necessary(var):
    assert isinstance(var, Variable)
    if convert_dtype(var.dtype) not in ['bool']:
        var = cast(var, dtype="bool")
    return var


def convert_var_dtype(var, dtype):
    if isinstance(var, Variable):
        src_dtype = convert_dtype(var.dtype)
        assert src_dtype in [
            'bool', 'float16', 'float32', 'float64', 'int32', 'int64', 'uint8'
        ], "The dtype of var {} is {}, which is not supported in the cast op.".format(
            var.name, src_dtype)
        assert dtype in [
            'bool', 'int', 'float'
        ], "The casted target dtype is {}, which is not supported in type casting.".format(
            dtype)
        cast_map = {
            'bool': 'bool',
            'int': 'int32',
            'float': 'float32',
        }
        return cast(var, dtype=cast_map[dtype])
    else:
        return eval('{}(var)'.format(dtype))


def convert_assert(cond, message=""):
    """
    A function representation of a Python ``assert`` statement.
    """
    if isinstance(cond, Variable):
        cond = cast(cond, "bool")
        # NOTE: message is not used because Paddle Assert has no corresponding parameter to use.
        return Assert(cond)
    else:
        assert cond, message


def convert_print(*args):
    """
    A function representing Python ``print`` statement. Note: this is a basic
    python function so we haven't handle sep, end, file and flush parameters of
    python function.
    """
    for var in args:
        if isinstance(var, Variable):
            var = Print(var)
        else:
            print(var)


def convert_pop(target, *args):
    """
    A function representation of a Python pop statement for a list or dict.

    Args:
        target(list|dict|Tensor): A variable to pop item from.
        *args(tuple): index or default value to parse.

    Returns:
        A item poped from target.
    """

    is_variable = isinstance(target, Variable)
    if is_variable:
        is_tensor_array = target.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY

    if is_variable and is_tensor_array:
        return _run_paddle_pop(target, *args)
    else:
        return _run_python_pop(target, *args)


def _run_paddle_pop(array, *args):
    if len(args) == 0:
        idx = -1
    else:
        idx = args[0]

    assert isinstance(idx, int)

    def cond(i, new_array):
        return less_than(i, arr_len)

    def body(i, new_array):
        item = array_read(array=array, i=i)
        array_write(item, array_length(new_array), new_array)
        i = increment(i)
        return i, new_array

    arr_len = array_length(array)
    if idx < 0:
        idx = idx + arr_len
    else:
        idx = fill_constant(shape=[1], dtype="int64", value=idx)

    pop_item = array_read(array, idx)

    new_array = _slice_tensor_array(array, 0, idx)
    i = idx + 1
    _, new_array = while_loop(cond, body, [i, new_array])
    assign(input=new_array, output=array)

    return pop_item


# TODO(liym27): A better way to slice tensor array.
#  Maybe support start == end for slice op.
def _slice_tensor_array(array, start, end):

    def true_fn():
        null_array = create_array("float32")
        return null_array

    def false_fn(array, start, end):
        new_array = slice(array, starts=[start], ends=[end], axes=[0])
        return new_array

    new_array = cond(start == end, true_fn, lambda: false_fn(array, start, end))
    return new_array


def _run_python_pop(target, *args):
    # 1. pop for a dict
    if len(args) == 2:
        idx, default = args
        return target.pop(idx, default)

    # 2. pop for a list or dict
    else:
        idx = args[0] if args else -1
        return target.pop(idx)
