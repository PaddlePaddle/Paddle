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

from paddle.fluid.framework import Variable
from paddle.fluid.layers import control_flow, logical_and, logical_or, logical_not
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import to_static_variable


def convert_while_loop(cond, body, loop_vars):
    """
    A function representation of a Python ``while`` statement.

    Args:
        cond(Callable): A callable object that returns a boolean variable to control whether to  execute the loop body.  It takes  ``loop_vars`` as arguments.
        body(Callable): A callable object that returns a tuple or list of variables with the same arguments ``loops_vars`` as ``cond`` .
        loop_vars(list|tuple): A list or tuple of variables passed to ``cond`` and ``body`` .

    Returns:
        A list or tuple of variables which returned by ``body`` .
    """

    pred = cond(*loop_vars)
    if isinstance(pred, Variable):
        loop_vars = _run_paddle_while_loop(cond, body, loop_vars)
    else:
        loop_vars = _run_py_while(cond, body, loop_vars)

    return loop_vars


def _run_paddle_while_loop(cond, body, loop_vars):
    loop_vars = [to_static_variable(var) for var in loop_vars]
    loop_vars = control_flow.while_loop(cond, body, loop_vars)
    return loop_vars


def _run_py_while(cond, body, loop_vars):
    while cond(*loop_vars):
        loop_vars = body(*loop_vars)
    return loop_vars


def convert_logical_and(x, y):
    """
    A function representation of a Python ``and`` statement.

    Args:
        x(bool|Variable): Left hand operand of ``and`` operator.
        y(bool|Variable): Right hand operand of ``and`` operator.

    Returns:
        A python bool variable or a bool Tensor.
    """

    if isinstance(x, Variable) and isinstance(y, Variable):
        return _run_paddle_logical_and(x, y)

    if not isinstance(x, Variable):
        return _run_py_logical_and(x, y)

    return _run_py_logical_and(y, x)


def _run_paddle_logical_and(x, y):
    return logical_and(x, y)


def _run_py_logical_and(x, y):
    assert not isinstance(x, Variable)
    # NOTE: Returns y if x is True
    return x and y


def convert_logical_or(x, y):
    """
    A function representation of a Python ``or`` statement.

    Args:
        x(bool|Variable): Left hand operand of ``or`` operator.
        y(bool|Variable): Right hand operand of ``or`` operator.

    Returns:
        A python bool variable or a bool Tensor.
    """

    if isinstance(x, Variable) and isinstance(y, Variable):
        return _run_paddle_logical_or(x, y)

    if not isinstance(x, Variable):
        return _run_py_logical_or(x, y)

    return _run_py_logical_or(y, x)


def _run_paddle_logical_or(x, y):
    return logical_or(x, y)


def _run_py_logical_or(x, y):
    assert not isinstance(x, Variable)
    # NOTE: Returns y if x is False
    return x or y


def convert_logical_not(x):
    """
    A function representation of a Python ``not`` statement.

    Args:
        x(bool|Variable): Operand of of ``not`` operator.

    Returns:
        A python bool variable or a bool Tensor.
    """

    if isinstance(x, Variable):
        return _run_paddle_logical_not(x)
    else:
        return _run_py_logical_not(x)


def _run_paddle_logical_not(x):
    return logical_not(x)


def _run_py_logical_not(x):
    return not x
