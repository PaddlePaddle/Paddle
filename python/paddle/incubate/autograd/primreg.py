# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class Registry:
    """A general registry object."""

    __slots__ = ['name', 'tab']

    def __init__(self, name):
        self.name = name
        self.tab = {}

    def register(self, name, value):
        assert (
            name not in self.tab
        ), f'name "{name}" should not be registered before.'
        self.tab[name] = value

    def lookup(self, name):
        return self.tab.get(name)


_primop_fn = Registry('primop_fn')
_orig2prim = Registry('orig2prim')
_prim2orig = Registry('prim2orig')
_primop_jvp = Registry('primop_jvp')
_primop_transpose = Registry('primop_transpose')
_primop_position_argnames = Registry('primop_position_argnames')
_composite_ops = Registry('composite')


def lookup_fn(optype):
    return _primop_fn.lookup(optype)


def lookup_orig2prim(optype):
    return _orig2prim.lookup(optype)


def lookup_prim2orig(optype):
    return _prim2orig.lookup(optype)


def lookup_jvp(optype):
    return _primop_jvp.lookup(optype)


def lookup_transpose(optype):
    return _primop_transpose.lookup(optype)


def lookup_composite(optype):
    return _composite_ops.lookup(optype)


def op_position_inputs(op):
    """
    Returns the position inputs of `op` as registered with REGISTER_FN.

    Args:
        op(Operator): The op that needs to get the inputs

    Returns:
        Tensor(s): Inputs of the op

    Examples:
        .. code-block:: python
            @REGISTER_FN('div_p', 'X', 'Y', 'Z')
            def div(x, y, out=None):
                return _simple_binop(LayerHelper('div_p', **locals()))

    The registered inputs are ['X', 'Y'] for div_p and accordingly this
    function will return inputs in the order of X then Y.

    """
    args = _primop_position_argnames.lookup(op.type)
    assert (
        args is not None
    ), f'args of {op.type} should not be None in op_position_inputs().'
    *input_names, _ = args

    inputs = []
    for name in input_names:
        vars = list(map(op.block.var, op.input(name)))
        assert (
            len(vars) >= 0
        ), f'len(vars) should be greater than or equal to 0, but len(vars)={len(vars)}.'
        if len(vars) > 1:
            inputs.append(vars)
        else:
            inputs.append(vars[0])

    return inputs


def op_position_output(op):
    """
    Returns the output of `op` as registered with REGISTER_FN.

    Args:
        op(Operator): The op that needs to get the output

    Returns:
        Tensor(s): Output of the op

    Examples:
        .. code-block:: python
            @REGISTER_FN('div_p', 'X', 'Y', 'Z')
            def div(x, y, out=None):
                return _simple_binop(LayerHelper('div_p', **locals()))

    The registered output is ['Z'] for div_p and accordingly this
    function will return output Z.

    """
    args = _primop_position_argnames.lookup(op.type)
    assert args is not None, 'args should not be None in op_position_output().'
    *_, output_name = args

    outvars = list(map(op.block.var, op.output(output_name)))
    assert (
        len(outvars) >= 0
    ), f'len(outvars) should be greater than or equal to 0, but len(outvars)={len(outvars)}.'
    if len(outvars) > 1:
        output = outvars
    else:
        output = outvars[0]

    return output


def REGISTER_FN(op_type, *position_argnames):
    """
    Decorator for registering the Python function for a primitive op.

    Args:
        op_type(str): The op name
        position_argnames(list[str]): Input and output names of the op

    Returns:
        wrapper: Inner wrapper function

    Examples:
        .. code-block:: python
        @REGISTER_FN('tanh_p', 'X', 'Y')
        def tanh(x, out=None):
            return _simple_unop(LayerHelper('tanh_p', **locals()))

    """

    if not isinstance(op_type, str):
        raise TypeError(f'op_type must be str, but got {type(op_type)}.')

    _primop_position_argnames.register(op_type, position_argnames)

    def wrapper(f):
        _primop_fn.register(op_type, f)
        return f

    return wrapper


def REGISTER_ORIG2PRIM(op_type):
    """
    Decorator for registering the lower function for an original op into sequence of primitive ops.

    Args:
        op_type(str): The op name

    Returns:
        wrapper: Inner wrapper function

    Examples:
        .. code-block:: python
            @REGISTER_ORIG2PRIM('tanh')
            def tanh_orig2prim(op):
                x, = get_input_var_list(op)
                return primops.tanh(x)

    """
    if not isinstance(op_type, str):
        raise TypeError(f'op_type must be str, but got {type(op_type)}.')

    def wrapper(f):
        def _lower(op, *args, **kwargs):
            assert (
                op.type == op_type
            ), f'op.type should be equal to op_type, but op.type is {op.type} and op_type is {op_type}'
            return f(op, *args, **kwargs)

        _orig2prim.register(op_type, _lower)

    return wrapper


def REGISTER_COMPOSITE(op_type):
    """
    Decorator for registering the lower function for an original op into sequence of primitive ops.

    Args:
        op_type(str): The op name

    Returns:
        wrapper: Inner wrapper function

    Examples:
        .. code-block:: python
            @REGISTER_COMPOSITE('softmax')
            def softmax_composite(x, axis):
                molecular = exp(x)
                denominator = broadcast_to(sum(molecular, axis=axis, keepdim=True), x.shape)
                res = divide(molecular, denominator)
                return res

    """
    if not isinstance(op_type, str):
        raise TypeError(f'op_type must be str, but got {type(op_type)}.')

    def wrapper(f):
        def _lower(op, *args, **kwargs):
            assert (
                op.type == op_type
            ), f'op.type should be equal to op_type, but op.type is {op.type} and op_type is {op_type}'
            return f(*args, **kwargs)

        _composite_ops.register(op_type, _lower)

    return wrapper


def REGISTER_PRIM2ORIG(op_type):
    """
    Decorator for registering the lower function for an primitive op into sequence of original ops.

    Args:
        op_type(str): The op name

    Returns:
        wrapper: Inner wrapper function

    Examples:
        .. code-block:: python
            @REGISTER_PRIM2ORIG('tanh_p')
            def tanh_prim2orig(op):
                x, = get_input_var_list(op)
                return paddle.tanh(x)

    """
    if not isinstance(op_type, str):
        raise TypeError(f'op_type must be str, but got {type(op_type)}.')

    def wrapper(f):
        def _lower(op, *args, **kwargs):
            assert (
                op.type == op_type
            ), f'op.type should be equal to op_type, but op.type is {op.type} and op_type is {op_type}'
            return f(op, *args, **kwargs)

        _prim2orig.register(op_type, _lower)

    return wrapper


def REGISTER_JVP(op_type):
    """
    Decorator for registering the JVP function for a primitive op.

    Args:
        op_type(str): The op name

    Returns:
        wrapper: Inner wrapper function

    Examples:
        .. code-block:: python
            @REGISTER_JVP('add_p')
            def add_jvp(op, x_dot, y_dot):
                return primops.add(x_dot, y_dot)

    """
    if not isinstance(op_type, str):
        raise TypeError(f'op_type must be str, but got {type(op_type)}.')

    def wrapper(f):
        def _jvp(op, *args, **kwargs):
            assert (
                op.type == op_type
            ), f'op.type should be equal to op_type, but op.type is {op.type} and op_type is {op_type}'
            return f(op, *args, **kwargs)

        _primop_jvp.register(op_type, _jvp)
        return f

    return wrapper


def REGISTER_TRANSPOSE(op_type):
    """
    Decorator for registering the transpose function for a primitive op
    that denotes a linear operation in the forward AD graph.

    Args:
        op_type(str): The op name

    Returns:
        wrapper: Inner wrapper function

    Examples:
        .. code-block:: python
            @REGISTER_TRANSPOSE('add_p')
            def add_transpose(op, z_bar):
                return z_bar, z_bar

    """
    if not isinstance(op_type, str):
        raise TypeError(f'op_type must be str, but got {type(op_type)}.')

    def wrapper(f):
        def _transpose(op, dot_checker, *args, **kwargs):
            assert (
                op.type == op_type
            ), f'op.type should be equal to op_type, but op.type is {op.type} and op_type is {op_type}'
            return f(op, dot_checker, *args, **kwargs)

        _primop_transpose.register(op_type, _transpose)
        return f

    return wrapper
