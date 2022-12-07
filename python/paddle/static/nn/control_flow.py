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
from functools import partial

import paddle
from paddle import _C_ops
from paddle.common_ops_import import LayerHelper
from paddle.fluid.control_flow import (
    ConditionalBlock,
    _to_sequence_except_dict,
    select_input,
    support_ret_buildin_type,
)
from paddle.fluid.data_feeder import check_type, check_variable_and_dtype
from paddle.fluid.tensor import assign
from paddle.fluid.utils import (
    assert_same_structure,
    flatten,
    is_sequence,
    map_structure,
    pack_sequence_as,
)
from paddle.framework import Variable, _non_static_mode, core, in_dygraph_mode

__all__ = [
    'Assert',
    'increment',
    'cond',
]


def Assert(cond, data=None, summarize=20, name=None):
    '''
    This API creates an op that asserts the given condition is true. If the
    condition is false, prints the tensors in data. ``summarize`` specifies the
    number of the elements in the tensors to print.

    Args:
        cond (Variable): The boolean condition tensor whose numel should be 1.
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

    Raises:
        TypeError: If ``cond`` is not boolean Variable.
        TypeError: If ``data`` is not a list or tuple or ``None``.
        TypeError: If ``summarize`` is not int.
        TypeError: If ``name`` is not a string or ``None`` .
        framework.core.EnforceNotMet: If the condition is False in running time.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.static.nn.control_flow import Assert

            x = paddle.full([2, 3], 2.0, 'float32')
            condition = paddle.max(x) < 1.0 # False
            Assert(condition, [x], 10, "example_assert_layer")

            exe = paddle.static.Executor()
            try:
                exe.run(paddle.static.default_main_program())
                # Print x and throws paddle.framework.core.EnforceNotMet exception
                # Example printed message for x:
                #
                # Variable: fill_constant_0.tmp_0
                #   - lod: {}
                #   - place: CPUPlace()
                #   - shape: [2, 3]
                #   - layout: NCHW
                #   - dtype: float
                #   - data: [2 2 2 2 2 2]
            except paddle.framework.core.EnforceNotMet as e:
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


def increment(x, value=1.0, in_place=True):
    """
    The OP is usually used for control flow to increment the data of :attr:`x` by an amount :attr:`value`.
    Notice that the number of elements in :attr:`x` must be equal to 1.

    Parameters:
        x (Variable): A tensor that must always contain only one element, its data type supports
            float32, float64, int32 and int64.
        value (float, optional): The amount to increment the data of :attr:`x`. Default: 1.0.
        in_place (bool, optional): Whether the OP should be performed in-place. Default: True.

    Returns:
        Variable: The elementwise-incremented tensor with the same shape and data type as :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle
          from paddle.static.nn.control_flow import increment

          counter = paddle.zeros(shape=[1], dtype='float32') # [0.]
          increment(counter) # [1.]
    """
    if in_dygraph_mode():
        return _C_ops.increment_(x, value)

    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'increment'
    )
    helper = LayerHelper("increment", **locals())
    if not in_place:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = x
    helper.append_op(
        type='increment',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'step': float(value)},
    )
    return out


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

    def _is_sequence_except_dict(x):
        """
        In this function, dict is not viewed as sequence.
        """
        if isinstance(x, dict):
            return False
        return is_sequence(x)

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

    Raises:
        TypeError: if ``true_fn`` or ``false_fn`` is not callable.
        ValueError: if ``true_fn`` and ``false_fn`` don't return the same nest
            structure of tensors.

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

    mask = paddle.cast(pred, dtype='int32')
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
