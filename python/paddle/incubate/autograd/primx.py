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

import logging
import typing
from collections import OrderedDict

import paddle
from paddle.base import framework
from paddle.base.core import ops_contain_none, prim_config
from paddle.base.framework import Operator, default_main_program
from paddle.incubate.autograd.utils import as_tensors

from .composite_rules import _composite
from .primreg import (
    lookup_composite,
    lookup_orig2prim,
    lookup_prim2orig,
)
from .primrules import _orig2prim, _prim2orig
from .utils import (
    flatten_and_remove_none,
    get_input_var_list,
    get_output_var_list,
    map_output_for_composite,
    prepare_python_api_arguments,
)


def topo_path(xs, ys, block=None):
    """Returns the list of ops on the path from `xs` to `ys` in topological
    order.

    TODO(Tongxin): supporting control flow and nested blocks.
    Args:
        xs: a list|tuple of vars as source
        ys: a list|tuple of vars as sink
        block: the program block containing the path, optional
    Returns:
        (path, unused_xs, unreached_ys): a tuple comprised of the resulting op
        path, the unused variables in `xs`, and the unreached variables in `ys`
    """

    block = default_main_program().current_block() if block is None else block

    path = []
    backpath = []
    reached_vars = OrderedDict()
    used_vars = OrderedDict()

    # Initialize reached vars
    for x in xs:
        assert (
            x is None or x.block == block
        ), 'x is not None and x.block != block'
        reached_vars[id(x)] = x

    # Reaching test, returning whether an op is reached from the given input
    reaching = lambda op: any(
        id(v) in reached_vars
        for v in flatten_and_remove_none(get_input_var_list(op))
    )

    # block.ops are supposedly in the order that preserves correct data
    # dependence.
    # Forward pass to identify all reached variables and ops
    for op in block.ops:
        if reaching(op):
            path.append(op)
            for var in flatten_and_remove_none(get_output_var_list(op)):
                reached_vars[id(var)] = var

    used_vars = OrderedDict((id(y), y) for y in ys if id(y) in reached_vars)
    back_reaching = lambda op: any(
        id(out) in used_vars
        for out in flatten_and_remove_none(get_output_var_list(op))
    )

    # Backward pass to find all used variables
    for op in reversed(path):
        if back_reaching(op):
            backpath.append(op)
            for var in flatten_and_remove_none(get_input_var_list(op)):
                used_vars[id(var)] = var

    unused_xs = [x for x in xs if id(x) not in used_vars]
    unreached_ys = [y for y in ys if id(y) not in reached_vars]

    return list(reversed(backpath)), unused_xs, unreached_ys


def output_vars_on_path(path):
    """Returns the output variables of all the ops on the path from `xs`
    to `ys`.

    Args:
        path: a list of ops on which to find the output variables

    Returns:
        vars: the output vars
    """
    vars = OrderedDict()
    for op in path:
        for out in flatten_and_remove_none(get_output_var_list(op)):
            vars[id(out)] = out

    return vars


class VarMap:
    """A general map data structure for linking variables to variables.

    An example is linking variables to their gradients.
    """

    __slots__ = ['name', 'varset', 'tab']

    def __init__(self, name, varset):
        self.name = name
        self.varset = varset
        self.tab = OrderedDict()

    def add(self, key_var, value_var):
        self.tab[id(key_var)] = id(value_var)

    def add_rec(self, key_vars, value_vars):
        if value_vars is None:
            return
        if isinstance(key_vars, paddle.base.framework.Variable):
            if not isinstance(value_vars, paddle.base.framework.Variable):
                raise TypeError(
                    f'value_vars must be Variable, but got {type(value_vars)}'
                )
            self.tab[id(key_vars)] = id(value_vars)
        else:
            assert len(key_vars) == len(value_vars), (
                f'len(key_vars) should be equal to len(value_vars), '
                f'but len(key_vars)={len(key_vars)} and len(value_vars)={len(value_vars)}.'
            )
            for key_var, value_var in zip(key_vars, value_vars):
                self.add_rec(key_var, value_var)

    def lookup(self, key_var):
        value_id = self.tab.get(id(key_var))
        if value_id is not None:
            return self.varset.get(value_id)
        else:
            return None

    def delete(self, key_var):
        varid = id(key_var)
        if varid in self.tab:
            del self.tab[id(key_var)]

    def delete_keyvars(self, key_vars):
        for var in key_vars:
            varid = id(var)
            if varid in self.tab:
                del self.tab[varid]

    def delete_valuevars(self, value_vars):
        ids = [id(v) for v in value_vars]
        keys = [k for k, v in self.tab.items() if v in ids]
        for k in keys:
            del self.tab[k]

    def contain_var(self, key_var):
        return self.tab.__contains__(id(key_var))

    def contain_value(self, value_var):
        return id(value_var) in self.tab.values()


# TODO(lml): supporting control flow, nested blocks, and block other than current block of main program.
class Transform:
    """An object that maintains the state of transformations applied to a
    primitive program."""

    def __init__(self, block):
        assert (
            block == default_main_program().current_block()
        ), 'only support transform on current block of main program.'
        self.block = block
        self.vars = self.init_vars(block)
        self.var2dot = VarMap('var2dot', self.vars)
        self.dot2bar = VarMap('dot2var', self.vars)

    def init_vars(self, block):
        vars = OrderedDict()
        for _, var in block.vars.items():
            vars[id(var)] = var
        return vars

    def add_vars(self, new_vars):
        self.vars.update({id(v): v for v in new_vars if v is not None})

    def add_vars_rec(self, new_vars):
        if new_vars is None:
            return
        if isinstance(new_vars, paddle.base.framework.Variable):
            self.vars.update({id(new_vars): new_vars})
            return
        if not isinstance(new_vars, list):
            raise TypeError(f'new_vars must be list, but got {type(new_vars)}')
        for var in new_vars:
            self.add_vars_rec(var)

    def erase_ops(self, ordered_indexes):
        block = self.block
        for op_index in reversed(ordered_indexes):
            block.desc._remove_op(op_index, op_index + 1)

        # remove from block.ops
        for op_index in reversed(ordered_indexes):
            del block.ops[op_index]

        block._sync_with_cpp()

    def erase_dots(self, vars_to_erase):
        for var in vars_to_erase:
            if id(var) in self.vars:
                del self.vars[id(var)]
        self.dot2bar.delete_keyvars(vars_to_erase)
        self.var2dot.delete_valuevars(vars_to_erase)
        block = self.block
        for var in vars_to_erase:
            name = var.name
            block.desc._remove_var(name.encode())
            del block.vars[name]
        block._sync_with_cpp()

    def var2dot_rec(self, vars):
        """Lookup var2dot recursively."""
        if isinstance(vars, paddle.base.framework.Variable):
            dot = self.var2dot.lookup(vars)
            return dot

        dots = [self.var2dot_rec(var) for var in vars]
        return dots

    def dot2bar_rec(self, dots):
        if isinstance(dots, paddle.base.framework.Variable):
            bar = self.dot2bar.lookup(dots)
            assert bar is not None, 'bar must be not None'
            return bar

        bars = [self.dot2bar_rec(dot) for dot in dots]
        return bars


# TODO(lml): supporting control flow, nested blocks, and block other than current block of main program.
def _lower(block, reverse, blacklist):
    # Some functions which are only used in _lower.
    def bind(args, to_bind, value_table):
        for i in range(len(args)):
            if isinstance(args[i], list):
                bind(args[i], to_bind, value_table)
            elif args[i] is not None and args[i].name in to_bind:
                args[i] = value_table[to_bind[args[i].name]]

    def bind_name(names, to_bind):
        return_list = []
        for name in names:
            if isinstance(name, list):
                return_list.append(bind_name(name, to_bind))
            else:
                return_list.append(to_bind[name] if name in to_bind else name)
        return return_list

    def expand_nested_list(xs):
        return_list = []
        for x in xs:
            if isinstance(x, list):
                return_list = return_list + expand_nested_list(x)
            else:
                return_list.append(x)
        return return_list

    # Step1: Do some preparatory work for lower
    lower_fn = _prim2orig if reverse else _orig2prim
    lookup_fn = lookup_prim2orig if reverse else lookup_orig2prim

    value_table = {}
    to_bind = {}
    to_bind_rev = {}
    for var in block.desc.all_vars():
        value_table[var.name()] = block.var(var.name())

    ops_to_remove = []
    vars_to_remove = set()

    # Step2: Process all ops in the target block
    for op_idx in range(len(block.ops)):
        op = block.ops[op_idx]
        ops_to_remove.append(op_idx)
        if lookup_fn(op.type) is not None and op.type not in blacklist:
            input_args = get_input_var_list(op)
            bind(input_args, to_bind, value_table)

            for orig_out, new_out in zip(
                expand_nested_list(get_output_var_list(op)),
                expand_nested_list(as_tensors(lower_fn(op, *input_args))),
            ):
                assert not (orig_out is None) ^ (
                    new_out is None
                ), "orig_out and new_out should match."
                vars_to_remove.add(new_out.name)
                value_table[new_out.name] = new_out
                to_bind[orig_out.name] = new_out.name
                to_bind_rev[new_out.name] = orig_out.name
        else:
            inputs = {}
            for i in range(len(op.input_names)):
                inputs[op.input_names[i]] = bind_name(
                    op.input(op.input_names[i]), to_bind
                )

            outputs = {}
            for i in range(len(op.output_names)):
                outputs[op.output_names[i]] = op.output(op.output_names[i])

            attrs = {}
            for name in sorted(op.attr_names):
                attrs[name] = op.attr(name)
            from paddle.base.dygraph.base import param_guard

            new_op_desc = block.desc.append_op()
            with param_guard(inputs), param_guard(outputs):
                op = Operator(
                    block=block,
                    desc=new_op_desc,
                    type=op.type,
                    inputs=inputs,
                    outputs=outputs,
                    attrs=attrs,
                )
            block.ops.append(op)

    # Step3: Do some post-processing work
    for op_idx in reversed(ops_to_remove):
        block.desc._remove_op(op_idx, op_idx + 1)
        del block.ops[op_idx]
    block._sync_with_cpp()

    for op_idx in range(len(block.ops)):
        op = block.ops[op_idx]
        for in_name in op.input_arg_names:
            if in_name in to_bind_rev:
                op._rename_input(in_name, to_bind_rev[in_name])

        for out_name in op.output_arg_names:
            if out_name in to_bind_rev:
                op._rename_output(out_name, to_bind_rev[out_name])

    for var_name in sorted(vars_to_remove):
        assert (
            var_name in to_bind_rev
        ), f'var_name "{var_name}" is not in to_bind_rev.'
        if var_name != to_bind_rev[var_name]:
            block.desc._remove_var(var_name.encode())
            del block.vars[var_name]
    block._sync_with_cpp()


def _lower_composite(
    block,
    filter_: typing.Callable[[framework.Operator], bool] = lambda x: True,
    start_idx=-1,
    backward_length=-1,
):
    """The operators in block which satisfy the filter condition will be decomposite into primitives."""

    def bind(args, to_bind, value_table):
        for i in range(len(args)):
            if isinstance(args[i], list):
                bind(args[i], to_bind, value_table)
            if not isinstance(args[i], paddle.base.framework.Variable):
                continue
            elif args[i] is not None and args[i].name in to_bind:
                args[i] = value_table[to_bind[args[i].name]]

    def bind_name(names, to_bind):
        return_list = []
        for name in names:
            if isinstance(name, list):
                return_list.append(bind_name(name, to_bind))
            else:
                return_list.append(to_bind[name] if name in to_bind else name)
        return return_list

    def expand_nested_list(xs):
        return_list = []
        for x in xs:
            if isinstance(x, list):
                return_list = return_list + expand_nested_list(x)
            else:
                return_list.append(x)
        return return_list

    if isinstance(block, paddle.base.framework.Block):
        logging.info("Atomize composite op to primitive ops begin.")

        # Step1: Do some preparatory work for lower
        lower_fn = _composite
        lookup_fn = lookup_composite

        value_table = {}
        to_bind = {}
        to_bind_rev = {}
        for var in block.desc.all_vars():
            value_table[var.name()] = block.var(var.name())

        ops_to_remove = []
        vars_to_remove = set()

        # if output var of composite rule is None, this means this var is not needed
        none_vars_to_remove = set()

        change = None

        # Only process required sliced block
        # If given start_idx, only ops[start_idx:] will be processed.
        # If given backward_length, only ops[:-backward_length] will be processed.
        # Note, start_idx and backward_length cannot be both given, because the length of non-processed part must be kept unchanged.
        length = len(block.ops)
        idx_list = range(length)
        assert (
            -1 <= backward_length <= length
        ), f'expect -1 <= backward_length <= {length}, but got backward_length: {backward_length}'
        assert (
            -1 <= start_idx <= length
        ), f'expect -1 <= start_idx <= {length}, but got start_idx: {start_idx}'
        assert not (
            backward_length > -1 and start_idx > -1
        ), f'got start_idx: {start_idx} and backward_length: {backward_length}'
        if backward_length > -1:
            idx_list = range(length - backward_length)
        if start_idx > -1:
            idx_list = range(start_idx, length)

        lower = lower_pre = False  # Flag of routing to lower or copy branch
        # Step2: Process all ops in the target block
        for op_idx in range(length):
            op = block.ops[op_idx]
            ops_to_remove.append(op_idx)
            op_name = op.type

            # NOTE: why need _sync_with_cpp here
            # _sync_with_cpp after every copied operator is very slow.
            # However, _sync_with_cpp only support continuous block currently.
            # The lowering transformation will generate program which is
            # crossed combination of copy block and lower block, such as
            # op1(copy) -> op2(copy) -> op3(lower) -> op4(lower) -> op5(copy) -> op6(copy)
            # It will cause _sync_with_cpp error.
            # So, _sync_with_cpp will be executed only once after every continuous copy block.
            lower = (
                (lookup_fn(op_name) is not None)
                and filter_(op)
                and op_idx in idx_list
            )
            if not lower_pre and lower:
                block._sync_with_cpp()
            lower_pre = lower

            if lower:
                change = True
                prim_config["composite_ops_record"].add(op_name)
                input_args = prepare_python_api_arguments(op)
                bind(input_args, to_bind, value_table)

                orig_outs = expand_nested_list(map_output_for_composite(op))
                new_outs = expand_nested_list(
                    as_tensors(lower_fn(op, *input_args))
                )
                assert len(orig_outs) == len(new_outs), (
                    f'when replace origin op {op_name} with composite rule, num of origin outs should be equal to new outs, '
                    f'but len(orig_outs) = {len(orig_outs)} and len(new_outs) = {len(new_outs)}'
                )

                for orig_out, new_out in zip(
                    orig_outs,
                    new_outs,
                ):
                    if (orig_out is None or new_out is None) and (
                        op_name not in ops_contain_none
                    ):
                        raise ValueError(
                            f"op {op_name} should not contain any None value. original outs={orig_outs} and its composite rule outs={new_outs}"
                        )
                    if orig_out is None:
                        # to keep same as phi op definition, orig_out may receive None
                        continue
                    elif new_out is not None:
                        assert orig_out.dtype == new_out.dtype, (
                            f'when replace origin op {op_name} with composite rule, origin out dtype should be equal to new out dtype, '
                            f'but orig_out: {orig_out.name}.dtype={orig_out.dtype} and new_out: {new_out.name}.dtype={new_out.dtype}'
                        )
                        assert (
                            -1 not in new_out.shape
                        ), f'when replace origin op {op_name} with composite rule, composite out shape has -1.'
                        assert orig_out.shape == new_out.shape, (
                            f'when replace origin op {op_name} with composite rule, origin out shape should be equal to new out shape, '
                            f'but orig_out: {orig_out.name}.shape={orig_out.shape} and new_out: {new_out.name}.shape={new_out.shape}'
                        )
                        assert not (orig_out is None) ^ (
                            new_out is None
                        ), "orig_out and new_out should match."
                        vars_to_remove.add(new_out.name)
                        value_table[new_out.name] = new_out
                        to_bind[orig_out.name] = new_out.name
                        to_bind_rev[new_out.name] = orig_out.name
                    else:
                        none_vars_to_remove.add(orig_out.name)
            else:
                op_desc = block.desc.append_op()
                op_desc.copy_from(op.desc)

        block._sync_with_cpp()
        # Step3: Do some post-processing work
        for op_idx in reversed(ops_to_remove):
            block.desc._remove_op(op_idx, op_idx + 1)
            del block.ops[op_idx]
        block._sync_with_cpp()

        for op_idx in range(len(block.ops)):
            op = block.ops[op_idx]
            for in_name in op.input_arg_names:
                if in_name in to_bind_rev:
                    op._rename_input(in_name, to_bind_rev[in_name])

            for out_name in op.output_arg_names:
                if out_name in to_bind_rev:
                    op._rename_output(out_name, to_bind_rev[out_name])

        for var_name in sorted(vars_to_remove):
            assert (
                var_name in to_bind_rev
            ), f'var_name "{var_name}" is not in to_bind_rev.'
            if var_name != to_bind_rev[var_name]:
                block.desc._remove_var(var_name.encode())
                del block.vars[var_name]
        block._sync_with_cpp()

        for var_name in sorted(none_vars_to_remove):
            block.desc._remove_var(var_name.encode())
            del block.vars[var_name]
        block._sync_with_cpp()

        for op in block.ops:
            if op._has_kernel(op.desc.type()):
                op.desc.infer_var_type(block.desc)
                op.desc.infer_shape(block.desc)
        # composite ops may contain other composite ops, thus, call _lower_composite again.
        if change:
            _lower_composite(
                block,
                filter_,
                start_idx=start_idx,
                backward_length=backward_length,
            )
        return

    elif isinstance(block, typing.Sequence):
        for item in block:
            _lower_composite(
                item,
                filter_,
                start_idx=start_idx,
                backward_length=backward_length,
            )
        return
    else:
        raise TypeError


@framework.static_only
def orig2prim(block=None):
    """
    Note:
        **This API is ONLY available in the static graph mode.**
        **Args block must be None or current block of main program.**

    All operators in the target block are processed as follows.
    If it is an original operator, it will be transformed into
    one or a series of automatic differential basic operators with
    equivalent function.

    Args:
        block(paddle.static.Block|None, optional): The
            target block to process on. Default None, and will
            process on the current block of main program.
    """

    block = default_main_program().current_block() if block is None else block
    assert (
        block == default_main_program().current_block()
    ), 'block is neither None nor current block of main program'
    _lower(block, reverse=False, blacklist=[])


@framework.static_only
def prim2orig(block=None, blacklist=None):
    """
    Note:
        **ONLY available in the static graph mode.**
        **Args block must be None or current block of main program.**

    All operators in the target block are processed as follows.
    If it is an automatic differential basic operator, it will be
    transformed into one or a series of original operators with
    equivalent function to support execution.

    Args:
        block(paddle.static.Block|None, optional): The
            target block to process on. Default None, and will
            process on the current block of main program.
        blacklist(list[string]|None, optional): The names of automatic
            differential basic operator that will not be transformed
            into original operators. Default None, and the blacklist
            is treated as empty list.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> from paddle.incubate.autograd import enable_prim, prim_enabled, prim2orig

            >>> paddle.enable_static()
            >>> enable_prim()

            >>> x = paddle.ones(shape=[2, 2], dtype='float32')
            >>> x.stop_gradients = False
            >>> y = x * x
            >>> dy_dx = paddle.static.gradients(y, x)
            >>> if prim_enabled():
            ...     prim2orig()
    """

    block = default_main_program().current_block() if block is None else block
    assert (
        block == default_main_program().current_block()
    ), 'block is neither None nor current block of main program'
    blacklist = [] if blacklist is None else blacklist
    _lower(block, reverse=True, blacklist=blacklist)
