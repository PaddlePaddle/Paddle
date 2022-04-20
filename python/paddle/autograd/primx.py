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

import paddle
from paddle.fluid import framework as framework
from paddle.fluid.framework import default_main_program, default_startup_program
from paddle.fluid import unique_name, core
from paddle.fluid.framework import Operator
from paddle import compat as cpt
from .primops import fill_const, add
from .primreg import op_position_inputs, op_position_output, lookup_orig2prim, lookup_prim2orig
from .primrules import get_input_vars, get_output_vars, _orig2prim, _prim2orig, _jvp, _transpose
from .primrules import get_input_var_list, get_output_var_list
from collections import OrderedDict


def flatten(inp):
    if inp is None or isinstance(inp, paddle.fluid.framework.Variable):
        return [inp]
    flattened = []
    for part in inp:
        flattened += flatten(part)
    return flattened


def topo_path(xs, ys, block=None):
    """ Returns the list of ops on the path from `xs` to `ys` in topological 
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

    if block is None:
        block = default_main_program().current_block()

    path = []
    backpath = []
    reached_vars = OrderedDict()
    used_vars = OrderedDict()

    # Initialize reached vars
    for x in xs:
        assert x is None or x.block == block
        reached_vars[id(x)] = x

    # Reaching test, returning whether an op is reached from the given input
    reaching = lambda op: any(id(v) in reached_vars for v in get_input_vars(op))

    # block.ops are supposedly in the order that preserves correct data
    # dependence.
    # Forward pass to identify all reached variables and ops
    for op in block.ops:
        if reaching(op):
            path.append(op)
            for var in get_output_vars(op):
                reached_vars[id(var)] = var

    used_vars = OrderedDict((id(y), y) for y in ys if id(y) in reached_vars)
    back_reaching = lambda op: any(id(out) in used_vars for out in get_output_vars(op))

    # Backward pass to find all used variables
    for op in reversed(path):
        if back_reaching(op):
            backpath.append(op)
            for var in get_input_vars(op):
                used_vars[id(var)] = var

    unused_xs = [x for x in xs if id(x) not in used_vars]
    unreached_ys = [y for y in ys if id(y) not in reached_vars]

    return list(reversed(backpath)), unused_xs, unreached_ys


def output_vars_on_path(path):
    """ Returns the output variables of all the ops on the path from `xs`
    to `ys`.
    
    Args:
        path: a list of ops on which to find the output variables

    Returns:
        vars: the output vars
    """
    vars = OrderedDict()
    for op in path:
        for out in get_output_vars(op):
            vars[id(out)] = out

    return vars


class VarMap(object):
    """ A general map data structure for linking variables to variables.
    
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
        if isinstance(key_vars, paddle.fluid.framework.Variable):
            assert isinstance(value_vars, paddle.fluid.framework.Variable)
            self.tab[id(key_vars)] = id(value_vars)
        else:
            assert len(key_vars) == len(value_vars)
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


class Transform(object):
    """ An object that maintains the state of transformations applied to a 
    primitve program. """

    def __init__(self, block):
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
        if isinstance(new_vars, paddle.fluid.framework.Variable):
            self.vars.update({id(new_vars): new_vars})
            return
        assert isinstance(new_vars, list)
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
            block.desc._remove_var(cpt.to_bytes(name))
            del block.vars[name]
        block._sync_with_cpp()

    def var2dot_rec(self, vars):
        """ Lookup var2dot recursively."""
        if isinstance(vars, paddle.fluid.framework.Variable):
            dot = self.var2dot.lookup(vars)
            return dot

        dots = [self.var2dot_rec(var) for var in vars]
        return dots

    def dot2bar_rec(self, dots, defaults=None):

        if isinstance(dots, paddle.fluid.framework.Variable):
            bar = self.dot2bar.lookup(dots)
            if bar is None and defaults is not None:
                bar = defaults
            return bar

        if defaults is None:
            defaults = [None for _ in range(dots)]

        bars = [
            self.dot2bar_rec(dot, default)
            for dot, default in zip(dots, defaults)
        ]
        return bars

    def linearize(self, xs, ys, xs_dot=None):
        """ Performs the linearization transform, a.k.a, forward mode AD 
        transform, on a primitive lowered program.
        
        Args:
            xs: a list of input variables
            ys: a list of output variables
            xs_dot: optional, a list of gradient input variables. The list size
                must be equal to `len(xs)`. The shape and dtype of each element
                must be the same as in `xs`

        Returns:
            (xs_dot, ys_dot): a tuple of two lists. `xs_dot` is the list of
            gradient inputs of the resulting linearized program. `ys_dot` is 
            the list gradient outputs of the resulting linearized program
            
        """
        if xs_dot is None:
            xs_dot = [fill_const(1.0, shape=x.shape, dtype=x.dtype) for x in xs]
            self.add_vars(xs_dot)
        else:
            assert len(xs) == len(xs_dot)

        for x, dot in zip(xs, xs_dot):
            assert x.dtype == dot.dtype
            assert x.shape == dot.shape
            self.var2dot.add(x, dot)

        path, unused_xs, _ = topo_path(xs, ys, self.block)

        # No need to track unused inputs
        for x in unused_xs:
            self.var2dot.delete(x)

        for op in path:
            # An input var may not be on the input-output path, which implies 
            # there may be None's in `ins_dot`. In this case we place
            # the original input in the position of the otherwise forward
            # gradient.
            ins = op_position_inputs(op)
            jvp_ins = self.var2dot_rec(ins)
            # apply op's forward ad rule
            outs_dot = _jvp(op, *jvp_ins)
            self.add_vars_rec(outs_dot)
            outs = op_position_output(op)
            self.var2dot.add_rec(outs, outs_dot)

        ys_dot = [self.var2dot.lookup(y) for y in ys]
        return xs_dot, ys_dot

    def transpose(self, ys_dot, xs_dot, ys_bar=None, retain_fwd=False):
        """ Performs the transpose transform, a.k.a, reverse mode AD 
        transform, on a linearized primitive program.

        Note, `transpose` is supposed to be used in couple with `linearize`.
        
        Args:
            ys_dot: a list of outputs of the linearized program.
            xs_dot: a list of inputs of the linearized program.
            ys_bar: optional, a list of inputs of the resulting transposed 
                program. The list size must be equal to `len(ys_dot)`. The shape
                and dtype of each element must be the same as in `ys_dot`

        Returns:
            (ys_bar, xs_bar): a tuple of two lists. `ys_bar` is the list of
            inputs of the resulting transposed program. `xs_bar` is 
            the list outputs of the resulting transposed program
            
        """
        assert all(v is not None for v in xs_dot), f'`xs_dot` includes None.'
        assert all(v is not None for v in ys_dot), f'`ys_dot` includes None.'

        if ys_bar is None:
            ys_bar = []
            for y in ys_dot:
                ys_bar.append(fill_const(1.0, shape=y.shape, dtype=y.dtype))
            self.add_vars(ys_bar)
        else:
            assert len(ys_dot) == len(ys_bar)
            for y_dot, y_bar in zip(ys_dot, ys_bar):
                assert y_dot.shape == y_bar.shape
                assert y_dot.dtype == y_bar.dtype

        for dot, bar in zip(ys_dot, ys_bar):
            self.dot2bar.add(dot, bar)

        # find all the relevant forward gradients
        path, unused_xs_dot, _ = topo_path(xs_dot, ys_dot, self.block)

        # No need to track unused inputs
        for dot in unused_xs_dot:
            self.dot2bar.delete(dot)

        dotvars = output_vars_on_path(path)
        dotvars.update((id(var), var) for var in xs_dot)

        is_dot = lambda v: id(v) in dotvars

        for op in reversed(path):
            out = op_position_output(op)
            out_bar_rec = self.dot2bar_rec(out, defaults=out)
            ins_bar_rec = _transpose(op, is_dot, out_bar_rec)

            # TODO(Tongxin): this is hacky. Tuple implies the Transpose rule
            # returns multiple entities. There should be better ways to handle
            # outputs.
            if isinstance(ins_bar_rec, tuple):
                ins_bar_rec = list(ins_bar_rec)
            else:
                ins_bar_rec = [ins_bar_rec]
            self.add_vars_rec(ins_bar_rec)

            ins_bar = flatten(ins_bar_rec)
            ins = get_input_vars(op)
            assert len(ins) == len(ins_bar)

            for dot, bar in zip(ins, ins_bar):
                if bar is not None:
                    # aggregate gradient
                    grad = self.dot2bar.lookup(dot)
                    if grad is None:
                        self.dot2bar.add(dot, bar)
                    else:
                        grad = add(grad, bar)
                        self.add_vars([grad])
                        self.dot2bar.add(dot, grad)

        xs_bar = [self.dot2bar.lookup(x) for x in xs_dot]

        if not retain_fwd and len(path) > 0:
            vars_to_remove = set()
            for op in path:
                vars_to_remove.update(get_output_vars(op))

            op_indexes = []

            block = self.block
            for i, op in enumerate(block.ops):
                if op in path:
                    op_indexes.append(i)
                    path.pop(0)
                    if len(path) == 0:
                        break

            self.erase_ops(op_indexes)
            self.erase_dots(vars_to_remove)

        return ys_bar, xs_bar


def _gradients(ys, xs, ys_bar=None):
    """ A drop-in replacement of paddle.gradients but instead computing
    on primitive ops.
    
    Args:
        ys: the target tensor or tensors
        xs: the input tensor or tensors
        ys_bar: the optional gradient tensors of `ys`
    
    Returns:
        xs_bar: a list gradients of input `xs`
    """

    ys, xs = to_tensors(ys), to_tensors(xs)
    block = ys[0].block
    # TODO(Tongxin) without any prior knowledge about whether the program
    # is completely lowered to primitive ops, it's mandatory to run the lowering
    # pass once and again. This is obviously inefficient and needs to be 
    # optimized.
    new_vars = xs + ys
    orig2prim(block, new_vars)

    ad = Transform(block)
    xs = new_vars[:len(xs)]
    ys = new_vars[len(xs):]
    
    xs_dot, ys_dot = ad.linearize(xs, ys)
    if any(var is None for var in ys_dot):
        assert False, f'Gradients cannot be computed. The given output `ys` does not depend on input `xs`.'
    ys_bar, xs_bar = ad.transpose(ys_dot, xs_dot, ys_bar)
    # remove xs_dot and their constructor ops

    op_indexes = []
    for var in xs_dot:
        if var is not None:
            op_index = block.ops.index(var.op)
            assert op_index >= 0
            op_indexes.append(op_index)

    ad.erase_ops(sorted(op_indexes))
    ad.erase_dots(xs_dot)

    # prim2orig(block, xs_bar)
    return xs_bar


def orig2prim(block=None, update_var_list=None):
    _lower(block, reverse=False, update_var_list=update_var_list)


def prim2orig(block=None, update_var_list=None):
    _lower(block, reverse=True, update_var_list=update_var_list)


def to_tensors(xs):
    if isinstance(xs, list or tuple):
        return xs
    else:
        return [xs]


def single_layer_list(xs):
    rt_l = []
    for x in xs:
        if isinstance(x, list):
            rt_l = rt_l + single_layer_list(x)
        else:
            rt_l.append(x)
    return rt_l


def bind(args, to_bind, vlt):
    for i in range(len(args)):
        if isinstance(args[i], list):
            bind(args[i], to_bind, vlt)
        elif args[i] is not None and args[i].name in to_bind:
            args[i] = vlt[to_bind[args[i].name]]


def _lower(block, reverse, update_var_list):
    lower_fn = _prim2orig if reverse else _orig2prim
    lookup_fn = lookup_prim2orig if reverse else lookup_orig2prim
    if block is None:
        program = default_main_program()
        assert program.num_blocks == 1, "The lower transform is designed to process only one block."
        block = program.current_block()

    vlt = {}
    to_bind = {}
    for var in block.desc.all_vars():
        vlt[var.name()] = block.var(var.name())

    ops_to_remove = []
    vars_to_remove = set()
    for op_idx in range(len(block.ops)):
        op = block.ops[op_idx]
        ops_to_remove.append(op_idx)
        if lookup_fn(op.type) is not None:
            input_args = get_input_var_list(op)
            bind(input_args, to_bind, vlt)

            for orig_out, new_out in zip(
                    single_layer_list(get_output_var_list(op)),
                    single_layer_list(to_tensors(lower_fn(op, *input_args)))):
                assert not (orig_out is None) ^ (
                    new_out is None), "orig_out and new_out should match."
                vars_to_remove.add(orig_out.name)
                vlt[new_out.name] = new_out
                to_bind[orig_out.name] = new_out.name
        else:

            def bind_name(names, to_bind):
                rt_l = []
                for name in names:
                    if isinstance(name, list):
                        rt_l.append(bind_name(name, to_bind))
                    else:
                        rt_l.append(to_bind[name] if name in to_bind else name)
                return rt_l

            inputs = {}
            for i in range(len(op.input_names)):
                inputs[op.input_names[i]] = bind_name(
                    op.input(op.input_names[i]), to_bind)
            # print(inputs)

            outputs = {}
            for i in range(len(op.output_names)):
                outputs[op.output_names[i]] = op.output(op.output_names[i])
            # print(outputs)

            attrs = {}
            for name in sorted(op.attr_names):
                attrs[name] = op.attr(name)
            from paddle.fluid.dygraph.base import param_guard
            new_op_desc = block.desc.append_op()
            with param_guard(inputs), param_guard(outputs):
                op = Operator(
                    block=block,
                    desc=new_op_desc,
                    type=op.type,
                    inputs=inputs,
                    outputs=outputs,
                    attrs=attrs)
            block.ops.append(op)

    if update_var_list is not None:
        for i in range(len(update_var_list)):
            if update_var_list[i].name in to_bind:
                update_var_list[i] = vlt[to_bind[update_var_list[i].name]]

    for op_idx in reversed(ops_to_remove):
        block._remove_op(op_idx)
    for var_name in vars_to_remove:
        block._remove_var(var_name)
