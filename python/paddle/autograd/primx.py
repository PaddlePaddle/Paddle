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

from paddle.fluid.framework import default_main_program, default_startup_program
from paddle.fluid import unique_name, core
from .primops import fill_const
from .primrules import get_input_vars, get_output_vars, _jvp, _transpose


def make_var(dtype, shape, block=None, namekey='', stop_gradient=False):
    """ Create a type inferred variable. """

    if block is None:
        block = default_main_program().current_block()

    name = unique_name.generate_with_ignorable_key(namekey + '%')

    return block.create_var(
        name=name,
        dtype=dtype,
        shape=shape,
        type=core.VarDesc.VarType.LOD_TENSOR,
        persistable=False,
        stop_gradient=stop_gradient)


def make_varlike(x, block=None, namekey='', stop_gradient=False):
    """ Make a variable using the dtype and shape of the given input. """
    return make_var(x.dtype, x.shape, block, namekey, stop_gradient)


def topo_path(xs, ys, block=None):
    """ Returns the ops in topological on the paths from input `xs` to 
    output `ys`. """

    if block is None:
        block = default_main_program().current_block()
    path = []
    def_vars = list(xs)
    sink_ops = {}
    for op in block.ops:
        if len(sink_ops) == len(ys):
            break
        ins = set(get_input_vars(op))
        if any(ins.intersection(def_vars)):
            path.append(op)
        outs = set(get_output_vars(op))
        for out in outs:
            if any(out is y for y in ys):
                # Found an output op
                assert not any(out is y for y in sink_ops)
                sink_ops[out] = op
            else:
                def_vars.append(op)
    if len(sink_ops) != len(ys):
        not_reachable = (var for var in ys if var not in sink_ops)
        raise f"Output vars: {' '.join(not_reachable)} are not reachable from inputs."
    return path


class Transform(object):
    """ An object that maintains the state of transformations applied to a 
    primitve program. """

    def __init__(self, block):
        self.block = block
        self.vardefs = {}
        self.varuses = {}
        self.var2dot = {}
        self.dot2bar = {}

    def get_var2dot(self, var):
        return getattr(self.var2dot, var, None)

    def set_var2dot(self, var, dot_var):
        self.var2dot[var] = dot_var

    def get_dot2bar(self, var):
        return getattr(self.dot2bar, var, None)

    def set_dot2bar(self, dot, bar):
        self.dot2bar[dot] = bar

    def check_dot(self, x):
        return x in self.var2dot.values()

    ## TODO(lml): finish these function
    def get_var_lookup_tab(self, block):
        tab = {}
        for var in block.allvars():
            tab[var.name] = var

    def bind(self):
        pass

    def update_vlt(self):
        pass

    def orig2prim(self, block=None):
        if block is None:
            block = default_main_program().current_block()
        vlt = get_var_lookup_tab(block)
        ops_to_remove = []
        for op_idx in range(len(block.ops)):
            op = block.ops(op_idx)
            if _orig2prim.lookup(op.type()) is not None:
                for name in op.input_arg_names():
                    if name in to_bind:
                        bind(name, to_bind[name])
                    update_vlt(vlt, to_bind[name], name)
                ops_to_remove.append(op_idx)
                for out_name, new_out in zip(get_output_names(op), _lower(op)):
                    to_bind[out_name] = new_out

        for op_idx in reversed(ops_to_remove):
            block.desc._remove_op(op_idx, op_idx + 1)
        ## TODO(lml): call correct interface to infer shape and dtype
        block.infer_shape()

    def update_defuse(self, op):
        for var in get_input_vars(op):
            if var not in self.varuses:
                self.varuses[var] = set([op])
            else:
                self.varuses[var].add(op)
        for var in get_output_vars(op):
            if var in self.vardefs:
                assert self.vardefs[var] == op, f'{var} is doubly assigned.'
            else:
                self.vardefs[var] = op

    def build_defuse(self, block=None):
        if block is None:
            block = default_main_program().current_block()
        for op in block.ops:
            if op.has_attr("sub_block"):
                sub_block_id = op._block_attr_id("sub_block")
                sub_block = block.program.block(sub_block_id)
                self.build_defuse(sub_block)
            else:
                self.update_defuse(op)

    def linearize(self, xs, ys, xs_dot=None):
        if xs_dot is None:
            xs_dot = []
            for x in xs:
                xs_dot.append(fill_const(1.0, shape=x.shape, dtype=x.dtype))
        else:
            assert len(xs) == len(xs_dot)
            for x, x_dot in zip(xs, xs_dot):
                assert x.shape == x_dot.shape
                assert x.dtype == x_dot.dtype

        map(self.set_var2dot, xs, xs_dot)
        for op in topo_path(xs, ys, self.block):
            xs_dot = list(map(self.get_var2dot, get_input_vars(op)))
            ys_dot = _jvp(op, *xs_dot)
            map(self.set_var2dot, op.get_output_vars(op), ys_dot)

        ys_dot = map(self.get_var2dot, ys)
        return xs_dot, ys_dot

    def transpose(self, ys_dot, xs_dot, ys_bar=None, retain_fwd=False):
        if ys_bar is None:
            ys_dot = []
            for y in ys_dot:
                ys_dot.append(fill_const(1.0, shape=y.shape, dtype=y.dtype))
        else:
            assert len(ys_dot) == len(ys_bar)
            for y_dot, y_bar in zip(ys_dot, ys_bar):
                assert y_dot.shape == y_bar.shape
                assert y_dot.dtype == y_bar.dtype

        map(self.set_dot2bar, ys_dot, ys_bar)
        for op in reversed(topo_path(xs_dot, ys_dot, self.block)):
            ys_bar = list(map(self.get_dot2bar, get_output_vars(op)))
            xs_bar = _transpose(op, self.check_dot, *ys_bar)
            map(self.set_dot2bar, op.get_input_vars(), xs_bar)

        xs_bar = list(map(self.get_dot2bar, xs_dot))

        if not retain_fwd:
            dots_to_remove = set()
            for op in topo_path(xs_dot, ys_dot):
                for var in get_input_vars(op):
                    if self.check_dot(var):
                        dots_to_remove.add(var)
                block = op.block
                op_idx = block.ops.index(op)
                block._remove_op(op_idx)
                # remove this op and input dots
            for k, v in self.var2dot:
                if v in dots_to_remove:
                    del self.var2dot[k]
            for k, v in self.dot2bar:
                if k in dots_to_remove:
                    del self.dot2bar[k]

        return ys_bar, xs_bar
