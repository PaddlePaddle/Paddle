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
from collections import OrderedDict


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
                    sink_ops[id(out)] = op
                else:
                    def_vars.append(out)
    if len(sink_ops) != len(ys):
        not_reachable = (var for var in ys if id(var) not in sink_ops)
        raise f"Output vars: {' '.join(not_reachable)} are not reachable from inputs."
    return path


class VarMap(object):
    __slots__ = ['name', 'varset', 'tab']

    def __init__(self, name, varset):
        self.name = name
        self.varset = varset
        self.tab = OrderedDict()
    
    def add(self, key_var, value_var):
        self.tab[id(key_var)] = id(value_var)
  
    def lookup(self, key_var):
        value_id = self.tab.get(id(key_var))
        value_var = self.varset.get(value_id)
        return value_var

    def delete(self, key_var):
        del self.tab[id(key_var)]

    def delete_keyvars(self, key_vars):
        for var in key_vars:
            del self.tab[id(var)]

    def delete_valuevars(self, value_vars):
        ids = [id(v) for v in value_vars]
        keys = [k for k, v in self.tab.items() if v in ids]
        for k in keys:
            del self.tab[k]

    def contain_var(self, key_var):
        return self.tab.__contains__(id(key_var))

    def contain_value(self, value_var):
        return self.tab.values().__contains__(id(value_var))


class Transform(object):
    """ An object that maintains the state of transformations applied to a 
    primitve program. """

    def __init__(self, block):
        self.block = block
        self.init_varset(block)
        self.var2dot = VarMap('var2dot', self.vars)
        self.dot2bar = VarMap('dot2var', self.vars)

    def init_varset(self, block):
        self.vars = OrderedDict()
        for _, var in block.vars.items():
            self.vars[id(var)] = var

    def update_varset(self, new_vars):
        self.vars.update({id(v) : v for v in new_vars if v is not None})

    def erase_dots(self, vars_to_erase):
        for var in vars_to_erase:
            del self.vars[id(var)]
        self.dot2bar.delete_keyvars(vars_to_erase)
        self.var2dot.delete_valuevars(vars_to_erase)
        for var in vars_to_erase:
            del var.block.vars[var.name]

    def is_dot(self, var):
        return self.var2dot.contain_value(var)

    def lower2prim(self):
        pass

    def linearize(self, xs, ys, xs_dot=None):
        if xs_dot is None:
            xs_dot = []
            for x in xs:
                xs_dot.append(fill_const(1.0, shape=x.shape, dtype=x.dtype))
        else:
            assert len(xs) == len(xs_dot)
            assert all(x.dtype == x_dot.dtype for x, x_dot in zip(xs, xs_dot))
            assert all(x.shape == x_dot.shape for x, x_dot in zip(xs, xs_dot))

        self.update_varset(xs_dot)
        for x, dot in zip(xs, xs_dot):
            self.var2dot.add(x, dot)
        for op in topo_path(xs, ys, self.block):
            ins = get_input_vars(op)
            ins_dot = [self.var2dot.lookup(var) for var in ins]
            jvp_ins = [x if dot is None else dot for dot, x in zip(ins_dot, ins)]
            outs_dot = _jvp(op, *jvp_ins)
            if not isinstance(outs_dot, list):
                outs_dot = [outs_dot]
            self.update_varset(outs_dot)

            for x, dot in zip(get_output_vars(op), outs_dot):
                self.var2dot.add(x, dot)
        
        ys_dot = [self.var2dot.lookup(y) for y in ys]
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

        self.update_varset(ys_bar)
        for dot, bar in zip(ys_dot, ys_bar):
            self.dot2bar.add(dot, bar)
        for op in reversed(topo_path(xs_dot, ys_dot, self.block)):
            outs_bar = [self.dot2bar.lookup(var) for var in get_output_vars(op)]
            ins_bar = _transpose(op, self.is_dot, *outs_bar)
            if isinstance(ins_bar, (list, tuple)):
                ins_bar = list(ins_bar)
            else:
                ins_bar = [ins_bar]
            self.update_varset(ins_bar)
            for dot, bar in zip(op.get_input_vars(), ins_bar):
                if bar is not None:
                    self.dot2bar.add(dot, bar)
        xs_bar = [self.dot2bar.lookup(x) for x in xs_dot]

        if not retain_fwd:
            dots_to_remove = set()
            for op in topo_path(xs_dot, ys_dot):
                for var in get_input_vars(op):
                    if self.is_dot(var):
                        dots_to_remove.add(var)
                block = op.block
                op_idx = block.ops.index(op)
                block._remove_op(op_idx)

            self.erase_dots(dots_to_remove)

        return ys_bar, xs_bar

