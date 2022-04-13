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
                sink_ops[id(out)] = op
            else:
                def_vars.append(out)
    if len(sink_ops) != len(ys):
        not_reachable = (var for var in ys if id(var) not in sink_ops)
        raise f"Output vars: {' '.join(not_reachable)} are not reachable from inputs."
    return path


class VarMap(object):
    __slots__ = ['tab']

    def __init__(self, name):
        self.name = name
        self.tab = {}
    
    def set(self, var, value):
        self.tab[id(var)] = value
  
    def lookup(self, var):
        return self.tab.get(id(var))
    
    def delete(self, var):
        del self.tab[id(var)]

    def contain_key(self, var):
        return self.tab.__contains__(id(var))

    def contain_value(self, value):
        return self.tab.values().__contains__(value)


class Transform(object):
    """ An object that maintains the state of transformations applied to a 
    primitve program. """

    def __init__(self, block):
        self.block = block
        self.vardefs = VarMap('vardefs')
        self.varuses = VarMap('varuses')
        self.var2dot = VarMap('var2dot')
        self.dot2bar = VarMap('dot2var')

    def is_dot(self, var):
        return self.var2dot.contain_value(var)

    def lower2prim(self):
        pass

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
            assert all(x.dtype == x_dot.dtype for x, x_dot in zip(xs, xs_dot))
            assert all(x.shape == x_dot.shape for x, x_dot in zip(xs, xs_dot))
        
        map(self.var2dot.set, xs, xs_dot)
        for op in topo_path(xs, ys, self.block):
            xs_dot = list(map(self.var2dot.lookup, get_input_vars(op)))
            ys_dot = _jvp(op, *xs_dot)
            map(self.var2dot.set, op.get_output_vars(op), ys_dot)
        
        ys_dot = map(self.var2dot.lookup, ys)
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
        
        map(self.dot2bar.set, ys_dot, ys_bar)
        for op in reversed(topo_path(xs_dot, ys_dot, self.block)):
            ys_bar = list(map(self.dot2bar.lookup, get_output_vars(op)))
            xs_bar = _transpose(op, self.is_dot, *ys_bar)
            map(self.dot2bar.set, op.get_input_vars(), xs_bar)

        xs_bar = list(map(self.dot2bar.lookup, xs_dot))

        if not retain_fwd:
            dots_to_remove = set()
            for op in topo_path(xs_dot, ys_dot):
                for var in get_input_vars(op):
                    if self.is_dot(var):
                        dots_to_remove.add(var)
                block = op.block
                op_idx = block.ops.index(op)
                block._remove_op(op_idx)

            for k, v in self.var2dot:
                if v in dots_to_remove:
                    del self.var2dot[k]
            for k, v in self.dot2bar:
                if k in dots_to_remove:
                    del self.dot2bar[k]
       
        return ys_bar, xs_bar

