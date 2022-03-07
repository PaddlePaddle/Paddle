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

import string
import threading
from typing import Callable, Any
import paddle


class Primitive(object):
    """ Primitive OP.
  
    In instance of `Primitive` identifies a primitive and provides
    interfaces for using the primitive.

    """

    def __init__(self, optype) -> None:
        self.optype = optype

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        runner = get_current_runner()
        runner.run_op(self, *args, **kwargs)


ADD = Primitive('add')
SUB = Primitive('sub')
MUL = Primitive('mul')
DIV = Primitive('div')
NEG = Primitive('neg')
TANH = Primitive('tanh')
BCAST = Primitive('broadcast')
REDUCE = Primitive('reduce')
SUM = Primitive('sum')
SLICE = Primitive('slice')
RESHAPE = Primitive('reshape')
MATMUL = Primitive('matmul')
IND_SELECT = Primitive('index_select')
TEN_SELECT = Primitive('tensor_select')
FILL = Primitive('fill')

nodemakers = {}
jvpmakers = {}
transposemakers = {}

def add_maker(x, y):
    out_var = make_var()
    node = PrimNode(ADD, out_var, x, y)
    node, out_var

def sub_maker(x, y):
    out_var = make_var()
    node = PrimNode(SUB, out_var, x, y)
    return node, out_var

def mul_maker(x, y):
    out_var = make_var()
    node = PrimNode(MUL, out_var, x, y)
    return node, out_var


nodemakers[ADD] = add_maker
nodemakers[SUB] = sub_maker
nodemakers[MUL] = mul_maker


def add_jvpmaker(x, y):
    return lambda tx, ty: ADD(tx, ty)


def sub_jvpmaker(x, y):
    return lambda tx, ty: SUB(tx, ty)


def mul_jvpmaker(x, y):
    return lambda tx, ty: ADD(MUL(x, ty), MUL(tx, y))


jvpmakers[ADD] = add_jvpmaker
jvpmakers[SUB] = sub_jvpmaker
jvpmakers[MUL] = mul_jvpmaker


def add_transposemaker(x, y):
    assert x.is_tangent and y.is_tangent
    return lambda t: t, lambda t: t


def sub_transposemaker(x, y):
    assert x.is_tangent and y.is_tangent
    return lambda t: t, lambda t: NEG(t)


def mul_transposemaker(x, y):
    assert x.is_tangent ^ y.is_tangent
    if x.is_tangent:
        return lambda t: (MUL(t, y), None)
    else:
        return lambda t: (None, MUL(x, t))


transposemakers[ADD] = add_transposemaker
transposemakers[SUB] = sub_transposemaker
transposemakers[MUL] = mul_transposemaker


class PrimNode(object):
    def __init__(self, primitive: Primitive, out_var, *in_vars, **kwargs) ->    None:
        self.op = primitive
        self.out_var = out_var
        self.in_vars = in_vars
        self.attributes = kwargs


class PrimGraph(object):
    def __init__(self) -> None:
        self.nodes = []

    def add_node(self, node, var):
        self.nodes.append(node)
        var.set_def(node)

class Var():
    def __init__(self, name, is_tangent=False) -> None:
        self.name = name
        self.def_node = None
        self.is_tangent = is_tangent

    def set_shape(self, shape):
        self.shape = shape

    def set_def(self, node: PrimNode):
        self.def_node = node


class Runner(object):
    def run_op(self, op, *args, **kwargs):
        raise f'This `process_op` method is missing in {type(self)}.'


class LowerToProgram(Runner):
    pass


class MakeGraph(Runner):
    def run_op(self, op, *args, **kwargs):
        var, node = op(*args, **kwargs)
        current_graph().add_node(node, var)

class JVP(Runner):
    def run_op(self, op, *args, **kwargs):
        jvpmaker = jvpmakers[op]
        jvp_fn = jvpmaker(*args, **kwargs)
        switch_runner('graph')
        out_dot = jvp_fn(*map(var2dot, args))
        switch_runner('jvp')
        return out_dot


class Transpose(Runner):

    def run_op(self, op, *args, **kwargs):
        transposemaker = transposemakers[op]
        transpose_fn = transposemaker(*args, **kwargs)
        switch_runner('graph')
        out_bar = make_var(is_tangent=True)
        in_bars = transpose_fn(out_bar)
        switch_runner('transpose')
        return out_bar, in_bars


def linearize(in_vars, out_vars):
    # create jvps for all nodes and update dot lookup table
    switch_runner('jvp')
    nodes = current_graph().nodes

    # (TODO) find entry nodes
    in_dots = (make_var(is_tangent=True) for var in in_vars)
    for var, dot in zip(in_vars, in_dots):
        set_var2dot(var, dot)

    out_dot = None

    for node in subtrace(nodes, in_vars, out_vars):
        out_dot = node.op(*node.in_vars, **node.attributes)
        set_var2dot(node.out_var, out_dot)

    return in_dots, out_dot

def transpose():
    # transpose all nodes and update bar lookup table
    switch_runner('transpose')


class ADRunnerState(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.graph = None
        self.vars = []
        self.var_lookup = {}
        self.dot_lookup = {}
        self.bar_lookup = {}
        self.runners = {'graph': MakeGraph(),
                        'jvp': JVP(),
                        'transpose': Transpose(),
                        'lower2prog': LowerToProgram()
                        }
        self.runner = None

    def switch_runner(self, kind):
        self.runner = self.runners[kind]


adrunner_state = ADRunnerState()


def switch_runner(kind):
    adrunner_state.switch_runner(kind)


def get_current_runner():
    return adrunner_state.runner


def make_var(is_tangent=False):
    name = f'%{len(adrunner_state.vars)}'
    var = Var(name, is_tangent)
    adrunner_state.vars.append(var)
    return var

def is_nodein(node):
    return all(v.def_node is None for v in node.in_vars)

def is_nodeout(node):
    pass

def subtrace(nodes, in_vars, out_vars):
    pass

def current_graph():
    return adrunner_state.graph

def var2dot(var):
    lookup_tab = adrunner_state.dot_lookup
    return lookup_tab[var] if var in lookup_tab else None

def set_var2dot(var, dot):
    lookup_tab = adrunner_state.dot_lookup
    lookup_tab[var] = dot
