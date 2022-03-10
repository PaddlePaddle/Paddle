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

# All automatic differential interfaces act on the same global program, and some contents are replaced, increased or decreased on it. After the transformation is completed, it is handed over to the compiler or converted back to the original operator system for execution.


# functional apis
def jvp(func, xs, v=None, allow_unused=True, batch=False):
    pass


def vjp(func, xs, v=None, allow_unused=True, batch=False):
    pass


def hvp(func, xs, v=None, allow_unused=True, batch=False):
    pass


def vhp(func, xs, v=None, allow_unused=True, batch=False):
    pass


class Jacobian(object):
    def __init__(self, func, xs, allow_unused=True, batch=False):
        pass


class Hessian(object):
    def __init__(self, func, xs, allow_unused=True, batch=False):
        pass


# procedural apis
def gradients(ys, xs, v):
    pass


class Optimizer(object):
    def minimize(self, ys):
        pass


# interior transforms
def subtrace(ys, xs):
    program = current_program()
    block = program.block[0]
    op_descs = []
    for op_desc in block:
        if xxx:
            op_descs.append()
    return op_descs


def origin2primitive(ys, xs):
    program = current_program()
    block = program.block[0]
    op_descs = subtrace(ys, xs)
    for op_desc in subtrace(ys, xs):
        new_op_desc = f(op_desc)
        block.erase(op_desc)
        block.append(new_op_desc)


def linearize(ys, xs):
    op_descs = subtrace(ys, xs)
    nodes = PrimGraph(op_descs)

    # create jvps for all nodes and update dot lookup table
    switch_runner('jvp')

    # (TODO) find entry nodes
    in_dots = (make_var(is_tangent=True) for var in in_vars)
    for var, dot in zip(in_vars, in_dots):
        set_var2dot(var, dot)

    out_dot = None

    for node in subtrace(nodes, in_vars, out_vars):
        out_dot = node.op(*node.in_vars, **node.attributes)
        set_var2dot(node.out_var, out_dot)

    return xs_dot, ys_dot


def transpose(ys, xs):
    # transpose all nodes and update bar lookup table
    switch_runner('transpose')
