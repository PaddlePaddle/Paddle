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

from paddle.framwork import default_main_program


# functional apis
def jvp(func, xs, v=None, create_graph=True, batch=False):
    pass


def vjp(func, xs, v=None, create_graph=True, batch=False):
    pass


def hvp(func, xs, v=None, create_graph=True, batch=False):
    pass


def vhp(func, xs, v=None, create_graph=True, batch=False):
    pass


class Jacobian(object):
    def __init__(self, func, xs, create_graph=True, batch=False):
        pass


class Hessian(object):
    def __init__(self, func, xs, create_graph=True, batch=False):
        pass


# procedural apis
def gradients(ys, xs, ys_bar):
    if ys_bar is None:
        ys_bar = [paddle.ones_like(y) for y in ys]
    convert2primitive(ys, xs)
    ys_dot, xs_dot = linearize(ys, xs)
    xs_bar = transpose(ys_bar)
    return xs_bar


class Optimizer(object):
    def minimize(self, ys):
        pass


# interior transforms
def subtrace(ys, xs):
    block = default_main_program().current_block()
    op_path = []
    for op_idx in range(0, block.desc.op_size()):
        if xxx:
            op_path.append(op_idx)
    return op_path


def convert2primitive(ys, xs):
    block = default_main_program().current_block()
    op_path = subtrace(ys, xs)
    switch_runner('primitive')
    for op_idx in op_path:
        op_desc = block.desc.op(op_idx)
        if not is_primitive(op_desc):
            runner = get_current_runner()
            runner.run_op(op_desc)
            ops_to_remove.append[op_idx]

    for op_idx in reversed(ops_to_remove):
        block.desc._remove_op(op_idx, op_idx + 1)


def linearize(ys, xs):
    op_descs = subtrace(ys, xs)

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


# for new_op_desc in to_insert:
#     _new_op_desc = new_block.desc.append_op()
#     _new_op_desc.copy_from(new_op_desc)
#     op = Operator(block=new_block, desc=_new_op_desc)
#     new_block.ops.append(op)
