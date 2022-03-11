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
def gradients(ys, xs, v):
    if ys_bar is None:
        ys_bar = [paddle.ones_like(y) for y in ys]
    convert2primitive(ys, xs)
    y_dots, x_dots = linearize(ys, xs)
    _, x_bars = transpose(y_dots, x_dots, v)
    return x_bars


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
        op = block.ops(op_idx)
        if not is_primitive(op):
            runner = get_current_runner()
            runner.run_op(op.type(), *op.ins, *op.outs, op.attrs)
            ops_to_remove.append[op_idx]

    for op_idx in reversed(ops_to_remove):
        block.desc._remove_op(op_idx, op_idx + 1)

    block.infer_shape()


def linearize(ys, xs, v=None):
    op_path = subtrace(ys, xs)

    # create jvps for all nodes and update dot lookup table
    switch_runner('jvp')

    x_dots = [make_var(
        is_tangent=True, ref_var=x) for x in xs] if v is None else v
    for x, x_dot in zip(xs, x_dots):
        set_var2dot(x, x_dot)

    y_dots = []

    for op_idx in op_path:
        op = block.ops(op_idx)
        runner = get_current_runner()
        y_dot = runner.run_op(op.type(), *op.ins, *op.outs, op.attrs)
        set_var2dot(op.outs, y_dot)
        y_dots += y_dot

    block.infer_shape()
    return y_dots, x_dots


def transpose(y_dots, x_dots, v=None):
    op_path = subtrace(y_dots, x_dots)
    # transpose all nodes and update bar lookup table
    switch_runner('transpose')
    y_bars = [make_var(
        is_tangent=False, ref_var=y_dot)
              for y_dot in y_dots] if v is None else v
    for y_bar, y_dot in zip(y_bars, y_dots):
        set_dot2bar(y_dot, y_bar)

    x_bars = []

    for op_idx in reversed(op_path):
        op = block.ops(op_idx)
        runner = get_current_runner()
        x_bar = runner.run_op(op.type(), *op.ins, *op.outs, op.attrs)
        set_dot2bar(op.ins, x_bar)
        x_bars += x_bar

    block.infer_shape()
    return y_bars, x_bars
