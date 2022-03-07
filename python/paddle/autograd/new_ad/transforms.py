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


def make_var(is_tangent=False):
    name = f'%{len(adrunner_state.vars)}'
    var = Var(name, is_tangent)
    adrunner_state.vars.append(var)
    return var
