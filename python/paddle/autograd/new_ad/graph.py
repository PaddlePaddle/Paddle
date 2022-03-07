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


class PrimGraph(object):
    def __init__(self) -> None:
        self.nodes = []

    def add_node(self, node, var):
        self.nodes.append(node)
        var.set_def(node)


class PrimNode(object):
    def __init__(self, primitive: Primitive, out_var, *in_vars,
                 **kwargs) -> None:
        self.op = primitive
        self.out_var = out_var
        self.in_vars = in_vars
        self.attributes = kwargs


class Var():
    def __init__(self, name, is_tangent=False) -> None:
        self.name = name
        self.def_node = None
        self.is_tangent = is_tangent

    def set_shape(self, shape):
        self.shape = shape

    def set_def(self, node: PrimNode):
        self.def_node = node


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
