#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

from paddle.utils import gast
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.base_transformer import BaseTransformer
from paddle.fluid.dygraph.dygraph_to_static.utils import create_funcDef_node, ast_to_source_code

DECORATOR_NAMES = ['declarative', 'to_static', 'dygraph_to_static_func']


class DecoratorTransformer(BaseTransformer):
    """
    Transform decorators.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Type of input node should be AstNodeWrapper, but received %s ." % type(
            wrapper_root)
        self.root = wrapper_root.node

        self.ancestor_nodes = []

    def transform(self):
        """
        Main function to transform AST.
        """
        self.visit(self.root)

    def visit(self, node):
        self.ancestor_nodes.append(node)
        ret = super(DecoratorTransformer, self).visit(node)
        self.ancestor_nodes.pop()
        return ret

    def visit_FunctionDef(self, node):
        assert isinstance(node, gast.FunctionDef)
        self.generic_visit(node)

        deco_list = node.decorator_list
        node.decorator_list = []

        decofun_str = '_orig_' + node.name
        for deco in reversed(deco_list):
            if isinstance(deco,
                          gast.Attribute) and deco.attr in DECORATOR_NAMES:
                continue
            elif isinstance(deco,
                            gast.Call) and deco.func.args[0].id == 'wraps':
                continue
            else:
                if isinstance(deco, gast.Attribute):
                    deco_name = deco.attr
                elif isinstance(deco, gast.Call):
                    deco_name = deco.func.args[0].id
                else:
                    deco_name = deco.id
            decofun_str = '_jst.Call({})({})'.format(deco_name, decofun_str)

        if decofun_str == '_orig_' + node.name:
            return node

        orig_func_node = create_funcDef_node(node.body,
                                             name='_orig_' + node.name,
                                             input_args=node.args,
                                             return_name_ids=[])

        decofun_str = '_decoed_{} = {}'.format(node.name, decofun_str)
        decofun_node = gast.parse(decofun_str).body[0]

        arg_str = ''
        for arg in node.args.args:
            arg_str += arg.id + ', '
        callfun_str = 'return _decoed_{}({})'.format(node.name, arg_str[0:-1])
        callfun_node = gast.parse(callfun_str).body[0]

        node.body = [orig_func_node, decofun_node, callfun_node]

        return node
