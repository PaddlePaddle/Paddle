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

import re

IGNORE_NAMES = [
    'declarative', 'to_static', 'dygraph_to_static_func', 'wraps',
    'staticmethod', 'classmethod'
]


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

    def visit_FunctionDef(self, node):
        assert isinstance(node, gast.FunctionDef)
        self.generic_visit(node)

        deco_list = node.decorator_list
        node.decorator_list = []

        # every decorator will append a node
        decofun_nodes = []
        # func to be decoed next time
        deco_target = '_orig_' + node.name
        # last decoed func
        decoed_func = ''

        for deco in reversed(deco_list):
            # skip INGNORE_NAMES
            if isinstance(deco, gast.Attribute):
                deco_name = deco.attr
            elif isinstance(deco, gast.Call):
                if hasattr(deco.func, 'args'):
                    deco_name = deco.func.args[0].id
                elif hasattr(deco.func, 'attr'):
                    deco_name = deco.func.attr
                else:
                    deco_name = deco.func.id
            else:
                deco_name = deco.id
            if deco_name in IGNORE_NAMES:
                continue

            # get function after decoration
            deco_full_name = ast_to_source_code(deco).strip()
            decoed_func = '_decoby_' + deco_name
            if isinstance(deco, gast.Call):
                # in this case , the deco_full_name will be like:
                # '_jst.Call(deco)(5)'
                rematch = re.match(r'\_jst\.Call\((.+?)\)\((.+?)\)',
                                   deco_full_name)
                re_name = rematch.group(1)
                re_args = rematch.group(2)
                re_args_with_func = deco_target + ', ' + re_args
                decofun_str = 'try:\n\t{0} = _jst.Call({1})({2})\nexcept:\n\t{0} = _jst.Call({1})({3})({4})'\
                    .format(decoed_func, re_name, re_args_with_func, re_args, deco_target)
            else:
                decofun_str = '{} = _jst.Call({})({})'.format(
                    decoed_func, deco_full_name, deco_target)

            decofun_nodes.extend(gast.parse(decofun_str).body)
            deco_target = decoed_func

        if not decofun_nodes:
            return node

        orig_func_node = gast.FunctionDef(name='_orig_' + node.name,
                                          args=node.args,
                                          body=node.body,
                                          decorator_list=[],
                                          returns=None,
                                          type_comment=None)

        args = [arg.id for arg in node.args.args]
        arg_str = ','.join(args)
        callfun_str = 'return {}({})'.format(decoed_func, arg_str)
        callfun_node = gast.parse(callfun_str).body[0]

        node.body = [orig_func_node] + decofun_nodes + [callfun_node]

        return node
