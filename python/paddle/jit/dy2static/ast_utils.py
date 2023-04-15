# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import ast
import collections
import inspect
import textwrap

import astor

from paddle.utils import gast


def ast_to_source_code(ast_node):
    """
    Transforms ast node into source code.
    """
    if not isinstance(ast_node, (gast.AST, ast.AST)):
        raise TypeError(
            "Type of ast_root should be gast.AST or ast.AST, but received %s."
            % type(ast_node)
        )
    if isinstance(ast_node, gast.AST):
        ast_node = gast.gast_to_ast(ast_node)

    # Do not wrap lines even if they are too long
    def pretty_source(source):
        return ''.join(source)

    source_code = astor.to_source(ast_node, pretty_source=pretty_source)
    return source_code


def modify_function_code(func, code_str='register_hook'):
    """
    Modify the function for the register hook
    """
    # 将函数代码解析为 AST 对象
    func_ast = ast.parse(textwrap.dedent(inspect.getsource(func)))

    # 从 AST 中提取函数定义节点
    func_def = next(
        (
            node
            for node in ast.walk(func_ast)
            if isinstance(node, ast.FunctionDef) and node.name == func.__name__
        ),
        None,
    )

    register_hook_pos_map = collections.defaultdict(list)
    assignment_pos_map = collections.defaultdict(list)
    # 获取代码中包含有 register_hook 的位置 并获取前面参数的位置
    for i in range(len(func_def.body) - 1, -1, -1):
        body = func_def.body[i]
        # print(ast.dump(body))
        # 获取某一片段代码的字符串形式
        body_str = astor.to_source(body)
        # 查看该代码片段是否有效（即 含有所要的 register_hook 且不是字符串或者被注释掉）
        if (
            '"""' not in body_str
            and code_str in body_str
            and not isinstance(body, ast.FunctionDef)
        ):
            # 查找 register_hook 所对应的参数信息
            param_name = body_str.split('.')[0]
            register_hook_pos_map[param_name].append(i)
        if isinstance(body, ast.Assign):
            for target in body.targets:
                assignment_pos_map[target.id].append(i)

    # 确定顺序
    order_map = {}
    for k, idx_list in register_hook_pos_map.items():
        for idx in idx_list:
            if k not in assignment_pos_map:
                order_map[idx] = 1
            else:
                for assignment_idx in assignment_pos_map[k]:
                    if idx > assignment_idx:
                        order_map[idx] = assignment_idx + 1
                        break
    code_order = [*range(len(func_def.body))]
    for k, v in sorted(order_map.items(), key=lambda x: x[1], reverse=True):
        if k == v:
            continue
        code_order.remove(k)
        code_order.insert(v, k)

    # 根据指定顺序重新排列函数代码
    new_body = [func_def.body[i] for i in code_order]
    func_def.body = new_body

    # 将修改后的 AST 对象转换为源代码字符串
    new_code = astor.to_source(func_ast)
    # print(new_code)
    return new_code
