# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import ast
import astor
import atexit
import copy
import gast
import imp
import inspect
import os
import six
import tempfile

from paddle.fluid import unique_name

dygraph_class_to_static_api = {
    "CosineDecay": "cosine_decay",
    "ExponentialDecay": "exponential_decay",
    "InverseTimeDecay": "inverse_time_decay",
    "NaturalExpDecay": "natural_exp_decay",
    "NoamDecay": "noam_decay",
    "PiecewiseDecay": "piecewise_decay",
    "PolynomialDecay": "polynomial_decay",
}

FOR_ITER_INDEX_PREFIX = '__for_loop_var_index'
FOR_ITER_VAR_SHAPE_PREFIX = '__for_loop_var_shape'


def _is_api_in_module_helper(obj, module_prefix):
    m = inspect.getmodule(obj)
    return m is not None and m.__name__.startswith(module_prefix)


def is_api_in_module(node, module_prefix):
    assert isinstance(node, gast.Call), "Input non-Call node for is_dygraph_api"
    func_str = astor.to_source(gast.gast_to_ast(node.func))
    try:
        # TODO(liym27):
        #  Consider a better to import modules like:
        #  source_file = inspect.getfile(dyfunc)
        #  import_statements = ImportVisitor(source_file).transform()
        #  import_str = "".join(import_statements)
        import paddle
        import paddle.fluid as fluid
        import paddle.fluid.layers as layers
        from paddle.fluid.dygraph import to_variable
        import paddle.fluid.dygraph as dygraph
        return eval("_is_api_in_module_helper({}, '{}')".format(func_str,
                                                                module_prefix))
    except NameError:
        return False


def is_dygraph_api(node):
    # Note: A api in module dygraph_to_static is not a real dygraph api.
    if is_api_in_module(node, "paddle.fluid.dygraph.dygraph_to_static"):
        return False

    return is_api_in_module(node, "paddle.fluid.dygraph")


def is_paddle_api(node):
    return is_api_in_module(node, "paddle.fluid")


# Is numpy_api cannot reuse is_api_in_module because of numpy module problem
def is_numpy_api(node):
    assert isinstance(node, gast.Call), "Input non-Call node for is_numpy_api"
    func_str = astor.to_source(gast.gast_to_ast(node.func))
    try:
        import numpy as np
        module_result = eval("_is_api_in_module_helper({}, '{}')".format(
            func_str, "numpy"))
        # BUG: np.random.uniform doesn't have module and cannot be analyzed
        # TODO: find a better way
        if not module_result:
            return func_str.startswith("numpy.") or func_str.startswith("np.")
    except NameError:
        return False


def is_control_flow_to_transform(node,
                                 static_analysis_visitor=None,
                                 var_name_to_type=None):
    """
    Determines whether the node is a PaddlePaddle control flow statement which needs to
    be transformed into a static graph control flow statement.
    """
    assert isinstance(node, gast.AST), \
        "The type of input node must be gast.AST, but received %s." % type(node)
    visitor = IsControlFlowVisitor(
        node, static_analysis_visitor, node_var_type_map=var_name_to_type)
    need_to_transform = visitor.transform()
    return need_to_transform


def _delete_keywords_from(node):
    assert isinstance(node, gast.Call)
    func_src = astor.to_source(gast.gast_to_ast(node.func))
    import paddle.fluid as fluid
    full_args = eval("inspect.getargspec({})".format(func_src))
    full_args_name = full_args[0]

    node.keywords = [k for k in node.keywords if k.arg in full_args_name]
    return


def to_static_api(dygraph_class):
    if dygraph_class in dygraph_class_to_static_api:
        return dygraph_class_to_static_api[dygraph_class]
    else:
        raise NotImplementedError("Paddle dygraph API {} cannot be converted "
                                  "to static graph at present.".format(
                                      dygraph_class))


def _add_keywords_to(node, dygraph_api_name):
    assert isinstance(node, gast.Call)
    if dygraph_api_name == "Linear":
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "output_dim":
                ast_keyword.arg = "size"

        node.keywords.append(
            gast.keyword(
                arg="num_flatten_dims",
                value=gast.Constant(
                    value=-1, kind=None)))

    if dygraph_api_name == "BilinearTensorProduct":
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "output_dim":
                ast_keyword.arg = "size"

    if dygraph_api_name == "PRelu":
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "input":
                ast_keyword.arg = "x"
    return


def is_to_variable(node):
    assert isinstance(node, gast.Call)
    if is_dygraph_api(node):
        api_name = ast_to_source_code(node.func).strip()
        return api_name.endswith("to_variable")
    return False


def to_static_ast(node, class_node):
    assert isinstance(node, gast.Call)
    assert isinstance(class_node, gast.Call)
    static_api = to_static_api(class_node.func.attr)

    node.func = gast.Attribute(
        attr=static_api,
        ctx=gast.Load(),
        value=gast.Attribute(
            attr='layers',
            ctx=gast.Load(),
            value=gast.Name(
                ctx=gast.Load(), id='fluid', annotation=None,
                type_comment=None)))

    update_args_of_func(node, class_node, 'forward')

    node.args.extend(class_node.args)
    node.keywords.extend(class_node.keywords)
    _add_keywords_to(node, class_node.func.attr)
    _delete_keywords_from(node)

    gast.fix_missing_locations(node)

    return node


def to_assign_node(node):
    # Transform dygraph api `fluid.dygraph.to_variable` to static api `fluid.layers.assign`.
    # NOTE:
    #   1. Api `to_variable` supports data type {float16, float32, float64, int16, int32, int64, uint8, uint16},
    #   but api `assign` only supports {float32, float64, int32, int64, bool};
    #   2. If the input of api `assign` is numpy.ndarray, its size cannot be greater than 1024 * 1024.
    assert isinstance(node, gast.Call)
    assign_api = gast.parse('fluid.layers.assign').body[0].value
    node.func = assign_api

    if node.args:
        node.args = [node.args[0]]
        node.keywords = []
    else:
        for idx, kw in enumerate(node.keywords):
            if kw.arg == 'value':
                node.keywords[idx].arg = 'input'
                node.keywords = [node.keywords[idx]]
                node.args = []
                break
    return node


def update_args_of_func(node, dygraph_node, method_name):
    assert isinstance(node, gast.Call)
    if method_name not in ["__init__", "forward"]:
        raise ValueError(
            "The method name of class to update args should be '__init__' or 'forward'"
        )

    class_src = astor.to_source(gast.gast_to_ast(dygraph_node.func))
    import paddle.fluid as fluid
    if method_name == "__init__" or eval(
            "issubclass({}, fluid.dygraph.Layer)".format(class_src)):
        full_args = eval("inspect.getargspec({}.{})".format(class_src,
                                                            method_name))
        full_args_name = [
            arg_name for arg_name in full_args[0] if arg_name != "self"
        ]
    else:
        full_args_name = []
    added_keywords = []
    for idx, arg in enumerate(node.args):
        added_keywords.append(gast.keyword(arg=full_args_name[idx], value=arg))

    node.args = []
    node.keywords = added_keywords + node.keywords


def create_api_shape_node(tensor_shape_node):
    assert isinstance(tensor_shape_node, (gast.Attribute, gast.Subscript))

    if isinstance(tensor_shape_node, gast.Attribute):
        api_shape_node = gast.Call(
            func=gast.parse('fluid.layers.shape').body[0].value,
            args=[tensor_shape_node.value],
            keywords=[])
        return api_shape_node

    if isinstance(tensor_shape_node, gast.Subscript):
        result_node = copy.deepcopy(tensor_shape_node)
        result_node.value = create_api_shape_node(result_node.value)
        return result_node


def get_constant_variable_node(name, value, shape=[1], dtype='int64'):
    return gast.parse('%s = fluid.layers.fill_constant(%s, "%s", %s)' %
                      (name, str(shape), dtype, str(value)))


def get_attribute_full_name(node):
    assert isinstance(
        node,
        gast.Attribute), "Input non-Attribute node to get attribute full name"
    return astor.to_source(gast.gast_to_ast(node)).strip()


def generate_name_node(name_ids, ctx=gast.Load()):
    """
    Generate list or gast.Tuple of ast.Name for Return statement.
    """
    if isinstance(name_ids, six.string_types):
        name_ids = [name_ids]
    if not isinstance(name_ids, (list, tuple, set)):
        raise TypeError('name_ids must be list or tuple or set, but received %s'
                        % type(type(name_ids)))
    gast_names = [
        gast.Name(
            id=name_id, ctx=ctx, annotation=None, type_comment=None)
        for name_id in name_ids
    ]
    if len(gast_names) == 1:
        name_node = gast_names[0]
    else:
        name_node = gast.Tuple(elts=gast_names, ctx=ctx)
    return name_node


def create_funcDef_node(nodes, name, input_args, return_name_ids):
    """
    Wrapper all statements of nodes into one ast.FunctionDef, which can be
    called by ast.Call.
    """
    nodes = copy.copy(nodes)
    # add return statement
    if return_name_ids:
        nodes.append(gast.Return(value=generate_name_node(return_name_ids)))
    else:
        nodes.append(gast.Return(value=None))
    func_def_node = gast.FunctionDef(
        name=name,
        args=input_args,
        body=nodes,
        decorator_list=[],
        returns=None,
        type_comment=None)
    return func_def_node


def index_in_list(array_list, item):
    try:
        return array_list.index(item)
    except ValueError:
        # Item not in array_list
        return -1


def create_assign_node(name, node):
    """
    Creates a `gast.Assign` node by given name_id as target and node as value.
    """
    targets = generate_name_node(name, ctx=gast.Store())
    assign_node = gast.Assign(targets=[targets], value=node)
    return targets, assign_node


class RenameTransformer(gast.NodeTransformer):
    def __init__(self, node):
        assert isinstance(
            node, gast.AST), "RenameTransformer only accepts gast.AST as input"
        self.root = node
        self.old_name = ""
        self.new_name = ""

    def rename(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name
        self.visit(self.root)

    def visit_Name(self, node):
        self.generic_visit(node)
        if node.id == self.old_name:
            node.id = self.new_name
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        attr_full_name = get_attribute_full_name(node)
        if attr_full_name == self.old_name:
            new_name_node = gast.parse(self.new_name).body[0].value
            return new_name_node
        return node


def ast_to_func(ast_root, dyfunc, delete_on_exit=True):
    """
    Transform modified AST of decorated function into python callable object.
    TODO: If only decorate one of inner function instead of decorating the main
    function, the other inner functions are invisible for the decorated function.
    """
    source = ast_to_source_code(ast_root)
    if six.PY2:
        source = source.encode('utf-8')
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    else:
        f = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8')
    with f:
        module_name = os.path.basename(f.name[:-3])
        f.write(source)

    if delete_on_exit:
        atexit.register(lambda: os.remove(f.name))
    module = imp.load_source(module_name, f.name)
    func_name = dyfunc.__name__
    if not hasattr(module, func_name):
        raise ValueError(
            'Function: %s doesn\'t exist in the Module transformed from AST.' %
            func_name)
    callable_func = getattr(module, func_name)
    # After transform dygraph function into callable_func saved in tmp file,
    # it lost the global variables from imported statements or defined in source file.
    # Recovers the necessary variables by `__globals__`.
    recover_globals_attribute(dyfunc, callable_func)

    return callable_func, f.name


def recover_globals_attribute(src_obj, dst_obj):
    attr_name = '__globals__'

    src_globals = getattr(src_obj, attr_name, {})
    dst_globals = getattr(dst_obj, attr_name, {})

    for k, v in src_globals.items():
        # ignore builtin attribute.
        if not (k.startswith('__') and k.endswith('__')):
            dst_globals[k] = v


def ast_to_source_code(ast_node):
    """
    Transformers ast node into source code.
    """
    if not isinstance(ast_node, (gast.AST, ast.AST)):
        raise TypeError(
            "Type of ast_root should be gast.AST or ast.AST, but received %s." %
            type(ast_node))
    if isinstance(ast_node, gast.AST):
        ast_node = gast.gast_to_ast(ast_node)
    source_code = astor.to_source(ast_node)
    return source_code


def is_candidate_node(node):
    """
    Nodes with specified type will be dependent on tensor.
    """
    is_compare_node = isinstance(node, (gast.Compare, gast.BoolOp, gast.UnaryOp,
                                        gast.For, gast.If, gast.While))
    # TODO(Aurelius84): `.numpy()` may be an customized function,
    # and should consider a more elegant way to solve this problem.
    has_numpy_attr = ".numpy()" in ast_to_source_code(node)
    return is_compare_node or has_numpy_attr


def compare_with_none(node):
    """
    Whether the comparator of `gast.Compare` node is `None`.
    """
    if isinstance(node, gast.Compare):
        for child in [node.left, node.comparators]:
            # node.comparators is a list.
            if isinstance(child, list):
                child = child[0]
            if (isinstance(child, gast.Constant) and child.value is None) or (
                    isinstance(child, gast.Name) and child.id == 'None'):
                return True
    return False


class IsControlFlowVisitor(gast.NodeVisitor):
    """
    Judge whether the ast_node of control flow from Dygraph code dependent on paddle Tensor.
    `ast_node` can be gast.If, gast.For, gast.While, gast.If.test(gast.Compare, gast.BoolOp, gast.UnaryOp).

    If returns True,
    gast.If.test must meet at least one of the following requirements:
        1. involves at least one var whose type is Tensor.
        2. the Tensor var calls `.numpy()[]` interface or Tensor.shape is [1].
        3. involves Tensor.shape[i] and the shape[i] is unknown in compile time.
    gast.While must meet at least one of the requirements 1 to 5:
        4. has `break` statement.
        5. has `continue` statement.
    gast.For must meet at least one of the requirements 4 to 6:
        6. calls `range` function in `for` statement and the argument of range is Tensor.
        TODO: Support non-range case

    The following examples should not be considered as control_flow_if:
        1. `if Tensor_var` or `if Tensor_var is None`
        2. if Tensor.shape[i] is determined with fixed value (not -1 or None)

    Note: pred in ConditionalBlock require variable, which means all vars should be Tensor
          or transformed into Tensor, like fill_constant(shape=[1], dtype='int32', value=Tensor.shape[i]).

    TODO: 1. need to deal with `tensor.shape[i]` which need to eval the data of shape[i],
             because reshape_op may be called before this statement.
    """

    def __init__(self,
                 ast_node,
                 static_analysis_visitor=None,
                 node_var_type_map=None):
        assert isinstance(
            ast_node, gast.AST
        ), "Type of input node should be gast.AST, but received %s." % type(
            ast_node)
        self.ast_root = ast_node
        if static_analysis_visitor is None:
            from .static_analysis import StaticAnalysisVisitor
            static_analysis_visitor = StaticAnalysisVisitor(ast_node)
        self.static_analysis_visitor = static_analysis_visitor
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )
        self.node_var_type_map = node_var_type_map

        self.is_control_flow_num = 0
        self._compare_node_tenor_set = set()

    def transform(self):
        node = self.ast_root
        if isinstance(node, gast.If):
            self._visit_If(node)
        elif isinstance(node, gast.For):
            self._visit_For(node)
        elif isinstance(node, gast.While):
            self._visit_While(node)
        else:
            self.visit(node)
        return self.is_control_flow_num > 0

    def _visit_If(self, node):
        assert isinstance(node, gast.If)
        self.visit(node.test)
        return

    def _visit_For(self, node):
        assert isinstance(node, gast.For)
        if not isinstance(node.iter, gast.Call):
            return

        # for in range(v.numpy()) or for in enumerate(v.numpy())
        if isinstance(node.iter.func, gast.Name):
            if node.iter.func.id == "range" or node.iter.func.id == "enumerate":
                for arg in node.iter.args:
                    self.visit(arg)
            else:
                return
        # for in v.numpy()
        elif isinstance(node.iter.func, gast.Attribute):
            if node.iter.func.attr == 'numpy':
                self._visit_Call(node.iter)
            else:
                return
        else:
            return

        for child_node in gast.walk(node):
            if isinstance(child_node, (gast.Continue, gast.Break)):
                self._visit_break_continue(child_node)
        return

    def _visit_While(self, node):
        assert isinstance(node, gast.While)
        test = node.test
        self.generic_visit(test)
        for child_node in gast.walk(node):
            if isinstance(child_node, (gast.Continue, gast.Break)):
                self._visit_break_continue(child_node)
        return

    def _visit_break_continue(self, node):
        assert isinstance(node, (gast.Break, gast.Continue))
        wrapper_node = self.node_to_wrapper_map.get(node)
        if not wrapper_node:
            # Transformed node is not in node_to_wrapper_map
            return

        while wrapper_node.parent:
            parent_node = wrapper_node.parent.node
            if isinstance(parent_node, (gast.For, gast.While)):
                if parent_node is self.ast_root:
                    self.is_control_flow_num += 1
                    return
                else:
                    return

            wrapper_node = wrapper_node.parent

        return

    def visit_BoolOp(self, node):
        for i, child in enumerate(node.values):
            self.visit(child)
        return node

    def visit_Compare(self, node):
        pre_control_flow_num = self.is_control_flow_num
        if not compare_with_none(node):
            self.generic_visit(node)
            for child in gast.walk(node):
                if isinstance(child, gast.Subscript):
                    self._visit_Subscript(child)
        if self.is_control_flow_num > pre_control_flow_num:
            self._compare_node_tenor_set.add(node)
        return node

    def _visit_Subscript(self, node):
        self.generic_visit(node)
        if hasattr(node, 'value') and isinstance(node.value, gast.Call):
            self._visit_Call(node.value)
        return node

    def _visit_Call(self, node):
        assert isinstance(node, gast.Call)
        if isinstance(node.func, gast.Attribute):
            attr_node = node.func
            if attr_node.attr == 'numpy':
                self.is_control_flow_num += 1

    def visit_Call(self, node):
        self._visit_Call(node)
        if is_paddle_api(node):
            self.is_control_flow_num += 1
        return node

    def visit_Name(self, node):
        if self._is_node_with_tensor(node, node.id):
            self.is_control_flow_num += 1
        return node

    def visit_Constant(self, node):
        if self._is_node_with_tensor(node, node.value):
            self.is_control_flow_num += 1
        return node

    def _is_node_with_tensor(self, node, name_id):
        from paddle.fluid.dygraph.dygraph_to_static.static_analysis import NodeVarType

        # Look up the node_var_type_map by name_id.
        if self.node_var_type_map:
            if name_id and isinstance(name_id, six.string_types):
                var_type = self.node_var_type_map.get(name_id, None)
                if var_type and var_type & NodeVarType.TENSOR_TYPES:
                    return True
        # if not found, look up the node_to_wrapper_map by node.
        wrapper_node = self.node_to_wrapper_map.get(node, None)
        if wrapper_node is not None:
            if wrapper_node.node_var_type & NodeVarType.TENSOR_TYPES:
                return True

        return False

    def get_compare_nodes_with_tensor(self):
        return self._compare_node_tenor_set


class NameNodeReplaceTransformer(gast.NodeTransformer):
    """
    This class replaces specified gast.Name node by replace_node.
    """

    def __init__(self, root_node, target_name, replace_node):
        assert isinstance(target_name, str)
        self.target_name = target_name
        self.replace_node = replace_node

        self.visit(root_node)

    def visit_Name(self, node):
        if node.id == self.target_name:
            return self.replace_node
        return node


class ForNodeVisitor(object):
    """
    This class parses python for statement, get transformed 3 statement components of for node
    three key statements:
        1). init_stmts: list[node], prepare nodes of for loop, may not only one
        2). cond_stmt: node, condition node to judge whether continue loop
        3). body_stmts: list[node], updated loop body, sometimes we should change
            the original statement in body, not just append new statement

    In this process, the semantics of for does not change.

    Now only can parse 3 type statements:
        1). for x in range(***)
        2). for x in var.numpy()
        3). for i, x enumerate(var.numpy())
    """

    def __init__(self, for_node):
        assert isinstance(
            for_node, gast.For
        ), "Input node for the initialization of ForNodeVisitor is not gast.For node."
        # 1. original for node
        self.node = for_node

        # 2. gast.For node main parts
        self.target = for_node.target
        # NOTE: type may be Node or list[Node]
        self.iter_args = for_node.iter if self.is_for_iter(
        ) else for_node.iter.args
        self.body = for_node.body

        # 3. key shared node or names
        # - x:
        #   - for x in range(***)
        #   - for x in var.numpy()
        #   - for i, x enumerate(var.numpy())
        self.iter_var_name = self._get_iter_var_name()

        # - created index var to slice Variable: __for_loop_var_index_0
        #   - for x in var.numpy()
        #   - for i, x enumerate(var.numpy())
        self.iter_idx_name = unique_name.generate(FOR_ITER_INDEX_PREFIX)

        # - created shape var to build loop condition: __for_loop_var_shape_0
        #   - for x in var.numpy()
        #   - for i, x enumerate(var.numpy())
        self.iter_var_shape_name = unique_name.generate(
            FOR_ITER_VAR_SHAPE_PREFIX)

        # - var.numpy()
        #   - for x in var.numpy()
        #   - for i, x enumerate(var.numpy())
        self.iter_node = self._get_iter_node()

        # - enumeate i:
        #   - for i, x enumerate(var.numpy())
        self.enum_idx_name = self._get_enum_idx_name()

        # - range/enumerate args length
        self.args_length = None

    def parse(self):
        self._args_check()
        if self.is_for_range_iter():
            return self._parse_for_range_stmts()
        elif self.is_for_iter():
            return self._parse_for_stmts()
        elif self.is_for_enumerate_iter():
            return self._parse_for_enumerate_stmts()
        else:
            raise None

    def is_for_range_iter(self):
        return isinstance(self.node.iter.func,
                          gast.Name) and self.node.iter.func.id == "range"

    def is_for_iter(self):
        return isinstance(
            self.node.iter.func,
            gast.Attribute) and self.node.iter.func.attr == 'numpy'

    def is_for_enumerate_iter(self):
        return isinstance(self.node.iter.func,
                          gast.Name) and self.node.iter.func.id == "enumerate"

    def _args_check(self):
        if self.is_for_range_iter():
            self.args_length = len(self.iter_args)
            assert self.args_length >= 1 and self.args_length <= 3, "range() function takes 1 to 3 arguments"
        elif self.is_for_enumerate_iter():
            self.args_length = len(self.iter_args)
            assert self.args_length >= 1 and self.args_length <= 2, "enumerate() function takes 1 to 2 arguments"
        else:
            self.args_length = None

    def _parse_for_range_stmts(self):
        init_stmts = []
        init_stmts.append(self._build_index_init_node())

        compare_node = self._build_compare_node()
        step_node = self._build_step_node()
        cond_stmt = self._build_cond_stmt(step_node, compare_node)

        body_stmts = self.body
        body_stmts.append(self._build_index_increase_node(step_node))

        return init_stmts, cond_stmt, body_stmts

    def _parse_for_stmts(self):
        init_stmts = []
        init_stmts.append(self._build_index_init_node())
        init_stmts.append(self._build_var_shape_assign_node())

        compare_node = self._build_compare_node()
        step_node = self._build_step_node()
        cond_stmt = self._build_cond_stmt(step_node, compare_node)

        body_stmts = self.body
        var_slice_node = self._build_var_slice_node()
        for body_node in body_stmts:
            NameNodeReplaceTransformer(body_node, self.iter_var_name,
                                       var_slice_node)
        body_stmts.append(self._build_index_increase_node(step_node))

        return init_stmts, cond_stmt, body_stmts

    def _parse_for_enumerate_stmts(self):
        init_stmts = []
        init_stmts.append(self._build_index_init_node())
        init_stmts.append(self._build_var_shape_assign_node())
        init_stmts.append(self._build_enum_init_node())

        compare_node = self._build_compare_node()
        step_node = self._build_step_node()
        cond_stmt = self._build_cond_stmt(step_node, compare_node)

        body_stmts = self.body
        var_slice_node = self._build_var_slice_node()
        for body_node in body_stmts:
            NameNodeReplaceTransformer(body_node, self.iter_var_name,
                                       var_slice_node)
        body_stmts.append(self._build_index_increase_node(step_node))
        body_stmts.append(self._build_enum_increase_node())

        return init_stmts, cond_stmt, body_stmts

    def _build_index_init_node(self):
        if self.is_for_range_iter():
            if self.args_length == 1:
                index_init_node = get_constant_variable_node(self.iter_var_name,
                                                             0)
            else:
                index_init_node = gast.Assign(
                    targets=[
                        gast.Name(
                            id=self.iter_var_name,
                            ctx=gast.Store(),
                            annotation=None,
                            type_comment=None)
                    ],
                    value=self.iter_args[0])
        else:
            index_init_node = get_constant_variable_node(self.iter_idx_name, 0)
        return index_init_node

    def _build_var_shape_assign_node(self):
        # get variable shape as iter length
        return gast.Assign(
            targets=[
                gast.Name(
                    id=self.iter_var_shape_name,
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None)
            ],
            value=create_api_shape_node(self.iter_node.func))

    def _build_enum_init_node(self):
        enum_init_node = get_constant_variable_node(
            name=self.enum_idx_name, value=0)
        if self.is_for_enumerate_iter() and self.args_length != 1:
            enum_init_node = gast.Assign(
                targets=[
                    gast.Name(
                        id=self.enum_idx_name,
                        ctx=gast.Store(),
                        annotation=None,
                        type_comment=None)
                ],
                value=self.iter_args[1])
        return enum_init_node

    def _build_compare_node(self):
        if self.is_for_range_iter():
            compare_node = self.iter_args[
                0] if self.args_length == 1 else self.iter_args[1]
        else:
            compare_node = gast.Subscript(
                value=gast.Name(
                    id=self.iter_var_shape_name,
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None),
                slice=gast.Index(value=gast.Constant(
                    value=0, kind=None)),
                ctx=gast.Load())
        return compare_node

    def _build_step_node(self):
        if self.is_for_range_iter():
            step_node = self.iter_args[
                2] if self.args_length == 3 else gast.Constant(
                    value=1, kind=None)
        else:
            step_node = gast.Constant(value=1, kind=None)
        return step_node

    def _build_cond_stmt(self, step_node, compare_node):
        return gast.Compare(
            left=gast.BinOp(
                left=gast.Name(
                    id=self.iter_var_name
                    if self.is_for_range_iter() else self.iter_idx_name,
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None),
                op=gast.Add(),
                right=step_node),
            ops=[gast.LtE()],
            comparators=[compare_node])

    def _build_index_increase_node(self, step_node):
        return gast.AugAssign(
            target=gast.Name(
                id=self.iter_var_name
                if self.is_for_range_iter() else self.iter_idx_name,
                ctx=gast.Store(),
                annotation=None,
                type_comment=None),
            op=gast.Add(),
            value=step_node)

    def _build_var_slice_node(self):
        return gast.Subscript(
            value=self.iter_node,
            slice=gast.Index(value=gast.Name(
                id=self.iter_idx_name,
                ctx=gast.Load(),
                annotation=None,
                type_comment=None)),
            ctx=gast.Load())

    def _build_enum_increase_node(self):
        return gast.AugAssign(
            target=gast.Name(
                id=self.enum_idx_name,
                ctx=gast.Store(),
                annotation=None,
                type_comment=None),
            op=gast.Add(),
            value=gast.Constant(
                value=1, kind=None))

    def _get_iter_var_name(self):
        if self.is_for_range_iter():
            return self.target.id
        elif self.is_for_iter():
            return self.target.id
        elif self.is_for_enumerate_iter():
            return self.target.elts[1].id
        return None

    def _get_iter_node(self):
        if self.is_for_iter():
            return self.iter_args
        elif self.is_for_enumerate_iter():
            return self.iter_args[0]
        return None

    def _get_enum_idx_name(self):
        if self.is_for_enumerate_iter():
            return self.target.elts[0].id
        return None
