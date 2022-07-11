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

from paddle.utils import gast
from paddle.fluid import unique_name
from paddle.fluid.dygraph.dygraph_to_static.utils import get_attribute_full_name
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import create_assign_node
from paddle.fluid.dygraph.dygraph_to_static.utils import ORIGI_INFO
from paddle.fluid.dygraph.dygraph_to_static.utils import FOR_ITER_INDEX_PREFIX
from paddle.fluid.dygraph.dygraph_to_static.utils import FOR_ITER_TUPLE_PREFIX
from paddle.fluid.dygraph.dygraph_to_static.utils import FOR_ITER_TUPLE_INDEX_PREFIX
from paddle.fluid.dygraph.dygraph_to_static.utils import FOR_ITER_VAR_LEN_PREFIX
from paddle.fluid.dygraph.dygraph_to_static.utils import FOR_ITER_VAR_NAME_PREFIX
from paddle.fluid.dygraph.dygraph_to_static.utils import FOR_ITER_ZIP_TO_LIST_PREFIX


class BaseTransformer(gast.NodeTransformer):

    def visit(self, node):
        if not isinstance(node, gast.AST):
            msg = ('Expected "gast.AST", but got "{}".').format(type(node))
            raise ValueError(msg)
        origin_info = getattr(node, ORIGI_INFO, None)

        result = super(BaseTransformer, self).visit(node)

        iter_result = result
        if iter_result is not node and iter_result is not None:
            if not isinstance(iter_result, (list, tuple)):
                iter_result = (iter_result, )
            if origin_info is not None:
                for n in iter_result:
                    setattr(n, ORIGI_INFO, origin_info)

        return result


class RenameTransformer(BaseTransformer):

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


class NameNodeReplaceTransformer(BaseTransformer):
    """
    This class replaces specified gast.Name node by replace_node.
    """

    def __init__(self, root_node, target_name, replace_node):
        assert isinstance(target_name, str)

        # NOTE(liym27):
        # Use gast.Name to replace gast.Name, otherwise, errors may occur.
        #
        # For examples:
        # If using a gast.Subscript to replace gast.Name, and the original gast.Name
        # is in the arguments of FunctionDef, an exception will be raised.
        #
        # ```
        # def func(x[i])) # x[i] can not be a argument
        #    # ...
        # ```

        assert isinstance(replace_node, gast.Name)
        self.target_name = target_name
        self.replace_node = replace_node

        self.visit(root_node)

    def visit_Name(self, node):
        if node.id == self.target_name:
            return self.replace_node
        return node

    def visit_Nonlocal(self, node):
        names = node.names

        def replace(s):
            if s == self.target_name: return self.replace_node.id
            return s

        node.names = list(map(replace, names))
        return node


class ForLoopTuplePreTransformer(BaseTransformer):
    """
    ForNodeVisitor parses 3 type statements (Here var is VarBase(Tensor) or python variable):
        1). for x in range(var[*]|var.numpy()[*])
        2). for x in var|var.numpy()
        3). for i, x in enumerate(var|var.numpy())

        We chose these 3 types because they are easier (x can be variable name iterating in var).
        However, users can write tuples in Python for loop, such as
        1). for var1, var2 in var|var.numpy()
        2). for t in enumerate(var|var.numpy())
        2). for i, (var1, var2, va3) in enumerate(var|var.numpy())

        To handle these case, this method will do the rewrite tuple pre-process:
        1). Non-enumerate case: for var1, var2 in var|var.numpy() will be re-written as:
          for FOR_ITER_TUPLE_PREFIX_x in var | var.numpy():
            var1 = FOR_ITER_TUPLE_PREFIX_x[0]
            var2 = FOR_ITER_TUPLE_PREFIX_x[1]
        2). Enumerate out tuple case: for t in enumerate(var|var.numpy) will be rewritten as:
          for FOR_ITER_TUPLE_INDEX_PREFIX_x, FOR_ITER_TUPLE_PREFIX_x in enumerate(var|var.numpy):
            t = (FOR_ITER_TUPLE_INDEX_PREFIX_x, FOR_ITER_TUPLE_PREFIX_x)
        3). Enumerate inner tuple case: for i, (var1, (var2, va3)) in enumerate(var|var.numpy()) will
        be re-written as:
          for i, FOR_ITER_TUPLE_PREFIX_x in var | var.numpy():
            var1 = FOR_ITER_TUPLE_PREFIX_x[0]
            var2 = FOR_ITER_TUPLE_PREFIX_x[1][0]
            var3 = FOR_ITER_TUPLE_PREFIX_x[1][1]
    """

    def __init__(self, wrapper_root):
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

    def transform(self):
        self.visit(self.root)

    def visit_For(self, node):
        if self.is_for_enumerate_iter(node):
            if isinstance(node.target, (gast.Name, gast.Attribute)):
                # Out tuple case
                out_tuple_name = ast_to_source_code(node.target).strip()
                tuple_iter_name = unique_name.generate(
                    FOR_ITER_TUPLE_INDEX_PREFIX)
                tuple_var_name = unique_name.generate(FOR_ITER_TUPLE_PREFIX)
                node.target = gast.Tuple(elts=[
                    gast.Name(id=tuple_iter_name,
                              ctx=gast.Store(),
                              annotation=None,
                              type_comment=None),
                    gast.Name(id=tuple_var_name,
                              ctx=gast.Store(),
                              annotation=None,
                              type_comment=None)
                ],
                                         ctx=gast.Store())
                node.body.insert(
                    0,
                    gast.Assign(targets=[
                        gast.Name(id=out_tuple_name,
                                  ctx=gast.Store(),
                                  annotation=None,
                                  type_comment=None)
                    ],
                                value=gast.Tuple(elts=[
                                    gast.Name(id=tuple_iter_name,
                                              ctx=gast.Load(),
                                              annotation=None,
                                              type_comment=None),
                                    gast.Name(id=tuple_var_name,
                                              ctx=gast.Load(),
                                              annotation=None,
                                              type_comment=None)
                                ],
                                                 ctx=gast.Load())))
            elif isinstance(node.target, (gast.List, gast.Tuple)) and len(
                    node.target.elts) >= 2 and isinstance(
                        node.target.elts[1], (gast.List, gast.Tuple)):
                # Inner tuple case
                inner_tuple_name = unique_name.generate(FOR_ITER_TUPLE_PREFIX)
                origin_inner_tuple_node = node.target.elts[1]
                node.target.elts[1] = gast.Name(id=inner_tuple_name,
                                                ctx=gast.Store(),
                                                annotation=None,
                                                type_comment=None)
                node.body[0:0] = self.tuple_to_stmts(origin_inner_tuple_node,
                                                     inner_tuple_name)
        elif self.is_for_iter(node) and isinstance(node.target,
                                                   (gast.List, gast.Tuple)):
            # Non-enumrate case:
            tuple_name = unique_name.generate(FOR_ITER_TUPLE_PREFIX)
            origin_tuple_node = node.target
            node.target = gast.Name(id=tuple_name,
                                    ctx=gast.Store(),
                                    annotation=None,
                                    type_comment=None)
            node.body[0:0] = self.tuple_to_stmts(origin_tuple_node, tuple_name)
        return node

    def tuple_to_stmts(self, node, tuple_name, idx=[]):
        if not isinstance(node, (gast.Tuple, gast.List)):
            value_node_str = tuple_name
            for i in idx:
                value_node_str = value_node_str + "[{}]".format(i)

            node_str = ast_to_source_code(node).strip()
            assign_node_str = "{} = {}".format(node_str, value_node_str)
            assign_node = gast.parse(assign_node_str).body[0]
            return [assign_node]

        # isinstance(node, (gast.Tuple, gast.List))
        ret = []
        for i, element in enumerate(node.elts):
            ret += self.tuple_to_stmts(node.elts[i], tuple_name, idx + [i])
        return ret

    def is_for_iter(self, for_node):
        assert isinstance(for_node,
                          gast.For), "Input node is not gast.For node."
        if isinstance(for_node.iter, (gast.Name, gast.Attribute)):
            return True
        elif isinstance(for_node.iter, gast.Call) and isinstance(
                for_node.iter.func,
                gast.Attribute) and for_node.iter.func.attr == 'numpy':
            return True
        elif isinstance(for_node.iter, gast.Subscript):
            return True
        else:
            return False

    def is_for_enumerate_iter(self, for_node):
        assert isinstance(for_node,
                          gast.For), "Input node is not gast.For node."
        return isinstance(for_node.iter, gast.Call) and isinstance(
            for_node.iter.func,
            gast.Name) and for_node.iter.func.id == "enumerate"


class SplitAssignTransformer(BaseTransformer):
    """
    This class transforms sequence assignments and multi-target assignments to normal assignments.
    """

    def __init__(self, ast_node):
        assert isinstance(ast_node, gast.AST)
        self.ast_root = ast_node

    def transform(self):
        self.visit(self.ast_root)

    def visit_Assign(self, node):
        target_nodes = node.targets
        if len(target_nodes) == 1:
            node = self._parse_sequence_assign(node)
        else:
            node = self._parse_multi_target_assign(node)
        return node

    def _parse_sequence_assign(self, node):
        """
        a, b = c, d
        ->
        a = c
        b = d
        """
        assert isinstance(node, gast.Assign)

        target_nodes = node.targets
        value_node = node.value
        if not isinstance(target_nodes[0], (gast.List, gast.Tuple)):
            return node
        if not isinstance(value_node, (gast.List, gast.Tuple)):
            return node

        targets = node.targets[0].elts
        values = node.value.elts
        if len(targets) != len(values):
            return node

        new_nodes = []
        for target, value in zip(targets, values):
            assign_node = gast.Assign(targets=[target], value=value)
            new_nodes.append(assign_node)

        return new_nodes

    def _parse_multi_target_assign(self, node):
        """
         Example 1:
         a = b = c
         ->
         b = c
         a = b

         Example 2:
         a, b = c, d = x
         ->
         c,d = x
         a = c
         b = d
         """
        assert isinstance(node, gast.Assign)

        target_nodes = node.targets
        value_node = node.value
        new_nodes = []
        for target in reversed(target_nodes):
            assign_node = gast.Assign(targets=[target], value=value_node)
            # NOTE: Because assign_node can be sequence assign statement like `a,b = c,d`,
            # it's necessary to visit this new assign_node
            parsed_node = self.visit_Assign(assign_node)
            if not isinstance(parsed_node, list):
                parsed_node = [parsed_node]

            new_nodes.extend(parsed_node)
            value_node = target

        return new_nodes


class ForNodeVisitor(object):
    """
    This class parses python for statement, get transformed 3 statement components of for node
    three key statements:
        1). init_stmts: list[node], prepare nodes of for loop, may not only one
        2). cond_stmt: node, condition node to judge whether continue loop
        3). body_stmts: list[node], updated loop body, sometimes we should change
            the original statement in body, not just append new statement

    In this process, the semantics of for does not change.

    Now only can parse 3 type statements (Here var is VarBase(Tensor) or python variable):
        1). for x in range(var[*]|var.numpy()[*])
        2). for x in var|var.numpy()
        3). for i, x enumerate(var|var.numpy())
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
        #   - for x in var|var.numpy()
        #   - for i, x enumerate(var|var.numpy())
        self.iter_var_name = self._get_iter_var_name()

        # - created index var to slice Variable: __for_loop_var_index_0
        #   - for x in var|var.numpy()
        #   - for i, x enumerate(var|var.numpy())
        self.iter_idx_name = unique_name.generate(FOR_ITER_INDEX_PREFIX)

        # - created shape var to build loop condition: __for_loop_var_len_0
        #   - for x in var|var.numpy()
        #   - for i, x enumerate(var|var.numpy())
        #   - for x in var
        self.iter_var_len_name = unique_name.generate(FOR_ITER_VAR_LEN_PREFIX)
        # - created zip to list var : __for_loop_iter_zip_0
        self.iter_zip_to_list_name = unique_name.generate(
            FOR_ITER_ZIP_TO_LIST_PREFIX)

        # - var.numpy()/var
        #   - for x in var|var.numpy()
        #   - for i, x enumerate(var|var.numpy())
        self.iter_node = self._get_iter_node()

        # - enumeate i:
        #   - for i, x enumerate(var|var.numpy())
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
            return None

    def is_for_range_iter(self):
        return isinstance(self.node.iter, gast.Call) and isinstance(
            self.node.iter.func,
            gast.Name) and self.node.iter.func.id == "range"

    def is_for_iter(self):
        if isinstance(self.node.iter,
                      (gast.Name, gast.Attribute, gast.List, gast.Tuple)):
            return True
        elif isinstance(self.node.iter, gast.Call) and isinstance(
                self.node.iter.func,
                gast.Attribute) and self.node.iter.func.attr == 'numpy':
            return True
        elif isinstance(self.node.iter, gast.Subscript):
            return True
        else:
            return False

    def is_for_enumerate_iter(self):
        return isinstance(self.node.iter, gast.Call) and isinstance(
            self.node.iter.func,
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
        init_stmts.extend(self._build_iter_node())
        init_stmts.append(self._build_index_init_node())
        init_stmts.append(self._build_var_len_assign_node())

        compare_node = self._build_compare_node()
        step_node = self._build_step_node()
        cond_stmt = self._build_cond_stmt(step_node, compare_node)

        body_stmts = self.body

        # NOTE(liym27): Here add a gast.Assign, and the target of it is gast.Name.
        # In NameNodeReplaceTransformer, using gast.Name to replace gast.Name is safe.
        target_node, assign_node = self._build_assign_var_slice_node()
        body_stmts[0:0] = [assign_node]
        for body_node in body_stmts:
            NameNodeReplaceTransformer(body_node, self.iter_var_name,
                                       target_node)
        body_stmts.append(self._build_index_increase_node(step_node))

        return init_stmts, cond_stmt, body_stmts

    def _parse_for_enumerate_stmts(self):
        init_stmts = []
        init_stmts.extend(self._build_iter_node())
        init_stmts.append(self._build_index_init_node())
        init_stmts.append(self._build_var_len_assign_node())
        init_stmts.append(self._build_enum_init_node())

        compare_node = self._build_compare_node()
        step_node = self._build_step_node()
        cond_stmt = self._build_cond_stmt(step_node, compare_node)

        body_stmts = self.body

        target_node, assign_node = self._build_assign_var_slice_node()
        body_stmts[0:0] = [assign_node]
        for body_node in body_stmts:
            NameNodeReplaceTransformer(body_node, self.iter_var_name,
                                       target_node)

        body_stmts.append(self._build_index_increase_node(step_node))
        body_stmts.append(self._build_enum_increase_node())

        return init_stmts, cond_stmt, body_stmts

    def _build_index_init_node(self):
        if self.is_for_range_iter():
            if self.args_length == 1:
                index_init_value_str = '0'
            else:
                index_init_value_str = ast_to_source_code(
                    self.iter_args[0]).strip()

            index_init_var_name = self.iter_var_name
        else:
            index_init_value_str = '0'
            index_init_var_name = self.iter_idx_name

        index_init_node_source_str = "{target} = {value}".format(
            target=index_init_var_name, value=index_init_value_str)

        index_init_node = gast.parse(index_init_node_source_str).body[0]

        return index_init_node

    def _build_var_len_assign_node(self):
        # get the length of iterable variable
        if isinstance(self.iter_node, gast.Call) and isinstance(
                self.iter_node.func,
                gast.Attribute) and self.iter_node.func.attr == 'numpy':
            iter_var_name = ast_to_source_code(
                self.iter_node.func.value).strip()
        else:
            iter_var_name = ast_to_source_code(self.iter_node).strip()

        convert_len_node_source_str = '{} = _jst.Len({})'.format(
            self.iter_var_len_name, iter_var_name)

        convert_len_node = gast.parse(convert_len_node_source_str).body[0]

        return convert_len_node

    def _build_iter_node(self):
        """
        Process special cases for iter_node inclue:
          - Case 1 (for zip):
            
            - for i, val in enumerate(zip(x, y))  # original code:
            
            - __for_loop_iter_zip_0 = list(zip(x, y))
            - for i, val in enumerate(__for_loop_iter_zip_0)
        """
        new_nodes = []
        if isinstance(self.iter_node, gast.Call) and isinstance(
                self.iter_node.func, gast.Name):
            if self.iter_node.func.id == 'zip':
                iter_var_name = ast_to_source_code(self.iter_node).strip()
                zip_to_list_str = "{target} = list({value})".format(
                    target=self.iter_zip_to_list_name, value=iter_var_name)
                zip_to_list_node = gast.parse(zip_to_list_str).body[0]
                new_nodes.append(zip_to_list_node)

                self.iter_node = gast.Name(id=self.iter_zip_to_list_name,
                                           ctx=gast.Load(),
                                           annotation=None,
                                           type_comment=None)

        return new_nodes

    def _build_enum_init_node(self):
        if self.is_for_enumerate_iter() and self.args_length != 1:
            init_value_str = ast_to_source_code(self.iter_args[1]).strip()
        else:
            init_value_str = '0'

        enum_init_node_source_str = "{} = {}".format(self.enum_idx_name,
                                                     init_value_str)
        enum_init_node = gast.parse(enum_init_node_source_str).body[0]
        return enum_init_node

    def _build_compare_node(self):
        if self.is_for_range_iter():
            compare_node = self.iter_args[
                0] if self.args_length == 1 else self.iter_args[1]
        else:
            compare_node = gast.Name(id=self.iter_var_len_name,
                                     ctx=gast.Load(),
                                     annotation=None,
                                     type_comment=None)
        return compare_node

    def _build_step_node(self):
        if self.is_for_range_iter():
            step_node = self.iter_args[
                2] if self.args_length == 3 else gast.Constant(value=1,
                                                               kind=None)
        else:
            step_node = gast.Constant(value=1, kind=None)
        return step_node

    def _build_cond_stmt(self, step_node, compare_node):
        if not isinstance(step_node, (gast.Constant, gast.UnaryOp)):
            raise NotImplementedError(
                "Dynamic-to-Static only supports the step value is a constant or negative constant in 'for-range' statements, "
                "such as '2', '-3'. But received: '{}'. Please fix code to be compatible with Dynamic-to-Static."
                .format(ast_to_source_code(step_node).strip()))

        if isinstance(step_node, gast.UnaryOp) or step_node.value < 0:
            # eg:
            # range(max, min, -2)
            # ->
            # i > min
            return gast.Compare(left=gast.Name(
                id=self.iter_var_name
                if self.is_for_range_iter() else self.iter_idx_name,
                ctx=gast.Load(),
                annotation=None,
                type_comment=None),
                                ops=[gast.Gt()],
                                comparators=[compare_node])
        else:
            # eg:
            # range(min, max, 2)
            # ->
            # i < max
            return gast.Compare(left=gast.Name(
                id=self.iter_var_name
                if self.is_for_range_iter() else self.iter_idx_name,
                ctx=gast.Load(),
                annotation=None,
                type_comment=None),
                                ops=[gast.Lt()],
                                comparators=[compare_node])

    def _build_index_increase_node(self, step_node):
        return gast.AugAssign(target=gast.Name(
            id=self.iter_var_name
            if self.is_for_range_iter() else self.iter_idx_name,
            ctx=gast.Store(),
            annotation=None,
            type_comment=None),
                              op=gast.Add(),
                              value=step_node)

    def _build_assign_var_slice_node(self):
        var_slice_str = "{}[{}]".format(
            ast_to_source_code(self.iter_node).strip(), self.iter_idx_name)
        var_slice_node = gast.parse(var_slice_str).body[0].value
        new_iter_var_name = unique_name.generate(FOR_ITER_VAR_NAME_PREFIX)
        target_node, assign_node = create_assign_node(new_iter_var_name,
                                                      var_slice_node)
        return target_node, assign_node

    def _build_enum_increase_node(self):
        return gast.AugAssign(target=gast.Name(id=self.enum_idx_name,
                                               ctx=gast.Store(),
                                               annotation=None,
                                               type_comment=None),
                              op=gast.Add(),
                              value=gast.Constant(value=1, kind=None))

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
