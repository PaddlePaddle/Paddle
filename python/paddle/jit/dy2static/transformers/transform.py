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

# gast is a generic AST to represent Python2 and Python3's Abstract Syntax Tree(AST).
# It provides a compatibility layer between the AST of various Python versions,
# as produced by ast.parse from the standard ast module.
# See details in https://github.com/serge-sans-paille/gast/

import os

from paddle.framework import use_pir_api

from .. import logging_utils
from ..utils import ast_to_source_code
from .assert_transformer import AssertTransformer
from .base import BaseTransformer
from .break_continue_transformer import (
    BreakContinueTransformer,
    BreakTransformOptimizer,
)
from .call_transformer import CallTransformer
from .cast_transformer import CastTransformer
from .create_variable_transformer import CreateVariableTransformer
from .decorator_transformer import DecoratorTransformer
from .early_return_transformer import EarlyReturnTransformer
from .ifelse_transformer import IfElseTransformer
from .logical_transformer import LogicalTransformer
from .loop_transformer import LoopTransformer
from .name_load_transformer import (
    AttributeJstTransformer,
    NameloadJstTransformer,
)
from .return_transformer import ReturnTransformer
from .super_transformer import SuperTransformer
from .tensor_shape_transformer import TensorShapeTransformer
from .tensorhook_transformer import RegisterHookTransformer
from .typehint_transformer import TypeHintTransformer

__all__ = []


def apply_optimization(transformers):
    """
    Judge whether to apply optimized transformation, such as BreakTransformOptimizer.
    And not all optimized transformations are applied by default. It's controlled by
    'export FLAGS_optim_transformation=1'
    """
    flag = str(os.environ.get('FLAGS_optim_transformation')) in [
        '1',
        'True',
        'true',
    ]
    if flag:
        transformers.insert(3, BreakTransformOptimizer)


class DygraphToStaticAst(BaseTransformer):
    """
    Main class to transform Dygraph to Static Graph
    """

    def __init__(self):
        self.translator_logger = logging_utils.TranslatorLogger()

    def get_static_ast(self, root):
        self.root = root
        self.decorate_func_name = None

        # inplace transfer
        self.transfer_from_node_type(self.root)
        return self.root

    def _apply(self, transformer, node, log_level):
        transformer(node).transform()
        self.translator_logger.log_transformed_code(
            log_level, self.root, transformer.__name__
        )

    def transfer_from_node_type(self, node):
        self.translator_logger.log(
            1, f"Source code: \n{ast_to_source_code(self.root)}"
        )
        # Generic transformation
        self.visit(node)

        transformers = [
            TypeHintTransformer,  # remove all typehint
            SuperTransformer,  # super() -> super(__class__, <first argument>)
            RegisterHookTransformer,
            EarlyReturnTransformer,
            AttributeJstTransformer,  # Tensor.size -> Tensor.size(), it's unnecessary in PIR mode
            TensorShapeTransformer,  # Tensor.shape -> paddle.shape(Tensor)
            BreakContinueTransformer,  # break/continue in loops
            ReturnTransformer,  # return in functions
            LogicalTransformer,  # logical and/or/not
            CreateVariableTransformer,  # create undefined var for if / while / for
            LoopTransformer,  # for/while -> while_op
            IfElseTransformer,  # if/else -> if_op
            AssertTransformer,  # assert statement
            CallTransformer,  # transform call recursively
            CastTransformer,  # type casting statement
            DecoratorTransformer,  # transform decorators to function call
            NameloadJstTransformer,
        ]

        if use_pir_api():
            # It's unnecessary in PIR mode
            transformers.remove(AttributeJstTransformer)

        apply_optimization(transformers)

        for index, transformer in enumerate(transformers):
            self._apply(transformer, node, log_level=index + 1)

        self.translator_logger.log_transformed_code(
            logging_utils.LOG_AllTransformer, self.root, "All Transformers"
        )

    def visit_FunctionDef(self, node):
        if self.decorate_func_name is None:
            self.decorate_func_name = node.name

        self.generic_visit(node)
        return node

    def get_module_name(self):
        """
        Return the main function name which will be used as module name
        in ast_to_func.
        """
        # Should consider BaseAPITransformer which add new module name in Yamei's PR.
        assert self.decorate_func_name, "decorate_func_name shall not be None."
        return self.decorate_func_name
