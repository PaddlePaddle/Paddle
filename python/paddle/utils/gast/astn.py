# Copyright (c) 2016, Serge Guelton

import ast
from . import gast


def _generate_translators(to):

    class Translator(ast.NodeTransformer):

        def _visit(self, node):
            if isinstance(node, ast.AST):
                return self.visit(node)
            elif isinstance(node, list):
                return [self._visit(n) for n in node]
            else:
                return node

        def generic_visit(self, node):
            class_name = type(node).__name__
            if not hasattr(to, class_name):
                # handle nodes that are not part of the AST
                return
            cls = getattr(to, class_name)
            new_node = cls(
                **{
                    field: self._visit(getattr(node, field))
                    for field in node._fields
                    if hasattr(node, field)
                }
            )

            for attr in node._attributes:
                try:
                    setattr(new_node, attr, getattr(node, attr))
                except AttributeError:
                    pass
            return new_node

    return Translator


AstToGAst = _generate_translators(gast)

GAstToAst = _generate_translators(ast)
