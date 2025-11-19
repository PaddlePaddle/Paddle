# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import re
from enum import Enum, auto


class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


class TokenType(Enum):
    IDENTIFIER = auto()
    NUMBER = auto()
    COLON = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    RARROW = auto()
    STRING = auto()
    EQUAL = auto()
    NEWLINE = auto()
    EOF = auto()


class Lexer:
    token_specification = [
        ('RARROW', r'->'),
        ('EQUAL', r'='),
        ('COLON', r':'),
        ('LBRACKET', r'\['),
        ('RBRACKET', r'\]'),
        ('COMMA', r','),
        ('NUMBER', r'\d+'),
        ('STRING', r'"[^"]*"|\'[^\']*\''),
        ('IDENTIFIER', r'[A-Za-z_][A-Za-z\.\$\_\*\d\^T]*'),
        ('SKIP', r'[ \t]+'),
        ('NEWLINE', r'[\r\n]+'),
        ('MISMATCH', r'.'),
    ]

    def __init__(self, context):
        from .macros import macro_registry

        self.macros = [list(d.values())[1] for d in macro_registry.macros]
        self.get_token = re.compile(
            '|'.join(
                f'(?P<{name}>{regex})'
                for name, regex in self.token_specification
            )
        ).match
        self.context = context

    def tokenize(self, text):
        pos = 0
        mo = self.get_token(text, pos)
        tokens = []
        if not text.endswith('\n'):
            text += '\n'
        while mo is not None:
            kind = mo.lastgroup
            value = mo.group()
            if kind == 'SKIP':
                pass
            elif kind == 'MISMATCH':
                raise RuntimeError(
                    f'Unexpected character {value!r} at position {pos}'
                )
            else:
                tokens.append(Token(TokenType[kind], value))
            pos = mo.end()
            mo = self.get_token(text, pos)
        return tokens

    def apply_macro(self, expression, macro):
        if isinstance(expression, str):
            expression = [expression]
        new_expression = []
        for expr in expression:
            results = macro(self.tokenize(expr), expr, self.context)
            if isinstance(results, str):
                new_expression.append(results)
            else:
                new_expression.extend(results)
        return new_expression

    def apply_single_macro_to_all(self, expressions, macro):
        new_expressions = []
        for expr in expressions:
            results = macro(self.tokenize(expr), expr, self.context)
            if isinstance(results, str):
                new_expressions.append(results)
            else:
                new_expressions.extend(results)
        return new_expressions

    def all_tokens(self, expressions):
        current_expressions = expressions
        for macro in self.macros:
            current_expressions = self.apply_single_macro_to_all(
                current_expressions, macro
            )

        tokens = []
        for expr in current_expressions:
            tokens.extend(self.tokenize(expr))
        return tokens
