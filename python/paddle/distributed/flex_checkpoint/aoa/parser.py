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

from .lexer import Token, TokenType


class Statement:
    def __init__(self, left_vars, right_vars, attrs):
        self.left_vars = left_vars  # List[Var]
        self.right_vars = right_vars  # List[Var]
        self.attrs = attrs  # List[Attribute]

    def __repr__(self):
        return f"Statement({self.left_vars} -> {self.right_vars}, attrs={self.attrs})"


class Var:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class Attribute:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __repr__(self):
        return f"{self.key}={self.value!r}"


class Parser:
    """
    AOA Grammar
    PROGRAM   ::= { STATEMENT }

    STATEMENT ::= VAR_LIST '->' VAR ',' ATTR_LIST       // meige
                | VAR '->' VAR_LIST ',' ATTR_LIST       // split
                | VAR '->' VAR ',' ATTR_LIST            // single variable mapping + attributes
                | VAR '->' VAR                          // single variable mapping, rename

    VAR_LIST  ::= VAR { ',' VAR }
    VAR       ::= IDENTIFIER
    ATTR_LIST ::= ATTRIBUTE { ',' ATTRIBUTE }
    ATTRIBUTE ::= IDENTIFIER '=' VALUE
    VALUE     ::= NUMBER | STRING
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def at_end(self):
        return self.peek().type == TokenType.EOF

    def peek(self, offset=0):
        if self.pos + offset >= len(self.tokens):
            return Token(TokenType.EOF, '')
        return self.tokens[self.pos + offset]

    def consume(self, expected_type=None):
        tok = self.peek()
        if expected_type and tok.type != expected_type:
            raise SyntaxError(
                f'Expected {expected_type}, got {tok.type} at pos {self.pos}'
            )
        self.pos += 1
        return tok

    def expect(self, expected_type):
        return self.consume(expected_type)

    def skip_newlines(self):
        while self.peek().type == TokenType.NEWLINE:
            self.consume()

    def parse_program(self):
        stmts = []
        self.skip_newlines()
        while not self.at_end():
            stmt = self.parse_statement()
            stmts.append(stmt)
            self.skip_newlines()
        return stmts

    def parse_statement(self):
        left_vars = [self.parse_var()]
        while self.peek().type == TokenType.COMMA:
            self.consume(TokenType.COMMA)
            left_vars.append(self.parse_var())
        self.expect(TokenType.RARROW)
        right_vars = [self.parse_var()]
        while self.peek().type == TokenType.COMMA:
            # Lookahead for attribute: IDENT '=' after COMMA means attribute starts
            if (
                self.peek(1).type == TokenType.IDENTIFIER
                and self.peek(2).type == TokenType.EQUAL
            ):
                break
            self.consume(TokenType.COMMA)
            right_vars.append(self.parse_var())
        attrs = []
        if self.peek().type == TokenType.COMMA:
            self.consume(TokenType.COMMA)
            attrs = self.parse_attr_list()
        return Statement(left_vars, right_vars, attrs)

    def parse_var(self):
        name = self.expect(TokenType.IDENTIFIER).value
        return Var(name)

    def parse_attr_list(self):
        attrs = [self.parse_attribute()]
        while self.peek().type == TokenType.COMMA:
            self.consume(TokenType.COMMA)
            attrs.append(self.parse_attribute())
        return attrs

    def parse_attribute(self):
        key = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.EQUAL)
        val_tok = self.consume()
        if val_tok.type == TokenType.NUMBER:
            val = int(val_tok.value)
        elif val_tok.type == TokenType.STRING:
            val = val_tok.value.strip('"').strip("'")
        else:
            raise SyntaxError(f'Unexpected value: {val_tok}')
        return Attribute(key, val)
