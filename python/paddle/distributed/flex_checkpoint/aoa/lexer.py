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

import math
import re
from enum import Enum, auto


def macro(name, priority):
    def decorator(func):
        macro_registry.register_macro(name, func, priority)
        return func

    return decorator


class MacroRegistry:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'macros'):
            self.macros = []

    def register_macro(self, name, func, priority):
        if any(macro['name'] == name for macro in self.macros):
            raise ValueError(f"Macro '{name}' is already registered.")
        self.macros.append({'name': name, 'func': func, 'priority': priority})
        self.macros.sort(key=lambda x: x['priority'], reverse=False)


macro_registry = MacroRegistry()


# star_macro must be called after layer_id_macro
@macro(name='star_macro', priority=3)
def star_macro(tokens, expression, context):
    STAR_TAG = "*"
    if STAR_TAG not in expression:
        return expression

    def _sort_keys_by_numeric_part(prefix, suffix, allkeys):
        pattern = re.compile(fr"{re.escape(prefix)}(\d+){re.escape(suffix)}")
        filtered_keys = []
        for key in allkeys:
            match = pattern.match(key)
            if match:
                num = int(match.group(1))
                filtered_keys.append((key, num))
        sorted_keys = sorted(filtered_keys, key=lambda x: x[1])
        return [key for key, _ in sorted_keys]

    pre_rarrow = True
    new_tokens = []
    for token in tokens:
        if token.type == TokenType.RARROW:
            pre_rarrow = False
        if token.type == TokenType.IDENTIFIER and STAR_TAG in token.value:
            prefix, suffix = token.value.split(STAR_TAG)
            allkeys = (
                context.get_all_dst_state_keys()
                if not pre_rarrow
                else context.get_all_dst_state_keys()
            )
            assert (
                len(allkeys) != 0
            ), f"No keys found with prefix {prefix} and suffix {suffix}!"
            keys = list(_sort_keys_by_numeric_part(prefix, suffix, allkeys))
            for key in keys:
                new_tokens.append(Token(TokenType.IDENTIFIER, key))
                if key != keys[-1]:
                    new_tokens.append(Token(TokenType.COMMA, ","))
        else:
            new_tokens.append(token)
    new_expression = "".join([token.value for token in new_tokens]) + "\n"
    return new_expression


@macro(name='layer_id_macro', priority=2)
def layer_id_macro(tokens, expression, context):
    LAYER_ID_MACRO_TAG = "$LAYER_ID"
    if LAYER_ID_MACRO_TAG not in expression:
        return expression

    name_with_layer_id = next(
        (
            token.value
            for token in tokens
            if token.type == TokenType.IDENTIFIER
            and LAYER_ID_MACRO_TAG in token.value
        ),
        None,
    )
    assert name_with_layer_id, "No $LAYER_ID found in NAME tokens"

    num_layers = context.get_num_hidden_layers(
        name_with_layer_id, LAYER_ID_MACRO_TAG
    )
    expanded_expressions = []

    for layer_id in range(num_layers):
        expr = ""
        for token in tokens:
            if token.type == TokenType.IDENTIFIER:
                if LAYER_ID_MACRO_TAG in token.value:
                    expr += token.value.replace(
                        LAYER_ID_MACRO_TAG, str(layer_id)
                    )
                elif token.value != "axis":
                    expr += f"{token.value}.layer.{layer_id}"
                else:
                    expr += token.value
            else:
                expr += token.value
        expanded_expressions.append(expr + "\n")

    return expanded_expressions


@macro(name='array_macro', priority=2)
def array_macro(tokens, expression, context):
    if "[" not in expression:
        return expression
    new_tokens = []
    idx = 0
    while idx < len(tokens):
        if tokens[idx].type == TokenType.LBRACKET:
            name = tokens[idx - 1].value
            assert (
                tokens[idx + 1].type == TokenType.NUMBER
                and tokens[idx + 2].type == TokenType.COLON
                and tokens[idx + 3].type == TokenType.NUMBER
                and tokens[idx + 4].type == TokenType.RBRACKET
            )
            new_tokens.pop()
            start = int(tokens[idx + 1].value)
            end = int(tokens[idx + 3].value)
            for i in range(start, end):
                new_tokens.append(
                    Token(TokenType.IDENTIFIER, name + "_" + str(i))
                )
                if i != end - 1:
                    new_tokens.append(Token(TokenType.COMMA, ","))
            idx += 5
        else:
            new_tokens.append(tokens[idx])
            idx += 1
    new_expression = "".join([token.value for token in new_tokens])
    new_expression += "\n"
    return new_expression


@macro(name='fused_qkv_macro', priority=1)
def fused_qkv_macro(tokens, expression, context):
    FUSED_QKV_TAG = "fused_qkv"
    if FUSED_QKV_TAG not in expression:
        return expression

    attn_head_num = None
    num_key_value_groups = None
    fused_qkv_pos = None
    rarrow_pos = None
    right_var_end_pos = None

    for idx, token in enumerate(tokens):
        if token.type == TokenType.IDENTIFIER:
            if token.value == "num_heads" and idx + 2 < len(tokens):
                attn_head_num = int(tokens[idx + 2].value)
            elif token.value == "num_key_value_groups" and idx + 2 < len(
                tokens
            ):
                num_key_value_groups = int(tokens[idx + 2].value)
            elif token.value == FUSED_QKV_TAG:
                fused_qkv_pos = idx
        elif token.type == TokenType.RARROW and rarrow_pos is None:
            rarrow_pos = idx
        if (
            right_var_end_pos is None
            and token.type == TokenType.IDENTIFIER
            and token.value
            in {FUSED_QKV_TAG, "num_heads", "num_key_value_groups"}
        ):
            right_var_end_pos = idx + 1

    assert attn_head_num and attn_head_num > 0, "num_heads must be positive."
    assert (
        num_key_value_groups and num_key_value_groups > 0
    ), "num_key_value_groups must be positive."
    assert fused_qkv_pos is not None, "No fused_qkv tag found in expression."
    assert rarrow_pos is not None, "No -> found in expression."
    assert (
        attn_head_num % num_key_value_groups == 0
    ), "num_heads must be divisible by num_key_value_groups."

    num_key_value_heads = attn_head_num // num_key_value_groups

    src_qkv_weight_name = tokens[0].value
    if fused_qkv_pos > 4:
        dst_qkv_weight_name = (
            "".join(
                token.value if token.type == TokenType.IDENTIFIER else "_"
                for token in tokens[rarrow_pos + 1 : right_var_end_pos]
            )
            + ".fused_qkv_tmp"
        )
    else:
        dst_qkv_weight_name = tokens[0].value

    src_state_shard_num = context.get_src_state_shard_num(src_qkv_weight_name)
    dst_state_shard_num = (
        context.get_dst_state_shard_num(dst_qkv_weight_name)
        if fused_qkv_pos == 4
        else 1
    )

    configs = [
        (src_state_shard_num, src_qkv_weight_name),
        (dst_state_shard_num, dst_qkv_weight_name),
    ]

    head_config = [
        ("Q", attn_head_num),
        ("K", num_key_value_heads),
        ("V", num_key_value_heads),
    ]

    def gen_expr(tp_degree, num_heads, tp_rank, comp):
        start = tp_rank * num_heads // tp_degree
        count = num_heads // tp_degree
        return ",".join(
            f"fused_qkv_tmp.{comp}_{i}" for i in range(start, start + count)
        )

    results = []
    for idx, (tp_degree, qkv_weight_name) in enumerate(configs):
        qkv_parts = [
            gen_expr(tp_degree, n, tp_rank, c)
            for tp_rank in range(tp_degree)
            for c, n in head_config
        ]
        if idx == 0:
            mapping = f"{qkv_weight_name} -> {','.join(qkv_parts)}, axis=1\n"
        else:
            mapping = f"{','.join(qkv_parts)} -> {qkv_weight_name}, axis=1\n"
        results.append(mapping)

    if fused_qkv_pos > 4:
        final_expr = (
            f"{dst_qkv_weight_name}->"
            + "".join(
                token.value
                for token in tokens[rarrow_pos + 1 : right_var_end_pos]
            )
            + ", axis=1\n"
        )
        results.append(final_expr)

    return results


@macro(name='fused_ffn_macro', priority=1)
def fused_ffn_macro(tokens, expression, context):
    FUSED_FFN_TAG = "fused_ffn"
    if FUSED_FFN_TAG not in expression:
        return expression
    assert (
        len(tokens) == 5 and tokens[4].value == FUSED_FFN_TAG
    ), "Invalid tokens for FUSED_FFN operation ！"
    src_ffn_weight_name = tokens[2].value
    dst_ffn_weight_name = tokens[0].value
    src_state_shard_num = context.get_src_state_shard_num(src_ffn_weight_name)
    dst_state_shard_num = context.get_dst_state_shard_num(dst_ffn_weight_name)
    splited_num = math.lcm(src_state_shard_num, dst_state_shard_num)

    configs = [
        (src_state_shard_num, src_ffn_weight_name),
        (dst_state_shard_num, dst_ffn_weight_name),
    ]

    split_config = [("GATE", splited_num), ("UP", splited_num)]

    def gen_expr(tp_degree, splited_num, tp_rank, comp):
        return ",".join(
            f"fused_ffn_tmp.{comp}_{tp_rank * splited_num // tp_degree + idx}"
            for idx in range(splited_num // tp_degree)
        )

    results = []
    for idx, (tp_degree, ffn_weight_name) in enumerate(configs):
        ffn_parts = [
            gen_expr(tp_degree, n, tp_rank, c)
            for tp_rank in range(tp_degree)
            for c, n in split_config
        ]
        if idx == 0:
            results.append(
                f"{ffn_weight_name}  -> {','.join(ffn_parts)}, axis=1 \n"
            )
        else:
            results.append(
                f"{','.join(ffn_parts)} -> {ffn_weight_name}, axis=1 \n"
            )
    return results


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
        ('IDENTIFIER', r'[A-Za-z][A-Za-z\.\$\_\*\d]*'),
        ('SKIP', r'[ \t]+'),
        ('NEWLINE', r'[\r\n]+'),
        ('MISMATCH', r'.'),
    ]

    def __init__(self, context):
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

    def apply_macros(self, expression):
        expressions = [expression]
        for macro in self.macros:
            expressions = self.apply_macro(expressions, macro)
        return expressions

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

    def all_tokens(self, expressions):
        tokens = []
        for expr in expressions:
            expanded_expressions = self.apply_macros(expr)
            for e in expanded_expressions:
                tokens.extend(self.tokenize(e))
        return tokens
