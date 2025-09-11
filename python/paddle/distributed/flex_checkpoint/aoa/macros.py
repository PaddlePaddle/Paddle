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

from .lexer import Token, TokenType


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

GLOBAL_ATTRIBUTE_KEYWORDS = [
    "axis",
    'fused_ffn',
    'fused_qkv_old',
    'num_heads',
    'num_key_value_groups',
    'permute',
]


# star_macro must be called after layer_id_macro
@macro(name='star_macro', priority=3)
def star_macro(tokens, expression, context):
    STAR_TAG = "*"
    if STAR_TAG not in expression:
        return expression

    def _sort_keys_by_numeric_part(prefix, suffix, allkeys):
        pattern = re.compile(rf"{re.escape(prefix)}(\d+){re.escape(suffix)}")
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
            assert len(allkeys) != 0, (
                f"No keys found with prefix {prefix} and suffix {suffix}!"
            )
            keys = list(_sort_keys_by_numeric_part(prefix, suffix, allkeys))
            for key in keys:
                new_tokens.append(Token(TokenType.IDENTIFIER, key))
                if key != keys[-1]:
                    new_tokens.append(Token(TokenType.COMMA, ","))
        else:
            new_tokens.append(token)
    new_expression = "".join([token.value for token in new_tokens])
    return new_expression


@macro(name='layer_id_macro', priority=1)
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
                elif token.value not in GLOBAL_ATTRIBUTE_KEYWORDS:
                    expr += f"{token.value}.layer.{layer_id}"
                else:
                    expr += token.value
            else:
                expr += token.value
        expanded_expressions.append(expr)

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
    return new_expression


@macro(name='fused_qkv_old_macro', priority=4)
def fused_qkv_old_macro(tokens, expression, context):
    FUSED_QKV_OLD_TAG = "fused_qkv_old"
    if not any(tkn.value == FUSED_QKV_OLD_TAG for tkn in tokens):
        return expression

    attn_head_num = None
    num_key_value_groups = None
    fused_qkv_old_pos = None
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
            elif token.value == FUSED_QKV_OLD_TAG:
                fused_qkv_old_pos = idx
        elif token.type == TokenType.RARROW and rarrow_pos is None:
            rarrow_pos = idx
        if (
            right_var_end_pos is None
            and token.type == TokenType.IDENTIFIER
            and token.value
            in {FUSED_QKV_OLD_TAG, "num_heads", "num_key_value_groups"}
        ):
            right_var_end_pos = idx + 1

    assert attn_head_num and attn_head_num > 0, "num_heads must be positive."
    assert num_key_value_groups and num_key_value_groups > 0, (
        "num_key_value_groups must be positive."
    )
    assert fused_qkv_old_pos is not None, (
        "No fused_qkv_old tag found in expression."
    )
    assert rarrow_pos is not None, "No -> found in expression."
    assert attn_head_num % num_key_value_groups == 0, (
        "num_heads must be divisible by num_key_value_groups."
    )

    results = []
    num_key_value_heads = num_key_value_groups
    if rarrow_pos == 1:
        src_qkv_weight_name = tokens[0].value
        if fused_qkv_old_pos > 4:
            dst_qkv_weight_name = None
        else:
            dst_qkv_weight_name = tokens[2].value

        src_state_shard_num = context.get_src_state_shard_num(
            src_qkv_weight_name
        )
        dst_state_shard_num = (
            context.get_dst_state_shard_num(dst_qkv_weight_name)
            if dst_qkv_weight_name is not None
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
                f"fused_qkv_old_tmp.{comp}_{i}"
                for i in range(start, start + count)
            )

        for idx, (tp_degree, qkv_weight_name) in enumerate(configs):
            qkv_parts = [
                gen_expr(tp_degree, n, tp_rank, c)
                for tp_rank in range(tp_degree)
                for c, n in head_config
            ]
            if idx == 0:
                mapping = f"{qkv_weight_name} -> {','.join(qkv_parts)}, axis=1"
                results.append(mapping)
            elif qkv_weight_name is not None:
                mapping = f"{','.join(qkv_parts)} -> {qkv_weight_name}, axis=1"
                results.append(mapping)

        if fused_qkv_old_pos > 4:

            def _generate_expr(prefix, count, target_name):
                elements = ",".join(
                    f"fused_qkv_old_tmp.{prefix}_{i}" for i in range(count)
                )
                return f"{elements} -> {target_name}, axis=1"

            q_name = tokens[2].value
            k_name = tokens[4].value
            v_name = tokens[6].value

            results.append(_generate_expr("Q", attn_head_num, q_name))
            results.append(_generate_expr("K", num_key_value_heads, k_name))
            results.append(_generate_expr("V", num_key_value_heads, v_name))
    elif rarrow_pos == 5:
        q_name = tokens[0].value
        k_name = tokens[2].value
        v_name = tokens[4].value
        dst_qkv_weight_name = tokens[6].value

        fused_qkv_tmp_name = f"{q_name}.{k_name}.{v_name}.tmp"
        results.append(
            f"{q_name},{k_name},{v_name}  ->  {fused_qkv_tmp_name}, axis=1"
        )
        dst_state_shard_num = context.get_dst_state_shard_num(
            dst_qkv_weight_name
        )

        configs = [
            (1, fused_qkv_tmp_name),
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
                f"fused_qkv_old_tmp.{comp}_{i}"
                for i in range(start, start + count)
            )

        for idx, (tp_degree, qkv_weight_name) in enumerate(configs):
            qkv_parts = [
                gen_expr(tp_degree, n, tp_rank, c)
                for tp_rank in range(tp_degree)
                for c, n in head_config
            ]
            if idx == 0:
                mapping = f"{qkv_weight_name} -> {','.join(qkv_parts)}, axis=1"
            else:
                mapping = f"{','.join(qkv_parts)} -> {qkv_weight_name}, axis=1"
            results.append(mapping)
    else:
        raise ValueError(
            f"Unsupported fused_qkv_old macro format: {expression}."
        )
    return results


@macro(name='fused_ffn_macro', priority=4)
def fused_ffn_macro(tokens, expression, context):
    FUSED_FFN_TAG = "fused_ffn"
    if not any(tkn.value == FUSED_FFN_TAG for tkn in tokens):
        return expression
    rarrow_pos = None
    fused_ffn_pos = None
    for idx, token in enumerate(tokens):
        if token.type == TokenType.RARROW and rarrow_pos is None:
            rarrow_pos = idx
        elif (
            token.type == TokenType.IDENTIFIER and token.value == FUSED_FFN_TAG
        ):
            fused_ffn_pos = idx
    assert rarrow_pos is not None, "No -> found in expression."
    assert fused_ffn_pos is not None, "No fused_ffn tag found in expression."
    results = []
    if rarrow_pos == 1:
        src_ffn_weight_name = tokens[0].value
        if fused_ffn_pos == 4:
            dst_ffn_weight_name = tokens[2].value
        else:
            dst_ffn_weight_name = None
        src_state_shard_num = context.get_src_state_shard_num(
            src_ffn_weight_name
        )
        dst_state_shard_num = (
            context.get_dst_state_shard_num(dst_ffn_weight_name)
            if dst_ffn_weight_name is not None
            else 1
        )
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

        for idx, (tp_degree, ffn_weight_name) in enumerate(configs):
            ffn_parts = [
                gen_expr(tp_degree, n, tp_rank, c)
                for tp_rank in range(tp_degree)
                for c, n in split_config
            ]
            if idx == 0:
                results.append(
                    f"{ffn_weight_name}  -> {','.join(ffn_parts)}, axis=1"
                )
            elif ffn_weight_name is not None:
                results.append(
                    f"{','.join(ffn_parts)} -> {ffn_weight_name}, axis=1"
                )
        if fused_ffn_pos > 4:

            def _generate_expr(prefix, count, target_name):
                elements = ",".join(
                    f"fused_ffn_tmp.{prefix}_{i}" for i in range(count)
                )
                return f"{elements} -> {target_name}, axis=1"

            gate_name = tokens[2].value
            up_name = tokens[4].value

            results.append(_generate_expr("GATE", splited_num, gate_name))
            results.append(_generate_expr("UP", splited_num, up_name))

    elif rarrow_pos == 3:
        gate_name = tokens[0].value
        up_name = tokens[2].value
        dst_ffn_weight_name = tokens[4].value

        fused_gate_up_tmp_name = f"{gate_name}.{up_name}.tmp"
        results.append(
            f"{gate_name},{up_name}  ->  {fused_gate_up_tmp_name}, axis=1"
        )
        dst_state_shard_num = context.get_dst_state_shard_num(
            dst_ffn_weight_name
        )

        configs = [
            (1, fused_gate_up_tmp_name),
            (dst_state_shard_num, dst_ffn_weight_name),
        ]

        split_config = [
            ("GATE", dst_state_shard_num),
            ("UP", dst_state_shard_num),
        ]

        def gen_expr(tp_degree, splited_num, tp_rank, comp):
            return ",".join(
                f"fused_ffn_tmp.{comp}_{tp_rank * splited_num // tp_degree + idx}"
                for idx in range(splited_num // tp_degree)
            )

        for idx, (tp_degree, ffn_weight_name) in enumerate(configs):
            ffn_parts = [
                gen_expr(tp_degree, n, tp_rank, c)
                for tp_rank in range(tp_degree)
                for c, n in split_config
            ]
            if idx == 0:
                results.append(
                    f"{ffn_weight_name}  -> {','.join(ffn_parts)}, axis=1"
                )
            else:
                results.append(
                    f"{','.join(ffn_parts)} -> {ffn_weight_name}, axis=1"
                )
    else:
        raise ValueError(f"Unsupported fused_ffn macro format: {expression}.")
    return results


@macro(name='transpose_macro', priority=5)
def transpose_macro(tokens, expression, context):
    TRANSPOSE_TAG = "^T"

    if TRANSPOSE_TAG not in expression:
        return expression

    transpose_vars = set()
    new_expression = ""
    rarrow_pos = None

    for idx, token in enumerate(tokens):
        if token.type == TokenType.RARROW:
            rarrow_pos = idx
            break

    assert rarrow_pos is not None, "No -> found in expression."

    for token in tokens[rarrow_pos + 1 :]:
        if token.type == TokenType.IDENTIFIER and token.value.endswith(
            TRANSPOSE_TAG
        ):
            raise ValueError(
                "Cannot assign to transpose (e.g., 'A -> B^T').\n"
                "B^T is not a real variable, just a view.\n"
                "Assign first:  A -> B\n"
                "Then transpose: B^T -> B"
            )
    for token in tokens:
        if token.type == TokenType.IDENTIFIER and token.value.endswith(
            TRANSPOSE_TAG
        ):
            var_name = token.value[: -len(TRANSPOSE_TAG)]
            transpose_vars.add(var_name)
            new_expression += var_name + "_transpose_tmp"
        else:
            new_expression += token.value

    results = [
        f'{var} -> {var}_transpose_tmp, permute = "[]"'
        for var in transpose_vars
    ]
    results.append(new_expression)
    return results


@macro(name='fused_qkv', priority=4)
def fused_qkv(tokens, expression, context):
    FUSED_QKV_TAG = "fused_qkv"
    if not any(tkn.value == FUSED_QKV_TAG for tkn in tokens):
        return expression

    attn_head_num = num_heads = None
    num_key_value_groups = None
    fused_qkv_pos = None
    rarrow_pos = None

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

    assert attn_head_num and attn_head_num > 0, (
        f"num_heads must be positive (got: {attn_head_num})"
    )
    assert num_key_value_groups and num_key_value_groups > 0, (
        f"num_key_value_groups must be positive (got: {num_key_value_groups})"
    )
    assert fused_qkv_pos is not None, "No fused_qkv tag found in expression."
    assert rarrow_pos is not None, "No -> found in expression."
    assert rarrow_pos == 1 or rarrow_pos == 5, (
        "Only support q,k,v -> fused_qkv or fused_qkv -> q,k,v patterns"
    )
    assert attn_head_num % num_key_value_groups == 0, (
        f"num_heads ({attn_head_num}) must be divisible by num_key_value_groups ({num_key_value_groups})."
    )

    num_key_value_heads = attn_head_num // num_key_value_groups

    def make_names(base, n):
        return [f"{base}{i}" for i in range(n)]

    results = []

    if rarrow_pos == 1:
        fused_qkv_var = tokens[0].value
        q_var = tokens[rarrow_pos + 1].value
        k_var = tokens[rarrow_pos + 3].value
        v_var = tokens[rarrow_pos + 5].value

        q_names = make_names(q_var, attn_head_num)
        k_names = make_names(k_var, num_key_value_groups)
        v_names = make_names(v_var, num_key_value_groups)

        fused_qkv_order = []
        for g in range(num_key_value_groups):
            fused_qkv_order.extend(
                q_names[g * num_key_value_heads : (g + 1) * num_key_value_heads]
            )
            fused_qkv_order.append(k_names[g])
            fused_qkv_order.append(v_names[g])
        results.append(
            f"{fused_qkv_var} -> {','.join(fused_qkv_order)}, axis=1"
        )

        results.append(f"{','.join(q_names)} -> {q_var}, axis=1")
        results.append(f"{','.join(k_names)} -> {k_var}, axis=1")
        results.append(f"{','.join(v_names)} -> {v_var}, axis=1")

        return results

    elif rarrow_pos == 5:
        q_var = tokens[0].value
        k_var = tokens[2].value
        v_var = tokens[4].value
        fused_qkv_var = tokens[rarrow_pos + 1].value

        q_names = make_names(q_var, attn_head_num)
        k_names = make_names(k_var, num_key_value_groups)
        v_names = make_names(v_var, num_key_value_groups)

        results.append(f"{q_var} -> {','.join(q_names)}, axis=1")
        results.append(f"{k_var} -> {','.join(k_names)}, axis=1")
        results.append(f"{v_var} -> {','.join(v_names)}, axis=1")

        fused_qkv_order = []
        for g in range(num_key_value_groups):
            fused_qkv_order.extend(
                q_names[g * num_key_value_heads : (g + 1) * num_key_value_heads]
            )
            fused_qkv_order.append(k_names[g])
            fused_qkv_order.append(v_names[g])
        results.append(
            f"{','.join(fused_qkv_order)} -> {fused_qkv_var}, axis=1"
        )
        return results

    else:
        return expression
