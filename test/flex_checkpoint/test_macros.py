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

from __future__ import annotations

import re
import unittest
from typing import TYPE_CHECKING

from paddle.distributed.flex_checkpoint.aoa.lexer import Lexer
from paddle.distributed.flex_checkpoint.aoa.macros import macro_registry

if TYPE_CHECKING:
    from collections.abc import Iterable


class MacroContext:
    def __init__(self):
        self.source_keys = {
            "embed_tokens.weight",
            "layers.0.self_attn.qkv_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.mlp.gate_up_fused_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.1.self_attn.qkv_proj.weight",
            "layers.1.self_attn.o_proj.weight",
            "layers.1.mlp.gate_up_fused_proj.weight",
            "layers.1.mlp.down_proj.weight",
            "layers.1.input_layernorm.weight",
            "layers.1.post_attention_layernorm.weight",
            "layers.0.experts.0.weight",
            "layers.0.experts.1.weight",
            "layers.1.experts.0.weight",
            "layers.1.experts.1.weight",
        }

    def get_all_dst_state_keys(self) -> Iterable[str]:
        return self.source_keys

    def get_all_src_state_keys(self) -> Iterable[str]:
        return self.source_keys

    def get_num_hidden_layers(
        self, name_with_layer_id: str, layer_id_macro_tag: str
    ) -> int:
        if layer_id_macro_tag not in name_with_layer_id:
            raise ValueError(
                f"layer_id_macro_tag '{layer_id_macro_tag}' not in name_with_layer_id '{name_with_layer_id}'"
            )
        prefix, suffix = name_with_layer_id.split(layer_id_macro_tag, 1)
        pattern = re.compile(rf"{re.escape(prefix)}(\d+){re.escape(suffix)}")
        max_layer = 0
        for key in self.get_all_dst_state_keys():
            match = pattern.fullmatch(key)
            if match:
                layer_num = int(match.group(1))
                max_layer = max(max_layer, layer_num)
        return max_layer + 1

    def get_src_state_shard_num(self, src_state_key: str) -> int:
        return 2

    def get_dst_state_shard_num(self, dst_state_key: str) -> int:
        return 4


def get_macro(macro_name):
    for macro in macro_registry.macros:
        if macro["name"] == macro_name:
            return macro["func"]
    raise ValueError(f"Macro '{macro_name}' not found.")


class TestMacro(unittest.TestCase):
    def setUp(self):
        self.lexer = Lexer(MacroContext())
        self.macro_func = None
        self.source = None
        self.expected_expanded = None

    def macro_name(self):
        raise NotImplementedError

    def source_code(self):
        raise NotImplementedError

    def expected(self):
        raise NotImplementedError

    def start_macro_test(self):
        self.macro_func = get_macro(self.macro_name())
        self.source = self.source_code()
        self.expected_expanded = self.expected()
        actual_expanded = self.lexer.apply_macro(self.source, self.macro_func)
        self.assertEqual(actual_expanded, self.expected_expanded)


class TestStarMacro(TestMacro):
    def macro_name(self):
        return "star_macro"

    def source_code(self):
        return "layers.1.experts.*.weight -> fused_experts, axis = 1"

    def expected(self):
        return [
            'layers.1.experts.0.weight,layers.1.experts.1.weight->fused_experts,axis=1\n'
        ]

    def test(self):
        self.start_macro_test()


class TestLayerIdMacro(TestMacro):
    def macro_name(self):
        return "layer_id_macro"

    def source_code(self):
        return "layers.$LAYER_ID.experts.0.weight -> test_layer_id, axis = 1"

    def expected(self):
        return [
            'layers.0.experts.0.weight->test_layer_id.layer.0,axis=1\n',
            'layers.1.experts.0.weight->test_layer_id.layer.1,axis=1\n',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedQkvOldMacro(TestMacro):
    def macro_name(self):
        return "fused_qkv_old_macro"

    def source_code(self):
        return "layers.1.self_attn.qkv_proj.weight -> layers.1.self_attn.qkv_proj.weight, fused_qkv_old, num_heads = 8, num_key_value_groups = 4"

    def expected(self):
        return [
            'layers.1.self_attn.qkv_proj.weight -> fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.V_3, axis=1',
            'fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_3 -> layers.1.self_attn.qkv_proj.weight, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedFfnMacro(TestMacro):
    def macro_name(self):
        return "fused_ffn_macro"

    def source_code(self):
        return "layers.1.mlp.gate_up_fused_proj.weight -> layers.1.mlp.gate_up_fused_proj.weight, fused_ffn"

    def expected(self):
        return [
            'layers.1.mlp.gate_up_fused_proj.weight  -> fused_ffn_tmp.GATE_0,fused_ffn_tmp.GATE_1,fused_ffn_tmp.UP_0,fused_ffn_tmp.UP_1,fused_ffn_tmp.GATE_2,fused_ffn_tmp.GATE_3,fused_ffn_tmp.UP_2,fused_ffn_tmp.UP_3, axis=1',
            'fused_ffn_tmp.GATE_0,fused_ffn_tmp.UP_0,fused_ffn_tmp.GATE_1,fused_ffn_tmp.UP_1,fused_ffn_tmp.GATE_2,fused_ffn_tmp.UP_2,fused_ffn_tmp.GATE_3,fused_ffn_tmp.UP_3 -> layers.1.mlp.gate_up_fused_proj.weight, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestTransposeMacro(TestMacro):
    def macro_name(self):
        return "transpose_macro"

    def source_code(self):
        return (
            "layers.1.mlp.down_proj.weight^T -> layers.1.mlp.down_proj.weight_T"
        )

    def expected(self):
        return [
            'layers.1.mlp.down_proj.weight -> layers.1.mlp.down_proj.weight_transpose_tmp, permute = "[]"',
            'layers.1.mlp.down_proj.weight_transpose_tmp->layers.1.mlp.down_proj.weight_T\n',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedQKVMacro(TestMacro):
    def macro_name(self):
        return "fused_qkv"

    def source_code(self):
        return "layers.1.self_attn.qkv_proj.weight -> Q, K, V, fused_qkv, num_heads = 8, num_key_value_groups = 2"

    def expected(self):
        return [
            'layers.1.self_attn.qkv_proj.weight -> Q0,Q1,Q2,Q3,K0,V0,Q4,Q5,Q6,Q7,K1,V1, axis=1',
            'Q0,Q1,Q2,Q3,Q4,Q5,Q6,Q7 -> Q, axis=1',
            'K0,K1 -> K, axis=1',
            'V0,V1 -> V, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedQKVMacro2(TestMacro):
    def macro_name(self):
        return "fused_qkv"

    def source_code(self):
        return "Q, K, V -> layers.1.self_attn.qkv_proj.weight, fused_qkv, num_heads = 8, num_key_value_groups = 8"

    def expected(self):
        return [
            'Q -> Q0,Q1,Q2,Q3,Q4,Q5,Q6,Q7, axis=1',
            'K -> K0,K1,K2,K3,K4,K5,K6,K7, axis=1',
            'V -> V0,V1,V2,V3,V4,V5,V6,V7, axis=1',
            'Q0,K0,V0,Q1,K1,V1,Q2,K2,V2,Q3,K3,V3,Q4,K4,V4,Q5,K5,V5,Q6,K6,V6,Q7,K7,V7 -> layers.1.self_attn.qkv_proj.weight, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedQkvOldMacro2(TestMacro):
    def macro_name(self):
        return "fused_qkv_old_macro"

    def source_code(self):
        return "Q,K,V -> layers.1.self_attn.qkv_proj.weight, fused_qkv_old, num_heads = 8, num_key_value_groups = 4"

    def expected(self):
        return [
            'Q,K,V  ->  Q.K.V.tmp, axis=1',
            'Q.K.V.tmp -> fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.V_3, axis=1',
            'fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_3 -> layers.1.self_attn.qkv_proj.weight, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedQkvOldMacro3(TestMacro):
    def macro_name(self):
        return "fused_qkv_old_macro"

    def source_code(self):
        return "fused_qkv_old_test_name -> q_test_name ,k_test_name, v_test_name, fused_qkv_old, num_heads = 8, num_key_value_groups = 4 "

    def expected(self):
        return [
            'fused_qkv_old_test_name -> fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.V_3, axis=1',
            'fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7 -> q_test_name, axis=1',
            'fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.K_3 -> k_test_name, axis=1',
            'fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.V_3 -> v_test_name, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedQkvOldMacro4(TestMacro):
    def macro_name(self):
        return "fused_qkv_old_macro"

    def source_code(self):
        return "fused_qkv_old_test_name ->  layers.1.self_attn.qkv_proj.weight,fused_qkv_old, num_heads = 8, num_key_value_groups = 8 "

    def expected(self):
        return [
            'fused_qkv_old_test_name -> fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.V_3,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_4,fused_qkv_old_tmp.K_5,fused_qkv_old_tmp.K_6,fused_qkv_old_tmp.K_7,fused_qkv_old_tmp.V_4,fused_qkv_old_tmp.V_5,fused_qkv_old_tmp.V_6,fused_qkv_old_tmp.V_7, axis=1',
            'fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.V_3,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.K_4,fused_qkv_old_tmp.K_5,fused_qkv_old_tmp.V_4,fused_qkv_old_tmp.V_5,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_6,fused_qkv_old_tmp.K_7,fused_qkv_old_tmp.V_6,fused_qkv_old_tmp.V_7 -> layers.1.self_attn.qkv_proj.weight, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedFfnMacro2(TestMacro):
    def macro_name(self):
        return "fused_ffn_macro"

    def source_code(self):
        return "layers.0.mlp.gate_up_fused_proj.weight -> layers.0.mlp.gate_proj.weight,layers.0.mlp.up_proj.weight, fused_ffn "

    def expected(self):
        return [
            'layers.0.mlp.gate_up_fused_proj.weight  -> fused_ffn_tmp.GATE_0,fused_ffn_tmp.UP_0,fused_ffn_tmp.GATE_1,fused_ffn_tmp.UP_1, axis=1',
            'fused_ffn_tmp.GATE_0,fused_ffn_tmp.GATE_1 -> layers.0.mlp.gate_proj.weight, axis=1',
            'fused_ffn_tmp.UP_0,fused_ffn_tmp.UP_1 -> layers.0.mlp.up_proj.weight, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedFfnMacro3(TestMacro):
    def macro_name(self):
        return "fused_ffn_macro"

    def source_code(self):
        return "layers.0.mlp.gate_up_fused_proj.weight -> layers.0.mlp.gate_proj.weight,layers.0.mlp.up_proj.weight, fused_ffn "

    def expected(self):
        return [
            'layers.0.mlp.gate_up_fused_proj.weight  -> fused_ffn_tmp.GATE_0,fused_ffn_tmp.UP_0,fused_ffn_tmp.GATE_1,fused_ffn_tmp.UP_1, axis=1',
            'fused_ffn_tmp.GATE_0,fused_ffn_tmp.GATE_1 -> layers.0.mlp.gate_proj.weight, axis=1',
            'fused_ffn_tmp.UP_0,fused_ffn_tmp.UP_1 -> layers.0.mlp.up_proj.weight, axis=1',
        ]

    def test(self):
        self.start_macro_test()


if __name__ == "__main__":
    unittest.main()
