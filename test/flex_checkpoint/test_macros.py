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

import unittest

from paddle.distributed.flex_checkpoint.aoa.aoa_engine import (
    AOAShardInfoContext,
)
from paddle.distributed.flex_checkpoint.aoa.lexer import Lexer
from paddle.distributed.flex_checkpoint.aoa.macros import macro_registry
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    ShardedWeightDesc,
)


class MacroContext:
    def __init__(self):
        self.source_keys = {
            "embed_tokens.weight",
            "layers.1.mlp.gate_up_fused_proj.weight",
            "layers.1.post_attention_layernorm.weight",
            "layers.2.self_attn.qkv_proj.weight",
            "layers.2.self_attn.o_proj.weight",
            "layers.2.mlp.gate_up_fused_proj.weight",
            "layers.2.mlp.down_proj.weight",
            "layers.2.input_layernorm.weight",
            "layers.1.mlp.gate_up_fused_proj.weight_test1",
            "layers.2.post_attention_layernorm.weight",
            "layers.1.experts.0.weight",
            "layers.0.qkv_proj.weight",
            "fused_qkv_old_test_name",
            "layers.shared.qkv_proj.weight",
            "layers.5.experts.0.up_gate_proj.weight",
            "layers.5.experts.1.up_gate_proj.weight",
            "layers.2.experts.0.weight",
            "layers.2.experts.1.weight",
            "layers.2.self_attn.qkv_proj.bias",
            "layers.2.mlp.gate_up_fused_proj.bias",
            "layers.3.experts.0.up_gate_proj.weight",
            "layers.3.experts.1.up_gate_proj.weight",
        }

        self.dst_keys = {
            "embed_tokens.weight",
            "layers.0.self_attn.qkv_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.mlp.gate_up_fused_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.1.mlp.gate_up_fused_proj.weight",
            "layers.1.mlp.gate_up_fused_proj.weight_test2",
            "layers.1.post_attention_layernorm.weight",
            "layers.0.experts.0.weight",
            "layers.0.experts.1.weight",
            "layers.1.experts.0.weight",
            "layers.0.q_proj.weight",
            "layers.0.k_proj.weight",
            "layers.0.v_proj.weight",
            "q_test_name",
            "k_test_name",
            "v_test_name",
            "layers.0.shared.q_proj.weight",
            "layers.0.shared.k_proj.weight",
            "layers.0.shared.v_proj.weight",
            "layers.1.shared.q_proj.weight",
            "layers.1.shared.k_proj.weight",
            "layers.1.shared.v_proj.weight",
            "layers.5.experts.0.gate_proj.weight",
            "layers.5.experts.1.gate_proj.weight",
            "layers.5.experts.0.up_proj.weight",
            "layers.5.experts.1.up_proj.weight",
            "layers.2.self_attn.qkv_proj.weight",
            "layers.2.self_attn.qkv_proj.bias",
            "layers.2.mlp.gate_up_fused_proj.bias",
            "layers.2.mlp.gate_up_fused_proj.weight",
            "layers.3.experts.0.up_gate_proj.weight",
            "layers.3.experts.1.up_gate_proj.weight",
        }

        # Build _ShardInfo mapping for AOAShardInfoContext based on existing keys
        def make_shard_info(keys: set[str], num_shards: int):
            shard_info: dict[str, list[ShardedWeightDesc]] = {}
            for k in keys:
                descs: list[ShardedWeightDesc] = []
                for i in range(num_shards):
                    descs.append(
                        ShardedWeightDesc(
                            key=k,
                            local_shape=(1,),
                            global_shape=(num_shards,),
                            global_offset=(i,),
                        )
                    )
                shard_info[k] = descs
            return shard_info

        self.source_state_shard_info = make_shard_info(self.source_keys, 2)
        self.destination_state_shard_info = make_shard_info(self.dst_keys, 4)

        self._ctx = AOAShardInfoContext(
            source_state_shard_info=self.source_state_shard_info,
            destination_state_shard_info=self.destination_state_shard_info,
        )

    def set_aoa_config_reverse(
        self,
    ):  # when aoa_config_reverse is True, the src and dst of AOAShardInfoContext are reversed
        self._ctx = AOAShardInfoContext(
            source_state_shard_info=self.destination_state_shard_info,
            destination_state_shard_info=self.source_state_shard_info,
        )
        self._ctx.aoa_config_reverse = True


def get_macro(macro_name):
    for macro in macro_registry.macros:
        if macro["name"] == macro_name:
            return macro["func"]
    raise ValueError(f"Macro '{macro_name}' not found.")


class TestMacro(unittest.TestCase):
    def setUp(self):
        self.macro_func = None
        self.source = None
        self.expected_expanded = None

    def macro_name(self):
        raise NotImplementedError

    def source_code(self):
        raise NotImplementedError

    def expected(self):
        raise NotImplementedError

    def start_macro_test(self, aoa_config_reverse: bool = False):
        self.macro_func = get_macro(self.macro_name())
        self.source = self.source_code()
        self.expected_expanded = self.expected()
        self.ctx = MacroContext()
        if aoa_config_reverse:
            self.ctx.set_aoa_config_reverse()
            self.lexer = Lexer(self.ctx._ctx)
            self.lexer.apply_macro(
                self.source, get_macro("get_var_mapping_chain_macro")
            )
        else:
            self.lexer = Lexer(self.ctx._ctx)
        actual_expanded = self.lexer.apply_macro(self.source, self.macro_func)
        self.assertEqual(actual_expanded, self.expected_expanded)


class TestStarMacro(TestMacro):
    def macro_name(self):
        return "star_macro"

    def source_code(self):
        return "layers.2.experts.*.weight -> fused_experts, axis = 1"

    def expected(self):
        return [
            'layers.2.experts.0.weight,layers.2.experts.1.weight->fused_experts,axis=1\n'
        ]

    def test(self):
        self.start_macro_test()


class TestLayerIdMacro(TestMacro):
    def macro_name(self):
        return "id_macro"

    def source_code(self):
        return "layers.$LAYER_ID.qkv_proj.weight->layers.$LAYER_ID.q_proj.weight,layer.$LAYER_ID.k_proj.weight,layer.$LAYER_ID.v_proj.weight\n"

    def expected(self):
        return [
            'layers.0.qkv_proj.weight->layers.0.q_proj.weight,layer.0.k_proj.weight,layer.0.v_proj.weight\n',
        ]

    def test(self):
        self.start_macro_test()


class Test_expert_id_Macro(TestMacro):
    def macro_name(self):
        return "id_macro"

    def source_code(self):
        return "layers.5.experts.$EXPERT_ID.up_gate_proj.weight -> layers.5.experts.$EXPERT_ID.gate_proj.weight, layers.5.experts.$EXPERT_ID.up_proj.weight"

    def expected(self):
        return [
            'layers.5.experts.0.up_gate_proj.weight->layers.5.experts.0.gate_proj.weight,layers.5.experts.0.up_proj.weight\n',
            'layers.5.experts.1.up_gate_proj.weight->layers.5.experts.1.gate_proj.weight,layers.5.experts.1.up_proj.weight\n',
        ]

    def test(self):
        self.start_macro_test()


class Test_ID_macro_reverse(TestMacro):
    def macro_name(self):
        return "id_macro"

    def source_code(self):
        return "layers.5.experts.$EXPERT_ID.up_gate_proj.weight -> layers.5.experts.$EXPERT_ID.gate_proj.weight, layers.5.experts.$EXPERT_ID.up_proj.weight"

    def expected(self):
        return [
            'layers.5.experts.0.up_gate_proj.weight->layers.5.experts.0.gate_proj.weight,layers.5.experts.0.up_proj.weight\n',
            'layers.5.experts.1.up_gate_proj.weight->layers.5.experts.1.gate_proj.weight,layers.5.experts.1.up_proj.weight\n',
        ]

    def test(self):
        self.start_macro_test(aoa_config_reverse=True)


class TestFusedQkvOldMacro(TestMacro):
    def macro_name(self):
        return "fused_qkv_old_macro"

    def source_code(self):
        return "layers.2.self_attn.qkv_proj.weight -> layers.2.self_attn.qkv_proj.weight, fused_qkv_old, num_heads = 8, num_key_value_groups = 4"

    def expected(self):
        return [
            'layers.2.self_attn.qkv_proj.weight -> fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.V_3, axis=1',
            'fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_3 -> layers.2.self_attn.qkv_proj.weight, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestTransposeMacro(TestMacro):
    def macro_name(self):
        return "transpose_macro"

    def source_code(self):
        return (
            "layers.2.mlp.down_proj.weight^T -> layers.2.mlp.down_proj.weight_T"
        )

    def expected(self):
        return [
            'layers.2.mlp.down_proj.weight -> layers.2.mlp.down_proj.weight_transpose_tmp, permute = "[]"',
            'layers.2.mlp.down_proj.weight_transpose_tmp->layers.2.mlp.down_proj.weight_T\n',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedQKVMacro(TestMacro):
    def macro_name(self):
        return "fused_qkv_macro"

    def source_code(self):
        return "layers.2.self_attn.qkv_proj.weight -> Q, K, V, fused_qkv, num_heads = 8, num_key_value_groups = 2"

    def expected(self):
        return [
            'layers.2.self_attn.qkv_proj.weight -> Q0,Q1,Q2,Q3,K0,V0,Q4,Q5,Q6,Q7,K1,V1, axis=1',
            'Q0,Q1,Q2,Q3,Q4,Q5,Q6,Q7 -> Q, axis=1',
            'K0,K1 -> K, axis=1',
            'V0,V1 -> V, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedQKVMacro2(TestMacro):
    def macro_name(self):
        return "fused_qkv_macro"

    def source_code(self):
        return "Q, K, V -> layers.2.self_attn.qkv_proj.weight, fused_qkv, num_heads = 8, num_key_value_groups = 8"

    def expected(self):
        return [
            'Q -> Q0,Q1,Q2,Q3,Q4,Q5,Q6,Q7, axis=1',
            'K -> K0,K1,K2,K3,K4,K5,K6,K7, axis=1',
            'V -> V0,V1,V2,V3,V4,V5,V6,V7, axis=1',
            'Q0,K0,V0,Q1,K1,V1,Q2,K2,V2,Q3,K3,V3,Q4,K4,V4,Q5,K5,V5,Q6,K6,V6,Q7,K7,V7 -> layers.2.self_attn.qkv_proj.weight, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedQkvOldMacro2(TestMacro):
    def macro_name(self):
        return "fused_qkv_old_macro"

    def source_code(self):
        return "Q,K,V -> layers.2.self_attn.qkv_proj.weight, fused_qkv_old, num_heads = 8, num_key_value_groups = 4"

    def expected(self):
        return [
            'Q,K,V  ->  Q.K.V.tmp, axis=1',
            'Q.K.V.tmp -> fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.V_3, axis=1',
            'fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_3 -> layers.2.self_attn.qkv_proj.weight, axis=1',
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
        return "fused_qkv_old_test_name ->  layers.2.self_attn.qkv_proj.weight,fused_qkv_old, num_heads = 8, num_key_value_groups = 8 "

    def expected(self):
        return [
            'fused_qkv_old_test_name -> fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.V_3,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_4,fused_qkv_old_tmp.K_5,fused_qkv_old_tmp.K_6,fused_qkv_old_tmp.K_7,fused_qkv_old_tmp.V_4,fused_qkv_old_tmp.V_5,fused_qkv_old_tmp.V_6,fused_qkv_old_tmp.V_7, axis=1',
            'fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.V_3,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.K_4,fused_qkv_old_tmp.K_5,fused_qkv_old_tmp.V_4,fused_qkv_old_tmp.V_5,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_6,fused_qkv_old_tmp.K_7,fused_qkv_old_tmp.V_6,fused_qkv_old_tmp.V_7 -> layers.2.self_attn.qkv_proj.weight, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedQkvOldMacro5(TestMacro):
    def macro_name(self):
        return "fused_qkv_old_macro"

    def source_code(self):
        return "layers.2.self_attn.qkv_proj.bias -> layers.2.self_attn.qkv_proj.bias, fused_qkv_old, num_heads = 8, num_key_value_groups = 4, axis = 0"

    def expected(self):
        return [
            'layers.2.self_attn.qkv_proj.bias -> fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.V_3, axis=0',
            'fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_3 -> layers.2.self_attn.qkv_proj.bias, axis=0',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedQkvOldMacro6(TestMacro):
    def macro_name(self):
        return "fused_qkv_old_macro"

    def source_code(self):
        return [
            "fused_qkv_old_test_name ->  A_TEST_NAME,fused_qkv_old, num_heads = 8, num_key_value_groups = 8 ",
            "A_TEST_NAME ->  layers.2.self_attn.qkv_proj.weight",
        ]

    def expected(self):
        return [
            'fused_qkv_old_test_name -> fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.V_3,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_4,fused_qkv_old_tmp.K_5,fused_qkv_old_tmp.K_6,fused_qkv_old_tmp.K_7,fused_qkv_old_tmp.V_4,fused_qkv_old_tmp.V_5,fused_qkv_old_tmp.V_6,fused_qkv_old_tmp.V_7, axis=1',
            'fused_qkv_old_tmp.Q_0,fused_qkv_old_tmp.Q_1,fused_qkv_old_tmp.K_0,fused_qkv_old_tmp.K_1,fused_qkv_old_tmp.V_0,fused_qkv_old_tmp.V_1,fused_qkv_old_tmp.Q_2,fused_qkv_old_tmp.Q_3,fused_qkv_old_tmp.K_2,fused_qkv_old_tmp.K_3,fused_qkv_old_tmp.V_2,fused_qkv_old_tmp.V_3,fused_qkv_old_tmp.Q_4,fused_qkv_old_tmp.Q_5,fused_qkv_old_tmp.K_4,fused_qkv_old_tmp.K_5,fused_qkv_old_tmp.V_4,fused_qkv_old_tmp.V_5,fused_qkv_old_tmp.Q_6,fused_qkv_old_tmp.Q_7,fused_qkv_old_tmp.K_6,fused_qkv_old_tmp.K_7,fused_qkv_old_tmp.V_6,fused_qkv_old_tmp.V_7 -> A_TEST_NAME, axis=1',
            'A_TEST_NAME ->  layers.2.self_attn.qkv_proj.weight',
        ]

    def test(self):
        self.start_macro_test(aoa_config_reverse=True)


class TestFusedFfnMacro(TestMacro):
    def macro_name(self):
        return "fused_ffn_macro"

    def source_code(self):
        return "layers.2.mlp.gate_up_fused_proj.weight -> layers.2.mlp.gate_up_fused_proj.weight, fused_ffn"

    def expected(self):
        return [
            'layers.2.mlp.gate_up_fused_proj.weight  -> fused_ffn_tmp.GATE_0,fused_ffn_tmp.GATE_1,fused_ffn_tmp.UP_0,fused_ffn_tmp.UP_1,fused_ffn_tmp.GATE_2,fused_ffn_tmp.GATE_3,fused_ffn_tmp.UP_2,fused_ffn_tmp.UP_3, axis=1',
            'fused_ffn_tmp.GATE_0,fused_ffn_tmp.UP_0,fused_ffn_tmp.GATE_1,fused_ffn_tmp.UP_1,fused_ffn_tmp.GATE_2,fused_ffn_tmp.UP_2,fused_ffn_tmp.GATE_3,fused_ffn_tmp.UP_3 -> layers.2.mlp.gate_up_fused_proj.weight, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedFfnMacro2(TestMacro):
    def macro_name(self):
        return "fused_ffn_macro"

    def source_code(self):
        return "layers.1.mlp.gate_up_fused_proj.weight -> layers.1.mlp.gate_proj.weight,layers.1.mlp.up_proj.weight, fused_ffn "

    def expected(self):
        return [
            'layers.1.mlp.gate_up_fused_proj.weight  -> fused_ffn_tmp.GATE_0,fused_ffn_tmp.UP_0,fused_ffn_tmp.GATE_1,fused_ffn_tmp.UP_1, axis=1',
            'fused_ffn_tmp.GATE_0,fused_ffn_tmp.GATE_1 -> layers.1.mlp.gate_proj.weight, axis=1',
            'fused_ffn_tmp.UP_0,fused_ffn_tmp.UP_1 -> layers.1.mlp.up_proj.weight, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedFfnMacro3(TestMacro):
    def macro_name(self):
        return "fused_ffn_macro"

    def source_code(self):
        return "layers.1.mlp.gate_up_fused_proj.weight -> layers.1.mlp.gate_proj.weight,layers.1.mlp.up_proj.weight, fused_ffn "

    def expected(self):
        return [
            'layers.1.mlp.gate_up_fused_proj.weight  -> fused_ffn_tmp.GATE_0,fused_ffn_tmp.UP_0,fused_ffn_tmp.GATE_1,fused_ffn_tmp.UP_1, axis=1',
            'fused_ffn_tmp.GATE_0,fused_ffn_tmp.GATE_1 -> layers.1.mlp.gate_proj.weight, axis=1',
            'fused_ffn_tmp.UP_0,fused_ffn_tmp.UP_1 -> layers.1.mlp.up_proj.weight, axis=1',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedFfnMacro4(TestMacro):
    def macro_name(self):
        return "fused_ffn_macro"

    def source_code(self):
        return "layers.2.mlp.gate_up_fused_proj.bias -> layers.2.mlp.gate_up_fused_proj.bias, fused_ffn, axis=0"

    def expected(self):
        return [
            'layers.2.mlp.gate_up_fused_proj.bias  -> fused_ffn_tmp.GATE_0,fused_ffn_tmp.GATE_1,fused_ffn_tmp.UP_0,fused_ffn_tmp.UP_1,fused_ffn_tmp.GATE_2,fused_ffn_tmp.GATE_3,fused_ffn_tmp.UP_2,fused_ffn_tmp.UP_3, axis=0',
            'fused_ffn_tmp.GATE_0,fused_ffn_tmp.UP_0,fused_ffn_tmp.GATE_1,fused_ffn_tmp.UP_1,fused_ffn_tmp.GATE_2,fused_ffn_tmp.UP_2,fused_ffn_tmp.GATE_3,fused_ffn_tmp.UP_3 -> layers.2.mlp.gate_up_fused_proj.bias, axis=0',
        ]

    def test(self):
        self.start_macro_test()


class TestFusedFfnMacro5(TestMacro):
    def macro_name(self):
        return "fused_ffn_macro"

    def source_code(self):
        return [
            "layers.1.mlp.gate_up_fused_proj.weight_test1 -> A_TEST_NAME, fused_ffn ",
            "A_TEST_NAME -> layers.1.mlp.gate_up_fused_proj.weight_test2",
        ]

    def expected(self):
        return [
            'layers.1.mlp.gate_up_fused_proj.weight_test1  -> fused_ffn_tmp.GATE_0,fused_ffn_tmp.GATE_1,fused_ffn_tmp.UP_0,fused_ffn_tmp.UP_1,fused_ffn_tmp.GATE_2,fused_ffn_tmp.GATE_3,fused_ffn_tmp.UP_2,fused_ffn_tmp.UP_3, axis=1',
            'fused_ffn_tmp.GATE_0,fused_ffn_tmp.UP_0,fused_ffn_tmp.GATE_1,fused_ffn_tmp.UP_1,fused_ffn_tmp.GATE_2,fused_ffn_tmp.UP_2,fused_ffn_tmp.GATE_3,fused_ffn_tmp.UP_3 -> A_TEST_NAME, axis=1',
            'A_TEST_NAME -> layers.1.mlp.gate_up_fused_proj.weight_test2',
        ]

    def test(self):
        self.start_macro_test(aoa_config_reverse=True)


class TestLayerIdOffsetMacro(TestMacro):
    def macro_name(self):
        return "layer_id_offset_macro"

    def source_code(self):
        return "layers.$LAYER_ID_OFFSET.experts.0.weight -> layers.$LAYER_ID_OFFSET.experts.0.weight, axis = 1"

    def expected(self):
        return [
            'layers.1.experts.0.weight->layers.0.experts.0.weight,axis=1\n',
            'layers.2.experts.0.weight->layers.1.experts.0.weight,axis=1\n',
        ]

    def test(self):
        self.start_macro_test()


class TestIdMacroCase0(TestMacro):
    def macro_name(self):
        return "id_macro"

    def source_code(self):
        return "layers.$LAYER_ID.qkv_proj.weight->layers.$LAYER_ID.q_proj.weight,layer.$LAYER_ID.k_proj.weight,layer.$LAYER_ID.v_proj.weight, fused_qkv_old, num_heads = 8, num_key_value_groups = 4\n"

    def expected(self):
        return [
            'layers.0.qkv_proj.weight->layers.0.q_proj.weight,layer.0.k_proj.weight,layer.0.v_proj.weight,fused_qkv_old,num_heads=8,num_key_value_groups=4\n',
        ]

    def test(self):
        self.start_macro_test()


class TestIdMacroCase1(TestMacro):
    def macro_name(self):
        return "id_macro"

    def source_code(self):
        return "layers.5.experts.$EXPERT_ID.up_gate_proj.weight -> layers.5.experts.$EXPERT_ID.gate_proj.weight, layers.5.experts.$EXPERT_ID.up_proj.weight, fused_ffn"

    def expected(self):
        return [
            'layers.5.experts.0.up_gate_proj.weight->layers.5.experts.0.gate_proj.weight,layers.5.experts.0.up_proj.weight,fused_ffn\n',
            'layers.5.experts.1.up_gate_proj.weight->layers.5.experts.1.gate_proj.weight,layers.5.experts.1.up_proj.weight,fused_ffn\n',
        ]

    def test(self):
        self.start_macro_test()


class TestIdMacroCase2(TestMacro):
    def macro_name(self):
        return "id_macro"

    def source_code(self):
        return "layers.$LAYER_ID.experts.$EXPERT_ID.up_gate_proj.weight -> layers.$LAYER_ID.experts.$EXPERT_ID.gate_proj.weight, fused_ffn"

    def expected(self):
        return [
            'layers.3.experts.0.up_gate_proj.weight->layers.3.experts.0.gate_proj.weight,fused_ffn\n',
            'layers.5.experts.0.up_gate_proj.weight->layers.5.experts.0.gate_proj.weight,fused_ffn\n',
            'layers.3.experts.1.up_gate_proj.weight->layers.3.experts.1.gate_proj.weight,fused_ffn\n',
            'layers.5.experts.1.up_gate_proj.weight->layers.5.experts.1.gate_proj.weight,fused_ffn\n',
        ]

    def test(self):
        self.start_macro_test()


class TestIdMacroCase3(TestMacro):
    def macro_name(self):
        return "id_macro"

    def source_code(self):
        return "layers.$LAYER_ID.experts.$EXPERT_ID.up_gate_proj.weight^T -> layers.$LAYER_ID.experts.$EXPERT_ID.gate_proj.weight, fused_ffn"

    def expected(self):
        return [
            'layers.3.experts.0.up_gate_proj.weight^T->layers.3.experts.0.gate_proj.weight,fused_ffn\n',
            'layers.5.experts.0.up_gate_proj.weight^T->layers.5.experts.0.gate_proj.weight,fused_ffn\n',
            'layers.3.experts.1.up_gate_proj.weight^T->layers.3.experts.1.gate_proj.weight,fused_ffn\n',
            'layers.5.experts.1.up_gate_proj.weight^T->layers.5.experts.1.gate_proj.weight,fused_ffn\n',
        ]

    def test(self):
        self.start_macro_test()


if __name__ == "__main__":
    unittest.main()
