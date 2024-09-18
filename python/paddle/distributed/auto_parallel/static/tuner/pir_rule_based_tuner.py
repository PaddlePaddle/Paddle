# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from abc import abstractmethod

import paddle
import paddle.nn
import paddle.nn.functional as F

_PIR_PATTERNS = {}


def register_pir_pattern(cls):
    """Register pir patterns"""

    def register():
        global _PIR_PATTERNS
        pattern = cls()
        _PIR_PATTERNS[pattern.name] = pattern

    register()

    return cls


class PIRBasePattern:
    """
    Base class of pattern.
    """

    _name = "base"

    def __init__(self):
        super().__init__()
        self.build()

    @property
    def name(self):
        return self.__class__._name

    @abstractmethod
    def build(self):
        pass


class DistInfos:
    def __init__(self):
        self.dist_infos = {}  # containing weight and bias
        self.dist_infos[1] = [
            [paddle.distributed.Replicate()],
            [paddle.distributed.Replicate()],
        ]
        self.dist_infos[2] = [
            [paddle.distributed.Replicate(), paddle.distributed.Replicate()],
            [paddle.distributed.Replicate(), paddle.distributed.Replicate()],
        ]

    def get_dist_info(self, mesh_num_dims=1):
        assert mesh_num_dims in [
            1,
            2,
        ], "mesh dims must be 1 or 2, for [dp], [mp] or [dp, mp]"
        return self.dist_infos[mesh_num_dims]

    def print_dist_infos(self):
        for mesh_num_dims, dist_infos in self.dist_infos.items():
            print(
                f"If mesh has {mesh_num_dims} dims, dist infos are {dist_infos}"
            )


class MpDistInfos(DistInfos):
    def __init__(self, mp_type=None):
        super().__init__()
        if mp_type == "column":
            self.dist_infos[1] = [
                [paddle.distributed.Shard(1)],
                [paddle.distributed.Shard(0)],
            ]
            self.dist_infos[2] = [
                [paddle.distributed.Replicate(), paddle.distributed.Shard(1)],
                [paddle.distributed.Replicate(), paddle.distributed.Shard(0)],
            ]
        elif mp_type == "row":
            self.dist_infos[1] = [
                [paddle.distributed.Shard(0)],
                [paddle.distributed.Replicate()],
            ]
            self.dist_infos[2] = [
                [paddle.distributed.Replicate(), paddle.distributed.Shard(0)],
                [
                    paddle.distributed.Replicate(),
                    paddle.distributed.Replicate(),
                ],
            ]


# Llama
# @register_pir_pattern
class PIRRotateHalfPattern(PIRBasePattern):
    """Rotate Half pattern"""

    name = "rotate_half"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        x_shape = [4, 1024, 32, 64]  # [batch, sequence, num_heads, head_size]
        tmp_x = paddle.randn(x_shape)
        tmp = paddle.static.data('tmp_x', x_shape, tmp_x.dtype)
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            x = paddle.reshape(tmp, x_shape)
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            out = paddle.concat([-x2, x1], axis=-1)

        self.pir_program = main_program
        print(f"in RotateHalfPattern, program is {self.pir_program}")
        paddle.disable_static()

        # # todo: how to design an efficient distributed infos for each pattern
        self.ops_dist_infos = None


# Llama
# @register_pir_pattern
class PIRApplyRotaryPosEmbPattern(PIRBasePattern):
    """ApplyRotaryPosEmb pattern"""

    name = "apply_rotary_pos_emb"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        batch_size = 4
        seq_length = 1024
        num_heads = 32
        head_size = 64
        q_shape = [batch_size, seq_length, num_heads, head_size]
        q = paddle.randn(q_shape)
        k_shape = [batch_size, seq_length, num_heads, head_size]
        k = paddle.randn(k_shape)
        cos_shape = [1, seq_length, 1, head_size]
        cos = paddle.randn(cos_shape)
        sin_shape = [1, seq_length, 1, head_size]
        sin = paddle.randn(sin_shape)
        position_ids_shape = [batch_size, seq_length]
        position_ids = paddle.randint(low=1, shape=position_ids_shape)

        # program construction
        with paddle.static.program_guard(main_program, start_program):
            q = paddle.static.data('q', q_shape, q.dtype)
            k = paddle.static.data('k', k_shape, k.dtype)
            cos = paddle.static.data('cos', cos_shape, cos.dtype)
            sin = paddle.static.data('sin', sin_shape, sin.dtype)
            position_ids = paddle.static.data(
                'position_ids', position_ids_shape, position_ids.dtype
            )

            if position_ids is None:
                cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
                sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
            else:
                cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
                sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
                cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
                sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)

        self.pir_program = main_program
        print(f"in ApplyRotaryPosEmbPattern, program is {self.pir_program}")
        paddle.disable_static()

        # # todo: how to design an efficient distributed infos for each pattern
        self.ops_dist_infos = None

    @staticmethod
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


# Llama
# @register_pir_pattern
class PIRQKVReshapePattern(PIRBasePattern):
    """QKV(not fused) and reshape pattern"""

    name = "qkv_reshape"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        batch_size = 4
        seq_length = 1024
        num_heads = 32
        head_size = 64
        hidden_size = num_heads * head_size
        hidden_states_shape = [batch_size, seq_length, hidden_size]
        hidden_states = paddle.randn(hidden_states_shape)
        weight_shape = [hidden_size, hidden_size]

        # program construction
        with paddle.static.program_guard(main_program, start_program):
            hidden_states = paddle.static.data(
                'hidden_states', hidden_states_shape, hidden_states.dtype
            )
            q_weight = paddle.create_parameter(weight_shape, "float32")
            k_weight = paddle.create_parameter(weight_shape, "float32")
            v_weight = paddle.create_parameter(weight_shape, "float32")

            q = paddle.matmul(hidden_states, q_weight)
            k = paddle.matmul(hidden_states, k_weight)
            v = paddle.matmul(hidden_states, v_weight)
            target_q_shape = [0, 0, num_heads, head_size]
            target_kv_shape = [0, 0, num_heads, head_size]
            q = q.reshape(shape=target_q_shape)
            k = k.reshape(shape=target_kv_shape)
            v = v.reshape(shape=target_kv_shape)

        self.pir_program = main_program
        print(f"in QKVReshape, program is {self.pir_program}")
        paddle.disable_static()

        # # todo: how to design an efficient distributed infos for each pattern
        self.ops_dist_infos = None


# Llama
# @register_pir_pattern
class PIRQKVRopePattern(PIRBasePattern):
    """QKV(not fused) and Rope pattern"""

    name = "qkv_rope"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        batch_size = 4
        seq_length = 1024
        num_heads = 32
        head_size = 64
        hidden_size = num_heads * head_size
        hidden_states_shape = [batch_size, seq_length, hidden_size]
        hidden_states = paddle.randn(hidden_states_shape)
        cos_cached_shape = [1, seq_length, 1, head_size]
        cos_cached = paddle.randn(cos_cached_shape)
        sin_cached_shape = [1, seq_length, 1, head_size]
        sin_cached = paddle.randn(sin_cached_shape)
        position_ids_shape = [batch_size, seq_length]
        position_ids = paddle.randint(low=1, shape=position_ids_shape)
        weight_shape = [hidden_size, hidden_size]

        # program construction
        with paddle.static.program_guard(main_program, start_program):
            hidden_states = paddle.static.data(
                'hidden_states', hidden_states_shape, hidden_states.dtype
            )
            q_weight = paddle.create_parameter(weight_shape, "float32")
            k_weight = paddle.create_parameter(weight_shape, "float32")
            v_weight = paddle.create_parameter(weight_shape, "float32")
            cos_cached = paddle.static.data(
                'cos_cached', cos_cached_shape, cos_cached.dtype
            )
            sin_cached = paddle.static.data(
                'sin_cached', cos_cached_shape, cos_cached.dtype
            )
            position_ids = paddle.static.data(
                'position_ids', position_ids_shape, position_ids.dtype
            )

            # qkv (not fused)
            q = paddle.matmul(hidden_states, q_weight)
            k = paddle.matmul(hidden_states, k_weight)
            v = paddle.matmul(hidden_states, v_weight)
            target_q_shape = [0, 0, num_heads, head_size]
            target_kv_shape = [0, 0, num_heads, head_size]
            q = q.reshape(shape=target_q_shape)
            k = k.reshape(shape=target_kv_shape)
            v = v.reshape(shape=target_kv_shape)

            # rope_emb
            cos = cos_cached[:, :seq_length, :, :]
            sin = sin_cached[:, :seq_length, :, :]

            # apply_rotary_pos_emb
            if position_ids is None:
                cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
                sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
            else:
                cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
                sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
                cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
                sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)
        self.pir_program = main_program
        print(f"in QKVRopePattern, program is {self.pir_program}")
        paddle.disable_static()

        # # todo: how to design an efficient distributed infos for each pattern
        self.ops_dist_infos = None

    @staticmethod
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


# Llama
# @register_pir_pattern
class PIRScaleDotProductPattern(PIRBasePattern):
    """Scale dot product attention pattern"""

    name = "scale_dot_product_attention"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        batch_size = 4
        seq_length = 1024
        num_heads = 32
        head_size = 64
        hidden_size = num_heads * head_size
        q_shape = [batch_size, seq_length, num_heads, head_size]
        q = paddle.randn(q_shape)
        k_shape = [batch_size, seq_length, num_heads, head_size]
        k = paddle.randn(k_shape)
        v_shape = [batch_size, seq_length, num_heads, head_size]
        v = paddle.randn(v_shape)
        attention_mask_shape = [batch_size, 1, seq_length, seq_length]
        attention_mask = paddle.randn(attention_mask_shape)

        # program construction
        with paddle.static.program_guard(main_program, start_program):
            q = paddle.static.data('q', q_shape, q.dtype)
            k = paddle.static.data('k', k_shape, k.dtype)
            v = paddle.static.data('v', v_shape, v.dtype)
            attention_mask = paddle.static.data(
                'attention_mask', attention_mask_shape, attention_mask.dtype
            )
            outputs = self.scale_dot_product_attention(q, k, v, attention_mask)

        self.pir_program = main_program
        print(f"in ScaleDotProductPattern, program is {self.pir_program}")
        paddle.disable_static()
        # # todo: how to design an efficient distributed infos for each pattern
        self.ops_dist_infos = None

    @staticmethod
    def scale_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
    ):

        bsz, q_len, num_heads, head_dim = query_states.shape
        _, kv_seq_len, _, _ = value_states.shape

        #  [ bz, seqlen, nhead, head_dim] -> [bs, nhead, seq_len, head_dim]
        query_states = paddle.transpose(query_states, [0, 2, 1, 3])
        # merge with the next tranpose
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        # matmul and devide by sqrt(head_dim)
        attn_weights = paddle.matmul(
            query_states / math.sqrt(head_dim),
            key_states.transpose([0, 1, 3, 2]),
        )

        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])

        attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(
            query_states.dtype
        )

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])

        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])

        return attn_output


# Llama
@register_pir_pattern
class PIRAttentionPattern(PIRBasePattern):
    """Attention pattern"""

    name = "attention"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        batch_size = 4
        seq_length = 1024
        num_heads = 32
        head_size = 64
        hidden_size = num_heads * head_size
        hidden_states_shape = [batch_size, seq_length, hidden_size]
        hidden_states = paddle.randn(hidden_states_shape)
        cos_cached_shape = [1, seq_length, 1, head_size]
        cos_cached = paddle.randn(cos_cached_shape)
        sin_cached_shape = [1, seq_length, 1, head_size]
        sin_cached = paddle.randn(sin_cached_shape)
        position_ids_shape = [batch_size, seq_length]
        position_ids = paddle.randint(low=1, shape=position_ids_shape)
        attention_mask_shape = [batch_size, 1, seq_length, seq_length]
        attention_mask = paddle.randn(attention_mask_shape)
        weight_shape = [hidden_size, hidden_size]

        # program construction
        with paddle.static.program_guard(main_program, start_program):
            hidden_states = paddle.static.data(
                'hidden_states', hidden_states_shape, hidden_states.dtype
            )
            q_weight = paddle.create_parameter(weight_shape, "float32")
            k_weight = paddle.create_parameter(weight_shape, "float32")
            v_weight = paddle.create_parameter(weight_shape, "float32")
            out_weight = paddle.create_parameter(weight_shape, "float32")
            cos_cached = paddle.static.data(
                'cos_cached', cos_cached_shape, cos_cached.dtype
            )
            sin_cached = paddle.static.data(
                'sin_cached', cos_cached_shape, cos_cached.dtype
            )
            position_ids = paddle.static.data(
                'position_ids', position_ids_shape, position_ids.dtype
            )
            attention_mask = paddle.static.data(
                'attention_mask', attention_mask_shape, attention_mask.dtype
            )

            # qkv (not fused)
            q = paddle.matmul(hidden_states, q_weight)
            k = paddle.matmul(hidden_states, k_weight)
            v = paddle.matmul(hidden_states, v_weight)
            target_q_shape = [0, 0, num_heads, head_size]
            target_kv_shape = [0, 0, num_heads, head_size]
            q = q.reshape(shape=target_q_shape)
            k = k.reshape(shape=target_kv_shape)
            v = v.reshape(shape=target_kv_shape)

            # rope_emb
            cos = cos_cached[:, :seq_length, :, :]
            sin = sin_cached[:, :seq_length, :, :]

            # apply_rotary_pos_emb
            if position_ids is None:
                cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
                sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
            else:
                cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
                sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
                cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
                sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)

            # scale_dot_product
            tmp = self.scale_dot_product_attention(
                q_embed, k_embed, v, attention_mask
            )

            # out_linear
            output = paddle.matmul(tmp, out_weight)

        self.pir_program = main_program
        print(f"in AttentionPattern, program is {self.pir_program}")
        paddle.disable_static()

        # # todo: how to design an efficient distributed infos for each pattern
        qkv_linear_dist_infos = MpDistInfos("column")
        out_linear_dist_infos = MpDistInfos("row")

        # # # build ops dist infos # # #
        ops_dist_infos = {
            (9,): qkv_linear_dist_infos,
            (10,): qkv_linear_dist_infos,
            (11,): qkv_linear_dist_infos,
            (77,): out_linear_dist_infos,
        }
        self.ops_dist_infos = ops_dist_infos

    @staticmethod
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x

    @staticmethod
    def scale_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
    ):

        bsz, q_len, num_heads, head_dim = query_states.shape
        _, kv_seq_len, _, _ = value_states.shape

        #  [ bz, seqlen, nhead, head_dim] -> [bs, nhead, seq_len, head_dim]
        query_states = paddle.transpose(query_states, [0, 2, 1, 3])
        # merge with the next tranpose
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        # matmul and devide by sqrt(head_dim)
        attn_weights = paddle.matmul(
            query_states / math.sqrt(head_dim),
            key_states.transpose([0, 1, 3, 2]),
        )

        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])

        attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(
            query_states.dtype
        )

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])

        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])

        return attn_output


# Llama
@register_pir_pattern
class PIRMLPPattern(PIRBasePattern):
    """MLP pattern"""

    name = "MLP"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        batch_size = 4
        seq_length = 1024
        hidden_size = 2048
        intermediate_size = 4096
        hidden_states_shape = [batch_size, seq_length, hidden_size]
        hidden_states = paddle.randn(hidden_states_shape)
        up_weight_shape = [hidden_size, intermediate_size]
        down_weight_shape = [intermediate_size, hidden_size]

        # program construction
        with paddle.static.program_guard(main_program, start_program):
            hidden_states = paddle.static.data(
                'hidden_states', hidden_states_shape, hidden_states.dtype
            )
            gate_weight = paddle.create_parameter(up_weight_shape, "float32")
            up_weight = paddle.create_parameter(up_weight_shape, "float32")
            down_weight = paddle.create_parameter(down_weight_shape, "float32")

            # qkv (not fused)
            gate = paddle.matmul(hidden_states, gate_weight)
            up = paddle.matmul(hidden_states, up_weight)
            tmp = paddle.incubate.nn.functional.swiglu(gate, up)
            out = paddle.matmul(tmp, down_weight)

        self.pir_program = main_program
        print(f"in MLPPattern, program is {self.pir_program}")
        paddle.disable_static()

        up_linear_dist_infos = MpDistInfos("column")
        down_linear_dist_infos = MpDistInfos("row")
        # up_linear_dist_infos = MpDistInfos()
        # down_linear_dist_infos = MpDistInfos()

        # # # build ops dist infos # # #
        ops_dist_infos = {
            (4,): up_linear_dist_infos,
            (5,): up_linear_dist_infos,
            (7,): down_linear_dist_infos,
        }
        self.ops_dist_infos = ops_dist_infos


# GPT
# @register_pir_pattern
class PIRQKVReshapePattern_2(PIRBasePattern):
    """QKV(not fused) and reshape pattern"""

    name = "qkv_reshape_2"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        batch_size = 4
        seq_length = 1024
        num_heads = 32
        head_size = 64
        hidden_size = num_heads * head_size
        hidden_states_shape = [batch_size, seq_length, hidden_size]
        hidden_states = paddle.randn(hidden_states_shape)
        weight_shape = [hidden_size, hidden_size]
        bias_shape = [hidden_size]

        # program construction
        with paddle.static.program_guard(main_program, start_program):
            hidden_states = paddle.static.data(
                'hidden_states', hidden_states_shape, hidden_states.dtype
            )
            q_weight = paddle.create_parameter(weight_shape, "float32")
            k_weight = paddle.create_parameter(weight_shape, "float32")
            v_weight = paddle.create_parameter(weight_shape, "float32")
            q_bias = paddle.create_parameter(bias_shape, "float32")
            k_bias = paddle.create_parameter(bias_shape, "float32")
            v_bias = paddle.create_parameter(bias_shape, "float32")

            q_tmp = paddle.matmul(hidden_states, q_weight)
            q = paddle.add(q_tmp, q_bias)
            k_tmp = paddle.matmul(hidden_states, k_weight)
            k = paddle.add(k_tmp, k_bias)
            v_tmp = paddle.matmul(hidden_states, v_weight)
            v = paddle.add(v_tmp, v_bias)
            target_q_shape = [0, 0, num_heads, head_size]
            target_kv_shape = [0, 0, num_heads, head_size]
            q = q.reshape(shape=target_q_shape)
            k = k.reshape(shape=target_kv_shape)
            v = v.reshape(shape=target_kv_shape)

        self.pir_program = main_program
        print(f"in QKVReshape2, program is {self.pir_program}")
        paddle.disable_static()

        # # todo: how to design an efficient distributed infos for each pattern
        self.ops_dist_infos = None


# GPT
# @register_pir_pattern
class PIRCoreAttnPattern(PIRBasePattern):
    """Core attention pattern"""

    name = "core_attn"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        batch_size = 4
        seq_length = 1024
        num_heads = 32
        head_size = 64
        hidden_size = num_heads * head_size
        q_shape = [batch_size, seq_length, num_heads, head_size]
        q = paddle.randn(q_shape)
        k_shape = [batch_size, seq_length, num_heads, head_size]
        k = paddle.randn(k_shape)
        v_shape = [batch_size, seq_length, num_heads, head_size]
        v = paddle.randn(v_shape)
        attention_mask_shape = [batch_size, 1, seq_length, seq_length]
        attention_mask = paddle.randn(attention_mask_shape)

        # program construction
        with paddle.static.program_guard(main_program, start_program):
            q = paddle.static.data('q', q_shape, q.dtype)
            k = paddle.static.data('k', k_shape, k.dtype)
            v = paddle.static.data('v', v_shape, v.dtype)
            attention_mask = paddle.static.data(
                'attention_mask', attention_mask_shape, attention_mask.dtype
            )
            outputs = self.core_attn(q, k, v, attention_mask)

        self.pir_program = main_program
        print(f"in CoreAttnPattern, program is {self.pir_program}")
        paddle.disable_static()
        # # todo: how to design an efficient distributed infos for each pattern
        self.ops_dist_infos = None

    @staticmethod
    def core_attn(
        q,
        k,
        v,
        attention_mask,
    ):
        bsz, q_len, num_heads, head_dim = q.shape
        perm = [0, 2, 1, 3]
        q = paddle.transpose(x=q, perm=perm)
        k = paddle.transpose(x=k, perm=perm)
        v = paddle.transpose(x=v, perm=perm)
        # scale dot product attention
        product = paddle.matmul(x=q * (head_dim**-0.5), y=k, transpose_y=True)

        if attention_mask is not None:
            product = product + attention_mask.astype(product.dtype)
            weights = F.softmax(product)
        else:
            weights = paddle.incubate.softmax_mask_fuse_upper_triangle(product)
        weights = F.dropout(
            weights, 0.5, training=True, mode="upscale_in_train"
        )

        out = paddle.matmul(weights, v)

        # combine heads
        out = paddle.transpose(
            out, perm=[0, 2, 1, 3]
        )  # bs, seq_len, num_head, head_dim
        out = paddle.reshape(x=out, shape=[0, 0, -1])  # bs, seq_len, dim

        return out


# # GPT
@register_pir_pattern
class PIRAttentio2nPattern(PIRBasePattern):
    """Attention2 pattern"""

    name = "attention2"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        batch_size = 4
        seq_length = 1024
        num_heads = 32
        head_size = 64
        hidden_size = num_heads * head_size
        hidden_states_shape = [batch_size, seq_length, hidden_size]
        hidden_states = paddle.randn(hidden_states_shape)
        weight_shape = [hidden_size, hidden_size]
        bias_shape = [hidden_size]
        attention_mask_shape = [batch_size, 1, seq_length, seq_length]
        attention_mask = paddle.randn(attention_mask_shape)

        # program construction
        with paddle.static.program_guard(main_program, start_program):
            hidden_states = paddle.static.data(
                'hidden_states', hidden_states_shape, hidden_states.dtype
            )
            q_weight = paddle.create_parameter(weight_shape, "float32")
            k_weight = paddle.create_parameter(weight_shape, "float32")
            v_weight = paddle.create_parameter(weight_shape, "float32")
            out_weight = paddle.create_parameter(weight_shape, "float32")
            q_bias = paddle.create_parameter(bias_shape, "float32")
            k_bias = paddle.create_parameter(bias_shape, "float32")
            v_bias = paddle.create_parameter(bias_shape, "float32")
            out_bias = paddle.create_parameter(bias_shape, "float32")
            attention_mask = paddle.static.data(
                'attention_mask', attention_mask_shape, attention_mask.dtype
            )

            # prepare qkv
            q_tmp = paddle.matmul(hidden_states, q_weight)
            q = paddle.add(q_tmp, q_bias)
            k_tmp = paddle.matmul(hidden_states, k_weight)
            k = paddle.add(k_tmp, k_bias)
            v_tmp = paddle.matmul(hidden_states, v_weight)
            v = paddle.add(v_tmp, v_bias)
            target_q_shape = [0, 0, num_heads, head_size]
            target_kv_shape = [0, 0, num_heads, head_size]
            q = q.reshape(shape=target_q_shape)
            k = k.reshape(shape=target_kv_shape)
            v = v.reshape(shape=target_kv_shape)

            # scale_dot_product
            tmp = self.core_attn(q, k, v, attention_mask)

            # out_linear
            out_tmp = paddle.matmul(tmp, out_weight)
            output = paddle.add(out_tmp, out_bias)

        self.pir_program = main_program
        print(f"in AttentionPattern2, program is {self.pir_program}")
        paddle.disable_static()

        # # todo: how to design an efficient distributed infos for each pattern
        qkv_linear_dist_infos = MpDistInfos("column")
        out_linear_dist_infos = MpDistInfos("row")

        # # # build ops dist infos # # #
        ops_dist_infos = {
            (10, 11): qkv_linear_dist_infos,
            (12, 13): qkv_linear_dist_infos,
            (14, 15): qkv_linear_dist_infos,
            (37, 38): out_linear_dist_infos,
        }
        self.ops_dist_infos = ops_dist_infos

    @staticmethod
    def core_attn(
        q,
        k,
        v,
        attention_mask,
    ):
        bsz, q_len, num_heads, head_dim = q.shape
        perm = [0, 2, 1, 3]
        q = paddle.transpose(x=q, perm=perm)
        k = paddle.transpose(x=k, perm=perm)
        v = paddle.transpose(x=v, perm=perm)
        # scale dot product attention
        product = paddle.matmul(x=q * (head_dim**-0.5), y=k, transpose_y=True)

        if attention_mask is not None:
            product = product + attention_mask.astype(product.dtype)
            weights = F.softmax(product)
        else:
            weights = paddle.incubate.softmax_mask_fuse_upper_triangle(product)
        weights = F.dropout(
            weights, 0.5, training=True, mode="upscale_in_train"
        )

        out = paddle.matmul(weights, v)

        # combine heads
        out = paddle.transpose(
            out, perm=[0, 2, 1, 3]
        )  # bs, seq_len, num_head, head_dim
        out = paddle.reshape(x=out, shape=[0, 0, -1])  # bs, seq_len, dim

        return out


# DemoNet, GPT
@register_pir_pattern
class PIRFFNPattern(PIRBasePattern):
    """FFN pattern"""

    name = "ffn"

    def __init__(self):
        super().__init__()

    def build(self):
        # # # build program # # #
        paddle.enable_static()
        # program init
        start_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        # data init
        x_shape = [4, 4]
        weight_shape = [4, 4]
        bias_shape = [4]
        x = paddle.randn(x_shape)
        # program construction
        with paddle.static.program_guard(main_program, start_program):
            x = paddle.static.data('x', x_shape, x.dtype)
            w1 = paddle.create_parameter(weight_shape, "float32")
            b1 = paddle.create_parameter(bias_shape, "float32")
            w2 = paddle.create_parameter(weight_shape, "float32")
            b2 = paddle.create_parameter(bias_shape, "float32")

            tmp_1 = paddle.matmul(x, w1)
            tmp_2 = paddle.add(tmp_1, b1)
            tmp_3 = paddle.nn.functional.gelu(tmp_2)
            tmp_4 = paddle.matmul(tmp_3, w2)
            out = paddle.add(tmp_4, b2)

        self.pir_program = main_program
        print(f"in FFNPattern, program is {self.pir_program}")
        paddle.disable_static()

        # # todo: how to design an efficient distributed infos for each pattern
        linear_1_dist_infos = MpDistInfos("column")
        linear_2_dist_infos = MpDistInfos("row")

        # # # build ops dist infos # # #
        ops_dist_infos = {
            (5, 6): linear_1_dist_infos,
            (8, 9): linear_2_dist_infos,
        }
        self.ops_dist_infos = ops_dist_infos


def match_pattern(pattern, pir_program):

    def _compare_op_node(src, tgt):
        """Compare whether two op nodes are equivalent."""
        if src.name() != tgt.name():
            return False

        return True

    def _match_core(src, tgt, is_op):
        # # for checking
        # print(f"current result is: {result}")
        # if is_op:
        #     print(f"comparing op in pattern: {src.name()}, op in program: {tgt.name()}")
        # else:
        #     print(f"comparing input/output in pattern: {src}, input/output in program: {tgt}")

        nonlocal not_matched
        # not support one input name or output name corresponding to multiple vars
        if not_matched:
            return

        if is_op:
            # print(f"comparing op: {src.name()}, with {tgt.name()}")
            # print(f"comparing op {src.name()}")
            # skip comparing data_op
            if src.name() == "pd_op.data":
                return
            # compare op
            if not _compare_op_node(src, tgt):
                # print(f"op not match")
                # print(f"because src name is {src.name()}, tgt name is {tgt.name()}")
                not_matched = True
                return

            src_id = src.get_parent_block().ops.index(src)
            tgt_id = tgt.get_parent_block().ops.index(tgt)
            # print(f"adding op {src.name()}")
            # print(f"comparing op: {src_id}, with {tgt_id}")
            result[src_id] = tgt_id

            # compare input operands num
            if src.num_operands() != tgt.num_operands():
                # print(f"num_operands not match")
                not_matched = True
                return
            # compare output results num
            if src.num_results() != tgt.num_results():
                # print(f"num_results not match")
                not_matched = True
                return

            # compare input operands
            src_operands = src.operands_source()
            for idx, src_operand in enumerate(src_operands):
                # print(f"compare op {src_id} src operand: {idx}")
                tgt_operand = tgt.operand_source(idx)
                _match_core(src_operand, tgt_operand, is_op=False)

            # compare output results
            src_results = src.results()
            for idx, src_result in enumerate(src_results):
                # print(f"compare op {src_id} tgt result: {idx}")
                tgt_result = tgt.result(idx)
                _match_core(src_result, tgt_result, is_op=False)

        else:
            # print(f"compare operand, not op")
            # compare tensor, from tensor to op

            # as input for op node
            src_as_input_ops = src.all_used_ops()
            tgt_as_input_ops = tgt.all_used_ops()
            # if len(src_as_input_ops) != len(tgt_as_input_ops):
            #     print(f"src_as_input_ops not match")
            #     not_matched = True
            #     return
            # todo: process src_as_input_ops < tgt_as_input_ops
            if len(src_as_input_ops) > len(tgt_as_input_ops):
                not_matched = True
                return
            if len(src_as_input_ops) == len(tgt_as_input_ops):
                for idx, src_as_input_op in enumerate(src_as_input_ops):
                    src_as_input_op_id = (
                        src_as_input_op.get_parent_block().ops.index(
                            src_as_input_op
                        )
                    )
                    if src_as_input_op_id in result.keys():
                        continue

                    tgt_as_input_op = tgt_as_input_ops[idx]
                    _match_core(src_as_input_op, tgt_as_input_op, is_op=True)

            # as output for op node
            src_as_output_op = src.get_defining_op()
            tgt_as_output_op = tgt.get_defining_op()
            if src_as_output_op is not None and tgt_as_output_op is not None:
                src_as_output_op_id = (
                    src_as_output_op.get_parent_block().ops.index(
                        src_as_output_op
                    )
                )
                if src_as_output_op_id not in result.keys():
                    _match_core(src_as_output_op, tgt_as_output_op, is_op=True)

    results = []
    result = {}
    matched_ids = set()
    matched_op_node_ids = set()

    # starts with a op node
    src_ops = pattern.pir_program.global_block().ops
    for op in src_ops:
        if op.name() != "pd_op.data" and op.name() != "builtin.parameter":
            src_start_op = op
            break
    # src_start_op = src_ops[0] # to be done, need to check pattern start op
    assert src_start_op is not None, "src_start_op is none"
    # print(f"start op of this pattern is: {src_start_op}")

    tgt_ops = pir_program.global_block().ops
    for idx, tgt_op in enumerate(tgt_ops):
        if tgt_op.name() == src_start_op.name():
            # print(f"program op index is {idx}")
            # print(f"start match core")
            not_matched = False
            result = {}
            _match_core(src_start_op, tgt_op, is_op=True)
            if not not_matched:
                # print(f"matched one, check whether to append")
                need_to_append = True
                for value in result.values():
                    if value in matched_op_node_ids:
                        result = {}
                        need_to_append = False
                        break
                if need_to_append:
                    # print(f"needs to append")
                    results.append(result)
                    for value in result.values():
                        matched_ids.add(value)
                        matched_op_node_ids.add(value)
                    result = {}
            else:
                # print(f"not matched")
                not_matched = False
                result = {}

    return results, matched_ids


def match_all_patterns(pir_program):
    matched_results = {}
    matched_ids = set()
    for pattern_name in _PIR_PATTERNS:
        pattern = _PIR_PATTERNS[pattern_name]
        results, matched = match_pattern(pattern, pir_program)
        for result in results:
            has_matched = False
            for id in result:
                if result[id] in matched_ids:
                    has_matched = True
                    break
            if not has_matched:
                for item in result:
                    matched_ids.add(result[id])
                if pattern.name not in matched_results:
                    matched_results[pattern.name] = []
                matched_results[pattern.name].append(result)

    return matched_results
