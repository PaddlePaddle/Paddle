# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import Shard

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None

BATCH_COUNT = 10
BATCH_SIZE = 16
SEQ_LEN = 128
NUM_HEADS = 8
HEAD_DIM = 64
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM


class RotaryAngle(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # [dim / 2]
        self.inv_freq = 1.0 / (
            self.base
            ** (
                paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32")
                / self.dim
            )
        )
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x, seq_len=None):
        # x: [bs, seq_len, num_heads, head_dim]
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        return (
            cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
            sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    if position_ids is None:
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryPositionEmbedding(nn.Layer):
    def __init__(self, seq_len, num_heads, head_dim, is_use_fused_rope=False):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rotary_angle = RotaryAngle(
            dim=self.head_dim, max_position_embeddings=self.seq_len
        )
        self.is_use_fused_rope = is_use_fused_rope
        self.hidden_size = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias_attr=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias_attr=False,
        )

    def forward(self, input):
        target_query_shape = [0, 0, self.num_heads, self.head_dim]
        query_states = self.q_proj(input).reshape(shape=target_query_shape)
        key_states = self.k_proj(input).reshape(shape=target_query_shape)

        cos, sin = self.rotary_angle(query_states, seq_len=self.seq_len)
        position_ids = paddle.arange(self.seq_len, dtype="int64").expand(
            (BATCH_SIZE, self.seq_len)
        )

        if self.is_use_fused_rope:
            query_states, key_states, _ = fused_rotary_position_embedding(
                query_states,
                key_states,
                v=None,
                sin=sin,
                cos=cos,
                position_ids=position_ids,
                use_neox_rotary_style=False,
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        return query_states, key_states


class TestLlamaRopeSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        paddle.set_device(self._backend)
        self.init_single_card_net_result()

    def mp_shard_fn(self, layer_name, layer, process_mesh):
        if layer_name == "q_proj" or layer_name == "k_proj":
            layer.weight = dist.shard_tensor(
                layer.weight, process_mesh, [Shard(1)]
            )

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def init_input_data(self):
        input = np.random.random([BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]).astype(
            self._dtype
        )
        input = paddle.to_tensor(input)
        return input

    def init_single_card_net_result(self):
        self.set_random_seed(self._seed)
        rotary_emb = RotaryPositionEmbedding(
            seq_len=SEQ_LEN,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
        )
        self.base_out, self.base_parameters = self.train_loop(rotary_emb)

    def train_loop(self, layer, shard_input=False):
        # run forward and backward
        input_dist_attr = [Shard(0)]

        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )
        for _ in range(BATCH_COUNT):
            input = self.init_input_data()
            if shard_input:
                input = dist.shard_tensor(input, self._mesh, input_dist_attr)
            query_states, key_states = layer(input)
            loss = paddle.sum(query_states + key_states)
            loss.backward()
            opt.step()
            opt.clear_grad()
        return loss, layer.parameters()

    def check_tensor_eq(self, a, b, rtol=1e-04, atol=1e-05, verbose=True):
        if a is None:
            assert b is None
            return
        np1 = a.astype("float32").numpy()
        np2 = b.astype("float32").numpy()
        np.testing.assert_allclose(
            np1, np2, rtol=rtol, atol=atol, verbose=verbose
        )

    def test_dp(self, is_use_fused_rope=False):
        self.set_random_seed(self._seed)
        dp_layer = RotaryPositionEmbedding(
            seq_len=SEQ_LEN,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            is_use_fused_rope=is_use_fused_rope,
        )

        dp_out, dp_parameters = self.train_loop(
            dp_layer,
            shard_input=True,
        )
        self.check_tensor_eq(dp_out, self.base_out)
        for param, param_base in zip(dp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def test_mp(self, is_use_fused_rope=False):
        self.set_random_seed(self._seed)
        mp_layer = RotaryPositionEmbedding(
            seq_len=SEQ_LEN,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            is_use_fused_rope=is_use_fused_rope,
        )
        mp_layer = dist.shard_layer(mp_layer, self._mesh, self.mp_shard_fn)
        mp_out, mp_parameters = self.train_loop(mp_layer)
        self.check_tensor_eq(mp_out, self.base_out)
        for param, param_base in zip(mp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def run_test_case(self):
        self.test_dp(is_use_fused_rope=False)
        self.test_mp(is_use_fused_rope=False)
        self.test_dp(is_use_fused_rope=True)
        self.test_mp(is_use_fused_rope=True)


if __name__ == '__main__':
    TestLlamaRopeSemiAutoParallel().run_test_case()
