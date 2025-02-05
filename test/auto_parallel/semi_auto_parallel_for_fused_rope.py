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

import numpy as np
from semi_auto_parallel_util import SemiAutoParallelTestBase

import paddle
import paddle.distributed as dist

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None


class TestFusedRopeApiForSemiAutoParallel(SemiAutoParallelTestBase):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._bs = 16
        self._seq_len = 128
        self._num_heads = 8
        self._head_dim = 64
        self._qkv_shape = [
            self._bs,
            self._seq_len,
            self._num_heads,
            self._head_dim,
        ]
        self._group_num = 4
        self._sin_cos_shape = [1, self._seq_len, 1, self._head_dim]
        self._position_ids_shape = [self._bs, self._seq_len]

    def check_placements(self, output, expected_placements):
        assert (
            output.placements == expected_placements
        ), f"{output.placements}  vs {expected_placements}"

    def test_only_q_input(self):
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        # [bs, seq_len, num_heads, head_dim]
        q = paddle.randn(self._qkv_shape, self._dtype)
        q.stop_gradient = False

        dist_q = dist.shard_tensor(q, self._mesh, dist.Shard(0))
        dist_q.stop_gradient = False
        dist_out_q, _, _ = fused_rotary_position_embedding(
            q=dist_q, use_neox_rotary_style=True
        )
        out_q, _, _ = fused_rotary_position_embedding(
            q, use_neox_rotary_style=True
        )
        self.check_tensor_eq(out_q, dist_out_q)
        self.check_placements(dist_out_q, [dist.Shard(0)])

        dist_out_q.backward()
        out_q.backward()
        self.check_tensor_eq(dist_q.grad, q.grad)

    def test_only_q_input_time_major(self):
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        # [seq_len, bs, num_heads, head_dim]
        qkv_shape = [self._seq_len, self._bs, self._num_heads, self._head_dim]
        q = paddle.randn(qkv_shape, self._dtype)
        q.stop_gradient = False

        dist_q = dist.shard_tensor(q, self._mesh, dist.Shard(0))
        dist_q.stop_gradient = False

        dist_out_q, _, _ = fused_rotary_position_embedding(
            q=dist_q, use_neox_rotary_style=True, time_major=True
        )
        out_q, _, _ = fused_rotary_position_embedding(
            q, use_neox_rotary_style=True, time_major=True
        )
        self.check_tensor_eq(out_q, dist_out_q)
        # NOTE: fused_rope have not supported shard on seq_len, so reshard to dist.Replicate
        self.check_placements(dist_out_q, [dist.Replicate()])

        dist_out_q.backward()
        out_q.backward()
        self.check_tensor_eq(dist_q.grad, q.grad)

    def test_common_case(self, is_gqa=False):
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        # [bs, seq_len, num_heads, head_dim]
        q = paddle.randn(self._qkv_shape, self._dtype)
        q.stop_gradient = False

        dist_q = dist.shard_tensor(q, self._mesh, dist.Shard(0))
        dist_q.stop_gradient = False
        if is_gqa:
            k_shape = [
                self._bs,
                self._seq_len,
                self._num_heads // self._group_num,
                self._head_dim,
            ]
        else:
            k_shape = self._qkv_shape
        k = paddle.randn(k_shape, self._dtype)
        k.stop_gradient = False
        dist_k = dist.shard_tensor(k, self._mesh, dist.Shard(2))
        dist_k.stop_gradient = False

        sin = paddle.randn(self._sin_cos_shape, self._dtype)
        sin.stop_gradient = True
        dist_sin = dist.shard_tensor(sin, self._mesh, dist.Replicate())
        dist_sin.stop_gradient = True

        cos = paddle.randn(self._sin_cos_shape, self._dtype)
        cos.stop_gradient = True
        dist_cos = dist.shard_tensor(cos, self._mesh, dist.Replicate())
        dist_cos.stop_gradient = True

        position_ids = paddle.arange(self._seq_len, dtype="int64").expand(
            (self._bs, self._seq_len)
        )
        position_ids.stop_gradient = True
        dist_position_ids = dist.shard_tensor(
            position_ids, self._mesh, dist.Shard(0)
        )
        dist_position_ids.stop_gradient = True

        dist_out_q, dist_out_k, _ = fused_rotary_position_embedding(
            q=dist_q,
            k=dist_k,
            sin=dist_sin,
            cos=dist_cos,
            position_ids=dist_position_ids,
            use_neox_rotary_style=False,
        )
        out_q, out_k, _ = fused_rotary_position_embedding(
            q=q,
            k=k,
            sin=sin,
            cos=cos,
            position_ids=position_ids,
            use_neox_rotary_style=False,
        )

        self.check_tensor_eq(out_q, dist_out_q)
        self.check_tensor_eq(out_k, dist_out_k)

        dist_out = paddle.sum(dist_out_q) + paddle.sum(dist_out_k)
        out = paddle.sum(out_q) + paddle.sum(out_k)
        dist_out.backward()
        out.backward()
        self.check_tensor_eq(dist_q.grad, q.grad)
        self.check_tensor_eq(dist_k.grad, k.grad)

    def test_common_case_time_major(self):
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        # [seq_len, bs, num_heads, head_dim]
        qkv_shape = [self._seq_len, self._bs, self._num_heads, self._head_dim]
        q = paddle.randn(qkv_shape, self._dtype)
        q.stop_gradient = False

        dist_q = dist.shard_tensor(q, self._mesh, dist.Shard(1))
        dist_q.stop_gradient = False

        k = paddle.randn(qkv_shape, self._dtype)
        k.stop_gradient = False
        dist_k = dist.shard_tensor(k, self._mesh, dist.Shard(2))
        dist_k.stop_gradient = False

        sin = paddle.randn(self._sin_cos_shape, self._dtype)
        sin.stop_gradient = True
        dist_sin = dist.shard_tensor(sin, self._mesh, dist.Replicate())
        dist_sin.stop_gradient = True

        cos = paddle.randn(self._sin_cos_shape, self._dtype)
        cos.stop_gradient = True
        dist_cos = dist.shard_tensor(cos, self._mesh, dist.Replicate())
        dist_cos.stop_gradient = True

        position_ids = paddle.arange(self._seq_len, dtype="int64").expand(
            (self._bs, self._seq_len)
        )
        position_ids.stop_gradient = True
        dist_position_ids = dist.shard_tensor(
            position_ids, self._mesh, dist.Shard(0)
        )
        dist_position_ids.stop_gradient = True

        dist_out_q, dist_out_k, _ = fused_rotary_position_embedding(
            q=dist_q,
            k=dist_k,
            sin=dist_sin,
            cos=dist_cos,
            position_ids=dist_position_ids,
            use_neox_rotary_style=False,
            time_major=True,
        )
        out_q, out_k, _ = fused_rotary_position_embedding(
            q=q,
            k=k,
            sin=sin,
            cos=cos,
            position_ids=position_ids,
            use_neox_rotary_style=False,
            time_major=True,
        )

        self.check_tensor_eq(out_q, dist_out_q)
        self.check_tensor_eq(out_k, dist_out_k)

        dist_out = dist_out_q + dist_out_k
        out = out_q + out_k
        dist_out.backward()
        out.backward()

        self.check_tensor_eq(dist_q.grad, q.grad)
        self.check_tensor_eq(dist_k.grad, k.grad)

    def test_common_case_time_major_shard_seq(self):
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        # [seq_len, bs, num_heads, head_dim]
        qkv_shape = [self._seq_len, self._bs, self._num_heads, self._head_dim]
        q = paddle.randn(qkv_shape, self._dtype)
        q.stop_gradient = False

        dist_q = dist.shard_tensor(q, self._mesh, dist.Shard(0))
        dist_q.stop_gradient = False

        k = paddle.randn(qkv_shape, self._dtype)
        k.stop_gradient = False
        dist_k = dist.shard_tensor(k, self._mesh, dist.Shard(2))
        dist_k.stop_gradient = False

        sin = paddle.randn(self._sin_cos_shape, self._dtype)
        sin.stop_gradient = True
        dist_sin = dist.shard_tensor(sin, self._mesh, dist.Replicate())
        dist_sin.stop_gradient = True

        cos = paddle.randn(self._sin_cos_shape, self._dtype)
        cos.stop_gradient = True
        dist_cos = dist.shard_tensor(cos, self._mesh, dist.Replicate())
        dist_cos.stop_gradient = True

        dist_out_q, dist_out_k, _ = fused_rotary_position_embedding(
            q=dist_q,
            k=dist_k,
            sin=dist_sin,
            cos=dist_cos,
            position_ids=None,
            use_neox_rotary_style=False,
            time_major=True,
        )
        out_q, out_k, _ = fused_rotary_position_embedding(
            q=q,
            k=k,
            sin=sin,
            cos=cos,
            position_ids=None,
            use_neox_rotary_style=False,
            time_major=True,
        )

        self.check_placements(dist_out_q, [dist.Shard(0)])
        self.check_placements(dist_out_k, [dist.Shard(0)])

        self.check_tensor_eq(out_q, dist_out_q)
        self.check_tensor_eq(out_k, dist_out_k)

        dist_out = dist_out_q + dist_out_k
        out = out_q + out_k
        dist_out.backward()
        out.backward()

        self.check_tensor_eq(dist_q.grad, q.grad)
        self.check_tensor_eq(dist_k.grad, k.grad)

    def run_test_case(self):
        if self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError(
                "fused_rotary_position_embedding only support gpu backend."
            )

        self.test_only_q_input()
        self.test_only_q_input_time_major()
        self.test_common_case()
        self.test_common_case(is_gqa=True)
        self.test_common_case_time_major()
        self.test_common_case_time_major_shard_seq()


if __name__ == '__main__':
    TestFusedRopeApiForSemiAutoParallel().run_test_case()
