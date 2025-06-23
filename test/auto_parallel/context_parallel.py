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


import os
import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.ring_attention import (
    shard_seq_load_balance,
    unshard_seq_load_balance,
)

dist.init_parallel_env()


class TestContextParallel:
    def __init__(self):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._sep_mesh = dist.ProcessMesh(
            list(range(self.world_size)), dim_names=["sep"]
        )

    def set_seed(self, seed):
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _test_cp_base(
        self,
        is_causal=True,
    ):
        mesh = dist.ProcessMesh(list(range(self.world_size)), dim_names=['sep'])
        dist.auto_parallel.set_mesh(mesh)
        self.set_seed(1024)
        bs = 2
        seq_len = 256  # flash_attn seq_len/card > 128
        dim = 16
        nheads = 2
        dtype = paddle.bfloat16
        q = paddle.rand(
            (bs, seq_len, nheads, dim),
            dtype=dtype,
        )
        k = paddle.rand(
            (bs, seq_len, nheads, dim),
            dtype=dtype,
        )
        v = paddle.rand(
            (bs, seq_len, nheads, dim),
            dtype=dtype,
        )
        q.stop_gradient = False
        k.stop_gradient = False
        v.stop_gradient = False

        with paddle.no_grad():
            dist.broadcast(q, src=0)
            dist.broadcast(k, src=0)
            dist.broadcast(v, src=0)
        # base compute
        output_ref = paddle.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal
        )
        loss_ref = output_ref.mean()
        loss_ref.backward()

        cp_q = q.detach().clone()
        cp_k = k.detach().clone()
        cp_v = v.detach().clone()
        placements = [dist.Replicate() for _ in range(len(mesh.dim_names))]

        # shard compute
        sharded_q = dist.shard_tensor(cp_q, mesh, placements)
        sharded_k = dist.shard_tensor(cp_k, mesh, placements)
        sharded_v = dist.shard_tensor(cp_v, mesh, placements)
        sharded_q = shard_seq_load_balance(sharded_q, 1)
        sharded_k = shard_seq_load_balance(sharded_k, 1)
        sharded_v = shard_seq_load_balance(sharded_v, 1)
        sharded_q.stop_gradient = False
        sharded_k.stop_gradient = False
        sharded_v.stop_gradient = False

        output_sharded = paddle.nn.functional.scaled_dot_product_attention(
            sharded_q, sharded_k, sharded_v, is_causal=is_causal, backend='p2p'
        )
        loss_sharded = paddle.mean(output_sharded)
        loss_sharded.backward()

        with paddle.no_grad():
            reorder_t = unshard_seq_load_balance(output_sharded, 1)
        np.testing.assert_allclose(
            loss_ref.numpy(), loss_sharded.numpy(), rtol=5e-06, atol=5e-06
        )
        np.testing.assert_allclose(
            output_ref.to("float32").numpy(),
            reorder_t.to("float32").numpy(),
            rtol=2e-01,
            atol=6e-02,
        )

        with paddle.no_grad():
            reorder_q_grad = unshard_seq_load_balance(sharded_q.grad, 1)
            reorder_k_grad = unshard_seq_load_balance(sharded_k.grad, 1)
            reorder_v_grad = unshard_seq_load_balance(sharded_v.grad, 1)

        rtol = 3e-05
        atol = 3e-05
        np.testing.assert_allclose(
            q.grad.to("float32").numpy(),
            reorder_q_grad.to("float32").numpy(),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            k.grad.to("float32").numpy(),
            reorder_k_grad.to("float32").numpy(),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            v.grad.to("float32").numpy(),
            reorder_v_grad.to("float32").numpy(),
            rtol=rtol,
            atol=atol,
        )

    def run_test_cases(self):
        # flash attention is not supported yet for cpu
        if os.getenv("backend") == "gpu":
            cuda_version_main = int(paddle.version.cuda().split(".")[0])
            device_prop_main = paddle.device.cuda.get_device_capability()[0]
            if cuda_version_main >= 11 and device_prop_main >= 8:
                self._test_cp_base()
                self._test_cp_base(is_causal=False)


if __name__ == '__main__':
    tester = TestContextParallel()
    tester.run_test_cases()
