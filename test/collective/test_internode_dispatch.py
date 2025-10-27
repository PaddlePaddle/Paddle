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

import hashlib
import os
import time

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.communication import deep_ep
from paddle.distributed.communication.group import Group

num_tokens = 4096
hidden = 7168
num_topk = 8
num_topk_groups = 4
num_experts = 256


def test_main(
    num_sms: int,
    local_rank: int,
    num_ranks: int,
    rank: int,
    buffer: deep_ep.Buffer,
    group: Group,
):
    # Settings
    assert num_experts % num_ranks == 0
    min_local_expert_id = (256 // num_ranks) * rank
    max_local_expert_id = (256 // num_ranks) * (rank + 1)
    if local_rank == 0:
        print(
            f'[config] num_tokens={num_tokens}, hidden={hidden}, '
            f'num_topk_groups={num_topk_groups}, num_topk={num_topk}',
            flush=True,
        )

    ############################################################################
    # random data
    ############################################################################

    paddle.seed(2025)
    global_x = paddle.randn(
        shape=[num_ranks, num_tokens, hidden], dtype=paddle.bfloat16
    )
    global_scores = paddle.randn(shape=[num_ranks, num_tokens, num_experts])
    (
        global_topk_weights,  # [num_ranks, num_tokens, num_topk]
        global_topk_idx,  # [num_ranks, num_tokens, num_topk]
    ) = paddle.topk(
        global_scores, num_topk, axis=-1, largest=True, sorted=False
    )

    x = global_x[rank]
    topk_idx = global_topk_idx[rank]
    topk_weights = global_topk_weights[rank]

    ############################################################################
    # get dispatch layout
    ############################################################################

    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)

    paddle.distributed.barrier(group)
    time.sleep(1)

    ############################################################################
    # do dispatching
    ############################################################################

    config = deep_ep.Buffer.get_dispatch_config(num_ranks)

    dispatch_args = {
        'x': x,
        'topk_idx': topk_idx,
        'topk_weights': topk_weights,
        'num_tokens_per_rank': num_tokens_per_rank,
        'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
        'is_token_in_rank': is_token_in_rank,
        'num_tokens_per_expert': num_tokens_per_expert,
        'config': config,
        'async_finish': False,
    }

    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_num_tokens_per_expert_list,
        handle,
        event,
    ) = buffer.dispatch(**dispatch_args)

    ############################################################################
    # validate result
    ############################################################################

    recv_tokens_md5sum = {
        hashlib.md5(token.tobytes()).hexdigest() for token in recv_x.numpy()
    }
    global_x_np = global_x.numpy()

    for src_rank in range(num_ranks):
        for src_token_idx in range(num_tokens):
            src_topk_idx = global_topk_idx[src_rank, src_token_idx]
            if paddle.any(
                (src_topk_idx >= min_local_expert_id)
                & (src_topk_idx < max_local_expert_id)
            ):
                src_token_md5sum = hashlib.md5(
                    global_x_np[src_rank, src_token_idx].tobytes()
                ).hexdigest()
                assert src_token_md5sum in recv_tokens_md5sum, (
                    f"Not receiving token from rank={src_rank} idx={src_token_idx}"
                )
                recv_tokens_md5sum.remove(src_token_md5sum)

    assert not recv_tokens_md5sum, (
        f"Unexpected tokens not owed to any source: {len(recv_tokens_md5sum)}"
    )


def test_loop(num_local_ranks):
    hcg = fleet.get_hybrid_communicate_group()
    ep_group = hcg.get_model_parallel_group()

    num_ranks = dist.get_world_size(ep_group)
    rank = dist.get_rank(ep_group)

    num_nodes = int(num_ranks / 8)
    local_rank = rank % 8
    print(
        f'local_rank:{local_rank}, num_local_ranks:{num_local_ranks}, '
        f'num_ranks:{num_ranks}, rank:{rank}'
    )

    assert num_local_ranks == 8 and num_ranks > 8

    for i in (10, 12, 14, 16, 18, 20):
        buffer = deep_ep.Buffer(
            ep_group,
            int(1e9),
            int(1e9),
            low_latency_mode=False,
            num_qps_per_rank=i,
        )
        test_main(
            i,
            local_rank,
            num_ranks,
            rank,
            buffer,
            ep_group,
        )
        del buffer


if __name__ == '__main__':
    num_processes = 8
    world_size = int(os.getenv('WORLD_SIZE', 1))
    mp_degree = world_size * num_processes
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "mp_degree": mp_degree,
    }
    fleet.init(is_collective=True, strategy=strategy)
    test_loop(num_processes)
