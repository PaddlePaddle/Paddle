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

import contextlib
import random
from functools import partial

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.communication import deep_ep


def bench_paddle(fn, num_warmups: int = 50, num_tests: int = 50):
    # clear
    cache = paddle.empty((int(256e6 // 4),), dtype="int32")
    cache.zero_()

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()
    del cache

    # Testing
    start_events = [
        paddle.device.Event(enable_timing=True) for _ in range(num_tests)
    ]
    end_events = [
        paddle.device.Event(enable_timing=True) for _ in range(num_tests)
    ]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
    paddle.device.synchronize()

    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )[1:]
    return np.average(times), np.min(times), np.max(times)


def paddle_per_token_cast_back(x_fp8: paddle.Tensor, x_scales: paddle.Tensor):
    x_fp32 = x_fp8.to("float32").view((x_fp8.shape[0], -1, 128))
    x_scales = x_scales.view((x_fp8.shape[0], -1, 1))
    return (x_fp32 * x_scales).view(x_fp8.shape).to("bfloat16")


def test_main(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    use_fp8: bool,
    rank: int,
    num_ranks: int,
    group: dist.communication.group,
    buffer: deep_ep.Buffer,
    seed: int = 0,
):
    paddle.seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks
    num_rdma_ranks = num_ranks / 8

    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert (
        num_ranks - rank_offset < 257
    ), 'Too many ranks (exceeding test precision limit)'

    x = paddle.ones((num_tokens, hidden), dtype="bfloat16") * (
        rank - rank_offset
    )
    x[:, -128:] = paddle.arange(0, num_tokens, dtype="bfloat16").view((-1, 1))
    scores = paddle.randn((num_tokens, num_experts), dtype="float32").abs_() + 1
    topk_idx = paddle.topk(
        scores, num_topk, axis=-1, largest=True, sorted=True
    )[1]
    topk_weights = paddle.randn((num_tokens, num_topk), dtype="float32").abs_()
    print("x: ", x, flush=True)
    print("scores: ", scores, flush=True)
    print("topk_idx: ", topk_idx, flush=True)
    print("topk_weights: ", topk_weights, flush=True)

    # Randomly mask some positions
    for i in range(10):
        topk_idx[
            random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)
        ] = -1

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections

    paddle.device.synchronize()
    dist.barrier()
    do_check = True
    hash_value, num_times = 0, 0
    all_times = 10000
    warp_up_time = 1
    for return_recv_hook in (False,):
        print(f"rank: {rank}, use_fp8: {use_fp8}", flush=True)
        for i in range(warp_up_time):
            (
                packed_recv_x,
                packed_recv_count,
                rdma_send_flags,
                handle,
                event,
                hook,
            ) = buffer.low_latency_dispatch_two_stage(
                x,
                topk_idx,
                topk_weights,
                num_tokens,
                num_experts,
                use_fp8=use_fp8,
                async_finish=not return_recv_hook,
                return_recv_hook=return_recv_hook,
            )
            packed_recv_x = (
                (packed_recv_x[0], packed_recv_x[1].contiguous())
                if use_fp8
                else packed_recv_x
            )
            if use_fp8:
                print(
                    "packed_recv_x: ", paddle.cast(packed_recv_x[0], "float32")
                )
            else:
                print("packed_recv_x: ", paddle.cast(packed_recv_x, "float32"))
            out = paddle.empty((num_tokens, hidden), dtype="bfloat16")
            if use_fp8:
                simulated_gemm_x = paddle_per_token_cast_back(
                    packed_recv_x[0].view((-1, hidden)),
                    packed_recv_x[1].view((-1, hidden // 128)),
                ).view(packed_recv_x[0].shape)
            else:
                simulated_gemm_x = packed_recv_x.clone()
            combined_x, event, hook = buffer.low_latency_combine_two_stage(
                simulated_gemm_x,
                topk_idx,
                topk_weights,
                handle,
                async_finish=not return_recv_hook,
                dispatch_use_fp8=use_fp8,
                return_recv_hook=return_recv_hook,
                out=out,
            )
            print(f"rank: {rank}, combined_x: {combined_x}", flush=True)
        dist.barrier()
        paddle.device.synchronize()

        def test_func(return_recv_hook: bool):
            (
                packed_recv_x,
                packed_recv_count,
                rdma_send_flags,
                handle,
                event,
                hook,
            ) = buffer.low_latency_dispatch_two_stage(
                x,
                topk_idx,
                topk_weights,
                num_tokens,
                num_experts,
                use_fp8=use_fp8,
                async_finish=not return_recv_hook,
                return_recv_hook=return_recv_hook,
            )
            combined_x, event, hook = buffer.low_latency_combine_two_stage(
                simulated_gemm_x,
                topk_idx,
                topk_weights,
                handle,
                async_finish=not return_recv_hook,
                dispatch_use_fp8=use_fp8,
                return_recv_hook=return_recv_hook,
                out=out,
            )

        # dispatch + combine
        avg_t, min_t, max_t = bench_paddle(
            partial(test_func, return_recv_hook=False),
            num_warmups=200,
            num_tests=10000,
        )
        print(
            f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
            f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us',
            flush=True,
        )
    return hash_value


# noinspection PyUnboundLocalVariable
def test_loop():
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    group = paddle.distributed.new_group(range(num_ranks))
    print("rank: ", rank, flush=True)
    print("num_ranks: ", num_ranks, flush=True)

    num_tokens, hidden, num_topk, num_experts = 128, 7168, 8, 64
    num_rdma_ranks = num_ranks / 8
    num_local_experts = num_experts / num_ranks
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint_two_stage(
        num_tokens, hidden, num_ranks, num_experts, num_topk
    )
    use_fp8 = True
    num_nvl_bytes = deep_ep.Buffer.get_low_latency_nvl_size_hint_two_stage(
        num_tokens, hidden, num_ranks, num_experts, num_topk, use_fp8
    )
    print(
        f'Allocating rdma buffer size: {num_rdma_bytes / 1e6} MB, nvl buffer size: {num_nvl_bytes / 1e6} MB...',
        flush=True,
    )
    buffer = deep_ep.Buffer(
        group,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_rdma_ranks,
    )
    test_main(
        num_tokens,
        hidden,
        num_experts,
        num_topk,
        use_fp8,
        rank,
        num_ranks,
        group,
        buffer,
        seed=1,
    )


def init_dist_env(world_size, seed=20):
    """
    初始化分布式环境。

    Args:
        world_size (int): 分布式训练任务的机器数量。
        seed (int): 随机种子。默认值为20。

    Returns:
        None。
    """
    context = contextlib.nullcontext()
    with context:
        # start to init distributed env
        strategy = fleet.DistributedStrategy()

        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": world_size,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        strategy.tensor_parallel_configs = {"tensor_init_seed": seed}

        fleet.init(is_collective=True, strategy=strategy)


if __name__ == '__main__':
    if dist.get_world_size() > 1:
        init_dist_env(dist.get_world_size())
    test_loop()
