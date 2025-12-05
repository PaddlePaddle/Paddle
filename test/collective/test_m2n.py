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

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.communication import deep_ep

num_max_tokens = 512


def bench_split(
    fn1,
    fn2,
    fn1_wait: bool = True,
    fn2_wait: bool = True,
    num_warmups: int = 50,
    num_tests: int = 50,
):
    # clear
    cache = paddle.empty((int(256e6 // 4),), dtype="int32")
    cache.zero_()

    # Warmup
    for _ in range(num_warmups):
        dist.barrier()
        req = fn1()
        if fn1_wait:
            req.wait()
        dist.barrier()
        req = fn2()
        if fn2_wait:
            req.wait()
        dist.barrier()

    # Flush L2
    cache.zero_()
    del cache

    # Testing
    start_events_fn1 = [
        paddle.device.Event(enable_timing=True) for _ in range(num_tests)
    ]
    end_events_fn1 = [
        paddle.device.Event(enable_timing=True) for _ in range(num_tests)
    ]
    start_events_fn2 = [
        paddle.device.Event(enable_timing=True) for _ in range(num_tests)
    ]
    end_events_fn2 = [
        paddle.device.Event(enable_timing=True) for _ in range(num_tests)
    ]
    for i in range(num_tests):
        # Record
        dist.barrier()
        start_events_fn1[i].record()
        req = fn1()
        end_events_fn1[i].record()
        if fn1_wait:
            req.wait()
        dist.barrier()
        start_events_fn2[i].record()
        req = fn2()
        end_events_fn2[i].record()
        if fn2_wait:
            req.wait()
        dist.barrier()
    paddle.device.synchronize()

    times_fn1 = np.array(
        [
            s.elapsed_time(e) / 1e3
            for s, e in zip(start_events_fn1, end_events_fn1)
        ]
    )[1:]
    times_fn2 = np.array(
        [
            s.elapsed_time(e) / 1e3
            for s, e in zip(start_events_fn2, end_events_fn2)
        ]
    )[1:]
    return (
        np.average(times_fn1),
        np.min(times_fn1),
        np.max(times_fn1),
        np.average(times_fn2),
        np.min(times_fn2),
        np.max(times_fn2),
    )


def bench_m2n(fn, num_warmups: int = 50, num_tests: int = 50):
    # clear
    cache = paddle.empty((int(256e6 // 4),), dtype="int32")
    cache.zero_()

    # Warmup
    for _ in range(num_warmups):
        dist.barrier()
        fn()
        dist.barrier()

    # Flush L2
    cache.zero_()
    del cache

    # Testing
    start_events_fn = [
        paddle.device.Event(enable_timing=True) for _ in range(num_tests)
    ]
    end_events_fn = [
        paddle.device.Event(enable_timing=True) for _ in range(num_tests)
    ]
    for i in range(num_tests):
        dist.barrier()
        start_events_fn[i].record()
        fn()
        end_events_fn[i].record()
        dist.barrier()
    paddle.device.synchronize()

    times_fn = np.array(
        [
            s.elapsed_time(e) / 1e3
            for s, e in zip(start_events_fn, end_events_fn)
        ]
    )[1:]
    return (
        np.average(times_fn),
        np.min(times_fn),
        np.max(times_fn),
    )


def per_token_cast_back(x_fp8: paddle.Tensor, x_scales: paddle.Tensor):
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
    a_start_rank: int,
    a_num_ranks: int,
    e_start_rank: int,
    e_num_ranks: int,
    group: dist.communication.group,
    buffer: deep_ep.Buffer,
    seed: int = 0,
):
    paddle.seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % e_num_ranks == 0
    num_local_experts = num_experts // e_num_ranks
    num_rdma_ranks = num_ranks / 8

    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, (
        'Too many ranks (exceeding test precision limit)'
    )

    x = paddle.ones((num_tokens, hidden), dtype="bfloat16") * (
        rank - rank_offset
    )
    # x[:, -128:] = paddle.arange(0, num_tokens, dtype="bfloat16").view((-1, 1))
    # x = paddle.randn((num_tokens, hidden), dtype="bfloat16")
    # x = paddle.ones((num_tokens, hidden), dtype="bfloat16") * 3
    topk_idx = paddle.randint(
        0, num_experts, shape=[num_tokens, num_topk], dtype="int64"
    )
    print(f"rank: {rank}, num_local_experts: {num_local_experts}")
    topk_weights = paddle.randn((num_tokens, num_topk), dtype="float32").abs_()
    # topk_weights = paddle.ones((num_tokens, num_topk), dtype="float32") * 5
    print("x: ", x, flush=True)
    print("topk_idx: ", topk_idx, flush=True)
    print("topk_weights: ", topk_weights, flush=True)

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections

    paddle.device.synchronize()
    dist.barrier()
    run_time = 1
    print("run_time: ", run_time)
    print("num_experts: ", num_experts)

    ref_recv_x = paddle.zeros(
        (e_num_ranks, num_local_experts, hidden), dtype=paddle.float32
    )  # [8, 3, 128]
    gbl_recv_x = paddle.zeros(
        (e_num_ranks, num_local_experts, hidden), dtype=paddle.float32
    )  # [8, 3, 128]
    ref_combin_x = paddle.zeros(
        (num_tokens, hidden), dtype=paddle.float32
    )  # [96, 8192]
    gbl_combin_x = paddle.zeros(
        (num_tokens, hidden), dtype=paddle.float32
    )  # [96, 8192]

    if rank >= a_start_rank and rank < a_start_rank + a_num_ranks:
        if not use_fp8:
            ref_recv_x.zero_()
            gbl_recv_x.zero_()
            ref_combin_x.zero_()
            gbl_combin_x.zero_()
            for i in range(num_tokens):
                for k, expert_id in enumerate(topk_idx[i]):
                    if expert_id == -1:
                        continue
                    erank_id = expert_id // num_local_experts  # 0-7
                    local_expert_id = expert_id % num_local_experts  # 0-2
                    ref_recv_x[erank_id, local_expert_id] += x[i].to(
                        paddle.float32
                    )
                    ref_combin_x[i] += (
                        x[i].to(paddle.float32) * topk_weights[i][k]
                    )

        packed_recv_x, handle, event, req = buffer.a2e_isend_two_stage(
            x,
            topk_idx,
            topk_weights,
            num_max_tokens,
            num_experts,
            use_fp8=use_fp8,
        )

        req.wait()
        dist.barrier()

        e2a_x, event, req = buffer.e2a_irecv_two_stage(
            topk_idx,
            topk_weights,
            handle,
            dispatch_use_fp8=use_fp8,
            out=None,
        )

        req.wait()
        dist.barrier()

        gbl_combin_x = e2a_x.to(paddle.float32)

        def a2e_isend_func():
            packed_recv_x, handle, event, req = buffer.a2e_isend_two_stage(
                x,
                topk_idx,
                topk_weights,
                num_max_tokens,
                num_experts,
                use_fp8=use_fp8,
            )
            return req

        def e2a_irecv_func():
            e2a_x, event, req = buffer.e2a_irecv_two_stage(
                topk_idx,
                topk_weights,
                handle,
                dispatch_use_fp8=use_fp8,
                out=None,
            )
            req.wait()
            return req

        avg_t_fn1, min_t_fn1, max_t_fn1, avg_t_fn2, min_t_fn2, max_t_fn2 = (
            bench_split(
                a2e_isend_func, e2a_irecv_func, fn1_wait=True, fn2_wait=False
            )
        )
        print(
            f'[rank: {rank}][a2e_isend_two_stage] '
            f'avg_t: {avg_t_fn1 * 1e6:.2f} us, min_t: {min_t_fn1 * 1e6:.2f} us, max_t: {max_t_fn1 * 1e6:.2f} us',
            flush=True,
        )
        print(
            f'[rank: {rank}][e2a_irecv_two_stage] '
            f'avg_t: {avg_t_fn2 * 1e6:.2f} us, min_t: {min_t_fn2 * 1e6:.2f} us, max_t: {max_t_fn2 * 1e6:.2f} us',
            flush=True,
        )

    if rank >= e_start_rank and rank < e_start_rank + e_num_ranks:
        (
            packed_recv_x,
            packed_recv_count,
            rdma_send_flags,
            handle,
            event,
            req,
        ) = buffer.a2e_irecv_two_stage(
            hidden,
            num_topk,
            num_max_tokens,
            num_experts,
            use_fp8=use_fp8,
        )
        req.wait()
        print(
            f'[rank: {rank}, packed_recv_count: {packed_recv_count}], packed_recv_x[1]: {packed_recv_x[1]}',
            flush=True,
        )
        dist.barrier()

        if not use_fp8:
            for local_expert_id in range(num_local_experts):
                gbl_recv_x[rank - e_start_rank, local_expert_id] = (
                    packed_recv_x[
                        local_expert_id, : packed_recv_count[local_expert_id]
                    ]
                    .to(paddle.float32)
                    .sum(0)
                )

        # e2a isend
        if use_fp8:
            simulated_gemm_x = per_token_cast_back(
                packed_recv_x[0].view((-1, hidden)),
                packed_recv_x[1].contiguous().view((-1, hidden // 128)),
            ).view(packed_recv_x[0].shape)
        else:
            simulated_gemm_x = packed_recv_x.clone()

        event, req = buffer.e2a_isend_two_stage(
            simulated_gemm_x,
            num_topk,
            handle,
            dispatch_use_fp8=use_fp8,
            out=None,
        )

        req.wait()
        dist.barrier()

        def a2e_irecv_func():
            (
                packed_recv_x,
                packed_recv_count,
                rdma_send_flags,
                handle,
                event,
                req,
            ) = buffer.a2e_irecv_two_stage(
                hidden,
                num_topk,
                num_max_tokens,
                num_experts,
                use_fp8=use_fp8,
            )
            # event.current_stream_wait()
            req.wait()
            return req

        def e2a_isend_func():
            event, req = buffer.e2a_isend_two_stage(
                simulated_gemm_x,
                num_topk,
                handle,
                dispatch_use_fp8=use_fp8,
                out=None,
            )
            return req

        avg_t_fn1, min_t_fn1, max_t_fn1, avg_t_fn2, min_t_fn2, max_t_fn2 = (
            bench_split(
                a2e_irecv_func, e2a_isend_func, fn1_wait=False, fn2_wait=True
            )
        )
        print(
            f'[rank: {rank}][a2e_irecv_two_stage] '
            f'avg_t: {avg_t_fn1 * 1e6:.2f} us, min_t: {min_t_fn1 * 1e6:.2f} us, max_t: {max_t_fn1 * 1e6:.2f} us',
            flush=True,
        )
        print(
            f'[rank: {rank}][e2a_isend_two_stage] '
            f'avg_t: {avg_t_fn2 * 1e6:.2f} us, min_t: {min_t_fn2 * 1e6:.2f} us, max_t: {max_t_fn2 * 1e6:.2f} us',
            flush=True,
        )

    if not use_fp8:
        dist.all_reduce(ref_recv_x, group=group)
        dist.all_reduce(gbl_recv_x, group=group)
        assert paddle.allclose(ref_recv_x, gbl_recv_x, rtol=1e-3, atol=1e-3), (
            f"[rank: {rank}], ref_recv_x: {ref_recv_x}, gbl_recv_x: {gbl_recv_x}"
        )
        print(
            f"[rank: {rank}], ref_recv_x: {ref_recv_x}, gbl_recv_x: {gbl_recv_x}"
        )
        assert paddle.allclose(
            ref_combin_x, gbl_combin_x, rtol=1.0, atol=1.0
        ), (
            f"[rank: {rank}], ref_combin_x: {ref_combin_x}, gbl_combin_x: {gbl_combin_x}"
        )
        print(
            f"[rank: {rank}], ref_combin_x: {ref_combin_x}, gbl_combin_x: {gbl_combin_x}"
        )
        print(f"rank: {rank} passed the check")
    dist.barrier()


def test_loop():
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    group = paddle.distributed.new_group(range(num_ranks))
    print("rank: ", rank, flush=True)
    print("num_ranks: ", num_ranks, flush=True)

    a_start_rank = 0
    a_num_ranks = 16
    e_start_rank = a_start_rank + a_num_ranks
    e_num_ranks = num_ranks - a_num_ranks
    # 64 * 3 / 48 = 4
    # 64 * 3 / 32 = 6
    # 64 * 3 / 24 = 8
    # 64 * 3 / 12 = 16
    num_tokens, hidden, num_topk, num_experts = 96, 8192, 8, 64

    assert num_tokens <= num_max_tokens, (
        "num_tokens must be less equal to num_max_tokens"
    )
    num_rdma_ranks = num_ranks / 8
    num_local_experts = num_experts / num_ranks
    num_rdma_bytes = deep_ep.M2NBuffer.get_low_latency_rdma_size_hint_two_stage(
        num_max_tokens,
        hidden,
        num_ranks,
        a_num_ranks,
        e_num_ranks,
        num_experts,
        num_topk,
    )

    use_fp8 = True
    num_nvl_bytes = deep_ep.M2NBuffer.get_low_latency_nvl_size_hint_two_stage(
        num_max_tokens,
        hidden,
        num_ranks,
        a_num_ranks,
        e_num_ranks,
        num_experts,
        num_topk,
        use_fp8,
    )
    print(
        f'Allocating rdma buffer size: {num_rdma_bytes / 1e6} MB, nvl buffer size: {num_nvl_bytes / 1e6} MB...',
        flush=True,
    )

    buffer = deep_ep.M2NBuffer(
        group,
        a_start_rank,
        a_num_ranks,
        e_start_rank,
        e_num_ranks,
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
        a_start_rank,
        a_num_ranks,
        e_start_rank,
        e_num_ranks,
        group,
        buffer,
        seed=1,
    )


def init_dist_env(world_size, seed=20):
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
