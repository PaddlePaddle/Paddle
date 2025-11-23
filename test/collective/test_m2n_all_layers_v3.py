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
import time

import paddle
import paddle.distributed as dist
from paddle import Tensor
from paddle.distributed import fleet
from paddle.distributed.communication import deep_ep
from paddle.incubate.fp8 import deep_gemm
from paddle.incubate.fp8.deep_gemm import (
    ceil_div,
    get_col_major_tma_aligned_tensor,
)

num_max_tokens = 512

M2N_DEBUG = False
M2N_ACC_DEBUG = False
M2N_DEVICE_SYNC = False


def per_token_cast_to_fp8(x: Tensor) -> tuple[Tensor, Tensor]:
    assert x.dim() == 2 and x.shape[1] % 128 == 0
    m, n = x.shape
    x_view = paddle.view(x, (m, -1, 128))
    x_abs = paddle.abs(x_view).astype(paddle.float32)
    x_amax = paddle.amax(x_abs, axis=2)
    x_amax = paddle.view(x_amax, (m, -1))
    x_amax = paddle.clip(x_amax, min=1e-4)
    scaled_x = x_view * (448.0 / x_amax.unsqueeze(2))
    scaled_x_converted = paddle.view(
        scaled_x.astype(paddle.float8_e4m3fn), (m, n)
    )

    x_amax_scaled = paddle.view((x_amax / 448.0), (m, -1))

    result = (scaled_x_converted, x_amax_scaled)
    return result


def per_block_cast_to_fp8(x: Tensor) -> tuple[Tensor, Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = paddle.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype
    )
    x_padded[:m, :n] = x
    x_view = paddle.view(x_padded, (-1, 128, x_padded.shape[1] // 128, 128))

    x_abs = paddle.abs(x_view).astype(paddle.float32)
    x_amax = paddle.amax(x_abs, axis=(1, 3), keepdim=True)
    x_amax = paddle.clip(x_amax, min=1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).astype(paddle.float8_e4m3fn)

    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (
        paddle.view(x_amax / 448.0, (x_view.shape[0], x_view.shape[2]))
    )


def construct(
    x: Tensor, y: Tensor
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor, Tensor]:
    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8


def per_token_cast_back(x_fp8: paddle.Tensor, x_scales: paddle.Tensor):
    x_fp32 = x_fp8.to("float32").view((x_fp8.shape[0], -1, 128))
    x_scales = x_scales.view((x_fp8.shape[0], -1, 1))
    return (x_fp32 * x_scales).view(x_fp8.shape).to("bfloat16")


A = paddle.randn((96, 7168), dtype="bfloat16")
B = paddle.randn((7168, 7168), dtype="bfloat16")
C = paddle.randn((96, 7168), dtype="bfloat16")

A_fp8, B_fp8 = construct(A, B)


def moe(x: Tensor, y: Tensor):
    [paddle.matmul(x, y) for _ in range(9)]
    return paddle.matmul(x, y)


def moe_fp8(x_fp8: Tensor, y_fp8: Tensor, out: Tensor):
    deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out, num_sms=108)
    [
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out, num_sms=108)
        for i in range(9)
    ]


def attention(x: Tensor, y: Tensor):
    return moe(x, y)


def attention_fp8(x_fp8: Tensor, y_fp8: Tensor, out: Tensor):
    deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out, num_sms=108)
    [
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out, num_sms=108)
        for i in range(9)
    ]


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

    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, (
        'Too many ranks (exceeding test precision limit)'
    )

    intermediate_size = hidden  # 28672
    num_micro_batches = 3
    GB = num_tokens * 3
    MB = num_tokens
    num_hidden_layers = 51
    moe_layer_start_index = 0
    num_benches = -1

    # x_fp8, y_fp8 = construct(x, y)
    # m, k = x.shape
    # n, k = y.shape
    # out = paddle.empty((m, n), dtype=paddle.bfloat16)

    # 整体思路
    # 1. 单层循环
    # 2. 以计算index为基准，通信index进行相应的偏移
    # 3. a2e 计算放到循环的开始位置, 最后一个micro batch循环不到, 放到循环结束单独处理
    # 4. e2a 计算放到循环的结束位置, 第一micro batch循环不到，放到循环开始之前单独处理
    # 5. 只在通信index有效的位置进行通信操作
    if rank >= a_start_rank and rank < a_start_rank + a_num_ranks:
        # x =
        xs = [
            paddle.ones((num_tokens, hidden), dtype="bfloat16") * (i + 2)
            for i in range(num_micro_batches)
        ]
        weights = paddle.eye(intermediate_size, hidden, dtype="bfloat16")

        topk_idx = paddle.randint(
            0, num_experts, shape=[num_tokens, num_topk], dtype="int64"
        )
        print(f"rank: {rank}, num_local_experts: {num_local_experts}")
        topk_weights = paddle.ones(
            (num_tokens, num_topk), dtype="float32"
        ).abs_()  # / num_topk

        a2e_send_result = [None] * num_micro_batches
        e2a_recv_result = [None] * num_micro_batches
        # for i in range(num_benches):
        i = -1
        while True:
            paddle.device.synchronize()
            dist.barrier()
            i += 1
            if num_benches > 0 and i >= num_benches:
                break
            # x = paddle.ones((num_tokens, hidden), dtype="bfloat16") * (
            #     rank + 1
            # )
            # loop
            for idx in range(
                moe_layer_start_index * num_micro_batches,
                num_hidden_layers * num_micro_batches,
            ):
                a2e_layer_idx = idx // num_micro_batches  # idx
                a2e_mb_idx = idx % num_micro_batches  # idx

                e2a_layer_idx_next = (
                    idx - num_micro_batches + 2
                ) // num_micro_batches  # idx - 2
                e2a_mb_idx_next = (
                    idx - num_micro_batches + 2
                ) % num_micro_batches  # idx - 2
                # attention
                # x = attention(x, weights) # 96 28672
                xs[a2e_mb_idx] = attention(xs[a2e_mb_idx], weights)
                if M2N_ACC_DEBUG:
                    print(
                        f"====== {i} compute attention {a2e_mb_idx}_{a2e_layer_idx}: {xs[a2e_mb_idx]}",
                        flush=True,
                    )

                if M2N_DEBUG:
                    print(
                        f"====== {i} compute attention {a2e_mb_idx}_{a2e_layer_idx}: {xs[a2e_mb_idx]}",
                        flush=True,
                    )

                # # attn 等待上一个micro batch数据接收完
                # if a2e_layer_idx_pre >=  moe_layer_start_index:
                #     _, _, event, hook = a2e_send_result[a2e_mb_idx_pre]
                #     # event.current_stream_wait()
                #     hook() # .current_stream_wait()
                #     if M2N_DEVICE_SYNC:
                #         paddle.device.synchronize()
                #     if M2N_DEBUG:
                #         print(f"{i} dispatch send wait attention {a2e_mb_idx_pre}_{a2e_layer_idx_pre} data end", flush=True)

                # attn 每一个micro batch均发送数据
                a2e_send_result[a2e_mb_idx] = buffer.a2e_isend_two_stage_v3(
                    xs[a2e_mb_idx],
                    topk_idx,
                    topk_weights,
                    num_max_tokens,
                    num_experts,
                    use_fp8=use_fp8,
                )
                if M2N_DEVICE_SYNC:
                    paddle.device.synchronize()
                if M2N_DEBUG:
                    print(
                        f"{i} dispatch send attention {a2e_mb_idx}_{a2e_layer_idx} data begin",
                        flush=True,
                    )

                _, _, event, hook = a2e_send_result[a2e_mb_idx]
                # event.current_stream_wait()
                hook()  # .current_stream_wait()
                if M2N_DEVICE_SYNC:
                    paddle.device.synchronize()
                if M2N_DEBUG:
                    print(
                        f"{i} dispatch send wait attention {a2e_mb_idx}_{a2e_layer_idx} data end",
                        flush=True,
                    )

                # attn 最后一层不在接收数据
                if (
                    e2a_layer_idx_next >= moe_layer_start_index
                    and e2a_layer_idx_next < num_hidden_layers - 1
                ):
                    _, handle, _, _ = a2e_send_result[e2a_mb_idx_next]
                    e2a_recv_result[e2a_mb_idx_next] = (
                        buffer.e2a_irecv_two_stage_v3(
                            topk_idx,
                            topk_weights,
                            handle,
                            dispatch_use_fp8=use_fp8,
                            out=None,
                        )
                    )
                    if M2N_DEVICE_SYNC:
                        paddle.device.synchronize()
                    if M2N_DEBUG:
                        print(
                            f"{i} combine recv moe {e2a_mb_idx_next}_{e2a_layer_idx_next} data begin",
                            flush=True,
                        )

                    e2a_x, event, hook = e2a_recv_result[e2a_mb_idx_next]
                    # event.current_stream_wait()
                    hook()  # .current_stream_wait()
                    # x = e2a_x
                    # print(f"{i} combine recv wait moe {e2a_mb_idx}_{e2a_layer_idx} data end, x: {x}", flush=True)
                    xs[e2a_mb_idx_next] = e2a_x

                    if M2N_DEVICE_SYNC:
                        paddle.device.synchronize()
                    if M2N_DEBUG:
                        print(
                            f"{i} combine recv wait moe {e2a_mb_idx_next}_{e2a_layer_idx_next} data end",
                            flush=True,
                        )

            print(f"==================== {i}", flush=True)
            # time.sleep(1)

    if rank >= e_start_rank and rank < e_start_rank + e_num_ranks:
        weights = paddle.eye(intermediate_size, hidden, dtype="bfloat16")
        a2e_recv_result = [None] * num_micro_batches
        e2a_send_result = [None] * num_micro_batches
        i = -1
        # for i in range(num_benches):
        while True:
            paddle.device.synchronize()
            dist.barrier()
            i += 1
            if num_benches > 0 and i >= num_benches:
                break
            # loop
            a2e_recv_result[0] = buffer.a2e_irecv_two_stage_v3(
                hidden,
                num_topk,
                num_max_tokens,
                num_experts,
                use_fp8=use_fp8,
            )
            if M2N_DEVICE_SYNC:
                paddle.device.synchronize()
            if M2N_DEBUG:
                print(
                    f"0 dispatch recv attention {0}_{0} data begin", flush=True
                )

            # moe 每一个micro batch 都等待数据接收完
            _, _, _, _, _, hook = a2e_recv_result[0]
            # event.current_stream_wait()
            hook().current_stream_wait()

            if M2N_DEVICE_SYNC:
                paddle.device.synchronize()
            if M2N_DEBUG:
                print(f"0 dispatch recv tion {0}_{0} data end", flush=True)

            for idx in range(
                moe_layer_start_index * num_micro_batches,
                num_hidden_layers * num_micro_batches,
            ):
                a2e_layer_idx = idx // num_micro_batches
                a2e_mb_idx = idx % num_micro_batches
                a2e_layer_idx_next = (idx + 1) // num_micro_batches
                a2e_mb_idx_next = (idx + 1) % num_micro_batches

                e2a_layer_idx = idx // num_micro_batches
                e2a_mb_idx = idx % num_micro_batches

                if idx < num_hidden_layers * num_micro_batches - 1:
                    a2e_recv_result[a2e_mb_idx_next] = (
                        buffer.a2e_irecv_two_stage_v3(
                            hidden,
                            num_topk,
                            num_max_tokens,
                            num_experts,
                            use_fp8=use_fp8,
                        )
                    )
                    if M2N_DEVICE_SYNC:
                        paddle.device.synchronize()
                    if M2N_DEBUG:
                        print(
                            f"{i} dispatch recv attention {a2e_mb_idx_next}_{a2e_layer_idx_next} data begin",
                            flush=True,
                        )

                    # moe 每一个micro batch 都等待数据接收完
                    _, _, _, _, _, hook = a2e_recv_result[a2e_mb_idx_next]
                    # event.current_stream_wait()
                    hook()  # .current_stream_wait()

                    # if use_fp8:
                    #     simulated_gemm_x = per_token_cast_back(
                    #         packed_recv_x[0].view((-1, hidden)),
                    #         packed_recv_x[1].contiguous().view((-1, hidden // 128)),
                    #     ).view(packed_recv_x[0].shape)
                    # else:
                    #     simulated_gemm_x = packed_recv_x.clone()

                    # paddle.device.synchronize()
                    # print(f"dispatch recv wait attention {a2e_mb_idx}_{a2e_layer_idx} data end, packed_recv_x: {packed_recv_x}", flush=True)
                    if M2N_DEVICE_SYNC:
                        paddle.device.synchronize()
                    if M2N_DEBUG:
                        print(
                            f"{i} dispatch recv wait attention {a2e_mb_idx_next}_{a2e_layer_idx_next} data end",
                            flush=True,
                        )

                moe(A, weights)
                if M2N_DEBUG:
                    print(
                        f"====== {i} compute moe {a2e_mb_idx}_{a2e_layer_idx}",
                        flush=True,
                    )

                # moe 启动发送上一个micro batch的数据
                if (
                    e2a_layer_idx >= moe_layer_start_index
                    and e2a_layer_idx < num_hidden_layers - 1
                ):
                    (
                        packed_recv_x,
                        packed_recv_count,
                        rdma_send_flags,
                        handle,
                        _,
                        _,
                    ) = a2e_recv_result[e2a_mb_idx]
                    if use_fp8:
                        simulated_gemm_x = per_token_cast_back(
                            packed_recv_x[0].view((-1, hidden)),
                            packed_recv_x[1]
                            .contiguous()
                            .view((-1, hidden // 128)),
                        ).view(packed_recv_x[0].shape)
                    else:
                        simulated_gemm_x = packed_recv_x
                    e2a_send_result[e2a_mb_idx] = buffer.e2a_isend_two_stage_v3(
                        simulated_gemm_x,
                        num_topk,
                        handle,
                        dispatch_use_fp8=use_fp8,
                        out=None,
                    )
                    if M2N_DEVICE_SYNC:
                        paddle.device.synchronize()
                    if M2N_DEBUG:
                        print(
                            f"{i} combine send moe {e2a_mb_idx}_{e2a_layer_idx} data begin",
                            flush=True,
                        )

                    if M2N_ACC_DEBUG:
                        print(
                            f"{i} combine send moe {e2a_mb_idx}_{e2a_layer_idx} data begin, simulated_gemm_x: {simulated_gemm_x}",
                            flush=True,
                        )

                    event, hook = e2a_send_result[e2a_mb_idx]
                    # event.current_stream_wait()
                    hook()  # .current_stream_wait()
                    if M2N_DEVICE_SYNC:
                        paddle.device.synchronize()
                    if M2N_DEBUG:
                        print(
                            f"{i} combine send wait moe {e2a_mb_idx}_{e2a_layer_idx} data end",
                            flush=True,
                        )

                # recv_count = packed_recv_count[0]
                # num_valid_tokens = recv_count.item()
                # moe(simulated_gemm_x[0][:num_valid_tokens], weights)

            print(f"==================== {i}", flush=True)
    time.sleep(10)
    # dist.barrier()


def test_loop():
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    group = paddle.distributed.new_group(range(num_ranks))
    print("rank: ", rank, flush=True)
    print("num_ranks: ", num_ranks, flush=True)

    a_start_rank = 0
    a_num_ranks = 8
    e_start_rank = a_start_rank + a_num_ranks
    e_num_ranks = num_ranks - a_num_ranks

    num_tokens, hidden, num_topk, num_experts = 96, 7168, 8, 64

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

    use_fp8 = False
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
