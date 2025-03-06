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
import re
import time
import unittest

import numpy as np
from legacy_test.test_collective_api_base import TestDistBase

import paddle
import paddle.distributed as dist
import paddle.distributed.communication.deep_ep as ep
from paddle.base import core
from paddle.base.core import Config
from paddle.distributed import fleet
from paddle.distributed.communication.group import Group


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


is_sm90 = (
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] == 9
    and paddle.device.cuda.get_device_capability()[1] == 0
)

is_sm_supported = is_sm90


def is_deep_ep_supported():
    if (
        not core.is_compiled_with_cuda()
        or get_cuda_version() < 12030
        or not is_sm_supported
    ):
        return False
    return True


def bench(fn, num_warmups: int = 20, num_tests: int = 30, post_fn=None):
    # Flush L2 cache with 256 MB data
    paddle.device.cuda.synchronize()
    cache = paddle.empty([int(256e6 // 4)], dtype=paddle.int32)

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [
        paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)
    ]
    end_events = [
        paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)
    ]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    paddle.device.cuda.synchronize()

    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )[1:]
    return np.average(times), np.min(times), np.max(times)


def per_token_cast_to_fp8(x: paddle.Tensor):
    assert x.dim() == 2 and x.shape[1] % 128 == 0
    # m, n = x.shape
    m = x.shape[0]
    n = x.shape[1]
    x_view = x.view([m, -1, 128])
    x_amax = (
        x_view.abs().cast(paddle.float32).amax(axis=2).view([m, -1]).clip(1e-4)
    )
    return (x_view * (448.0 / x_amax.unsqueeze(2))).cast(
        paddle.float8_e4m3fn
    ).view([m, n]), (x_amax / 448.0).view([m, -1])


def per_token_cast_back(x_fp8: paddle.Tensor, x_scales: paddle.Tensor):
    x_fp32 = x_fp8.cast(paddle.float32).view([x_fp8.shape[0], -1, 128])
    x_scales = x_scales.view([x_fp8.shape[0], -1, 1])
    return (x_fp32 * x_scales).view(x_fp8.shape).cast(paddle.bfloat16)


def inplace_unique(x: paddle.Tensor, num_slots: int):
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = paddle.zeros([x.shape[0], num_slots + 1], dtype=x.dtype).to(
        x.place
    )
    # bin_count.scatter_add_(1, x_padded, paddle.ones_like(x_padded))
    bin_count.put_along_axis_(
        axis=1,
        indices=x_padded,
        values=paddle.ones_like(x_padded),
        reduce='add',
        include_self=True,
    )

    bin_count = bin_count[:, :num_slots]
    sorted_bin_count = paddle.sort(bin_count, axis=-1, descending=True)
    sorted_bin_idx = paddle.argsort(bin_count, axis=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = paddle.sort(sorted_bin_idx, descending=True, axis=-1)
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.shape[1])
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def test_main(
    num_sms: int,
    local_rank: int,
    num_local_ranks: int,
    num_ranks: int,
    rank: int,
    buffer: ep.Buffer,
    group: Group,
):
    # Settings
    num_tokens, hidden, num_topk, num_experts = (
        4096,
        7168,
        8,
        (256 // num_ranks) * num_ranks,
    )
    assert num_experts % num_ranks == 0 and num_local_ranks == 8

    # Random data
    x = paddle.ones(shape=[num_tokens, hidden], dtype=paddle.bfloat16) * rank
    x_pure_rand = paddle.randn(
        shape=[num_tokens, hidden], dtype=paddle.bfloat16
    )
    x_e4m3 = per_token_cast_to_fp8(x)
    scores = (
        paddle.randn([num_tokens, num_experts], dtype=paddle.float32).abs() + 1
    )
    topk_idx = paddle.topk(
        scores, num_topk, axis=-1, largest=True, sorted=False
    )[1]
    topk_weights = (
        paddle.ones([num_tokens, num_topk], dtype=paddle.float32) * rank
    )
    topk_weights_pure_rand = paddle.randn(
        [num_tokens, num_topk], dtype=paddle.float32
    )
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Expert meta
    num_tokens_per_expert = paddle.zeros(
        [
            num_experts,
        ],
        dtype=paddle.int32,
    )
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = paddle.empty(
        [
            num_ranks,
        ],
        dtype=paddle.int32,
    )
    token_idx_in_rank = paddle.full(
        (num_ranks, num_tokens), -1, dtype=paddle.int64
    )
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).cast(paddle.int32).max(axis=-1)
        count = token_sel.sum().item()
        tokens = paddle.argsort(token_sel.cast(paddle.int32), descending=True)
        tokens[:count] = paddle.sort(tokens[:count])
        token_idx_in_rank[i][tokens[:count]] = paddle.arange(
            count, dtype=paddle.int64
        )
    token_idx_in_rank = token_idx_in_rank.t().contiguous().cast(paddle.int32)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    (
        ref_num_tokens_per_rank,
        _,
        ref_num_tokens_per_expert,
        ref_is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)
    assert paddle.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert paddle.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert paddle.allclose(ref_is_token_in_rank, is_token_in_rank)
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    paddle.distributed.barrier(group)
    time.sleep(1)

    # Config
    nvl_buffer_size = 256
    config = Config(num_sms, 8, nvl_buffer_size)

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, rank_prefix_matrix):
        assert paddle.allclose(check_x.amin(axis=1), check_x.amax(axis=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = rank_prefix_matrix[i][rank].item()
            assert (
                check_x[check_start:check_end, :].int() - i
            ).sum().item() == 0
            check_start = check_end

    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x in (x_pure_rand, x, x_e4m3):
                for with_topk in (False, True):
                    dispatch_args = {
                        'x': current_x,
                        'num_tokens_per_rank': num_tokens_per_rank,
                        'is_token_in_rank': is_token_in_rank,
                        'num_tokens_per_expert': num_tokens_per_expert,
                        'config': config,
                        'async_finish': async_mode,
                    }
                    if with_topk:
                        dispatch_args.update(
                            {
                                'topk_idx': topk_idx,
                                'topk_weights': (
                                    topk_weights_pure_rand
                                    if current_x is x_pure_rand
                                    else topk_weights
                                ),
                            }
                        )
                    if previous_mode:
                        dispatch_args.update(
                            {'previous_event': buffer.capture()}
                        )
                    (
                        recv_x,
                        recv_topk_idx,
                        recv_topk_weights,
                        recv_num_tokens_per_expert_list,
                        handle,
                        event,
                    ) = buffer.dispatch(**dispatch_args)
                    event.current_stream_wait() if async_mode else ()
                    recv_x = (
                        per_token_cast_back(*recv_x)
                        if isinstance(recv_x, tuple)
                        else recv_x
                    )

                    # Checks
                    rank_prefix_matrix = handle[0]
                    assert (
                        gbl_num_tokens_per_rank[rank].item() == recv_x.shape[0]
                    ), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.shape[0]}'
                    assert (
                        gbl_num_tokens_per_expert.view([num_ranks, -1])[
                            rank
                        ].tolist()
                        == recv_num_tokens_per_expert_list
                    )
                    if current_x is not x_pure_rand:
                        pass
                        # check_data(recv_x, rank_prefix_matrix)
                    if with_topk:
                        # Check `topk_idx`
                        assert (
                            recv_topk_idx.equal(-1)
                            | (
                                (recv_topk_idx >= 0)
                                & (recv_topk_idx < (num_experts // num_ranks))
                            )
                        ).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(
                            recv_num_tokens_per_expert_list
                        ):
                            assert recv_topk_idx.equal(i).sum().item() == count

                        # Check `topk_weights`
                        if current_x is not x_pure_rand:
                            recv_topk_weights[
                                recv_topk_idx.equal(-1)
                            ] = recv_topk_weights.amax(
                                axis=1, keepdim=True
                            ).expand_as(
                                recv_topk_weights
                            )[
                                recv_topk_idx.equal(-1)
                            ]
                            # check_data(recv_topk_weights, rank_prefix_matrix)

                    # Test cached dispatch (must without top-k staffs)
                    # NOTES: handle must be refreshed
                    if not with_topk:
                        dispatch_args = {
                            'x': current_x,
                            'handle': handle,
                            'config': config,
                            'async_finish': async_mode,
                        }
                        if previous_mode:
                            dispatch_args.update(
                                {'previous_event': buffer.capture()}
                            )
                        recv_x, _, _, _, _, event = buffer.dispatch(
                            **dispatch_args
                        )
                        event.current_stream_wait() if async_mode else ()
                        recv_x = (
                            per_token_cast_back(*recv_x)
                            if isinstance(recv_x, tuple)
                            else recv_x
                        )
                        if current_x is not x_pure_rand:
                            pass
                            # check_data(recv_x, rank_prefix_matrix)

                    # Test combine
                    combine_args = {
                        'x': recv_x,
                        'handle': handle,
                        'config': config,
                        'async_finish': async_mode,
                    }
                    if with_topk:
                        combine_args.update({'topk_weights': recv_topk_weights})
                    if previous_mode:
                        dispatch_args.update(
                            {'previous_event': buffer.capture()}
                        )
                    combined_x, combined_topk_weights, event = buffer.combine(
                        **combine_args
                    )
                    event.current_stream_wait() if async_mode else ()
                    # check_x = combined_x.cast(paddle.float32) / is_token_in_rank.sum(axis=1).unsqueeze(1)
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    # assert calc_diff(check_x, ref_x) < 5e-6
                    if with_topk:
                        pass
                        # check_topk_weights = combined_topk_weights if (current_x is x_pure_rand) else (combined_topk_weights / is_token_in_rank.sum(axis=1).unsqueeze(1))
                        # ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                        # assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    # For later tuning
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes

                    if local_rank == 0:
                        print(' passed', flush=True)
    if local_rank == 0:
        print()

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        nvl_recv_bytes = (
            (dispatch_bf16_nvl_recv_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_nvl_recv_bytes
        )
        for nvl_chunk_size in range(4, 33, 4):
            config = Config(num_sms, nvl_chunk_size, nvl_buffer_size)
            tune_args = {'x': current_x, 'handle': handle, 'config': config}
            t = bench(lambda: buffer.dispatch(**tune_args))[0]
            if t < best_time:
                best_time, best_results = t, (num_sms, nvl_chunk_size)

        if isinstance(current_x, tuple):
            # Gather FP8 the best config from rank 0
            best_dispatch_results = paddle.to_tensor(
                [best_results[0], best_results[1]], dtype=paddle.int32
            )
            all_best_fp8_results_list = [
                paddle.zeros_like(best_dispatch_results)
                for _ in range(paddle.distributed.get_world_size(group))
            ]
            dist.all_gather(
                all_best_fp8_results_list,
                best_dispatch_results,
                group=group,
            )
            best_dispatch_results = all_best_fp8_results_list[0].tolist()
    dispatch_config = Config(
        best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size
    )

    dispatch_args = {
        'x': x,
        'num_tokens_per_rank': num_tokens_per_rank,
        'is_token_in_rank': is_token_in_rank,
        'num_tokens_per_expert': num_tokens_per_expert,
        'config': (dispatch_config if dispatch_config is not None else config),
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in range(1, 5, 1):
        config = Config(num_sms, nvl_chunk_size, nvl_buffer_size)
        tune_args = {'x': recv_x, 'handle': handle, 'config': config}
        t = bench(lambda: buffer.combine(**tune_args))[0]
        if local_rank == 0:
            if t < best_time:
                best_time, best_results = t, (num_sms, nvl_chunk_size)


@unittest.skipIf(
    not is_deep_ep_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 12.2"
    "and device's compute capability must be 9.0",
)
class TestCollectiveDeepEPAllToAllIntranode(TestDistBase):
    def test_loop():
        hcg = fleet.get_hybrid_communicate_group()
        ep_group = hcg.get_model_parallel_group()
        local_rank = dist.get_rank(ep_group)
        num_local_ranks = dist.get_world_size(ep_group)
        rank = local_rank
        num_ranks = num_local_ranks
        paddle.seed(rank)

        test_ll_compatibility, num_rdma_bytes = False, 0
        if test_ll_compatibility:
            ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = (
                16,
                5120,
                256,
                9,
            )
            num_rdma_bytes = ep.Buffer.get_low_latency_rdma_size_hint(
                ll_num_tokens, ll_hidden, num_ranks, ll_num_experts
            )

        buffer = ep.Buffer(
            ep_group,
            int(1e9),
            num_rdma_bytes,
            low_latency_mode=test_ll_compatibility,
            num_qps_per_rank=(
                ll_num_experts // num_ranks if test_ll_compatibility else 1
            ),
        )

        for i in (24,):
            test_main(
                i,
                local_rank,
                num_local_ranks,
                num_ranks,
                rank,
                buffer,
                ep_group,
            )


if __name__ == "__main__":
    unittest.main()
