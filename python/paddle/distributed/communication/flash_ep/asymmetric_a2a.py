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

import paddle
from paddle.base.core import (
    get_flash_ep_coalesce_rdma_layout,
    get_flash_ep_coalesce_rdma_schedule,
)

from .buffer import Buffer

FP8_ALIGN = 128


class FlashEPBuffer:
    """
    FlashEPBuffer
    """

    def __init__(self):
        """
        __init__
        """
        self._buffer = None

    def get_buffer(self, group, hidden_bytes):
        """Get or create a buffer for all-to-all communication.

        Args:
            group (paddle.distributed.ProcessGroup): Process group for communication
            hidden_bytes (int): Number of hidden bytes needed

        Returns:
            Buffer: Communication buffer
        """
        num_nvl_bytes, num_rdma_bytes = 0, 0
        for config in (
            Buffer.get_dispatch_config(group.world_size),
            Buffer.get_combine_config(group.world_size),
        ):
            # Split long line for PEP8 compliance
            num_nvl_bytes = max(
                config.get_nvl_buffer_size_hint(hidden_bytes, group.world_size),
                num_nvl_bytes,
            )
            num_rdma_bytes = max(
                config.get_rdma_buffer_size_hint(
                    hidden_bytes, group.world_size
                ),
                num_rdma_bytes,
            )

        # Allocate buffer if not existed or not enough buffer
        # NOTES: the adaptive routing configuration of the network **must be off**
        if (
            self._buffer is None
            or self._buffer.group != group
            or self._buffer.num_nvl_bytes < num_nvl_bytes
            or self._buffer.num_rdma_bytes < num_rdma_bytes
        ):
            self._buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
        return self._buffer

    def clear_buffer(self):
        """
        clear_buffer to remove memory allocation caused by flashep
        """
        if self._buffer is not None:
            del self._buffer
            self._buffer = None


flashep_buffer = FlashEPBuffer()


def get_hidden_bytes(x) -> int:
    if isinstance(x, tuple):
        x = x[0]
    return x.shape[1] * max(x.element_size(), 2)


def _local_dispatch_func(kernel_inputs, local_expert_id):
    out_dispatched_hidden_states_list = []
    out_dispatched_topk_weights_list = []
    out_dispatched_indices_list = []
    out_details_metas_list = []

    # 反向计算时, 是不是很多信息不需要重复计算了？前向可以记录下一些信息
    for input in kernel_inputs:
        (
            dispatched_hidden_states,
            dispatched_topk_weights,  # optional, maybe None
            dispatched_indices,
            details_metas,
        ) = input

        mask = (dispatched_indices == local_expert_id).any(axis=-1)
        hidden_states = dispatched_hidden_states[mask]
        indices = dispatched_indices[mask]

        if dispatched_topk_weights is not None:
            topk_weights = dispatched_topk_weights[mask]

            prob_indices = paddle.nonzero(indices == local_expert_id)
            topk_weights = paddle.gather_nd(topk_weights, prob_indices)
            assert (
                topk_weights.shape[0] == hidden_states.shape[0]
            ), f"topk_weights: {topk_weights.shape} vs {hidden_states.shape}"
            out_dispatched_topk_weights_list.append(topk_weights)

        out_dispatched_hidden_states_list.append(hidden_states)
        out_dispatched_indices_list.append(indices)
        out_details_metas_list.append(details_metas[mask])

    if out_dispatched_hidden_states_list:
        return (
            paddle.concat(out_dispatched_hidden_states_list, axis=0),
            (
                paddle.concat(out_dispatched_topk_weights_list, axis=0)
                if out_dispatched_topk_weights_list
                else None
            ),  # optional, maybe None
            (
                paddle.concat(out_dispatched_indices_list, axis=0)
                if out_dispatched_indices_list
                else None
            ),  # optional, maybe None
            paddle.concat(out_details_metas_list, axis=0),
        )
    else:
        # 注意处理输出为0size的情况
        return (
            paddle.empty(
                [0, dispatched_hidden_states.shape[1]],
                dispatched_hidden_states.dtype,
            ),
            paddle.empty(
                [0, dispatched_topk_weights.shape[1]],
                dispatched_topk_weights.dtype,
            ),
            paddle.empty(
                [0, dispatched_indices.shape[1]], dispatched_indices.dtype
            ),
            paddle.empty([0, details_metas.shape[1]], details_metas.dtype),
        )


def local_dispatch_func(dispatch_history, local_expert_id, out_len):
    kernel_inputs = []
    for dispatch in dispatch_history:
        dispatched_hidden_states, dispatched_topk_weights, states = dispatch

        dispatched_indices = states["dispatched_indices"]
        details_metas = states["handle"][-3]
        details_metas = paddle.view(details_metas, "int32")
        kernel_inputs.append(
            (
                dispatched_hidden_states,
                dispatched_topk_weights,
                dispatched_indices,
                details_metas,
            )
        )

    (
        out_dispatched_hidden_states,
        out_dispatched_topk_weights,
        out_dispatched_indices,
        out_details_metas,
    ) = _local_dispatch_func(kernel_inputs, local_expert_id)
    assert out_dispatched_hidden_states.shape[0] == out_len
    pad_len = (out_len + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN - out_len
    out_dispatched_hidden_states = paddle.nn.functional.pad(
        out_dispatched_hidden_states, [0, pad_len, 0, 0], value=0
    )
    if out_dispatched_topk_weights is not None:
        out_dispatched_topk_weights = paddle.nn.functional.pad(
            out_dispatched_topk_weights, [0, pad_len, 0, 0], value=0
        )
    if out_dispatched_indices is not None:
        out_dispatched_indices = paddle.nn.functional.pad(
            out_dispatched_indices, [0, pad_len, 0, 0], value=-1
        )
    out_details_metas = paddle.nn.functional.pad(
        out_details_metas, [0, pad_len, 0, 0], value=0
    )
    return (
        out_dispatched_hidden_states,
        out_dispatched_topk_weights,
        out_dispatched_indices,
        out_details_metas,
        out_len,
    )


def local_combine_func(
    tokens,
    probs,  # optional, maybe None
    indices,  # optional, maybe None
    details_metas,
    combine_notify_infos,
    combine_buffers,
    local_expert_id,
    ori_len,
):
    # combine_buffers是一个len为num_stages的list。list中的每个元素都是一个长为2的tuple, 第一个元素为tokens, 第二个元素为probs
    recv_gbl_channel_prefix_matrix_list = []

    tokens = tokens[:ori_len]
    if probs is not None:
        probs = probs[:ori_len]
    if indices is not None:
        indices = indices[:ori_len]
    details_metas = details_metas[:ori_len]
    for info in combine_notify_infos:
        combine_handle = info["handle"]
        recv_gbl_channel_prefix_matrix = combine_handle[5]
        recv_gbl_channel_prefix_matrix_list.append(
            recv_gbl_channel_prefix_matrix
        )

    token_num = tokens.shape[0]
    for id in range(token_num):
        meta = details_metas[id, :]
        channel_id = meta[0].item()
        nvl_head = meta[1].item()
        src_rank = meta[2].item()
        stage_idx_to_combine = meta[
            3
        ].item()  # 表明这个token需要在第几次流水线进行combine

        channel_offset = recv_gbl_channel_prefix_matrix_list[
            stage_idx_to_combine
        ][src_rank][channel_id].item()
        offset = channel_offset + nvl_head

        # combine tokens
        combine_buffers[stage_idx_to_combine][0][offset, :] += tokens[
            id, :
        ].astype("float32")

        if probs is not None:
            # combine probs
            for k_id in range(indices.shape[1]):
                if indices[id, k_id] == local_expert_id:
                    break
            combine_buffers[stage_idx_to_combine][1][offset, k_id] = probs[
                id
            ]  # 不会重复, 所以直接加上去就行


def fused_get_schedule_and_layout_func(
    token_indices,
    local_expert_to_stage_map,
    num_experts,
    nranks,
    num_loop_stage,
):
    dispatch_schedule_map, combine_schedule_map = (
        get_flash_ep_coalesce_rdma_schedule(
            token_indices,
            local_expert_to_stage_map,
            nranks,
            num_experts,
            num_loop_stage,
        )
    )

    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
    ) = get_flash_ep_coalesce_rdma_layout(
        token_indices,
        dispatch_schedule_map,
        combine_schedule_map,
        nranks,
        num_experts,
        num_loop_stage,
    )

    return (
        dispatch_schedule_map,
        combine_schedule_map,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
    )


def fused_notify_dispatch_func(
    x,
    token_indices,
    num_tokens_per_rank,
    num_tokens_per_rdma_rank,
    num_tokens_per_expert,
    is_token_in_rank,
    buffer,
    num_loop_stage,
):
    notify_infos = []
    for loop_idx in range(num_loop_stage):
        (
            dispatch_num_recv_tokens_per_expert,
            _,
            _,
            dispatch_handle,
        ) = buffer.internode_notify_dispatch(
            x,
            token_indices,
            num_tokens_per_rank[loop_idx][0],
            num_tokens_per_rdma_rank[loop_idx][0],
            num_tokens_per_expert[loop_idx][0],
            is_token_in_rank[loop_idx][0],
        )

        dispatch_states = {}
        dispatch_states["num_recv_tokens_per_expert_list"] = (
            dispatch_num_recv_tokens_per_expert
        )
        dispatch_states["handle"] = dispatch_handle

        notify_infos.append(dispatch_states)
    return notify_infos


def fused_notify_combine_func(
    x,
    token_indices,
    num_tokens_per_rank,
    num_tokens_per_rdma_rank,
    is_token_in_rank,
    combine_schedule_map,
    num_loop_stage,
    buffer,
    group,
):
    num_combine_tokens_list = []
    asymm_recv_rdma_counter_list = []  # 每一次combine时，从rdma收到的token总数
    recv_rdma_rank_prefix_sum = []
    recv_rdma_channel_prefix_matrix = []
    recv_gbl_channel_prefix_matrix = []
    send_rdma_head = (
        []
    )  # token在channel内部的偏移量。dispatch的发送方在非对称combine接收时需要用到信息，由dispatch发送方先计算好然后留着自己用，同时也发送给dispatch接收方
    send_nvl_head = (
        []
    )  # token在channel内部的偏移量。dispatch的接收方在非对称combine发送时需要用到信息，由dispatch发送方先计算好然后发送给dispatch接收方
    for loop_idx in range(num_loop_stage):
        (
            num_combine_tokens_,
            moe_recv_rdma_counter_,
            recv_rdma_rank_prefix_sum_,
            recv_rdma_channel_prefix_matrix_,
            recv_gbl_channel_prefix_matrix_,
            send_rdma_head_,
            send_nvl_head_,
        ) = buffer.internode_notify_combine(
            x,
            token_indices,
            num_tokens_per_rank[loop_idx][1],
            num_tokens_per_rdma_rank[loop_idx][1],
            is_token_in_rank[loop_idx][1],
        )

        num_combine_tokens_list.append(num_combine_tokens_)
        asymm_recv_rdma_counter_list.append(moe_recv_rdma_counter_)
        recv_rdma_rank_prefix_sum.append(recv_rdma_rank_prefix_sum_)
        recv_rdma_channel_prefix_matrix.append(recv_rdma_channel_prefix_matrix_)
        recv_gbl_channel_prefix_matrix.append(recv_gbl_channel_prefix_matrix_)
        send_rdma_head.append(send_rdma_head_)
        send_nvl_head.append(send_nvl_head_)

    asymm_recv_rdma_counter_loop_prefix_sum = paddle.cumsum(
        paddle.to_tensor(asymm_recv_rdma_counter_list, dtype="int32")
    )
    asymm_recv_rdma_rank_prefix_sum = paddle.stack(recv_rdma_rank_prefix_sum)
    asymm_recv_rdma_channel_prefix_matrix = paddle.stack(
        recv_rdma_channel_prefix_matrix
    )
    asymm_send_rdma_head = paddle.stack(send_rdma_head)
    asymm_send_nvl_head = paddle.stack(send_nvl_head)

    # 创建一个buffer, 真正dispatch的时候往里面填充
    asymm_aggregated_nvl_head = paddle.empty(
        [sum(asymm_recv_rdma_counter_list), 8], dtype="int32"
    )

    # 传给dispatch来用
    asymmetric_handle = (
        combine_schedule_map,
        asymm_recv_rdma_counter_loop_prefix_sum,
        asymm_recv_rdma_rank_prefix_sum,
        asymm_recv_rdma_channel_prefix_matrix,
        asymm_send_rdma_head,
        asymm_send_nvl_head,
        asymm_aggregated_nvl_head,
    )

    notify_infos = []
    asymm_recv_rdma_start_idx = 0
    for loop_idx in range(num_loop_stage):
        asymm_recv_rdma_end_idx = (
            asymm_recv_rdma_start_idx + asymm_recv_rdma_counter_list[loop_idx]
        )
        # 留着给combine时用
        combine_handle = (
            None,
            None,
            None,
            recv_rdma_channel_prefix_matrix[loop_idx],  # 其实可以用切片
            recv_rdma_rank_prefix_sum[loop_idx],
            recv_gbl_channel_prefix_matrix[loop_idx],
            None,
            None,
            send_rdma_head[loop_idx],
            asymm_aggregated_nvl_head[
                asymm_recv_rdma_start_idx:asymm_recv_rdma_end_idx
            ],  # inplace的切片
        )
        asymm_recv_rdma_start_idx = asymm_recv_rdma_end_idx

        combine_states = {}
        combine_states["handle"] = combine_handle
        combine_states["num_combine_tokens"] = num_combine_tokens_list[loop_idx]
        notify_infos.append(combine_states)

    return notify_infos, asymmetric_handle


def fused_notify_func(
    x,
    token_indices,
    num_tokens_per_rank,
    num_tokens_per_rdma_rank,
    num_tokens_per_expert,
    is_token_in_rank,
    combine_schedule_map,
    buffer,
    num_loop_stage,
):
    combine_num_combine_tokens_list = []
    combine_asymm_recv_rdma_counter_list = (
        []
    )  # 每一次combine时，从rdma收到的token总数
    combine_recv_rdma_rank_prefix_sum = []
    combine_recv_rdma_channel_prefix_matrix = []
    combine_recv_gbl_channel_prefix_matrix = []
    combine_send_rdma_head = (
        []
    )  # token在channel内部的偏移量。dispatch的发送方在非对称combine接收时需要用到信息，由dispatch发送方先计算好然后留着自己用，同时也发送给dispatch接收方
    combine_send_nvl_head = (
        []
    )  # token在channel内部的偏移量。dispatch的接收方在非对称combine发送时需要用到信息，由dispatch发送方先计算好然后发送给dispatch接收方

    dispatch_notify_infos = []
    for loop_idx in range(num_loop_stage):
        dispatch_notify_info, combine_notify_info = (
            buffer.internode_fused_notify(
                x,
                token_indices,
                num_tokens_per_rank[loop_idx][0],
                num_tokens_per_rdma_rank[loop_idx][0],
                num_tokens_per_expert[loop_idx][0],
                is_token_in_rank[loop_idx][0],
                num_tokens_per_rank[loop_idx][1],
                num_tokens_per_rdma_rank[loop_idx][1],
                is_token_in_rank[loop_idx][1],
            )
        )
        (
            dispatch_num_recv_tokens_per_expert,
            _,
            _,
            dispatch_handle,
        ) = dispatch_notify_info
        (
            combine_num_combine_tokens_,
            combine_moe_recv_rdma_counter_,
            combine_recv_rdma_rank_prefix_sum_,
            combine_recv_rdma_channel_prefix_matrix_,
            combine_recv_gbl_channel_prefix_matrix_,
            combine_send_rdma_head_,
            combine_send_nvl_head_,
        ) = combine_notify_info

        dispatch_states = {}
        dispatch_states["num_recv_tokens_per_expert_list"] = (
            dispatch_num_recv_tokens_per_expert
        )
        dispatch_states["handle"] = dispatch_handle

        combine_num_combine_tokens_list.append(combine_num_combine_tokens_)
        combine_asymm_recv_rdma_counter_list.append(
            combine_moe_recv_rdma_counter_
        )
        combine_recv_rdma_rank_prefix_sum.append(
            combine_recv_rdma_rank_prefix_sum_
        )
        combine_recv_rdma_channel_prefix_matrix.append(
            combine_recv_rdma_channel_prefix_matrix_
        )
        combine_recv_gbl_channel_prefix_matrix.append(
            combine_recv_gbl_channel_prefix_matrix_
        )
        combine_send_rdma_head.append(combine_send_rdma_head_)
        combine_send_nvl_head.append(combine_send_nvl_head_)

        dispatch_notify_infos.append(dispatch_states)

    asymm_recv_rdma_counter_loop_prefix_sum = paddle.cumsum(
        paddle.to_tensor(combine_asymm_recv_rdma_counter_list, dtype="int32")
    )
    asymm_recv_rdma_rank_prefix_sum = paddle.stack(
        combine_recv_rdma_rank_prefix_sum
    )
    asymm_recv_rdma_channel_prefix_matrix = paddle.stack(
        combine_recv_rdma_channel_prefix_matrix
    )
    asymm_send_rdma_head = paddle.stack(combine_send_rdma_head)
    asymm_send_nvl_head = paddle.stack(combine_send_nvl_head)

    # 创建一个buffer, 真正dispatch的时候往里面填充
    asymm_aggregated_nvl_head = paddle.empty(
        [sum(combine_asymm_recv_rdma_counter_list), 8], dtype="int32"
    )

    # 传给dispatch来用
    asymmetric_handle = (
        combine_schedule_map,
        asymm_recv_rdma_counter_loop_prefix_sum,
        asymm_recv_rdma_rank_prefix_sum,
        asymm_recv_rdma_channel_prefix_matrix,
        asymm_send_rdma_head,
        asymm_send_nvl_head,
        asymm_aggregated_nvl_head,
    )
    combine_notify_infos = []
    asymm_recv_rdma_start_idx = 0
    for loop_idx in range(num_loop_stage):
        asymm_recv_rdma_end_idx = (
            asymm_recv_rdma_start_idx
            + combine_asymm_recv_rdma_counter_list[loop_idx]
        )
        # 留着给combine时用
        combine_handle = (
            None,
            None,
            None,
            combine_recv_rdma_channel_prefix_matrix[loop_idx],  # 其实可以用切片
            combine_recv_rdma_rank_prefix_sum[loop_idx],
            combine_recv_gbl_channel_prefix_matrix[loop_idx],
            None,
            None,
            combine_send_rdma_head[loop_idx],
            asymm_aggregated_nvl_head[
                asymm_recv_rdma_start_idx:asymm_recv_rdma_end_idx
            ],  # inplace的切片
        )
        asymm_recv_rdma_start_idx = asymm_recv_rdma_end_idx

        combine_states = {}
        combine_states["handle"] = combine_handle
        combine_states["num_combine_tokens"] = combine_num_combine_tokens_list[
            loop_idx
        ]
        combine_notify_infos.append(combine_states)

    return dispatch_notify_infos, combine_notify_infos, asymmetric_handle


def notify_dispatch_and_combine(
    x,
    token_indices,
    local_expert_to_stage_map,
    num_experts,
    group,
    num_loop_stage,
):
    nranks = group.nranks
    buffer = flashep_buffer.get_buffer(group, get_hidden_bytes(x))

    (
        _,
        combine_schedule_map,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
    ) = fused_get_schedule_and_layout_func(
        token_indices,
        local_expert_to_stage_map,
        num_experts,
        nranks,
        num_loop_stage,
    )

    dispatch_notify_infos, combine_notify_infos, asymmetric_handle = (
        fused_notify_func(
            x,
            token_indices,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            combine_schedule_map,
            buffer,
            num_loop_stage,
        )
    )

    # 统计本地每个expert收到的token总数
    tokens_per_expert_list = [
        sum(
            dispatch_states["num_recv_tokens_per_expert_list"][i]
            for dispatch_states in dispatch_notify_infos
        )
        for i in range(num_experts // group.nranks)
    ]
    return (
        dispatch_notify_infos,
        combine_notify_infos,
        tokens_per_expert_list,
        asymmetric_handle,
    )


def dispatch_func(
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
    async_finish=False,
    handle=None,
    asymmetric_handle=None,
):
    assert handle is not None
    buffer = flashep_buffer.get_buffer(group, get_hidden_bytes(x))

    (
        recv_x,
        recv_token_indices,
        recv_token_probs,
        num_recv_tokens_per_expert_list,
        handle,
        event,
    ) = buffer.dispatch(
        x,
        handle=handle,
        asymmetric_handle=asymmetric_handle,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_experts=num_experts,
        async_finish=async_finish,
    )

    states = {}
    states["dispatched_indices"] = recv_token_indices
    states["tokens_per_expert"] = paddle.to_tensor(
        num_recv_tokens_per_expert_list
    )
    states["num_recv_tokens_per_expert_list"] = num_recv_tokens_per_expert_list
    states["handle"] = handle

    if not async_finish:
        event = None
    return recv_x, recv_token_probs, states, event


def combine_func(
    x,
    group,
    handle,
    topk_weights=None,
    output=None,
    output_topk_weights=None,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
):
    buffer = flashep_buffer.get_buffer(group, get_hidden_bytes(x))
    combined_x, combined_weight, event = buffer.combine(
        x,
        topk_weights=topk_weights,
        handle=handle,
        output=output,
        output_topk_weights=output_topk_weights,
        async_finish=async_finish,
        previous_event=previous_event,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    if not async_finish:
        event = None
    return combined_x, combined_weight, event
