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
    local_combine_backward,
    local_combine_forward,
    local_dispatch_backward,
    local_dispatch_forward,
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


def local_dispatch_forward_func(dispatch_history, local_expert_id, out_len):
    dispatched_hidden_states_list = []
    dispatched_indices_list = []
    dispatched_topk_weights_list = []
    details_metas_list = []
    fp8_scales_list = []

    use_fp8 = False
    for dispatch in dispatch_history:
        (
            dispatched_hidden_states,
            dispatched_scales,
            dispatched_topk_weights,
            states,
        ) = dispatch
        dispatched_indices = states["dispatched_indices"]
        details_metas = states["handle"][-3]
        if details_metas.shape[0] == 0:
            details_metas = paddle.empty([0, 4], dtype="int32")
        else:
            details_metas = paddle.view(details_metas, "int32")

        if dispatched_scales is not None:
            use_fp8 = True

        dispatched_hidden_states_list.append(dispatched_hidden_states)
        fp8_scales_list.append(dispatched_scales)
        dispatched_topk_weights_list.append(dispatched_topk_weights)
        dispatched_indices_list.append(dispatched_indices.astype("int32"))
        details_metas_list.append(details_metas)

    (
        out_dispatched_hidden_states,
        out_dispatched_topk_weights,
        out_details_metas,
        out_fp8_scales,
    ) = local_dispatch_forward(
        dispatched_hidden_states_list,
        dispatched_topk_weights_list,
        dispatched_indices_list,
        details_metas_list,
        fp8_scales_list if use_fp8 else None,
        local_expert_id,
        out_len,
        FP8_ALIGN,
    )

    return (
        out_dispatched_hidden_states,
        out_fp8_scales,
        out_dispatched_topk_weights,
        out_details_metas,
        out_len,
    )


def local_dispatch_backward_func(dispatch_history, local_expert_id, out_len):
    dispatched_hidden_states_list = []
    dispatched_indices_list = []
    details_metas_list = []

    for dispatch in dispatch_history:
        dispatched_hidden_states, _, _, states = dispatch

        dispatched_indices = states["dispatched_indices"]
        details_metas = states["handle"][-3]
        if details_metas.shape[0] == 0:
            details_metas = paddle.empty([0, 4], dtype="int32")
        else:
            details_metas = paddle.view(details_metas, "int32")

        dispatched_hidden_states_list.append(dispatched_hidden_states)
        dispatched_indices_list.append(dispatched_indices.astype("int32"))
        details_metas_list.append(details_metas)

    (
        out_dispatched_hidden_states,
        out_dispatched_indices,
        out_details_metas,
    ) = local_dispatch_backward(
        dispatched_hidden_states_list,
        dispatched_indices_list,
        details_metas_list,
        local_expert_id,
        out_len,
        FP8_ALIGN,
    )

    return (
        out_dispatched_hidden_states,
        out_dispatched_indices,
        out_details_metas,
        out_len,
    )


def local_combine_forward_func(
    tokens,
    details_metas,
    combine_notify_infos,
    combine_buffers,
    ori_len,
    is_buffer_active,
):
    recv_gbl_channel_prefix_matrix_list = []

    for info in combine_notify_infos:
        combine_handle = info["handle"]
        recv_gbl_channel_prefix_matrix = combine_handle[5]
        recv_gbl_channel_prefix_matrix_list.append(
            recv_gbl_channel_prefix_matrix
        )

    if tokens.shape[0] == 0:
        return

    combine_buffers = local_combine_forward(
        combine_buffers,
        tokens,
        details_metas,
        recv_gbl_channel_prefix_matrix_list,
        ori_len,
        is_buffer_active,
    )


def local_combine_backward_func(
    tokens,
    probs,
    indices,
    details_metas,
    combine_notify_infos,
    combine_buffers,
    combine_probs,
    local_expert_id,
    ori_len,
    is_buffer_active,
):
    recv_gbl_channel_prefix_matrix_list = []

    for info in combine_notify_infos:
        combine_handle = info["handle"]
        recv_gbl_channel_prefix_matrix = combine_handle[5]
        recv_gbl_channel_prefix_matrix_list.append(
            recv_gbl_channel_prefix_matrix
        )

    if tokens.shape[0] == 0:
        return

    local_combine_backward(
        combine_buffers,
        combine_probs,
        tokens,
        indices,
        probs,
        details_metas,
        recv_gbl_channel_prefix_matrix_list,
        local_expert_id,
        ori_len,
        is_buffer_active,
    )


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
    # The total number of tokens received via RDMA during each combine operation.
    combine_asymm_recv_rdma_counter_list = []
    combine_recv_rdma_rank_prefix_sum = []
    combine_recv_rdma_channel_prefix_matrix = []
    combine_recv_gbl_channel_prefix_matrix = []
    # The intra-channel offset of a token. This is precomputed by the dispatch sender,
    # which uses it internally for asymmetric combine reception and also transmits it
    # to the dispatch receiver.
    combine_send_rdma_head = []
    # The token's offset within the channel. The dispatch receiver uses this information
    # for asymmetric combine sends. It is calculated by the dispatch sender and then
    # transmitted to the dispatch receiver.
    combine_send_nvl_head = []

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

    # Create a buffer, which will be populated during the actual dispatch.
    asymm_aggregated_nvl_head = paddle.empty(
        [sum(combine_asymm_recv_rdma_counter_list), 8], dtype="int32"
    )

    # Pass it to dispatch for use.
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
        # Pass it to combine for use.
        combine_handle = (
            None,
            None,
            None,
            combine_recv_rdma_channel_prefix_matrix[loop_idx],
            combine_recv_rdma_rank_prefix_sum[loop_idx],
            combine_recv_gbl_channel_prefix_matrix[loop_idx],
            None,
            None,
            combine_send_rdma_head[loop_idx],
            asymm_aggregated_nvl_head[
                asymm_recv_rdma_start_idx:asymm_recv_rdma_end_idx
            ],  # In-place slice
        )
        asymm_recv_rdma_start_idx = asymm_recv_rdma_end_idx

        combine_states = {}
        combine_states["handle"] = combine_handle
        combine_states["num_combine_tokens"] = combine_num_combine_tokens_list[
            loop_idx
        ]
        combine_notify_infos.append(combine_states)

    return dispatch_notify_infos, combine_notify_infos, asymmetric_handle


def notify_dispatch_and_combine_func(
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

    # Count the total number of tokens received by each local expert.
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
    scale=None,
    async_finish=False,
    handle=None,
    asymmetric_handle=None,
):
    assert handle is not None
    buffer = flashep_buffer.get_buffer(group, get_hidden_bytes(x))

    if scale is not None:
        x = (x, scale)

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
    if isinstance(recv_x, tuple):
        recv_x, scale = recv_x
    else:
        scale = None
    return recv_x, scale, recv_token_probs, states, event


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
