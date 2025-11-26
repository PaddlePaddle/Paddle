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

import logging
from itertools import accumulate

import paddle

from .asymmetric_a2a import (
    combine_func,
    dispatch_func,
    get_flashep_rowmap_func,
    local_combine_backward_func,
    local_combine_forward_func,
    local_dispatch_backward_func,
    local_dispatch_forward_func,
    notify_dispatch_and_combine_func,
)
from .utils import (
    get_event_from_calc_stream,
)

logger = logging.getLogger(__name__)

DETAILS_METAS_OFFSET = -3


def build_pipeline_stage_infos(
    local_num_experts,
    num_loop_stage,
    expert_num_for_dispatch_stage=None,  # Indicates the expert num per pipeline stage.
    expert_num_for_combine_stage=None,  # Indicates the expert num per pipeline stage.
):
    assert num_loop_stage is not None, "num_loop_stage must be not None"
    assert (
        num_loop_stage > 0 and num_loop_stage <= local_num_experts
    ), "num_loop_stage must be greater than 0 and less than local_num_experts"
    assert local_num_experts > 0, "local_num_experts must be greater than 0"

    if (
        expert_num_for_dispatch_stage is None
        or expert_num_for_combine_stage is None
    ):
        expert_num_per_stage = local_num_experts // num_loop_stage
        expert_num_for_dispatch_stage = [
            expert_num_per_stage for _ in range(num_loop_stage)
        ]
        expert_num_for_combine_stage = [
            expert_num_per_stage for _ in range(num_loop_stage)
        ]
    else:
        assert (
            len(expert_num_for_dispatch_stage) == num_loop_stage
            and len(expert_num_for_combine_stage) == num_loop_stage
        )
        assert (
            sum(expert_num_for_dispatch_stage) == local_num_experts
            and sum(expert_num_for_combine_stage) == local_num_experts
        )

    expert_num_for_dispatch_stage_prefix = list(
        accumulate(expert_num_for_dispatch_stage)
    )
    expert_num_for_combine_stage_prefix = list(
        accumulate(expert_num_for_combine_stage)
    )

    local_expert_to_stage_map = [[0, 0] for _ in range(local_num_experts)]
    for i in range(local_num_experts):
        for loop_idx, prefix in enumerate(expert_num_for_dispatch_stage_prefix):
            if prefix > i:
                break
        local_expert_to_stage_map[i][0] = loop_idx
        for loop_idx, prefix in enumerate(expert_num_for_combine_stage_prefix):
            if prefix > i:
                break
        local_expert_to_stage_map[i][1] = loop_idx

    expert_num_for_dispatch_stage_prefix = expert_num_for_dispatch_stage_prefix
    expert_num_for_combine_stage_prefix = expert_num_for_combine_stage_prefix
    local_expert_to_stage_map = paddle.to_tensor(
        local_expert_to_stage_map, dtype="int32"
    )
    num_loop_stage = num_loop_stage

    logger.info(
        f"expert_num_for_dispatch_stage_prefix: {expert_num_for_dispatch_stage_prefix}"
    )
    logger.info(
        f"expert_num_for_combine_stage_prefix: {expert_num_for_combine_stage_prefix}"
    )
    logger.info(f"local_expert_to_stage_map: {local_expert_to_stage_map}")

    pipeline_stage_infos = {}
    pipeline_stage_infos["expert_num_for_dispatch_stage_prefix"] = (
        expert_num_for_dispatch_stage_prefix
    )
    pipeline_stage_infos["expert_num_for_combine_stage_prefix"] = (
        expert_num_for_combine_stage_prefix
    )
    pipeline_stage_infos["local_expert_to_stage_map"] = (
        local_expert_to_stage_map
    )
    pipeline_stage_infos["num_loop_stage"] = num_loop_stage
    return pipeline_stage_infos


def _get_expert_dependencies(
    local_expert_id, expert_num_for_dispatch_stage_prefix
):
    # Find the dispatch round on which the current expert depends
    dispatch_stage_idx = next(
        (
            i
            for i, expert_prefix in enumerate(
                expert_num_for_dispatch_stage_prefix
            )
            if expert_prefix > local_expert_id
        ),
        None,
    )
    return dispatch_stage_idx


def _init_combine_buffer(
    seq_len,
    hidden_size,
    topk,
    local_num_experts,
    combine_notify_infos,
    expert_num_for_combine_stage_prefix,
    has_prob=False,
):
    combine_buffers = []
    combine_probs = []

    # During each global combine, accumulate to output_tokens in-place with fp32 precision.
    output_tokens = paddle.zeros([seq_len, hidden_size], "float32")
    output_topk_weights = (
        paddle.zeros([seq_len, topk], "float32") if has_prob else None
    )

    num_loop_stage = len(combine_notify_infos)
    for stage_idx in range(num_loop_stage):
        combine_out_len = combine_notify_infos[stage_idx]["num_combine_tokens"]
        combine_buffers.append(
            paddle.zeros([combine_out_len, hidden_size], dtype="float32")
        )
        if has_prob:
            combine_probs.append(
                paddle.zeros([combine_out_len, topk], dtype="float32")
            )

    # Indicates whether the current buffer will be written with data again.
    is_buffer_active = [1 for _ in range(num_loop_stage)]
    # Indicates which buffer to send in the global combine a2a after each local combine.
    expert_result_buffer = [-1 for _ in range(local_num_experts)]
    for stage_idx, local_expert_id_prefix in enumerate(
        expert_num_for_combine_stage_prefix
    ):
        expert_result_buffer[local_expert_id_prefix - 1] = stage_idx

    return (
        combine_buffers,
        combine_probs,
        output_tokens,
        output_topk_weights,
        is_buffer_active,
        expert_result_buffer,
    )


class FlashEPFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        token_probs,
        token_indices,
        pipeline_stage_infos,
        num_experts,
        expert_funcs,
        group,
        flash_ep_fp8_dispatch_a2a=False,
        flash_ep_split_expert_bw=False,
    ):
        scale = None
        if flash_ep_fp8_dispatch_a2a:
            hidden_states, scale = (
                paddle.incubate.nn.functional.fp8_quant_blockwise(
                    hidden_states, output_scale_transpose=False
                )
            )

        expert_num_for_dispatch_stage_prefix = pipeline_stage_infos[
            "expert_num_for_dispatch_stage_prefix"
        ]
        expert_num_for_combine_stage_prefix = pipeline_stage_infos[
            "expert_num_for_combine_stage_prefix"
        ]
        local_expert_to_stage_map = pipeline_stage_infos[
            "local_expert_to_stage_map"
        ]
        ctx.num_loop_stage = pipeline_stage_infos["num_loop_stage"]

        # Perform the meta information exchange for all pipeline stages in one go.
        (
            dispatch_notify_infos,
            combine_notify_infos,
            tokens_per_expert_list,
            asymmetric_handle,
        ) = notify_dispatch_and_combine_func(
            hidden_states,
            token_indices,
            local_expert_to_stage_map,
            num_experts,
            group=group,
            num_loop_stage=ctx.num_loop_stage,
        )

        ctx.expert_nodes = []
        for expert in expert_funcs:
            if hasattr(expert, "build_expert_node"):
                expert = expert.build_expert_node()
            ctx.expert_nodes.append(expert)

        ctx.pipeline_stage_infos = pipeline_stage_infos
        ctx.local_num_experts = num_experts // group.nranks
        ctx.num_experts = num_experts
        ctx.group = group
        ctx.token_probs_shape = token_probs.shape
        ctx.token_probs_dtype = token_probs.dtype
        ctx.topk = token_indices.shape[1]
        ctx.seq_len = hidden_states.shape[0]
        ctx.hidden_size = hidden_states.shape[1]
        ctx.dispatch_notify_infos = dispatch_notify_infos
        ctx.combine_notify_infos = combine_notify_infos
        ctx.tokens_per_expert_list = tokens_per_expert_list
        ctx.flash_ep_split_expert_bw = flash_ep_split_expert_bw

        dispatch_history = []
        dispatch_events = []
        ctx.dispatched_indices = []
        ctx.details_metas = []
        for stage_idx in range(ctx.num_loop_stage):
            dispatch_states = dispatch_notify_infos[stage_idx]
            # global dispatch a2a
            tokens, scale_, probs, states, event = dispatch_func(
                hidden_states,
                token_indices,
                token_probs,
                num_experts,
                group,
                scale=scale,
                async_finish=True,
                handle=dispatch_states["handle"],
                asymmetric_handle=asymmetric_handle,
                num_loop_stage=ctx.num_loop_stage,
            )

            ctx.dispatched_indices.append(states["dispatched_indices"])
            ctx.details_metas.append(states["handle"][DETAILS_METAS_OFFSET])
            dispatch_history.append((tokens, scale_, probs, states))
            dispatch_events.append(event)

        # Create combine-related buffers and control signals.
        (
            combine_buffers,
            _,
            output_tokens,
            _,
            is_buffer_active,
            expert_result_buffer,
        ) = _init_combine_buffer(
            ctx.seq_len,
            ctx.hidden_size,
            ctx.topk,
            ctx.local_num_experts,
            ctx.combine_notify_infos,
            expert_num_for_combine_stage_prefix,
        )

        ctx.probs = []
        ctx.output_rowmap_list = []
        ctx.output_rowmap_len_list = []
        for local_expert_id in range(ctx.local_num_experts):
            # Identify prior data dependencies and perform multi-stream synchronization.
            dispatch_stage_idx = _get_expert_dependencies(
                local_expert_id, expert_num_for_dispatch_stage_prefix
            )
            if dispatch_events[dispatch_stage_idx]:
                dispatch_events[dispatch_stage_idx].calc_stream_wait(group.id)
            if dispatch_stage_idx == len(ctx.output_rowmap_list):
                output_rowmap, output_rowmap_len = get_flashep_rowmap_func(
                    ctx.dispatched_indices[dispatch_stage_idx],
                    ctx.local_num_experts,
                )
                ctx.output_rowmap_list.append(output_rowmap)
                ctx.output_rowmap_len_list.append(output_rowmap_len)

            # Local dispatch data redistribution
            (
                tokens,
                scale_,
                probs,
                details_metas,
                ori_len,
            ) = local_dispatch_forward_func(
                dispatch_history[: dispatch_stage_idx + 1],
                ctx.output_rowmap_list[: dispatch_stage_idx + 1],
                ctx.output_rowmap_len_list[: dispatch_stage_idx + 1],
                ctx.local_num_experts,
                local_expert_id,
                out_len=tokens_per_expert_list[local_expert_id],
                num_loop_stage=ctx.num_loop_stage,
            )
            ctx.probs.append(probs)

            # Expert Computation
            tokens = ctx.expert_nodes[local_expert_id].forward(
                tokens, probs, scale=scale_
            )

            # Local combine data aggregation
            local_combine_forward_func(
                tokens,
                details_metas,
                combine_notify_infos,
                combine_buffers,
                ori_len,
                is_buffer_active,
                ctx.group,
                ctx.num_loop_stage,
            )
            tokens, scale_, probs, details_metas = None, None, None, None

            # When a buffer is scheduled for all-to-all communication,
            # update the combine-related control signals.
            if expert_result_buffer[local_expert_id] != -1:
                stage_idx = expert_result_buffer[local_expert_id]
                is_buffer_active[stage_idx] = 0
                tokens = combine_buffers[stage_idx].astype("bfloat16")
                combine_buffers[stage_idx] = paddle.empty(
                    [0, tokens.shape[1]], combine_buffers[stage_idx].dtype
                )  # TODO: Set None
                combine_states = combine_notify_infos[stage_idx]
                _, _, event = combine_func(
                    tokens,
                    group,
                    previous_event=get_event_from_calc_stream(group.id),
                    async_finish=True,
                    allocate_on_comm_stream=True,
                    output=output_tokens,
                    handle=combine_states["handle"],
                    num_loop_stage=ctx.num_loop_stage,
                )

        if event:
            event.calc_stream_wait(group.id)

        return output_tokens.astype("bfloat16"), tokens_per_expert_list

    @staticmethod
    def backward(ctx, output_grad):
        expert_num_for_dispatch_stage_prefix = ctx.pipeline_stage_infos[
            "expert_num_for_dispatch_stage_prefix"
        ]
        expert_num_for_combine_stage_prefix = ctx.pipeline_stage_infos[
            "expert_num_for_combine_stage_prefix"
        ]

        dispatch_history = []
        dispatch_events = []
        for stage_idx in range(ctx.num_loop_stage):
            dispatch_states = ctx.dispatch_notify_infos[stage_idx]
            tokens, _, _, states, event = dispatch_func(
                output_grad,
                None,  # token_indices
                None,  # token_probs
                ctx.num_experts,
                ctx.group,
                async_finish=True,
                handle=dispatch_states["handle"],
                num_loop_stage=ctx.num_loop_stage,
            )
            states["dispatched_indices"] = ctx.dispatched_indices[stage_idx]
            handle = list(states["handle"])
            handle[DETAILS_METAS_OFFSET] = ctx.details_metas[stage_idx]
            states["handle"] = tuple(handle)
            dispatch_history.append((tokens, None, None, states))
            dispatch_events.append(event)

        (
            combine_buffers,
            combine_probs,
            output_tokens,
            output_topk_weights,
            is_buffer_active,
            expert_result_buffer,
        ) = _init_combine_buffer(
            ctx.seq_len,
            ctx.hidden_size,
            ctx.topk,
            ctx.local_num_experts,
            ctx.combine_notify_infos,
            expert_num_for_combine_stage_prefix,
            has_prob=True,
        )

        backward_w_callbacks = []
        for local_expert_id in range(ctx.local_num_experts):
            dispatch_stage_idx = _get_expert_dependencies(
                local_expert_id, expert_num_for_dispatch_stage_prefix
            )

            if dispatch_events[dispatch_stage_idx]:
                dispatch_events[dispatch_stage_idx].calc_stream_wait(
                    ctx.group.id
                )

            (
                tokens,
                indices,
                details_metas,
                ori_len,
            ) = local_dispatch_backward_func(
                dispatch_history[: dispatch_stage_idx + 1],
                ctx.output_rowmap_list[: dispatch_stage_idx + 1],
                ctx.output_rowmap_len_list[: dispatch_stage_idx + 1],
                ctx.local_num_experts,
                local_expert_id,
                out_len=ctx.tokens_per_expert_list[local_expert_id],
                num_loop_stage=ctx.num_loop_stage,
            )

            if ctx.flash_ep_split_expert_bw:
                tokens, probs, backward_w_callback = ctx.expert_nodes[
                    local_expert_id
                ].backward(
                    tokens,
                    ctx.probs[local_expert_id],
                    split_expert_bw=ctx.flash_ep_split_expert_bw,
                )
                backward_w_callbacks.append(backward_w_callback)
            else:
                tokens, probs = ctx.expert_nodes[local_expert_id].backward(
                    tokens, ctx.probs[local_expert_id]
                )

            local_combine_backward_func(
                tokens,
                probs,
                indices,
                details_metas,
                ctx.combine_notify_infos,
                combine_buffers,
                combine_probs,
                local_expert_id,
                ori_len,
                is_buffer_active,
                ctx.group,
                ctx.num_loop_stage,
            )
            tokens, indices, details_metas = None, None, None

            if expert_result_buffer[local_expert_id] != -1:
                stage_idx = expert_result_buffer[local_expert_id]
                is_buffer_active[stage_idx] = 0
                tokens = combine_buffers[stage_idx].astype("bfloat16")
                probs = combine_probs[stage_idx]
                combine_buffers[stage_idx] = paddle.empty(
                    [0, tokens.shape[1]], combine_buffers[stage_idx].dtype
                )
                combine_probs[stage_idx] = paddle.empty(
                    [0, probs.shape[1]], combine_probs[stage_idx].dtype
                )
                combine_states = ctx.combine_notify_infos[stage_idx]
                _, _, event = combine_func(
                    tokens,
                    ctx.group,
                    topk_weights=probs,
                    previous_event=get_event_from_calc_stream(ctx.group.id),
                    async_finish=True,
                    allocate_on_comm_stream=True,
                    output=output_tokens,
                    output_topk_weights=output_topk_weights,
                    handle=combine_states["handle"],
                    num_loop_stage=ctx.num_loop_stage,
                )

        if ctx.flash_ep_split_expert_bw:
            for backward_w_callback in backward_w_callbacks:
                backward_w_callback()

        if event:
            event.calc_stream_wait(ctx.group.id)

        output_tokens = output_tokens.astype("bfloat16")
        return output_tokens, output_topk_weights, None
