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
    local_combine_func,
    local_dispatch_func,
    notify_dispatch_and_combine,
)
from .utils import (
    get_event_from_calc_stream,
)

logger = logging.getLogger(__name__)


def build_pipeline_stage_infos(
    local_num_experts,
    num_loop_stage,
    expert_num_for_dispatch_stage=None,  # dispatch时，每个流水线stage传输了几个专家
    expert_num_for_combine_stage=None,  # combine时，每个流水线stage传输了几个专家
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
    ):
        expert_num_for_dispatch_stage_prefix = pipeline_stage_infos[
            "expert_num_for_dispatch_stage_prefix"
        ]
        expert_num_for_combine_stage_prefix = pipeline_stage_infos[
            "expert_num_for_combine_stage_prefix"
        ]
        local_expert_to_stage_map = pipeline_stage_infos[
            "local_expert_to_stage_map"
        ]
        num_loop_stage = pipeline_stage_infos["num_loop_stage"]
        ctx.pipeline_stage_infos = pipeline_stage_infos

        ctx.expert_nodes = []
        for expert in expert_funcs:
            if hasattr(expert, "build_expert_node"):
                expert = expert.build_expert_node()
            ctx.expert_nodes.append(expert)

        local_num_experts = num_experts // group.nranks
        ctx.local_num_experts = local_num_experts
        ctx.num_experts = num_experts
        ctx.group = group
        ctx.token_probs_shape = token_probs.shape
        ctx.token_probs_dtype = token_probs.dtype

        (
            dispatch_notify_infos,
            combine_notify_infos,
            tokens_per_expert_list,
            asymmetric_handle,
        ) = notify_dispatch_and_combine(
            hidden_states,
            token_indices,
            local_expert_to_stage_map,
            num_experts,
            group=group,
            num_loop_stage=num_loop_stage,
        )

        ctx.dispatch_notify_infos = dispatch_notify_infos
        ctx.combine_notify_infos = combine_notify_infos
        ctx.tokens_per_expert_list = tokens_per_expert_list

        combine_buffers = []
        for stage_idx in range(num_loop_stage):
            combine_out_len = combine_notify_infos[stage_idx][
                "num_combine_tokens"
            ]
            combine_buffers.append(
                (
                    paddle.zeros(
                        [combine_out_len, hidden_states.shape[1]],
                        dtype="float32",
                    ),
                )
            )

        dispatch_history = []
        dispatch_events = []
        ctx.dispatched_indices = []
        ctx.details_metas = []
        for stage_idx in range(num_loop_stage):
            dispatch_states = dispatch_notify_infos[stage_idx]
            tokens, probs, states, event = dispatch_func(
                hidden_states,
                token_indices,
                token_probs,
                num_experts,
                group,
                async_finish=True,
                handle=dispatch_states["handle"],
                asymmetric_handle=asymmetric_handle,
            )

            ctx.dispatched_indices.append(states["dispatched_indices"])
            ctx.details_metas.append(states["handle"][-3])
            dispatch_history.append((tokens, probs, states))
            dispatch_events.append(event)

        combine_events = []
        ctx.probs = []
        for local_expert_id in range(local_num_experts):
            # 找到当前专家依赖的dispatch轮数
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

            if dispatch_events[dispatch_stage_idx]:
                dispatch_events[dispatch_stage_idx].calc_stream_wait(group.id)

            out_len = tokens_per_expert_list[local_expert_id]
            (
                tokens,
                probs,
                _,
                details_metas,
                ori_len,
            ) = local_dispatch_func(
                dispatch_history[: dispatch_stage_idx + 1],
                local_expert_id,
                out_len,
            )

            ctx.probs.append(probs)

            # tokens = ctx.expert_nodes[local_expert_id].forward(tokens, probs, num_sms=112)
            tokens = ctx.expert_nodes[local_expert_id].forward(
                tokens,
                probs,
                [tokens.shape[0]],
                out_len,
            )

            local_combine_func(
                tokens,
                None,  # probs
                None,  # indices
                details_metas,
                combine_notify_infos,
                combine_buffers,
                local_expert_id,
                ori_len,
            )
            combine_events.append(get_event_from_calc_stream(group.id))

        # 每次global combine的时候, 用fp32的精度inplace累加到output_tokens上
        output_tokens = paddle.zeros(hidden_states.shape, token_probs.dtype)
        for stage_idx in range(num_loop_stage):
            local_expert_id = expert_num_for_combine_stage_prefix[stage_idx]

            previous_event = combine_events[local_expert_id - 1]

            combine_states = combine_notify_infos[stage_idx]
            _, _, event = combine_func(
                combine_buffers[stage_idx][0].astype("bfloat16"),
                group,
                previous_event=previous_event,
                async_finish=True,
                allocate_on_comm_stream=True,
                output=output_tokens,
                handle=combine_states["handle"],
            )
            combine_buffers[stage_idx] = None  # 及时释放显存

        if event:
            event.calc_stream_wait(group.id)

        return output_tokens.astype(hidden_states.dtype), tokens_per_expert_list

    @staticmethod
    def backward(ctx, output_grad):
        expert_num_for_dispatch_stage_prefix = ctx.pipeline_stage_infos[
            "expert_num_for_dispatch_stage_prefix"
        ]
        expert_num_for_combine_stage_prefix = ctx.pipeline_stage_infos[
            "expert_num_for_combine_stage_prefix"
        ]
        num_loop_stage = ctx.pipeline_stage_infos["num_loop_stage"]

        combine_buffers = []
        for stage_idx in range(num_loop_stage):
            combine_out_len = ctx.combine_notify_infos[stage_idx][
                "num_combine_tokens"
            ]
            combine_buffers.append(
                (
                    paddle.zeros(
                        [combine_out_len, output_grad.shape[1]], dtype="float32"
                    ),
                    paddle.zeros(
                        [combine_out_len, ctx.token_probs_shape[1]],
                        dtype="float32",
                    ),
                )
            )

        dispatch_history = []
        dispatch_events = []
        for stage_idx in range(num_loop_stage):
            dispatch_states = ctx.dispatch_notify_infos[stage_idx]
            tokens, _, states, event = dispatch_func(
                output_grad,
                None,  # token_indices
                None,  # token_probs
                ctx.num_experts,
                ctx.group,
                async_finish=True,
                handle=dispatch_states["handle"],
            )
            states["dispatched_indices"] = ctx.dispatched_indices[stage_idx]
            handle = list(states["handle"])
            handle[-3] = ctx.details_metas[stage_idx]
            states["handle"] = tuple(handle)
            dispatch_history.append((tokens, None, states))
            dispatch_events.append(event)

        combine_events = []
        for local_expert_id in range(ctx.local_num_experts):
            # 找到当前专家依赖的dispatch轮数
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

            if dispatch_events[dispatch_stage_idx]:
                dispatch_events[dispatch_stage_idx].calc_stream_wait(
                    ctx.group.id
                )

            out_len = ctx.tokens_per_expert_list[local_expert_id]
            (
                tokens,
                _,
                indices,
                details_metas,
                ori_len,
            ) = local_dispatch_func(
                dispatch_history[: dispatch_stage_idx + 1],
                local_expert_id,
                out_len,
            )

            tokens, probs = ctx.expert_nodes[local_expert_id].backward(
                tokens, ctx.probs[local_expert_id]
            )

            local_combine_func(
                tokens,
                probs,
                indices,
                details_metas,
                ctx.combine_notify_infos,
                combine_buffers,
                local_expert_id,
                ori_len,
            )
            combine_events.append(get_event_from_calc_stream(ctx.group.id))

        output_tokens = paddle.zeros(output_grad.shape, ctx.token_probs_dtype)
        output_topk_weights = paddle.zeros(
            ctx.token_probs_shape, ctx.token_probs_dtype
        )
        for stage_idx in range(num_loop_stage):
            local_expert_id = expert_num_for_combine_stage_prefix[stage_idx]

            previous_event = combine_events[local_expert_id - 1]

            combine_states = ctx.combine_notify_infos[stage_idx]
            _, _, event = combine_func(
                combine_buffers[stage_idx][0].astype("bfloat16"),
                ctx.group,
                topk_weights=combine_buffers[stage_idx][1],
                previous_event=previous_event,
                async_finish=True,
                allocate_on_comm_stream=True,
                output=output_tokens,
                output_topk_weights=output_topk_weights,
                handle=combine_states["handle"],
            )
            combine_buffers[stage_idx] = None  # 及时释放显存

        if event:
            event.calc_stream_wait(ctx.group.id)

        output_tokens = output_tokens.astype(output_grad.dtype)
        return output_tokens, output_topk_weights, None
