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
from abc import ABC, abstractmethod
from itertools import accumulate

import paddle

from .asymmetric_a2a import (
    all_in_one_combine_func,
    dispatch_func,
    get_flashep_rowmap_func,
    local_combine_backward_func,
    local_combine_forward_func,
    local_dispatch_backward_func,
    local_dispatch_forward_func,
    notify_dispatch_and_combine_func,
    set_tokens_ready_func,
)
from .utils import (
    get_event_from_calc_stream,
)

logger = logging.getLogger(__name__)

DETAILS_METAS_OFFSET = -3


def init_pipeline_stage_infos(
    num_local_experts,
    num_pipeline_stages,
    experts_per_dispatch_stage=None,
    experts_per_combine_stage=None,
):
    """
    To build pipeline stage information for FlashEP parallel processing.

    Based on the number of local experts (num_local_experts) and pipeline stages (num_pipeline_stages),
    calculates the number of experts responsible for each stage, along with corresponding prefix sums,
    to determine which pipeline stage each expert belongs to in dispatch and combine stages respectively.

    Args:
        num_local_experts (int):
            Total number of experts on each rank.

        num_pipeline_stages (int):
            Number of pipeline stages, must satisfy 0 < num_pipeline_stages <= num_local_experts.

        experts_per_dispatch_stage (list[int] or None):
            List of length num_pipeline_stages, indicating how many experts each stage is responsible for
            during the dispatch phase. If None, experts are evenly distributed.

        experts_per_combine_stage (list[int] or None):
            Same as above, for the combine stage. If None, experts are evenly distributed.

    Examples:
        >>> num_local_experts=6
        >>> num_pipeline_stages=3
        >>> experts_per_dispatch_stage=[1, 2, 3]
        >>> experts_per_combine_stage=[2, 2, 2]
        # This configuration means:
        # - Single GPU has 6 experts, FlashEP specifies 3 pipeline stages
        # - Dispatch phase: 1st stage sends 1 expert, 2nd stage sends 2 experts, 3rd stage sends 3 experts
        # - Combine phase: Each stage combines 2 experts

    Returns:
        dict: Dictionary containing pipeline stage information, including:
            - dispatch_stage_cumsum: Prefix sum of expert counts for dispatch stage
            - combine_stage_cumsum: Prefix sum of expert counts for combine stage
            - expert_stage_mapping: Mapping from local experts to pipeline stages
            - num_pipeline_stages: Number of pipeline loop stages
    """
    assert num_pipeline_stages is not None, (
        "num_pipeline_stages must be not None"
    )
    assert (
        num_pipeline_stages > 0 and num_pipeline_stages <= num_local_experts
    ), (
        "num_pipeline_stages must be greater than 0 and less than num_local_experts"
    )
    assert num_local_experts > 0, "num_local_experts must be greater than 0"

    if experts_per_dispatch_stage is None or experts_per_combine_stage is None:
        experts_per_stage = num_local_experts // num_pipeline_stages
        experts_per_dispatch_stage = [
            experts_per_stage for _ in range(num_pipeline_stages)
        ]
        experts_per_combine_stage = [
            experts_per_stage for _ in range(num_pipeline_stages)
        ]
    else:
        assert (
            len(experts_per_dispatch_stage) == num_pipeline_stages
            and len(experts_per_combine_stage) == num_pipeline_stages
        )
        assert (
            sum(experts_per_dispatch_stage) == num_local_experts
            and sum(experts_per_combine_stage) == num_local_experts
        )

    dispatch_stage_cumsum = list(accumulate(experts_per_dispatch_stage))
    combine_stage_cumsum = list(accumulate(experts_per_combine_stage))

    expert_stage_mapping = [[0, 0] for _ in range(num_local_experts)]
    for i in range(num_local_experts):
        for stage_idx, prefix in enumerate(dispatch_stage_cumsum):
            if prefix > i:
                break
        expert_stage_mapping[i][0] = stage_idx
        for stage_idx, prefix in enumerate(combine_stage_cumsum):
            if prefix > i:
                break
        expert_stage_mapping[i][1] = stage_idx

    logger.info(f"dispatch_stage_cumsum: {dispatch_stage_cumsum}")
    logger.info(f"combine_stage_cumsum: {combine_stage_cumsum}")
    logger.info(f"expert_stage_mapping: {expert_stage_mapping}")

    pipeline_stage_infos = {
        "dispatch_stage_cumsum": dispatch_stage_cumsum,
        "combine_stage_cumsum": combine_stage_cumsum,
        "expert_stage_mapping": paddle.to_tensor(
            expert_stage_mapping, dtype="int32"
        ),
        "num_pipeline_stages": num_pipeline_stages,
    }
    return pipeline_stage_infos


def _get_expert_dependencies(local_expert_id, dispatch_stage_cumsum):
    """
    To find the dispatch round on which the current expert depends
    """
    dispatch_stage_idx = next(
        (
            i
            for i, expert_prefix in enumerate(dispatch_stage_cumsum)
            if expert_prefix > local_expert_id
        ),
        None,
    )
    return dispatch_stage_idx


def _init_combine_buffer(
    seq_len,
    hidden_size,
    topk,
    num_local_experts,
    combine_notify_infos,
    combine_stage_cumsum,
    has_prob=False,
):
    combine_buffers = []
    combine_probs = []

    # During each global combine, accumulate to output_tokens in-place with fp32 precision.
    output_tokens = paddle.zeros([seq_len, hidden_size], "float32")
    output_topk_weights = (
        paddle.zeros([seq_len, topk], "float32") if has_prob else None
    )

    num_pipeline_stages = len(combine_notify_infos)
    for stage_idx in range(num_pipeline_stages):
        combine_out_len = combine_notify_infos[stage_idx]["num_combine_tokens"]
        combine_buffers.append(
            paddle.zeros([combine_out_len, hidden_size], dtype="float32")
        )
        if has_prob:
            combine_probs.append(
                paddle.zeros([combine_out_len, topk], dtype="float32")
            )

    # Indicates whether the current buffer will be written with data again.
    is_buffer_active = [1 for _ in range(num_pipeline_stages)]
    # Indicates which buffer to send in the global combine a2a after each local combine.
    is_buffer_ready = [-1 for _ in range(num_local_experts)]
    for stage_idx, local_expert_id_prefix in enumerate(combine_stage_cumsum):
        is_buffer_ready[local_expert_id_prefix - 1] = stage_idx

    return (
        combine_buffers,
        combine_probs,
        output_tokens,
        output_topk_weights,
        is_buffer_active,
        is_buffer_ready,
    )


def _quantize_if_needed(input, need_quant):
    scale = None
    if need_quant:
        input, scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            input, output_scale_transpose=False
        )
    return input, scale


def _extract_expert_interface(expert_nodes, flash_ep_recompute_local_dispatch):
    res = []
    for expert in expert_nodes:
        if hasattr(expert, "build_expert_node"):
            expert = expert.build_expert_node()
        res.append(expert)
        assert hasattr(expert, "forward")
        assert hasattr(expert, "backward")

        if flash_ep_recompute_local_dispatch:
            assert hasattr(expert, "clear_input_tensors")
            assert hasattr(expert, "set_input_tensors")
    return res


class BaseExpertNode(ABC):
    @abstractmethod
    def forward(self, tokens, probs, scale=None):
        pass

    @abstractmethod
    def backward(self, grad_tokens, grad_probs, split_expert_b=False):
        pass

    def set_input_tensors(self, tokens, probs):
        pass

    def clear_input_tensors():
        pass


class FlashEPFunction(paddle.autograd.PyLayer):
    """
    FlashEP (Flash Expert Parallel) Function.

    An expert parallel implementation based on communication-computation pipeline overlapping,
    optimizing Mixture of Experts (MoE) model performance through a 6-stage pipeline execution:
    1. Meta Information Exchange - Coordinates communication patterns between experts
    2. Global Dispatch All-to-All Communication - Distributes tokens across nodes
    3. Local Dispatch Data Redistribution - Reorganizes data within nodes
    4. Expert Computation - Executes expert forward/backward computation
    5. Local Combine Data Aggregation - Aggregates expert outputs within nodes
    6. Global Combine All-to-All Communication - Collects expert outputs across nodes

    Designed as non-intrusive - users only need to provide expert computation nodes
    to benefit from pipeline overlapping performance gains.
    """

    @staticmethod
    def forward(
        ctx,
        input,
        token_probs,
        token_indices,
        pipeline_stage_infos,
        num_experts,
        expert_nodes,
        group,
        flash_ep_fp8_dispatch_a2a=False,
        flash_ep_split_expert_bw=False,
        flash_ep_recompute_local_dispatch=False,
    ):
        """
        FlashEP forward propagation function.

        Executes expert parallel computation through a 6-stage pipeline,
        achieving communication-computation overlapping optimization.

        Args:
            ctx: Context object for storing information required by backward propagation
            input (Tensor): Input activations, 2D tensor with shape [sequence_length, hidden_size]
            token_probs (Tensor): Routing weights, 2D tensor with shape [sequence_length, topk]
            token_indices (Tensor): Routing dispatch results, 2D tensor with shape [sequence_length, topk]
            pipeline_stage_infos (dict): Pipeline metadata obtained from init_pipeline_stage_infos, containing:
                - dispatch_stage_cumsum: Cumulative sum of dispatch stages
                - combine_stage_cumsum: Cumulative sum of combine stages
                - expert_stage_mapping: Expert to stage mapping
                - num_pipeline_stages: Number of pipeline stages
            num_experts (int): Total number of global experts
            expert_nodes (list): Expert computation nodes, must inherit and implement BaseExpertNode interface
            group: Expert parallel communication group
            flash_ep_fp8_dispatch_a2a (bool, optional): Whether to use FP8 precision for forward dispatch communication.
                Reduces communication volume with no precision loss
            flash_ep_split_expert_bw (bool, optional): Whether to separate expert gradient computation during backward propagation.
                Enables overlapping of expert gradient computation with communication for higher performance
            flash_ep_recompute_local_dispatch (bool, optional): Whether to recompute local dispatch results.
                Only retains global dispatch results in forward, recomputes local dispatch in backward to reduce memory usage,
                with no precision loss

        Returns:
            Tuple[Tensor, list]:
                - Output tensor with same shape as input
                - List of token counts per expert

        Raises:
            AssertionError: When input tensor dimensions do not meet requirements
        """
        # Barrier head to avoid DeepEP timeout when training large models.
        paddle.distributed.barrier(group)

        assert len(input.shape) == 2, "input must be 2D tensor."
        assert len(token_probs.shape) == 2, "token_probs must be 2D tensor."
        assert len(token_indices.shape) == 2, "token_indices must be 2D tensor."

        input, scale = _quantize_if_needed(input, flash_ep_fp8_dispatch_a2a)

        dispatch_stage_cumsum = pipeline_stage_infos["dispatch_stage_cumsum"]
        combine_stage_cumsum = pipeline_stage_infos["combine_stage_cumsum"]
        expert_stage_mapping = pipeline_stage_infos["expert_stage_mapping"]
        ctx.num_pipeline_stages = pipeline_stage_infos["num_pipeline_stages"]

        # Phase 1: Meta Information Exchange.
        (
            dispatch_notify_infos,
            combine_notify_infos,
            tokens_per_expert_list,
            asymmetric_handle,
        ) = notify_dispatch_and_combine_func(
            input,
            token_indices,
            expert_stage_mapping,
            num_experts,
            group=group,
            num_pipeline_stages=ctx.num_pipeline_stages,
        )

        ctx.pipeline_stage_infos = pipeline_stage_infos
        ctx.num_local_experts = num_experts // group.nranks
        ctx.num_experts = num_experts
        ctx.group = group
        ctx.token_probs_shape = token_probs.shape
        ctx.token_probs_dtype = token_probs.dtype
        ctx.topk = token_indices.shape[1]
        ctx.seq_len = input.shape[0]
        ctx.hidden_size = input.shape[1]
        ctx.dispatch_notify_infos = dispatch_notify_infos
        ctx.combine_notify_infos = combine_notify_infos
        ctx.tokens_per_expert_list = tokens_per_expert_list
        ctx.flash_ep_split_expert_bw = flash_ep_split_expert_bw
        ctx.flash_ep_recompute_local_dispatch = (
            flash_ep_recompute_local_dispatch
        )

        ctx.expert_nodes = _extract_expert_interface(
            expert_nodes, flash_ep_recompute_local_dispatch
        )

        dispatch_history = []
        dispatch_events = []
        ctx.dispatched_indices = []
        ctx.details_metas = []
        # Phase 2: Global Dispatch A2A Communication Pipeline.
        for stage_idx in range(ctx.num_pipeline_stages):
            dispatch_states = dispatch_notify_infos[stage_idx]
            tokens, scale_, probs, states, event = dispatch_func(
                input,
                token_indices,
                token_probs,
                num_experts,
                group,
                scale=scale,
                async_finish=True,
                handle=dispatch_states["handle"],
                asymmetric_handle=asymmetric_handle,
                pipeline_stage_id=stage_idx,
                num_pipeline_stages=ctx.num_pipeline_stages,
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
            is_buffer_ready,
        ) = _init_combine_buffer(
            ctx.seq_len,
            ctx.hidden_size,
            ctx.topk,
            ctx.num_local_experts,
            ctx.combine_notify_infos,
            combine_stage_cumsum,
        )

        previous_event = get_event_from_calc_stream(group.id)

        ctx.probs = []
        ctx.output_rowmap_list = []
        ctx.output_rowmap_offset_list = []
        token_list = []
        probs_list = []
        handle_list = []
        is_tokens_ready = paddle.zeros([ctx.num_pipeline_stages], dtype="int32")
        for local_expert_id in range(ctx.num_local_experts):
            # Identify prior data dependencies and perform multi-stream synchronization.
            dispatch_stage_idx = _get_expert_dependencies(
                local_expert_id, dispatch_stage_cumsum
            )
            if dispatch_events[dispatch_stage_idx]:
                dispatch_events[dispatch_stage_idx].calc_stream_wait(group.id)
            if dispatch_stage_idx == len(ctx.output_rowmap_list):
                output_rowmap, output_rowmap_offset = get_flashep_rowmap_func(
                    ctx.dispatched_indices[dispatch_stage_idx],
                    ctx.num_local_experts,
                )
                ctx.output_rowmap_list.append(output_rowmap)
                ctx.output_rowmap_offset_list.append(output_rowmap_offset)

            # Phase 3: Local dispatch data redistribution.
            (
                tokens,
                scale_,
                probs,
                details_metas,
                original_token_count,
            ) = local_dispatch_forward_func(
                dispatch_history[: dispatch_stage_idx + 1],
                ctx.output_rowmap_list[: dispatch_stage_idx + 1],
                ctx.output_rowmap_offset_list[: dispatch_stage_idx + 1],
                ctx.num_local_experts,
                local_expert_id,
                out_len=ctx.tokens_per_expert_list[local_expert_id],
                num_pipeline_stages=ctx.num_pipeline_stages,
            )
            ctx.probs.append(probs)

            # Phase 4: Expert Computation.
            tokens = ctx.expert_nodes[local_expert_id].forward(
                tokens, probs, scale=scale_
            )
            if ctx.flash_ep_recompute_local_dispatch:
                ctx.expert_nodes[local_expert_id].clear_input_tensors()

            # Phase 5: Local combine data aggregation.
            local_combine_forward_func(
                tokens,
                details_metas,
                combine_notify_infos,
                combine_buffers,
                original_token_count,
                is_buffer_active,
                ctx.group,
                ctx.num_pipeline_stages,
            )
            tokens, scale_, probs, details_metas = None, None, None, None

            if is_buffer_ready[local_expert_id] != -1:
                # When a buffer is scheduled for all-to-all communication,
                # update the combine-related control signals.
                stage_idx = is_buffer_ready[local_expert_id]
                is_buffer_active[stage_idx] = 0
                tokens = combine_buffers[stage_idx].astype("bfloat16")
                combine_buffers[stage_idx] = paddle.empty(
                    [0, tokens.shape[1]], combine_buffers[stage_idx].dtype
                )  # TODO: Set None
                combine_states = combine_notify_infos[stage_idx]
                token_list.append(tokens)
                probs_list.append(None)
                handle_list.append(combine_states["handle"])
                set_tokens_ready_func(is_tokens_ready, stage_idx)

        event = all_in_one_combine_func(
            token_list,
            group,
            handle_list,
            topk_weights_list=probs_list,
            output=output_tokens,
            output_topk_weights=None,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
            num_pipeline_stages=ctx.num_pipeline_stages,
            is_tokens_ready=is_tokens_ready,
        )

        if ctx.flash_ep_recompute_local_dispatch:
            ctx.dispatch_history = dispatch_history
        dispatch_history = None

        if event:
            event.calc_stream_wait(group.id)

        return output_tokens.astype("bfloat16"), ctx.tokens_per_expert_list

    @staticmethod
    def backward(ctx, output_grad):
        # Barrier head to avoid DeepEP timeout when training large models.
        paddle.distributed.barrier(ctx.group)

        dispatch_stage_cumsum = ctx.pipeline_stage_infos[
            "dispatch_stage_cumsum"
        ]
        combine_stage_cumsum = ctx.pipeline_stage_infos["combine_stage_cumsum"]

        dispatch_history = []
        dispatch_events = []
        for stage_idx in range(ctx.num_pipeline_stages):
            dispatch_states = ctx.dispatch_notify_infos[stage_idx]
            tokens, _, _, states, event = dispatch_func(
                output_grad,
                None,  # token_indices
                None,  # token_probs
                ctx.num_experts,
                ctx.group,
                async_finish=True,
                handle=dispatch_states["handle"],
                pipeline_stage_id=stage_idx,
                num_pipeline_stages=ctx.num_pipeline_stages,
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
            is_buffer_ready,
        ) = _init_combine_buffer(
            ctx.seq_len,
            ctx.hidden_size,
            ctx.topk,
            ctx.num_local_experts,
            ctx.combine_notify_infos,
            combine_stage_cumsum,
            has_prob=True,
        )

        previous_event = get_event_from_calc_stream(ctx.group.id)

        token_list = []
        probs_list = []
        handle_list = []
        backward_w_callbacks = []
        is_tokens_ready = paddle.zeros([ctx.num_pipeline_stages], dtype="int32")
        for local_expert_id in range(ctx.num_local_experts):
            dispatch_stage_idx = _get_expert_dependencies(
                local_expert_id, dispatch_stage_cumsum
            )

            if dispatch_events[dispatch_stage_idx]:
                dispatch_events[dispatch_stage_idx].calc_stream_wait(
                    ctx.group.id
                )

            (
                tokens,
                indices,
                details_metas,
                original_token_count,
            ) = local_dispatch_backward_func(
                dispatch_history[: dispatch_stage_idx + 1],
                ctx.output_rowmap_list[: dispatch_stage_idx + 1],
                ctx.output_rowmap_offset_list[: dispatch_stage_idx + 1],
                ctx.num_local_experts,
                local_expert_id,
                out_len=ctx.tokens_per_expert_list[local_expert_id],
                num_pipeline_stages=ctx.num_pipeline_stages,
            )

            if ctx.flash_ep_recompute_local_dispatch:
                (
                    fwd_tokens,
                    fwd_scale_,
                    _,
                    _,
                    _,
                ) = local_dispatch_forward_func(
                    ctx.dispatch_history[: dispatch_stage_idx + 1],
                    ctx.output_rowmap_list[: dispatch_stage_idx + 1],
                    ctx.output_rowmap_offset_list[: dispatch_stage_idx + 1],
                    ctx.num_local_experts,
                    local_expert_id,
                    out_len=ctx.tokens_per_expert_list[local_expert_id],
                    num_pipeline_stages=ctx.num_pipeline_stages,
                )
                ctx.expert_nodes[local_expert_id].set_input_tensors(
                    fwd_tokens, scale=fwd_scale_
                )
                fwd_tokens, fwd_scale_ = None, None

            if ctx.flash_ep_split_expert_bw:
                tokens, probs, cb = ctx.expert_nodes[local_expert_id].backward(
                    tokens,
                    ctx.probs[local_expert_id],
                    split_expert_bw=ctx.flash_ep_split_expert_bw,
                )
                backward_w_callbacks.append(cb)
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
                original_token_count,
                is_buffer_active,
                ctx.group,
                ctx.num_pipeline_stages,
            )
            tokens, probs, indices, details_metas = None, None, None, None

            if is_buffer_ready[local_expert_id] != -1:
                # When a buffer is scheduled for all-to-all communication,
                # update the combine-related control signals.
                stage_idx = is_buffer_ready[local_expert_id]
                tokens = combine_buffers[stage_idx].astype("bfloat16")
                probs = combine_probs[stage_idx]
                combine_buffers[stage_idx] = paddle.empty(
                    [0, tokens.shape[1]], combine_buffers[stage_idx].dtype
                )
                combine_probs[stage_idx] = paddle.empty(
                    [0, probs.shape[1]], combine_probs[stage_idx].dtype
                )
                combine_states = ctx.combine_notify_infos[stage_idx]
                token_list.append(tokens)
                probs_list.append(probs)
                handle_list.append(combine_states["handle"])
                set_tokens_ready_func(is_tokens_ready, stage_idx)

        event = all_in_one_combine_func(
            token_list,
            ctx.group,
            handle_list,
            topk_weights_list=probs_list,
            output=output_tokens,
            output_topk_weights=output_topk_weights,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
            num_pipeline_stages=ctx.num_pipeline_stages,
            is_tokens_ready=is_tokens_ready,
        )

        ctx.dispatch_history = None

        if ctx.flash_ep_split_expert_bw:
            for cb in backward_w_callbacks:
                cb()

        # paddle.device.synchronize()
        if event:
            event.calc_stream_wait(ctx.group.id)

        return output_tokens.astype("bfloat16"), output_topk_weights, None
