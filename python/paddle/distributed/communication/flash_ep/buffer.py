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

# The file has been adapted from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import paddle
import paddle.distributed as dist
from paddle.base.core import (
    FEBuffer as CppBuffer,
    FEConfig as Config,
)

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group

from paddle.base.core import EventHandle

from .utils import EventOverlap


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
        - high-throughput internode all-to-all (dispatch and combine, using RDMA and NVLink)

    Attributes:
        num_sms: the SMs used in high-throughput kernels.
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        num_nvl_bytes: the buffer size for intranode NVLink communication.
        num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
        runtime: the C++ runtime.
    """

    num_sms: int = 20

    def __init__(
        self,
        group: Group,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 12,
        num_pipeline_stages: int = 1,
    ) -> None:
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
            low_latency_mode: whether to enable low-latency mode.
            num_qps_per_rank: the number of QPs for RDMA, the low-latency mode requires that this number equals
                to the number of local experts.
        """

        # Initialize the CPP runtime
        self.rank = group.rank
        self.group_size = group.world_size
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.num_pipeline_stages = num_pipeline_stages
        self.runtime = CppBuffer(
            self.rank,
            self.group_size,
            num_pipeline_stages,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode,
            group.id,
        )

        # Synchronize device IDs
        device_ids = []
        local_device_id = self.runtime.get_local_device_id()
        dist.all_gather_object(device_ids, local_device_id, group)

        # Synchronize IPC handles
        ipc_handles = []
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        dist.all_gather_object(ipc_handles, local_ipc_handle, group)

        assert self.runtime.get_num_rdma_ranks() > 1, (
            f"current group:{group}, rank:{self.rank}"
        )
        assert not low_latency_mode, "current low_latency_mode is not supported"

        # Synchronize NVSHMEM unique IDs
        root_unique_id = None
        if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
            # Enable IBGDA
            assert num_qps_per_rank > 0
            num_qps = num_qps_per_rank
            internode_ll_two_stage_num_max_qps = 32
            if num_qps_per_rank < internode_ll_two_stage_num_max_qps:
                num_qps = internode_ll_two_stage_num_max_qps
            if not os.getenv("NVSHMEM_DISABLE_P2P"):
                os.environ['NVSHMEM_DISABLE_P2P'] = '1'
            os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
            os.environ['NVSHMEM_IBGDA_NIC_HANDLER'] = 'gpu'
            os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps}'
            # Make sure QP depth is always larger than the number of on-flight WRs, so that we can skip WQ slot check
            os.environ['NVSHMEM_QP_DEPTH'] = '1024'
            # NOTES: NVSHMEM initialization requires at least 256 MiB
            os.environ['NVSHMEM_CUMEM_GRANULARITY'] = f'{2**29}'

            nvshmem_unique_ids = []
            if (low_latency_mode and self.rank == 0) or (
                not low_latency_mode and self.runtime.get_rdma_rank() == 0
            ):
                root_unique_id = self.runtime.get_local_nvshmem_unique_id()
            dist.all_gather_object(nvshmem_unique_ids, root_unique_id, group)
            root_unique_id = nvshmem_unique_ids[
                0 if low_latency_mode else self.runtime.get_root_rdma_rank(True)
            ]

        # Make CPP runtime available
        self.runtime.sync(device_ids, ipc_handles, root_unique_id)
        assert self.runtime.is_available()

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """
        Set the number of SMs to use in high-throughput kernels.

        Arguments:
            new_num_sms: the new number to be set.
        """

        assert new_num_sms % 2 == 0, 'The SM count must be even'
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture a CUDA event on the current stream

        Returns:
            event: the captured event.
        """
        return EventOverlap(EventHandle())

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """
        config_map = {
            2: Config(Buffer.num_sms, 16, 256, 6, 128),
            4: Config(Buffer.num_sms, 16, 256, 6, 128),
            8: Config(Buffer.num_sms, 6, 64, 6, 128),
            16: Config(Buffer.num_sms, 16, 288, 20, 128),
            24: Config(Buffer.num_sms, 8, 288, 32, 128),
            32: Config(Buffer.num_sms, 8, 288, 32, 128),
            64: Config(Buffer.num_sms, 20, 288, 28, 128),
            128: Config(Buffer.num_sms, 20, 560, 32, 128),
            144: Config(Buffer.num_sms, 32, 720, 12, 128),
            160: Config(Buffer.num_sms, 28, 720, 12, 128),
        }
        assert num_ranks in config_map, (
            f'Unsupported number of EP ranks: {num_ranks}'
        )
        return config_map[num_ranks]

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """
        config_map = {
            2: Config(Buffer.num_sms, 6, 256, 6, 128),
            4: Config(Buffer.num_sms, 6, 256, 6, 128),
            8: Config(Buffer.num_sms, 6, 64, 6, 128),
            16: Config(Buffer.num_sms, 2, 288, 28, 128),
            24: Config(Buffer.num_sms, 1, 288, 20, 128),
            32: Config(Buffer.num_sms, 1, 288, 20, 128),
            64: Config(Buffer.num_sms, 1, 288, 20, 128),
            128: Config(Buffer.num_sms, 1, 560, 12, 128),
            144: Config(Buffer.num_sms, 2, 720, 8, 128),
            160: Config(Buffer.num_sms, 2, 720, 8, 128),
        }
        assert num_ranks in config_map, (
            f'Unsupported number of EP ranks: {num_ranks}'
        )
        return config_map[num_ranks]

    # noinspection PyTypeChecker
    def dispatch(
        self,
        x: paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor],
        handle: tuple | None = None,
        num_tokens_per_rank: paddle.Tensor | None = None,
        num_tokens_per_rdma_rank: paddle.Tensor | None = None,
        is_token_in_rank: paddle.Tensor | None = None,
        num_tokens_per_expert: paddle.Tensor | None = None,
        topk_idx: paddle.Tensor | None = None,
        topk_weights: paddle.Tensor | None = None,
        expert_alignment: int = 1,
        config: Config | None = None,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        num_experts: int = 0,
        asymmetric_handle: tuple | None = None,
        pipeline_stage_id: int = 0,
    ) -> tuple[
        tuple[paddle.Tensor, paddle.Tensor] | paddle.Tensor,
        paddle.Tensor | None,
        paddle.Tensor | None,
        list[int],
        tuple,
        EventOverlap,
    ]:
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `paddle.Tensor` or tuple of `paddle.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `paddle.bfloat16`; for the second type, the first element of the tuple must be shaped as
                `[num_tokens, hidden]` with type `paddle.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `paddle.float`.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `int64`, the expert indices selected by each token,
                `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `float`, the expert weights of each token to dispatch.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
                received token count.
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = (
            self.get_dispatch_config(self.group_size)
            if config is None
            else config
        )

        assert self.runtime.get_num_rdma_ranks() > 1

        return self.internode_dispatch(
            x,
            handle,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            topk_idx,
            topk_weights,
            expert_alignment,
            config,
            previous_event,
            async_finish,
            allocate_on_comm_stream,
            num_experts,
            asymmetric_handle=asymmetric_handle,
            pipeline_stage_id=pipeline_stage_id,
        )

    # noinspection PyTypeChecker
    def combine(
        self,
        x: paddle.Tensor,
        handle: tuple,
        topk_weights: paddle.Tensor | None = None,
        output: paddle.Tensor | None = None,
        output_topk_weights: paddle.Tensor | None = None,
        config: Config | None = None,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        pipeline_stage_id: int = 0,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None, EventOverlap]:
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `[num_tokens, hidden]` with `bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            topk_weights: `[num_tokens, num_topk]` with `float`, the tokens' top-k weights for reducing to its original ranks.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = (
            self.get_combine_config(self.group_size)
            if config is None
            else config
        )

        assert self.runtime.get_num_rdma_ranks() > 1

        return self.internode_combine(
            x,
            handle,
            topk_weights,
            output,
            output_topk_weights,
            config,
            previous_event,
            async_finish,
            allocate_on_comm_stream,
            pipeline_stage_id,
        )

    # noinspection PyTypeChecker
    def internode_dispatch(
        self,
        x: paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor],
        handle: tuple | None = None,
        num_tokens_per_rank: paddle.Tensor | None = None,
        num_tokens_per_rdma_rank: paddle.Tensor | None = None,
        is_token_in_rank: paddle.Tensor | None = None,
        num_tokens_per_expert: paddle.Tensor | None = None,
        topk_idx: paddle.Tensor | None = None,
        topk_weights: paddle.Tensor | None = None,
        expert_alignment: int = 1,
        config: Config | None = None,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        num_experts: int = 0,
        asymmetric_handle=None,
        pipeline_stage_id: int = 0,
    ) -> tuple[
        tuple[paddle.Tensor, paddle.Tensor] | paddle.Tensor,
        paddle.Tensor | None,
        paddle.Tensor | None,
        list[int],
        tuple,
        EventOverlap,
    ]:
        """
        Internode dispatch implementation, for more details, please refer to the `dispatch` docs.
        Normally, you should not directly call this function.
        """
        assert config is not None

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert num_experts > 0

            if asymmetric_handle is not None:
                (
                    asymm_send_combine_schedule_map,
                    asymm_recv_rdma_counter_loop_prefix_sum,
                    asymm_recv_rdma_rank_prefix_sum,
                    asymm_recv_rdma_channel_prefix_matrix,
                    asymm_send_rdma_head,
                    asymm_send_nvl_head,
                    asymm_aggregated_nvl_head,
                ) = asymmetric_handle
            else:
                asymm_send_combine_schedule_map = None
                asymm_recv_rdma_counter_loop_prefix_sum = None
                asymm_recv_rdma_rank_prefix_sum = None
                asymm_recv_rdma_channel_prefix_matrix = None
                asymm_send_rdma_head = None
                asymm_send_nvl_head = None
                asymm_aggregated_nvl_head = None

            (
                is_token_in_rank,
                rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
            ) = handle
            num_recv_tokens = recv_src_meta.shape[0]
            num_rdma_recv_tokens = send_nvl_head.shape[0]
            (
                recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                _,
                _,
                _,
                _,
                _,
                _,
                recv_src_meta,
                _,
                _,
                event,
            ) = self.runtime.internode_dispatch(
                x,
                x_scales,
                topk_idx,
                topk_weights,
                None,
                None,
                is_token_in_rank,
                None,
                num_recv_tokens,
                num_rdma_recv_tokens,
                rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                asymm_send_combine_schedule_map,
                asymm_recv_rdma_counter_loop_prefix_sum,
                asymm_recv_rdma_rank_prefix_sum,
                asymm_recv_rdma_channel_prefix_matrix,
                asymm_send_rdma_head,
                asymm_send_nvl_head,
                asymm_aggregated_nvl_head,
                expert_alignment,
                config,
                getattr(previous_event, 'event', None),
                async_finish,
                allocate_on_comm_stream,
                num_experts,
                pipeline_stage_id,
            )
            handle = (
                is_token_in_rank,
                rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                EventOverlap(event),
            )
        else:
            assert (
                num_tokens_per_rank is not None
                and is_token_in_rank is not None
                and num_tokens_per_expert is not None
            )
            num_experts = num_tokens_per_expert.shape[0]
            (
                recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
                event,
            ) = self.runtime.internode_dispatch(
                x,
                x_scales,
                topk_idx,
                topk_weights,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                expert_alignment,
                config,
                getattr(previous_event, 'event', None),
                async_finish,
                allocate_on_comm_stream,
                num_experts,
            )
            handle = (
                is_token_in_rank,
                rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                EventOverlap(event),
            )

    def internode_fused_notify(
        self,
        x: paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor],
        topk_idx: paddle.Tensor | None = None,
        dispatch_num_tokens_per_rank: paddle.Tensor | None = None,
        dispatch_num_tokens_per_rdma_rank: paddle.Tensor | None = None,
        dispatch_num_tokens_per_expert: paddle.Tensor | None = None,
        dispatch_is_token_in_rank: paddle.Tensor | None = None,
        combine_num_tokens_per_rank: paddle.Tensor | None = None,
        combine_num_tokens_per_rdma_rank: paddle.Tensor | None = None,
        combine_is_token_in_rank: paddle.Tensor | None = None,
        expert_alignment: int = 1,
        config: Config | None = None,
    ):
        # Default config
        config = (
            self.get_dispatch_config(self.group_size)
            if config is None
            else config
        )
        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)

        (
            dispatch_num_recv_tokens_per_expert_list,
            dispatch_num_recv_tokens_list,
            dispatch_num_rdma_recv_tokens_list,
            dispatch_rdma_channel_prefix_matrix,
            dispatch_gbl_channel_prefix_matrix,
            dispatch_recv_rdma_rank_prefix_sum,
            dispatch_recv_gbl_rank_prefix_sum,
            combine_num_recv_tokens_list,
            combine_num_rdma_recv_tokens_list,
            combine_recv_rdma_rank_prefix_sum,
            combine_recv_rdma_channel_prefix_matrix,
            combine_recv_gbl_channel_prefix_matrix,
            combine_send_rdma_head,
            combine_send_nvl_head,
            combine_num_rdma_recv_tokens_cumsum,
        ) = self.runtime.internode_fused_notify(
            x,
            x_scales,
            topk_idx,
            dispatch_num_tokens_per_rank,
            dispatch_num_tokens_per_rdma_rank,
            dispatch_num_tokens_per_expert,
            dispatch_is_token_in_rank,
            combine_num_tokens_per_rank,
            combine_num_tokens_per_rdma_rank,
            combine_is_token_in_rank,
            expert_alignment,
            config,
        )

        dispatch_notify_infos = []
        for stage_idx in range(self.num_pipeline_stages):
            handle = (
                dispatch_is_token_in_rank[stage_idx],
                dispatch_rdma_channel_prefix_matrix[stage_idx],
                dispatch_gbl_channel_prefix_matrix[stage_idx],
                None,
                dispatch_recv_rdma_rank_prefix_sum[stage_idx],
                None,
                dispatch_recv_gbl_rank_prefix_sum[stage_idx],
                paddle.empty(
                    [dispatch_num_recv_tokens_list[stage_idx], 0]
                ),  # TODO: Just used to pass shape, can be simplified later
                None,
                paddle.empty(
                    [dispatch_num_rdma_recv_tokens_list[stage_idx], 0]
                ),
            )

            dispatch_states = {}
            dispatch_states["num_recv_tokens_per_expert_list"] = (
                dispatch_num_recv_tokens_per_expert_list[stage_idx]
            )
            dispatch_states["handle"] = handle
            dispatch_notify_infos.append(dispatch_states)

        asymm_aggregated_nvl_head = paddle.empty(
            [sum(combine_num_rdma_recv_tokens_list), 8], dtype="int32"
        )

        asymmetric_handle = (
            combine_num_rdma_recv_tokens_cumsum,
            combine_recv_rdma_rank_prefix_sum,
            combine_recv_rdma_channel_prefix_matrix,
            combine_send_rdma_head,
            combine_send_nvl_head,
            asymm_aggregated_nvl_head,
        )

        combine_notify_infos = []
        asymm_recv_rdma_start_idx = 0
        for stage_idx in range(self.num_pipeline_stages):
            asymm_recv_rdma_end_idx = (
                asymm_recv_rdma_start_idx
                + combine_num_rdma_recv_tokens_list[stage_idx]
            )

            combine_handle = (
                None,
                None,
                None,
                combine_recv_rdma_channel_prefix_matrix[stage_idx],
                combine_recv_rdma_rank_prefix_sum[stage_idx],
                combine_recv_gbl_channel_prefix_matrix[stage_idx],
                None,
                None,
                combine_send_rdma_head[stage_idx],
                asymm_aggregated_nvl_head[
                    asymm_recv_rdma_start_idx:asymm_recv_rdma_end_idx
                ],  # inplace slice
            )
            asymm_recv_rdma_start_idx = asymm_recv_rdma_end_idx

            combine_states = {}
            combine_states["handle"] = combine_handle
            combine_states["num_combine_tokens"] = combine_num_recv_tokens_list[
                stage_idx
            ]
            combine_notify_infos.append(combine_states)

        return dispatch_notify_infos, combine_notify_infos, asymmetric_handle

    # noinspection PyTypeChecker
    def internode_combine(
        self,
        x: paddle.Tensor,
        handle: tuple | list,
        topk_weights: paddle.Tensor | None = None,
        output: paddle.Tensor | None = None,
        output_topk_weights: paddle.Tensor | None = None,
        config: Config | None = None,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        pipeline_stage_id: int = 0,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None, EventOverlap]:
        """
        Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.
        """
        assert config is not None

        # Unpack handle
        (
            _,
            _,
            _,
            rdma_channel_prefix_matrix,
            rdma_rank_prefix_sum,
            gbl_channel_prefix_matrix,
            gbl_rank_prefix_sum,
            _,
            send_rdma_head,
            send_nvl_head,
        ) = handle

        # Launch the kernel
        combined_x, combined_topk_weights, event = (
            self.runtime.internode_combine(
                x,
                topk_weights,
                rdma_channel_prefix_matrix,
                rdma_rank_prefix_sum,
                gbl_channel_prefix_matrix,
                send_rdma_head,
                send_nvl_head,
                output,
                output_topk_weights,
                config,
                getattr(previous_event, 'event', None),
                async_finish,
                allocate_on_comm_stream,
                pipeline_stage_id,
            )
        )
        return combined_x, combined_topk_weights, EventOverlap(event)
