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
from typing import TYPE_CHECKING, Callable

import paddle
import paddle.distributed as dist
from paddle.base.core import (
    Buffer as CppBuffer,
    Config,
    get_low_latency_rdma_size_hint,
)

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group

from paddle.base.core import EventHandle

from .utils import EventOverlap


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
        - high-throughput intranode all-to-all (dispatch and combine, using NVLink)
        - high-throughput internode all-to-all (dispatch and combine, using RDMA and NVLink)
        - low-latency all-to-all (dispatch and combine, using RDMA)

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
        num_qps_per_rank: int = 1,
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
        self.runtime = CppBuffer(
            self.rank,
            self.group_size,
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

        # Synchronize NVSHMEM unique IDs
        root_unique_id = None
        if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
            # Enable IBGDA for the low latency mode, which refers to "no package forwarding between NVLink and RDMA"
            if low_latency_mode:
                assert num_qps_per_rank > 0
                os.environ['NVSHMEM_DISABLE_P2P'] = '1'
                os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
                os.environ['NVSHMEM_IBGDA_NIC_HANDLER'] = 'gpu'
                os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = (
                    f'{num_qps_per_rank}'
                )
                # Make sure QP depth is always larger than the number of on-flight WRs, so that we can skip WQ slot check
                os.environ['NVSHMEM_QP_DEPTH'] = '1024'
                # NOTES: NVSHMEM initialization requires at least 256 MiB
                os.environ['NVSHMEM_CUMEM_GRANULARITY'] = f'{2 ** 29}'

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
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_ranks: int,
        num_experts: int,
    ) -> int:
        """
        Get a minimum size requirement for the RDMA buffer. The size calculation will be done with BF16.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_ranks: the number of EP group ranks.
            num_experts: the number of all experts.

        Returns:
            size: the RDMA buffer size recommended.
        """
        return get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts
        )

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
            8: Config(Buffer.num_sms, 6, 256, 6, 128),
            16: Config(Buffer.num_sms, 16, 288, 20, 128),
            24: Config(Buffer.num_sms, 8, 288, 32, 128),
            32: Config(Buffer.num_sms, 8, 288, 32, 128),
            64: Config(Buffer.num_sms, 20, 288, 28, 128),
            128: Config(Buffer.num_sms, 20, 560, 32, 128),
            144: Config(Buffer.num_sms, 32, 720, 12, 128),
            160: Config(Buffer.num_sms, 28, 720, 12, 128),
        }
        assert (
            num_ranks in config_map
        ), f'Unsupported number of EP ranks: {num_ranks}'
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
            8: Config(Buffer.num_sms, 6, 256, 6, 128),
            16: Config(Buffer.num_sms, 2, 288, 28, 128),
            24: Config(Buffer.num_sms, 1, 288, 20, 128),
            32: Config(Buffer.num_sms, 1, 288, 20, 128),
            64: Config(Buffer.num_sms, 1, 288, 20, 128),
            128: Config(Buffer.num_sms, 1, 560, 12, 128),
            144: Config(Buffer.num_sms, 2, 720, 8, 128),
            160: Config(Buffer.num_sms, 2, 720, 8, 128),
        }
        assert (
            num_ranks in config_map
        ), f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]

    # noinspection PyTypeChecker
    def get_dispatch_layout(
        self,
        topk_idx: paddle.Tensor,
        num_experts: int,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> tuple[
        paddle.Tensor,
        paddle.Tensor | None,
        paddle.Tensor,
        paddle.Tensor,
        EventOverlap,
    ]:
        """
        Calculate the layout required for later communication.

        Arguments:
            topk_idx: `[num_tokens, num_topk]`, dtype must be `int64`, the expert indices selected by each token,
                `-1` means no selections.
            num_experts: the number of experts.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            num_tokens_per_rank: `[num_ranks]` with `int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            num_tokens_per_expert: `[num_experts]` with `int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `bool`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = self.runtime.get_dispatch_layout(
            topk_idx,
            num_experts,
            getattr(previous_event, 'event', None),
            async_finish,
            allocate_on_comm_stream,
        )
        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            EventOverlap(event),
        )

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

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
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
            )

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            (
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                is_token_in_rank,
                send_head,
            ) = handle
            num_recv_tokens = recv_src_idx.shape[0]
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, event = (
                self.runtime.intranode_dispatch(
                    x,
                    x_scales,
                    None,
                    None,
                    None,
                    is_token_in_rank,
                    None,
                    num_recv_tokens,
                    rank_prefix_matrix,
                    channel_prefix_matrix,
                    expert_alignment,
                    config,
                    getattr(previous_event, 'event', None),
                    async_finish,
                    allocate_on_comm_stream,
                )
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                None,
                None,
                None,
                None,
                EventOverlap(event),
            )
        else:
            assert (
                num_tokens_per_rank is not None
                and is_token_in_rank is not None
                and num_tokens_per_expert is not None
            )
            (
                recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                send_head,
                event,
            ) = self.runtime.intranode_dispatch(
                x,
                x_scales,
                topk_idx,
                topk_weights,
                num_tokens_per_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                0,
                None,
                None,
                expert_alignment,
                config,
                getattr(previous_event, 'event', None),
                async_finish,
                allocate_on_comm_stream,
            )
            handle = (
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                is_token_in_rank,
                send_head,
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                EventOverlap(event),
            )

    # noinspection PyTypeChecker
    def combine(
        self,
        x: paddle.Tensor,
        handle: tuple,
        topk_weights: paddle.Tensor | None = None,
        config: Config | None = None,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
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

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_combine(
                x,
                handle,
                topk_weights,
                config,
                previous_event,
                async_finish,
                allocate_on_comm_stream,
            )

        # NOTES: the second `_` is for the sending side, so we should use the third one
        (
            rank_prefix_matrix,
            _,
            channel_prefix_matrix,
            src_idx,
            is_recv_token_in_rank,
            send_head,
        ) = handle

        # Launch the kernel
        recv_x, recv_topk_weights, event = self.runtime.intranode_combine(
            x,
            topk_weights,
            src_idx,
            rank_prefix_matrix,
            channel_prefix_matrix,
            send_head,
            config,
            getattr(previous_event, 'event', None),
            async_finish,
            allocate_on_comm_stream,
        )
        return recv_x, recv_topk_weights, EventOverlap(event)

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
            assert topk_idx is None and topk_weights is None
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
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, _, _, _, _, event = (
                self.runtime.internode_dispatch(
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
                    expert_alignment,
                    config,
                    getattr(previous_event, 'event', None),
                    async_finish,
                    allocate_on_comm_stream,
                )
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                None,
                None,
                None,
                None,
                EventOverlap(event),
            )
        else:
            assert (
                num_tokens_per_rank is not None
                and is_token_in_rank is not None
                and num_tokens_per_expert is not None
            )
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
                expert_alignment,
                config,
                getattr(previous_event, 'event', None),
                async_finish,
                allocate_on_comm_stream,
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

    # noinspection PyTypeChecker
    def internode_combine(
        self,
        x: paddle.Tensor,
        handle: tuple | list,
        topk_weights: paddle.Tensor | None = None,
        config: Config | None = None,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None, EventOverlap]:
        """
        Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.
        """
        assert config is not None

        # Unpack handle
        (
            is_combined_token_in_rank,
            _,
            _,
            rdma_channel_prefix_matrix,
            rdma_rank_prefix_sum,
            gbl_channel_prefix_matrix,
            gbl_rank_prefix_sum,
            src_meta,
            send_rdma_head,
            send_nvl_head,
        ) = handle

        # Launch the kernel
        combined_x, combined_topk_weights, event = (
            self.runtime.internode_combine(
                x,
                topk_weights,
                src_meta,
                is_combined_token_in_rank,
                rdma_channel_prefix_matrix,
                rdma_rank_prefix_sum,
                gbl_channel_prefix_matrix,
                send_rdma_head,
                send_nvl_head,
                config,
                getattr(previous_event, 'event', None),
                async_finish,
                allocate_on_comm_stream,
            )
        )
        return combined_x, combined_topk_weights, EventOverlap(event)

    def barrier_all(self):
        self.runtime.barrier_all()

    def clean_low_latency_buffer(
        self,
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
    ) -> None:
        """
        As low-latency kernels require part of the buffer to be zero-initialized, so it is vital to clean the buffer
            if the buffer is dirty at some time.
        For example, after running the normal dispatch/combine, you must run this function before executing any
            low-latency kernel.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_experts: the number of all experts.
        """
        self.runtime.clean_low_latency_buffer(
            num_max_dispatch_tokens_per_rank, hidden, num_experts
        )

    # noinspection PyTypeChecker
    def low_latency_dispatch(
        self,
        x: paddle.Tensor,
        topk_idx: paddle.Tensor,
        expertwise_scale: paddle.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        use_fp8: bool = True,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> tuple[
        tuple[paddle.Tensor, paddle.Tensor],
        paddle.Tensor,
        tuple,
        EventOverlap,
        Callable,
    ]:
        """
        A low-latency implementation for dispatching with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Even for ranks in the same node, NVLink are fully disabled for simplicity.
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you can not hold more than 2
            low-latency kernels' result tensor at a single moment.

        Arguments:
            x: `paddle.Tensor` with `paddle.bfloat16`, shaped as `[num_tokens, hidden]`, only several hidden shapes are
                supported. The number of tokens to be dispatched must be less than `num_max_dispatch_tokens_per_rank`.
            topk_idx: `paddle.Tensor` with `paddle.int64`, shaped as `[num_tokens, num_topk]`, only several top-k shapes
                are supported. `-1` indices (not selecting any expert) are supported.
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            num_experts: the number of all experts.
            use_fp8: whether to enable FP8 casting, with this, the received data will be a tuple of FP8 tensor and scaling factors.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you not set this flag, the kernel will ensure the data's arrival.

        Returns:
            recv_x: a tuple with received tokens for each expert. The first element is a `paddle.Tensor` shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `paddle.float8_e4m3fn`.
                The second tensor is the corresponding scales for the first element with shape
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 128]` with `paddle.float32`.
                Notice that, the last-two-dimension of the scaling tensors are in column-major for TMA compatibility.
                Moreover, not all tokens are valid, only some of the `num_max_dispatch_tokens_per_rank * num_ranks` are,
                as we do not synchronize CPU received count with GPU (also not incompatible with CUDA graph).
            recv_count: a tensor shaped `[num_local_experts]` with type `paddle.int32`, indicating how many tokens each
                expert receive. As mentioned before, all not tokens are valid in `recv_x`.
            handle: the communication handle to be used in the `low_latency_combine` function.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        (
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            event,
            hook,
        ) = self.runtime.low_latency_dispatch(
            x,
            topk_idx,
            expertwise_scale,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            use_fp8,
            async_finish,
            return_recv_hook,
        )
        handle = (
            packed_recv_src_info,
            packed_recv_layout_range,
            num_max_dispatch_tokens_per_rank,
            num_experts,
        )
        tensors_to_record = (
            x,
            topk_idx,
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
        )
        return (
            (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x,
            packed_recv_count,
            handle,
            EventOverlap(event, tensors_to_record if async_finish else None),
            hook,
        )

    # noinspection PyTypeChecker
    def low_latency_combine(
        self,
        x: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        handle: tuple,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> tuple[paddle.Tensor, EventOverlap, Callable]:
        """
        A low-latency implementation for combining tokens (reduce **with weights**) with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Even for ranks in the same node, NVLink are fully disabled for simplicity.
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you can not hold more than 2
            low-latency kernels' result tensor at a single moment.

        Arguments:
            x: `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `paddle.bfloat16`,
                the local calculated tokens to be sent to this original rank and reduced.
            topk_idx: `[num_combined_tokens, num_topk]` with `paddle.int64`, the expert indices selected by the dispatched
                tokens. `-1` indices (not selecting any expert) are supported. Note that, `num_combined_tokens` equals
                to the number of dispatched tokens.
            topk_weights: `[num_combined_tokens, num_topk]` with `paddle.float32`, the expert weights selected by the dispatched
                tokens. The received tokens will be reduced with the weights in this tensor.
            handle: the communication handle given by the `dispatch` function.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you not set this flag, the kernel will ensure the data's arrival.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, num_topk]` and type `paddle.bfloat16`.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            num_experts,
        ) = handle
        combined_x, event, hook = self.runtime.low_latency_combine(
            x,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            async_finish,
            return_recv_hook,
        )
        tensors_to_record = (
            x,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            combined_x,
        )
        return (
            combined_x,
            EventOverlap(event, tensors_to_record if async_finish else None),
            hook,
        )
