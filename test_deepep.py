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
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.autograd import PyLayer
from paddle.base import core
from paddle.distributed import fleet
from paddle.distributed.communication import deep_ep
from paddle.distributed.communication.group import Group

_buffer = None


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


def get_buffer(group: Group, hidden_bytes: int):
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        deep_ep.Buffer.get_dispatch_config(group.world_size),
        deep_ep.Buffer.get_combine_config(group.world_size),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.world_size),
            num_nvl_bytes,
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.world_size),
            num_rdma_bytes,
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = deep_ep.Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def get_hidden_bytes(x: paddle.Tensor) -> int:
    return x.shape[1] * max(x.element_size(), 2)


class FusedDispatch(PyLayer):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        previous_event=None,
    ):
        """Forward pass of fused dispatch."""
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs.cast(paddle.float32),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        ctx.group = group
        ctx.handle = handle
        ctx.event = event
        tokens_per_expert = paddle.to_tensor(num_recv_tokens_per_expert_list)

        states = {}
        states["dispatched_indices"] = recv_token_indices
        states["tokens_per_expert"] = tokens_per_expert
        states["handle"] = handle

        return recv_x, recv_token_probs, states

    @staticmethod
    def backward(ctx, grad_output, grad_token_probs):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle

        grad_x, grad_token_probs, event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.cast(paddle.float32),
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return grad_x, None, grad_token_probs


class NewFusedDispatch(PyLayer):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        previous_event=None,
    ):
        """Forward pass of fused dispatch."""
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        (
            num_recv_tokens_per_expert_list,
            num_recv_tokens,
            num_rdma_recv_tokens,
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_rank_prefix_sum,
            handle,
        ) = buffer.internode_notify_dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs.cast(paddle.float32),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            is_token_in_rank=is_token_in_rank,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            handle,
            event,
        ) = buffer.internode_dispatch_after_notify(
            x,
            rdma_channel_prefix_matrix=rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix=gbl_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum=recv_rdma_rank_prefix_sum,
            recv_gbl_rank_prefix_sum=recv_gbl_rank_prefix_sum,
            topk_idx=token_indices,
            topk_weights=token_probs.cast(paddle.float32),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            is_token_in_rank=is_token_in_rank,
            num_recv_tokens=num_recv_tokens,
            num_rdma_recv_tokens=num_rdma_recv_tokens,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        ctx.group = group
        ctx.handle = handle
        ctx.event = event
        tokens_per_expert = paddle.to_tensor(num_recv_tokens_per_expert_list)

        states = {}
        states["dispatched_indices"] = recv_token_indices
        states["tokens_per_expert"] = tokens_per_expert
        states["handle"] = handle

        return recv_x, recv_token_probs, states

    @staticmethod
    def backward(ctx, grad_output, grad_token_probs):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle

        grad_x, grad_token_probs, event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.cast(paddle.float32),
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return grad_x, None, grad_token_probs


class FusedCombine(PyLayer):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, states, previous_event=None):
        """Forward pass of fused combine."""
        handle = states["handle"]
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, event = buffer.combine(
            x,
            handle=handle,
            async_finish=False,
            previous_event=None,
            allocate_on_comm_stream=False,
        )
        ctx.handle = handle
        ctx.group = group
        ctx.previous_event = previous_event

        return combined_x

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of fused combine."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=ctx.previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return grad_x


def fused_dispatch(
    x,
    token_indices,
    token_probs,
    num_experts,
    group: Group,
    previous_event=None,
):
    return FusedDispatch.apply(
        x.contiguous(),
        token_indices,
        token_probs,
        num_experts,
        group,
        previous_event,
    )


def new_fused_dispatch(
    x,
    token_indices,
    token_probs,
    num_experts,
    group: Group,
    previous_event=None,
):
    return NewFusedDispatch.apply(
        x.contiguous(),
        token_indices,
        token_probs,
        num_experts,
        group,
        previous_event,
    )


def fused_combine(x, group, handle, previous_event=None):
    states = {}
    states["handle"] = handle
    return FusedCombine.apply(x, group, states, previous_event)


class TestDeepEP(unittest.TestCase):
    def setUp(self):
        self.expert_parallel_degree = paddle.distributed.get_world_size()

        self.rank = dist.get_rank()
        paddle.seed(42 + self.rank)
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "mp_degree": self.expert_parallel_degree,
        }
        fleet.init(is_collective=True, strategy=strategy)
        self.group = (
            dist.fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )

    def get_inputs(self, seq_len, hidden_size, num_experts, topk):
        hidden_states = paddle.randn([seq_len, hidden_size]).astype("bfloat16")
        probs = (
            paddle.randn([seq_len, num_experts], dtype=paddle.float32).abs() + 1
        )
        topk_weights, topk_idx = paddle.topk(probs, topk, axis=-1, sorted=True)
        return hidden_states, topk_weights, topk_idx

    def _test_case(self):
        seq_len = 2048
        hidden_size = 1024
        topk = 8
        num_experts = 32

        local_num_experts = num_experts // self.expert_parallel_degree

        hidden_states, topk_weights, topk_idx = self.get_inputs(
            seq_len, hidden_size, num_experts, topk
        )

        print("hidden_states:", hidden_states)
        dispatched_hidden_states, dispatched_probs, states = fused_dispatch(
            hidden_states, topk_idx, topk_weights, num_experts, self.group
        )
        dispatched_hidden_states *= dispatched_probs.sum(
            axis=-1, keepdim=True
        ).astype("bfloat16")
        combined_hidden_states = fused_combine(
            dispatched_hidden_states, self.group, states["handle"]
        )
        print("combined_hidden_states:", combined_hidden_states)

    def test_new_dispathc(self):
        seq_len = 2048
        hidden_size = 1024
        topk = 8
        num_experts = 32

        local_num_experts = num_experts // self.expert_parallel_degree

        hidden_states, topk_weights, topk_idx = self.get_inputs(
            seq_len, hidden_size, num_experts, topk
        )

        dispatched_hidden_states, dispatched_probs, states = fused_dispatch(
            hidden_states, topk_idx, topk_weights, num_experts, self.group
        )
        dispatched_hidden_states *= dispatched_probs.sum(
            axis=-1, keepdim=True
        ).astype("bfloat16")
        combined_hidden_states = fused_combine(
            dispatched_hidden_states, self.group, states["handle"]
        )
        print("combined_hidden_states:", combined_hidden_states)

        dispatched_hidden_states, dispatched_probs, states = new_fused_dispatch(
            hidden_states, topk_idx, topk_weights, num_experts, self.group
        )
        dispatched_hidden_states *= dispatched_probs.sum(
            axis=-1, keepdim=True
        ).astype("bfloat16")
        new_combined_hidden_states = fused_combine(
            dispatched_hidden_states, self.group, states["handle"]
        )
        print(
            "new dispatch combined_hidden_states:", new_combined_hidden_states
        )

        np.testing.assert_allclose(
            combined_hidden_states, new_combined_hidden_states
        )


if __name__ == "__main__":
    unittest.main()
