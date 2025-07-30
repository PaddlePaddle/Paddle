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
# limitations under the License


import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import _C_ops


def shard_seq_load_balance(tensor, seq_dim):
    # dtensor Replicate() -> reorder -> Shard(seq_dim)
    placements = tensor.placements
    process_mesh = tensor.process_mesh
    cp_index = process_mesh.dim_names.index('sep')
    cp_degree = process_mesh.shape[cp_index]
    if cp_degree > 1:
        # split
        sliced_datas = paddle.split(
            tensor, num_or_sections=cp_degree * 2, axis=seq_dim
        )
        # resort [q0,q1,q2,q3] -> [q0,q3,q1,q2]
        indices = []
        for i in range(cp_degree):
            indices.append(i)
            indices.append(cp_degree * 2 - 1 - i)
        reorder_indices = indices
        reordered = [sliced_datas[i] for i in reorder_indices]
        reordered_tensor = paddle.concat(reordered, axis=seq_dim)
        # reshard q/k/v -> Shard(seq_dim)
        placements[cp_index] = paddle.distributed.Shard(seq_dim)  # seq_dim:1
        tensor = paddle.distributed.reshard(
            reordered_tensor, process_mesh, placements
        )
    return tensor


def unshard_seq_load_balance(tensor, seq_dim):
    # dtensor Shard(seq_dim) -> Replicate() -> reorder
    placements = tensor.placements
    process_mesh = tensor.process_mesh
    cp_index = process_mesh.dim_names.index('sep')
    cp_degree = process_mesh.shape[cp_index]
    all_tensor = dist.reshard(tensor, process_mesh, [dist.Replicate()])
    sliced_datas = paddle.split(
        all_tensor, num_or_sections=cp_degree * 2, axis=seq_dim
    )
    reorder_indices = []
    for i in range(cp_degree):
        reorder_indices.append(i)
        reorder_indices.append(cp_degree * 2 - 1 - i)
    inverse_indices = [0] * len(reorder_indices)
    for idx, v in enumerate(reorder_indices):
        inverse_indices[v] = idx
    restored = [sliced_datas[i] for i in inverse_indices]
    return paddle.concat(restored, axis=seq_dim)


class RingCommunicator:
    def __init__(self, group, local_key, local_value):
        self._k_buffer = [
            local_key.clone().contiguous(),
            local_key.clone().contiguous(),
        ]
        self._v_buffer = [
            local_value.clone().contiguous(),
            local_value.clone().contiguous(),
        ]

        self._next_buffer_idx = 0

        self.group = group
        mesh = dist.auto_parallel.get_mesh()
        process_id = dist.get_rank()
        self.group_rank = mesh.get_rank_by_dim_and_process_id("sep", process_id)
        self.cp_size = mesh.get_dim_size("sep")
        cp_index = mesh.dim_names.index("sep")

        self.send_rank = self.group.ranks[
            (self.group_rank + 1) % self.cp_size
        ]  # 1%2=1
        self.recv_rank = self.group.ranks[(self.group_rank - 1) % self.cp_size]

        self._reqs = []

    def wait(self):
        paddle.device.synchronize()

    def add_to_buffers(self, key, value):
        if key.shape != self._k_buffer[self._next_buffer_idx].shape:
            self._k_buffer[self._next_buffer_idx][:, : key.shape[1], :, :].add_(
                key
            )
            self._v_buffer[self._next_buffer_idx][:, : key.shape[1], :, :].add_(
                value
            )
        else:
            self._k_buffer[self._next_buffer_idx].add_(key)
            self._v_buffer[self._next_buffer_idx].add_(value)

    def get_buffers(self):
        return (
            self._k_buffer[self._next_buffer_idx],
            self._v_buffer[self._next_buffer_idx],
        )

    def send_recv(self):
        send_k_op = dist.P2POp(
            dist.isend,
            self._k_buffer[self._next_buffer_idx].contiguous(),
            self.send_rank,
            self.group,
        )
        send_v_op = dist.P2POp(
            dist.isend,
            self._v_buffer[self._next_buffer_idx].contiguous(),
            self.send_rank,
            self.group,
        )
        recv_k_op = dist.P2POp(
            dist.irecv,
            self._k_buffer[(self._next_buffer_idx + 1) % 2],
            self.recv_rank,
            self.group,
        )
        recv_v_op = dist.P2POp(
            dist.irecv,
            self._v_buffer[(self._next_buffer_idx + 1) % 2],
            self.recv_rank,
            self.group,
        )

        self._next_buffer_idx = (self._next_buffer_idx + 1) % 2

        ops = [send_k_op, send_v_op, recv_k_op, recv_v_op]

        self._reqs = dist.batch_isend_irecv(ops)


def update_out_and_lse(
    old_out, old_lse, block_out, block_lse, second_chunk_only=False
):
    if second_chunk_only:
        second_chunk_out = old_out[:, old_out.shape[1] // 2 :, :, :]
        second_chunk_lse = old_lse[:, old_lse.shape[1] // 2 :, :, :]
        second_chunk_out, second_chunk_lse = update_out_and_lse(
            second_chunk_out, second_chunk_lse, block_out, block_lse
        )
        old_out[:, old_out.shape[1] // 2 :, :, :] = second_chunk_out
        old_lse[:, old_lse.shape[1] // 2 :, :, :] = second_chunk_lse
        return old_out, old_lse
    else:
        block_out, block_lse = paddle.cast(block_out, "float32"), paddle.cast(
            block_lse, "float32"
        )
        with paddle.amp.auto_cast(enable=False):
            return old_out - (old_out - block_out) * F.sigmoid(
                block_lse - old_lse
            ), old_lse - F.log_sigmoid(old_lse - block_lse)


def get_chunk_id(rank, cp_size):
    return rank, (2 * cp_size - 1 - rank)


def concat_masks(attn_masks_list, rank, cp_size):
    assert len(attn_masks_list) == 2 * cp_size
    first_chunk_id, second_chunk_id = get_chunk_id(rank, cp_size)
    return paddle.concat(
        [attn_masks_list[first_chunk_id], attn_masks_list[second_chunk_id]],
        axis=3,
    )


def ring_flash_attention_forward_func(
    group,
    local_query,
    local_key,
    local_value,
    attn_mask=None,
    dropout=0.0,
    is_causal=False,
    fixed_seed_offset=None,
    training=True,
):
    cp_size = group.world_size
    group_rank = group.rank

    comm_buffer = RingCommunicator(group, local_key, local_value)
    local_q_seq_len = local_query.shape[1]
    if attn_mask is not None:
        attn_masks_list = paddle.split(
            attn_mask, num_or_sections=cp_size * 2, axis=3
        )
    if is_causal:
        local_query_second_chunk = local_query[
            :, local_q_seq_len // 2 :, :, :
        ].contiguous()
    for step in range(cp_size):
        block_k, block_v = comm_buffer.get_buffers()
        if step != cp_size - 1:
            comm_buffer.send_recv()
        if not is_causal:
            # out [bs, seq, nhead, headdim]
            # lse [bs, nhead, seq]
            block_out, _, block_lse, _ = _C_ops.flash_attn(
                local_query,
                block_k,
                block_v,
                fixed_seed_offset,
                (
                    None
                    if attn_mask is None
                    else concat_masks(
                        attn_masks_list, (group_rank - step) % cp_size, cp_size
                    )
                ),
                dropout,
                False,
                False,
                not training,
                "",
            )
            paddle.unsqueeze_(paddle.transpose_(block_lse, [0, 2, 1]), axis=-1)

            if step == 0:
                out, lse = block_out, block_lse
            else:
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            if step == 0:
                block_out, _, block_lse, _ = _C_ops.flash_attn(
                    local_query,
                    block_k,
                    block_v,
                    fixed_seed_offset,
                    None,
                    dropout,
                    True,
                    False,
                    not training,
                    "",
                )
                paddle.unsqueeze_(
                    paddle.transpose_(block_lse, [0, 2, 1]), axis=-1
                )
                out, lse = block_out, block_lse
            elif step > group_rank:
                block_out, _, block_lse, _ = _C_ops.flash_attn(
                    local_query_second_chunk,
                    block_k,
                    block_v,
                    fixed_seed_offset,
                    None,
                    dropout,
                    False,
                    False,
                    not training,
                    "",
                )
                block_lse = block_lse[:, :, 0 : (local_q_seq_len // 2)]
                paddle.unsqueeze_(
                    paddle.transpose_(block_lse, [0, 2, 1]), axis=-1
                )
                out, lse = update_out_and_lse(
                    out, lse, block_out, block_lse, True
                )
            else:
                block_out, _, block_lse, _ = _C_ops.flash_attn(
                    local_query,
                    block_k[:, : local_q_seq_len // 2, :, :],
                    block_v[:, : local_q_seq_len // 2, :, :],
                    fixed_seed_offset,
                    None,
                    dropout,
                    False,
                    False,
                    not training,
                    "",
                )
                paddle.unsqueeze_(
                    paddle.transpose_(block_lse, [0, 2, 1]), axis=-1
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        paddle.device.synchronize()
    out = paddle.cast(out, local_query.dtype)
    lse = paddle.transpose_(paddle.squeeze(lse, axis=-1), [0, 2, 1])
    return out, lse


def ring_flash_attention_backward_func(
    group,
    local_out_grad,
    local_query,
    local_key,
    local_value,
    local_out,
    lse,
    attn_mask,
    dropout=0.0,
    is_causal=False,
    fixed_seed_offset=None,
):
    cp_size = group.world_size
    group_rank = group.rank

    lse = lse.contiguous()

    local_q_seq_len = local_query.shape[1]
    query_grad_buffer = paddle.zeros_like(local_query)
    key_grad_buffer = paddle.zeros_like(local_key)
    value_grad_buffer = paddle.zeros_like(local_value)

    kv_comm_buffer = RingCommunicator(group, local_key, local_value)
    grad_comm_buffer = RingCommunicator(
        group, key_grad_buffer, value_grad_buffer
    )
    if is_causal:
        local_query_second_chunk = local_query[:, local_q_seq_len // 2 :, :, :]
        local_out_second_chunk = local_out[:, local_q_seq_len // 2 :, :, :]
        lse_second_chunk = lse[:, :, local_q_seq_len // 2 :].contiguous()
        out_grad_second_chunk = local_out_grad[:, local_q_seq_len // 2 :, :, :]

    if attn_mask is not None:
        attn_masks_list = paddle.split(
            attn_mask, num_or_sections=cp_size * 2, axis=3
        )
    for step in range(cp_size):
        block_k, block_v = kv_comm_buffer.get_buffers()
        if step != cp_size - 1:
            kv_comm_buffer.send_recv()

        if not is_causal:
            block_q_grad, block_k_grad, block_v_grad = _C_ops.flash_attn_grad(
                local_query,
                block_k,
                block_v,
                local_out,
                lse,
                fixed_seed_offset,
                (
                    None
                    if attn_mask is None
                    else concat_masks(
                        attn_masks_list, (group_rank - step) % cp_size, cp_size
                    )
                ),
                local_out_grad,
                dropout,
                False,
            )
            query_grad_buffer.add_(block_q_grad)
        else:
            if step == 0:
                block_q_grad, block_k_grad, block_v_grad = (
                    _C_ops.flash_attn_grad(
                        local_query,
                        block_k,
                        block_v,
                        local_out,
                        lse,
                        fixed_seed_offset,
                        None,
                        local_out_grad,
                        dropout,
                        True,
                    )
                )
                query_grad_buffer.add_(block_q_grad)
            elif step > group_rank:
                block_q_grad, block_k_grad, block_v_grad = (
                    _C_ops.flash_attn_grad(
                        local_query_second_chunk,
                        block_k,
                        block_v,
                        local_out_second_chunk,
                        lse_second_chunk,
                        fixed_seed_offset,
                        None,
                        out_grad_second_chunk,
                        dropout,
                        False,
                    )
                )
                query_grad_buffer[:, local_q_seq_len // 2 :, :, :].add_(
                    block_q_grad
                )
            else:
                block_q_grad, block_k_grad, block_v_grad = (
                    _C_ops.flash_attn_grad(
                        local_query,
                        block_k[:, : local_q_seq_len // 2, :, :],
                        block_v[:, : local_q_seq_len // 2, :, :],
                        local_out,
                        lse,
                        fixed_seed_offset,
                        None,
                        local_out_grad,
                        dropout,
                        False,
                    )
                )
                query_grad_buffer.add_(block_q_grad)
        paddle.device.synchronize()

        grad_comm_buffer.add_to_buffers(
            block_k_grad.contiguous(), block_v_grad.contiguous()
        )
        grad_comm_buffer.send_recv()

        grad_comm_buffer.wait()
    key_grad_buffer, value_grad_buffer = grad_comm_buffer.get_buffers()

    return query_grad_buffer, key_grad_buffer, value_grad_buffer


class RingFlashAttention(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask=None,
        dropout=0.0,
        is_causal=False,
        fixed_seed_offset=None,
        training=True,
    ):
        if dropout > 0.0:
            raise NotImplementedError(
                "Dropout is not supported in ring attention yet."
            )
        mesh = dist.auto_parallel.get_mesh()
        cp_index = mesh.dim_names.index('sep')
        process_id = dist.get_rank()
        rank = mesh.get_rank_by_dim_and_process_id("sep", process_id)
        dist.init_parallel_env()

        group = mesh._get_group("sep")
        local_query = dist.auto_parallel.api.dtensor_to_local(
            query, query.process_mesh, query.placements
        )
        local_key = dist.auto_parallel.api.dtensor_to_local(
            key, key.process_mesh, key.placements
        )
        local_value = dist.auto_parallel.api.dtensor_to_local(
            value, value.process_mesh, value.placements
        )
        if attn_mask is not None:
            is_causal = False

        out, lse = ring_flash_attention_forward_func(
            group,
            local_query,
            local_key,
            local_value,
            attn_mask,
            dropout,
            is_causal,
            fixed_seed_offset,
            training,
        )
        ctx.save_for_backward(group, query, key, value, out, lse, attn_mask)
        ctx.fixed_seed_offset = fixed_seed_offset
        ctx.dropout = dropout
        ctx.is_causal = is_causal
        out_dtensor = dist.auto_parallel.api.dtensor_from_local(
            out, query.process_mesh, query.placements
        )
        return out_dtensor.contiguous()

    @staticmethod
    def backward(ctx, out_grad):
        mesh = dist.auto_parallel.get_mesh()
        cp_index = mesh.dim_names.index('sep')
        group, query, key, value, out, lse, attn_mask = ctx.saved_tensor()
        fixed_seed_offset = ctx.fixed_seed_offset
        dropout = ctx.dropout
        is_causal = ctx.is_causal

        if fixed_seed_offset is None:
            fixed_seed_offset = paddle.to_tensor(
                [0, 0], place=paddle.CPUPlace(), dtype=paddle.int64
            )
        local_query = dist.auto_parallel.api.dtensor_to_local(
            query, query.process_mesh, query.placements
        )
        local_key = dist.auto_parallel.api.dtensor_to_local(
            key, key.process_mesh, key.placements
        )
        local_value = dist.auto_parallel.api.dtensor_to_local(
            value, value.process_mesh, value.placements
        )
        local_out_grad = dist.auto_parallel.api.dtensor_to_local(
            out_grad, out_grad.process_mesh, out_grad.placements
        )
        query_grad, key_grad, value_grad = ring_flash_attention_backward_func(
            group,
            local_out_grad,
            local_query,
            local_key,
            local_value,
            out,
            lse,
            attn_mask,
            dropout,
            is_causal,
            fixed_seed_offset,
        )
        query_grad_dtensor = dist.auto_parallel.api.dtensor_from_local(
            query_grad, query.process_mesh, query.placements
        )
        key_grad_dtensor = dist.auto_parallel.api.dtensor_from_local(
            key_grad, key.process_mesh, key.placements
        )
        value_grad_dtensor = dist.auto_parallel.api.dtensor_from_local(
            value_grad, value.process_mesh, value.placements
        )

        if attn_mask is not None and not attn_mask.stop_gradient:
            return (
                query_grad_dtensor,
                key_grad_dtensor,
                value_grad_dtensor,
                None,
            )
        else:
            return query_grad_dtensor, key_grad_dtensor, value_grad_dtensor
