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


class RingCommunicator:
    def __init__(self, group, local_key, local_value):
        self._k_buffer = [paddle.zeros_like(local_key) for _ in range(2)]
        self._v_buffer = [paddle.zeros_like(local_value) for _ in range(2)]

        self._k_buffer[0] = local_key.clone()
        self._v_buffer[0] = local_value.clone()

        self._next_buffer_idx = 0

        self.group = group
        mesh = dist.auto_parallel.get_mesh()
        process_id = dist.get_rank()
        self.group_rank = mesh.get_rank_by_dim_and_process_id("sep", process_id)
        self.cp_size = mesh.get_dim_size("sep")
        cp_index = mesh.dim_names.index("sep")

        # print(f'RingFlashAttention mesh:{mesh}, rank:{process_id},   cpsize:{mesh.get_dim_size("sep")},  cprank:{self.group_rank},  mesh.mesh:{mesh.mesh}, cp_group:{self.group}, cp_index:{cp_index}')

        self.send_rank = self.group.ranks[
            (self.group_rank + 1) % self.cp_size
        ]  # 1%2=1
        self.recv_rank = self.group.ranks[(self.group_rank - 1) % self.cp_size]
        # print(f'self.group_rank : {self.group_rank}, self.send_rank:{self.send_rank},   self.recv_rank:{self.recv_rank}')

        self._reqs = []

    def wait(self):
        # TODO(zhangyuqin1998)：batch_isend_irecv异步流下，无法wait，需要修复。对性能有影响。
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
            self._k_buffer[self._next_buffer_idx],
            self.send_rank,
            self.group,
        )
        send_v_op = dist.P2POp(
            dist.isend,
            self._v_buffer[self._next_buffer_idx],
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
    query,
    key,
    value,
    attn_mask=None,
    dropout=0.0,
    is_causal=False,
    fixed_seed_offset=None,
    training=True,
):
    cp_size = group.world_size
    group_rank = group.rank
    # print(f'ring_flash_attention_forward_func cp_size:{cp_size}, group_rank:{group_rank}')

    mesh = dist.auto_parallel.get_mesh()
    # cp_size = mesh.get_dim_size("sep")
    # process_id = dist.get_rank()
    # group_rank = mesh.get_rank_by_dim_and_process_id("sep",process_id)

    local_query = dist.auto_parallel.api.dtensor_to_local(
        query, mesh, query.placements
    )
    local_key = dist.auto_parallel.api.dtensor_to_local(
        key, mesh, key.placements
    )
    local_value = dist.auto_parallel.api.dtensor_to_local(
        value, mesh, value.placements
    )

    comm_buffer = RingCommunicator(group, local_key, local_value)
    local_q_seq_len = local_query.shape[1]
    print(
        f'ring_flash_attention_forward_func query shape:{local_query.shape}, key shape:{local_key.shape}, value shape:{local_value.shape} '
    )
    if attn_mask is not None:
        attn_masks_list = paddle.split(
            attn_mask, num_or_sections=cp_size * 2, axis=3
        )
    if is_causal:
        local_query_second_chunk = local_query[:, local_q_seq_len // 2 :, :, :]
        # print(f'local_query_second_chunk shape:{local_query_second_chunk.shape}')
    for step in range(cp_size):
        block_k, block_v = comm_buffer.get_buffers()
        # print(f'step {step} get_buffers finished')

        if step != cp_size - 1:
            comm_buffer.send_recv()
        # print(f'step {step} send_recv finished')
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
                # print(f'step {step} 11 local_query:{local_query.shape},   block_k:{block_k.shape},  block_v:{block_v.shape}')
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
                # print(f'step {step} 22 local_query:{local_query.shape},   block_k:{block_k.shape},  block_v:{block_v.shape}')
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
                # print(f'step {step} 33 local_query:{local_query.shape},   block_k:{block_k.shape},  block_v:{block_v.shape}')
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

        # print(f'step {step} send_recv wait')
        # if step != cp_size - 1:
        #     comm_buffer.wait()
        paddle.device.synchronize()
        # print(f'step {step} send_recv wait finished')
    # print(f'forward 1111 out:{out.shape}, lse:{lse.shape}')
    out = paddle.cast(out, local_query.dtype)
    lse = paddle.transpose_(paddle.squeeze(lse, axis=-1), [0, 2, 1])
    # print(f'forward output out:{out.shape}, lse:{lse.shape}')
    return out, lse


def ring_flash_attention_backward_func(
    group,
    out_grad,
    query,
    key,
    value,
    local_out,
    lse,
    attn_mask,
    dropout=0.0,
    is_causal=False,
    fixed_seed_offset=None,
):
    cp_size = group.world_size
    group_rank = group.rank
    # print(f'ring_flash_attention_backward_func cp_size:{cp_size}, group_rank:{group_rank}')
    mesh = dist.auto_parallel.get_mesh()
    # cp_size = mesh.get_dim_size("sep")
    # process_id = dist.get_rank()
    # group_rank = mesh.get_rank_by_dim_and_process_id("sep",process_id)
    # print(f'ring_flash_attention_backward_func query placements:{query.placements}, key placements:{key.placements}, value placements:{value.placements} , out :{local_out.shape} ')

    local_query = dist.auto_parallel.api.dtensor_to_local(
        query, mesh, query.placements
    )
    local_key = dist.auto_parallel.api.dtensor_to_local(
        key, mesh, key.placements
    )
    local_value = dist.auto_parallel.api.dtensor_to_local(
        value, mesh, value.placements
    )
    local_out_grad = dist.auto_parallel.api.dtensor_to_local(
        out_grad, mesh, out_grad.placements
    )

    local_q_seq_len = local_query.shape[1]
    query_grad_buffer = paddle.zeros_like(local_query)
    key_grad_buffer = paddle.zeros_like(local_key)
    value_grad_buffer = paddle.zeros_like(local_value)

    kv_comm_buffer = RingCommunicator(group, local_key, local_value)
    grad_comm_buffer = RingCommunicator(
        group, key_grad_buffer, value_grad_buffer
    )
    print(
        f'ring_flash_attention_backward_func query_grad_buffer shape:{query_grad_buffer.shape}'
    )
    if is_causal:
        local_query_second_chunk = local_query[:, local_q_seq_len // 2 :, :, :]
        local_out_second_chunk = local_out[:, local_q_seq_len // 2 :, :, :]
        lse_second_chunk = lse[:, :, local_q_seq_len // 2 :]
        out_grad_second_chunk = local_out_grad[:, local_q_seq_len // 2 :, :, :]
        print(
            f'is_casual: local_query_second_chunks shape:{local_query_second_chunk.shape}, local_out_second_chunk shape:{local_out_second_chunk.shape}, lse_second_chunk shape:{lse_second_chunk.shape}, out_grad_second_chunk shape:{out_grad_second_chunk.shape}'
        )

    if attn_mask is not None:
        attn_masks_list = paddle.split(
            attn_mask, num_or_sections=cp_size * 2, axis=3
        )

    # try:
    #     from paddlenlp_ops import flash_attn_bwd
    # except (ImportError, ModuleNotFoundError):
    #     from paddlenlp.utils.log import logger

    #     logger.warning(
    #         "if you run ring_flash_attention.py, please ensure you install "
    #         "the paddlenlp_ops by following the instructions "
    #         "provided at https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md"
    #     )

    for step in range(cp_size):
        block_k, block_v = kv_comm_buffer.get_buffers()
        # print(f'ring_flash_attention_backward_func {step} local_query:{local_query}, block_k:{block_k}, block_v:{block_v}, local_out:{local_out}, lse:{lse}, local_out_grad:{local_out_grad}')

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

        # if step != cp_size - 1:
        #     kv_comm_buffer.wait()
        # if step != 0:
        #     grad_comm_buffer.wait()
        paddle.device.synchronize()

        grad_comm_buffer.add_to_buffers(block_k_grad, block_v_grad)
        grad_comm_buffer.send_recv()

    grad_comm_buffer.wait()
    key_grad_buffer, value_grad_buffer = grad_comm_buffer.get_buffers()

    return query_grad_buffer, key_grad_buffer, value_grad_buffer


def reshard_tensor_with_dim(grad_tensor, dim_name):
    placements = grad_tensor.placements
    process_mesh = grad_tensor.process_mesh
    cp_index = process_mesh.dim_names.index(dim_name)
    print(
        f'reshard_tensor_with_dim {dim_name} process_mesh:{process_mesh}, placements:{placements}, cp_index:{cp_index}'
    )
    if placements[cp_index] == dist.Shard(1):
        # allgather q k v
        print(f'allgather query_grad {grad_tensor.placements}')
        placements[cp_index] = dist.Replicate()
    return dist.reshard(grad_tensor, process_mesh, placements)


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
        # print(f'RingFlashAttention forward attn_mask:{attn_mask}, dropout_p:{dropout}, is_causal:{is_causal} ')
        if dropout > 0.0:
            raise NotImplementedError(
                "Dropout is not supported in ring attention yet."
            )
        mesh = dist.auto_parallel.get_mesh()
        cp_index = mesh.dim_names.index('sep')
        # print(f'forward mesh:{mesh}, cp_index:{cp_index}')
        process_id = dist.get_rank()
        rank = mesh.get_rank_by_dim_and_process_id("sep", process_id)
        dist.init_parallel_env()
        # sub_mesh = mesh.get_submesh_with_dim("sep")
        # 同时创建group会重名，

        # reorder_mesh = mesh.get_mesh_with_dim("sep")._mesh.reshape(
        #     mesh.get_dim_size("sep"), -1
        # )
        # curr_rank = paddle.distributed.get_rank()
        # print(f'RingFlashAttention forward, curr_rank:{curr_rank}, reorder_mesh:{reorder_mesh}')
        # rank_list = mesh.mesh.flatten().tolist()
        # groups = {}
        # for r in rank_list:
        #     col_idx = np.argmax(reorder_mesh == r) % reorder_mesh.shape[-1]
        #     if col_idx in groups:
        #         continue
        #     pg = paddle.distributed.new_group(reorder_mesh[:, col_idx])
        #     groups[col_idx] = pg
        # print(f'RingFlashAttention forward, group_set:{groups}')
        # col_idx = np.argmax(reorder_mesh == curr_rank) % reorder_mesh.shape[-1]
        # group = groups[col_idx]
        group = mesh._get_group("sep")

        # sub_mesh = ProcessMesh(reorder_mesh[:, col_idx], [dim_name])
        # group = sub_mesh.get_group("sep")
        # print(f'RingFlashAttention mesh:{mesh}, rank:{process_id},   cpsize:{mesh.get_dim_size("sep")},  "cprank:{rank},  group:{group}')

        if attn_mask is not None:
            is_causal = False
        out, lse = ring_flash_attention_forward_func(
            group,
            query,
            key,
            value,
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
        # print(f'query:{query}, out:{out}, query.process_mesh:{query.process_mesh}, query.placements:{query.placements}')
        out_dtensor = dist.auto_parallel.api.dtensor_from_local(
            out, query.process_mesh, query.placements
        )
        # print(f'out_dtensor:{out_dtensor}, query:{query}, out:{out}')
        return out_dtensor

    @staticmethod
    def backward(ctx, out_grad):
        mesh = dist.auto_parallel.get_mesh()
        cp_index = mesh.dim_names.index('sep')
        # print(f'backward mesh:{mesh}, cp_index:{cp_index}')
        group, query, key, value, out, lse, attn_mask = ctx.saved_tensor()
        fixed_seed_offset = ctx.fixed_seed_offset
        dropout = ctx.dropout
        is_causal = ctx.is_causal

        if fixed_seed_offset is None:
            fixed_seed_offset = paddle.to_tensor(
                [0, 0], place=paddle.CPUPlace(), dtype=paddle.int64
            )
        # out_grad dim_names::[dp,mp,cp]}, placements=[Shard(dim=0), Shard(dim=2), Shard(dim=1)]
        # print(f'backward group:{group}, query:{query}, out:{out}, out_grad:{out_grad}, query.process_mesh:{query.process_mesh}, query.placements:{query.placements}')
        query_grad, key_grad, value_grad = ring_flash_attention_backward_func(
            group,
            out_grad,
            query,
            key,
            value,
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
        # print(f'backward query_grad_dtensor:{query_grad_dtensor}, key_grad_dtensor:{key_grad_dtensor}, value_grad_dtensor:{value_grad_dtensor}')

        # query_grad_dtensor = reshard_tensor_with_dim(query_grad_dtensor, "sep")
        # key_grad_dtensor = reshard_tensor_with_dim(key_grad_dtensor, "sep")
        # value_grad_dtensor = reshard_tensor_with_dim(value_grad_dtensor, "sep")
        # print(f'backward query_grad_dtensor222222:{query_grad_dtensor}, key_grad_dtensor:{key_grad_dtensor}, value_grad_dtensor:{value_grad_dtensor}')

        if attn_mask is not None and not attn_mask.stop_gradient:
            return (
                query_grad_dtensor,
                key_grad_dtensor,
                value_grad_dtensor,
                None,
            )
        else:
            return query_grad_dtensor, key_grad_dtensor, value_grad_dtensor
