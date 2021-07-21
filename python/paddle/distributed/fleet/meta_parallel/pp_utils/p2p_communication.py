# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

_groups = None
_hcg = None
_tensor_shape = (2, 1024, 768)


def initialize_p2p_groups(hcg):
    global _groups, _hcg
    _groups = [
        paddle.distributed.new_group(ranks=group)
        for group in hcg.get_p2p_groups()
    ]
    _hcg = hcg


def _is_valid_communciate(src_stage, dest_stage):
    first_stage = 0
    last_stage = _hcg.get_pipe_parallel_world_size() - 1
    assert abs(src_stage-dest_stage) == 1 or \
        (src_stage == first_stage and dest_stage == last_stage) or \
        (src_stage == last_stage and dest_stage == first_stage)


def partial_send_operator(tensor,
                          dst=0,
                          mp_ranks=1,
                          mp_rank_id=0,
                          group=None,
                          use_calc_stream=True):

    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id
    return paddle.fluid.core.ops.partial_send(
        tensor, 'use_calc_stream', use_calc_stream, 'ring_id', ring_id, 'peer',
        dst, 'num', mp_ranks, 'id', mp_rank_id)


def partial_recv_operator(tensor,
                          src=0,
                          mp_ranks=1,
                          mp_rank_id=0,
                          group=None,
                          use_calc_stream=True):

    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    return paddle.fluid.core.ops.partial_recv(
        tensor, 'use_calc_stream', use_calc_stream, 'ring_id', ring_id, 'peer',
        src, 'num', mp_ranks, 'id', mp_rank_id, 'dtype', tensor.dtype,
        'out_shape', tensor.shape)


def partial_allgather_operator(tensor,
                               mp_ranks=1,
                               mp_rank_id=0,
                               group=None,
                               use_calc_stream=True):
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    return paddle.fluid.core.ops.partial_allgather_(
        tensor, 'use_calc_stream', use_calc_stream, 'ring_id', ring_id,
        'nranks', mp_ranks, 'rank', mp_rank_id)


def send(tensor, dest_stage):
    global _groups, _hcg
    src_stage = _hcg.get_stage_id()
    _is_valid_communciate(src_stage, dest_stage)
    group = _get_send_recv_group(src_stage, dest_stage)
    return paddle.distributed.send(
        tensor, dst=1 if dest_stage > src_stage else 0, group=group)


def recv(tensor, src_stage):
    global _groups, _hcg
    dest_stage = _hcg.get_stage_id()

    _is_valid_communciate(src_stage, dest_stage)
    group = _get_send_recv_group(src_stage, dest_stage)
    return paddle.distributed.recv(
        tensor, src=0 if dest_stage > src_stage else 1, group=group)


def send_partial(tensor, dest_stage, mp_degree, mp_rank):
    global _groups, _hcg
    src_stage = _hcg.get_stage_id()
    _is_valid_communciate(src_stage, dest_stage)
    group = _get_send_recv_group(src_stage, dest_stage)
    return partial_send_operator(
        tensor,
        dst=1 if dest_stage > src_stage else 0,
        mp_ranks=mp_degree,
        mp_rank_id=mp_rank,
        group=group)


def recv_partial(tensor, src_stage, mp_degree, mp_rank):
    global _groups, _hcg
    dest_stage = _hcg.get_stage_id()

    _is_valid_communciate(src_stage, dest_stage)
    group = _get_send_recv_group(src_stage, dest_stage)
    return partial_recv_operator(
        tensor,
        src=0 if dest_stage > src_stage else 1,
        mp_ranks=mp_degree,
        mp_rank_id=mp_rank,
        group=group)


def _get_send_recv_group(src_stage, dest_stage):
    global _groups, _hcg
    stage_id = None
    first_stage = 0
    last_stage = _hcg.get_pipe_parallel_world_size() - 1
    #if (src_stage == first_stage and dest_stage == last_stage) or \
    #        (dest_stage == first_stage and src_stage == last_stage):
    #    stage_id = last_stage
    #if src_stage > dest_stage:
    #    stage_id = dest_stage
    #else:
    #    stage_id = src_stage
    #group_id = _hcg.get_rank_from_stage(stage_id=stage_id)
    group_id = _hcg.get_rank_from_stage(stage_id=src_stage)
    return _groups[group_id]


def _communicate(tensor_send_next, tensor_send_prev, recv_prev, recv_next):

    tensor_recv_prev = None
    tensor_recv_next = None

    global _tensor_shape, _groups, _hcg
    tensor_chunk_shape = _tensor_shape
    dtype = "float32"

    current_stage = _hcg.get_stage_id()
    prev_stage = current_stage - 1
    next_stage = current_stage + 1

    if recv_prev:
        tensor_recv_prev = paddle.empty(shape=tensor_chunk_shape, dtype=dtype)
    if recv_next:
        tensor_recv_next = paddle.empty(shape=tensor_chunk_shape, dtype=dtype)

    if tensor_send_prev is not None:
        group = _get_send_recv_group(
            src_stage=current_stage, dest_stage=prev_stage)
        #print("group msg:", group)
        paddle.distributed.wait(tensor_send_prev, use_calc_stream=True)
        paddle.distributed.send(
            tensor_send_prev, dst=0, group=group, use_calc_stream=False)
    if tensor_recv_prev is not None:
        group = _get_send_recv_group(
            src_stage=prev_stage, dest_stage=current_stage)
        #print("group msg:", group)
        paddle.distributed.recv(
            tensor_recv_prev, src=0, group=group, use_calc_stream=True)
        #print("tensor_recv_prev", tensor_recv_prev.numpy())

    if tensor_send_next is not None:
        group = _get_send_recv_group(
            src_stage=current_stage, dest_stage=next_stage)
        #print("group msg:", group)
        paddle.distributed.wait(tensor_send_next, use_calc_stream=True)
        paddle.distributed.send(
            tensor_send_next, dst=1, group=group, use_calc_stream=False)
    if tensor_recv_next is not None:
        group = _get_send_recv_group(
            src_stage=next_stage, dest_stage=current_stage)
        #print("group msg:", group)
        paddle.distributed.recv(
            tensor_recv_next, src=1, group=group, use_calc_stream=True)
        #print("tensor_recv_next", tensor_recv_next.numpy())

    return tensor_recv_prev, tensor_recv_next


def recv_forward():
    if _hcg.is_first_stage:
        input_tensor = None
    else:
        input_tensor, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False)
    return input_tensor


def recv_backward():
    if _hcg.is_last_stage:
        output_tensor_grad = None
    else:
        _, output_tensor_grad = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True)
    return output_tensor_grad


def send_forward(output_tensor):
    if not _hcg.is_last_stage:
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False)


def send_backward(input_tensor_grad):
    if not _hcg.is_first_stage:
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False)


def send_forward_recv_backward(output_tensor):
    if _hcg.is_last_stage:
        output_tensor_grad = None
    else:
        _, output_tensor_grad = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True)
    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad):
    if _hcg.is_first_stage:
        input_tensor = None
    else:
        input_tensor, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False)
    return input_tensor


# def send_forward_recv_forward(output_tensor, recv_prev, timers=None):
#     """Batched recv from previous rank and send to next rank in pipeline."""
#     if timers is not None:
#         timers('forward-send-forward-recv').start()
#     input_tensor, _ = _communicate(
#         tensor_send_next=output_tensor,
#         tensor_send_prev=None,
#         recv_prev=recv_prev,
#         recv_next=False)
#     if timers is not None:
#         timers('forward-send-forward-recv').stop()
#     return input_tensor

# def send_backward_recv_backward(input_tensor_grad, recv_next, timers=None):
#     """Batched recv from next rank and send to previous rank in pipeline."""
#     if timers is not None:
#         timers('backward-send-backward-recv').start()
#     _, output_tensor_grad = _communicate(
#         tensor_send_next=None,
#         tensor_send_prev=input_tensor_grad,
#         recv_prev=False,
#         recv_next=recv_next)
#     if timers is not None:
#         timers('backward-send-backward-recv').stop()
#     return output_tensor_grad

# def send_forward_backward_recv_forward_backward(
#         output_tensor, input_tensor_grad, recv_prev,
#         recv_next, timers=None):
#     """Batched send and recv with previous and next ranks in pipeline."""
#     if timers is not None:
#         timers('forward-backward-send-forward-backward-recv').start()
#     input_tensor, output_tensor_grad = _communicate(
#         tensor_send_next=output_tensor,
#         tensor_send_prev=input_tensor_grad,
#         recv_prev=recv_prev,
#         recv_next=recv_next)
#     if timers is not None:
#         timers('forward-backward-send-forward-backward-recv').stop()
#     return input_tensor, output_tensor_grad
