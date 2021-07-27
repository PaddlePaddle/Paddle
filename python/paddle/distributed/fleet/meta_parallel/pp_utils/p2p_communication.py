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
from .utils import is_float_tensor, get_tensor_dtype, paddle_2_number, number_2_dtype

_groups = None
_hcg = None
_tensor_shape = (2, 1024, 768)


class SendRecvMeta:
    def __init__(self):
        self.send_shape_message = None
        self.send_dtype_messgae = None

        self.recv_shape_message = None
        self.recv_dtype_message = None

        self.has_send_meta = False
        self.has_recv_meta = False

    def recv_meta(self, group):
        tensor_type = paddle.to_tensor([0])
        paddle.distributed.recv(tensor_type, src=0, group=group)

        tensor_type = tensor_type.item()
        if tensor_type == 0:
            # recv len(shape)
            dims = paddle.to_tensor([0])
            paddle.distributed.recv(dims, src=0, group=group)
            dims = dims.item()

            # recv shape
            shape = paddle.to_tensor([0] * dims)
            paddle.distributed.recv(shape, src=0, group=group)
            shape = shape.numpy().tolist()

            # recv dtype
            dtype = paddle.to_tensor([0])
            paddle.distributed.recv(dtype, src=0, group=group)

            self.recv_shape_message = shape
            self.recv_dtype_message = dtype.item()

    def send_meta(self, tensor, group):
        if isinstance(tensor, paddle.Tensor):
            tensor_type = paddle.to_tensor([0])

            # send tensor type
            paddle.distributed.send(tensor_type, dst=1, group=group)

            # send len(shape)
            dims = paddle.to_tensor(len(tensor.shape))
            paddle.distributed.send(dims, dst=1, group=group)

            # send shape
            shape = paddle.to_tensor(tensor.shape)
            paddle.distributed.send(shape, dst=1, group=group)

            # send dtype
            dtype = paddle.to_tensor(paddle_2_number(tensor.dtype))
            paddle.distributed.send(dtype, dst=1, group=group)

    def set_send_message(self, tensor):
        self.send_shape_message = tensor.shape
        self.send_dtype_message = paddle_2_number(tensor.dtype)


_send_recv_meta = SendRecvMeta()


def initialize_p2p_groups(hcg):
    global _groups, _hcg
    _hcg = hcg


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


def _get_send_recv_group(src_stage, dest_stage):
    global _groups, _hcg
    stage_id = None
    first_stage = 0
    last_stage = _hcg.get_pipe_parallel_world_size() - 1
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
        # tensor_recv_prev = paddle.empty(shape=tensor_chunk_shape, dtype=dtype)
        tensor_recv_prev = paddle.empty(
            shape=_send_recv_meta.recv_shape_message,
            dtype=number_2_dtype(_send_recv_meta.recv_dtype_message))

    if recv_next:
        # tensor_recv_next = paddle.empty(shape=tensor_chunk_shape, dtype=dtype)
        tensor_recv_next = paddle.empty(
            shape=_send_recv_meta.send_shape_message,
            dtype=number_2_dtype(_send_recv_meta.send_dtype_message))

    if tensor_send_prev is not None:
        # group = _get_send_recv_group(
        # src_stage=current_stage, dest_stage=prev_stage)
        #print("group msg:", group)
        paddle.distributed.wait(tensor_send_prev, use_calc_stream=True)
        paddle.distributed.send(
            tensor_send_prev,
            dst=0,
            group=_hcg.send_prev_group,
            use_calc_stream=False)
    if tensor_recv_prev is not None:
        # group = _get_send_recv_group(
        # src_stage=prev_stage, dest_stage=current_stage)
        #print("group msg:", group)
        paddle.distributed.recv(
            tensor_recv_prev,
            src=0,
            group=_hcg.recv_prev_group,
            use_calc_stream=True)
        #print("tensor_recv_prev", tensor_recv_prev.numpy())

    if tensor_send_next is not None:
        # group = _get_send_recv_group(
        # src_stage=current_stage, dest_stage=next_stage)
        #print("group msg:", group)
        paddle.distributed.wait(tensor_send_next, use_calc_stream=True)
        paddle.distributed.send(
            tensor_send_next,
            dst=1,
            group=_hcg.send_next_group,
            use_calc_stream=False)
    if tensor_recv_next is not None:
        # group = _get_send_recv_group(
        # src_stage=next_stage, dest_stage=current_stage)
        #print("group msg:", group)
        paddle.distributed.recv(
            tensor_recv_next,
            src=1,
            group=_hcg.recv_next_group,
            use_calc_stream=True)
        #print("tensor_recv_next", tensor_recv_next.numpy())

    return tensor_recv_prev, tensor_recv_next


def recv_forward():
    if _hcg.is_first_stage:
        input_tensor = None
    else:
        # check recv forward
        if not _send_recv_meta.has_recv_meta:
            _send_recv_meta.recv_meta(_hcg.recv_prev_group)
            _send_recv_meta.has_recv_meta = True

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

        if not _send_recv_meta.has_send_meta:
            _send_recv_meta.set_send_message(output_tensor)
            _send_recv_meta.send_meta(output_tensor, _hcg.send_next_group)
            _send_recv_meta.has_send_meta = True

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
        if not _send_recv_meta.has_send_meta:
            _send_recv_meta.set_send_message(output_tensor)
            _send_recv_meta.send_meta(output_tensor, _hcg.send_next_group)
            _send_recv_meta.has_send_meta = True

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
