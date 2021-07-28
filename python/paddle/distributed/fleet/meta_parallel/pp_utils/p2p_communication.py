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


class SendRecvMeta:
    def __init__(self):
        self.send_shape_message = None
        self.send_dtype_message = None

        self.recv_shape_message = None
        self.recv_dtype_message = None

        self.has_send_meta = False
        self.has_recv_meta = False

    def _recv_shape_dtype(self, group):
        # recv len(shape)
        dims = paddle.to_tensor([0])
        paddle.distributed.recv(dims, src=0, group=group)
        dims = dims.item()

        # recv shape
        shape = paddle.to_tensor([0] * dims)
        paddle.distributed.recv(shape, src=0, group=group)

        # recv dtype
        dtype = paddle.to_tensor([0])
        paddle.distributed.recv(dtype, src=0, group=group)
        return shape.numpy().tolist(), dtype.item()

    def recv_meta(self, group):
        tensor_type = paddle.to_tensor([0])
        paddle.distributed.recv(tensor_type, src=0, group=group)
        tensor_type = tensor_type.item()

        if tensor_type == 0:
            shape, dtype = self._recv_shape_dtype(group)
            self.recv_shape_message = shape
            self.recv_dtype_message = dtype

        elif tensor_type == 1:
            num = paddle.to_tensor([0])
            paddle.distributed.recv(num, src=0, group=group)
            num = num.item()
            shapes = []
            dtypes = []
            for i in range(num):
                shape, dtype = self._recv_shape_dtype(group)
                shapes.append(shape)
                dtypes.append(dtype)

            self.recv_shape_message = tuple(shapes)
            self.recv_dtype_message = tuple(dtypes)

    def _send_dims_shape_dtype(self, tensor, group):
        # send len(shape)
        dims = paddle.to_tensor(len(tensor.shape))
        paddle.distributed.send(dims, dst=1, group=group)

        # send shape
        shape = paddle.to_tensor(tensor.shape)
        paddle.distributed.send(shape, dst=1, group=group)

        # send dtype
        dtype = paddle.to_tensor(paddle_2_number(tensor.dtype))
        paddle.distributed.send(dtype, dst=1, group=group)

    def send_meta(self, tensor, group):
        if isinstance(tensor, paddle.Tensor):
            tensor_type = paddle.to_tensor([0])
            # send tensor type
            paddle.distributed.send(tensor_type, dst=1, group=group)

            self._send_dims_shape_dtype(tensor, group)
        elif isinstance(tensor, tuple):
            tensor_type = paddle.to_tensor([1])
            # send tensor type
            paddle.distributed.send(tensor_type, dst=1, group=group)

            nums = paddle.to_tensor(len(tensor))
            paddle.distributed.send(nums, dst=1, group=group)

            for d in tensor:
                assert isinstance(d, paddle.Tensor)
                self._send_dims_shape_dtype(d, group=group)

    def set_send_message(self, tensor):
        if isinstance(tensor, paddle.Tensor):
            self.send_shape_message = tensor.shape
            self.send_dtype_message = paddle_2_number(tensor.dtype)
        elif isinstance(tensor, tuple):
            self.send_shape_message = tuple(
                [d.shape for d in tensor if not d.stop_gradient])
            self.send_dtype_message = tuple(
                [paddle_2_number(d.dtype) for d in tensor])


_send_recv_meta = SendRecvMeta()


def initialize_p2p_groups(hcg):
    global _groups, _hcg
    _hcg = hcg


def _communicate(tensor_send_next, tensor_send_prev, recv_prev, recv_next):
    global _groups, _hcg

    tensor_recv_prev = None
    tensor_recv_next = None

    recv_shape_msg = _send_recv_meta.recv_shape_message
    recv_dtype_msg = _send_recv_meta.recv_dtype_message
    send_shape_msg = _send_recv_meta.send_shape_message
    send_dtype_msg = _send_recv_meta.send_dtype_message

    if recv_prev:
        if isinstance(recv_shape_msg, tuple):
            tensor_recv_prev = []
            for idx, shape in enumerate(recv_shape_msg):
                tensor_recv_prev.append(
                    paddle.empty(
                        shape=shape, dtype=number_2_dtype(recv_dtype_msg[idx])))
            tensor_recv_prev = tuple(tensor_recv_prev)
        else:
            tensor_recv_prev = paddle.empty(
                shape=recv_shape_msg, dtype=number_2_dtype(recv_dtype_msg))

    if recv_next:
        if isinstance(send_shape_msg, tuple):
            tensor_recv_next = []
            for idx, shape in enumerate(send_shape_msg):
                tensor_recv_next.append(
                    paddle.empty(
                        shape=shape, dtype=number_2_dtype(send_dtype_msg[idx])))
            tensor_recv_next = tuple(tensor_recv_next)
        else:
            tensor_recv_next = paddle.empty(
                shape=send_shape_msg, dtype=number_2_dtype(send_dtype_msg))

    if tensor_send_prev is not None:
        if isinstance(tensor_send_prev, tuple):
            for d in tensor_send_prev:
                paddle.distributed.wait(d, use_calc_stream=True)
                paddle.distributed.send(
                    d, dst=0, group=_hcg.send_prev_group, use_calc_stream=False)
        else:
            paddle.distributed.wait(tensor_send_prev, use_calc_stream=True)
            paddle.distributed.send(
                tensor_send_prev,
                dst=0,
                group=_hcg.send_prev_group,
                use_calc_stream=False)

    if tensor_recv_prev is not None:
        if isinstance(tensor_recv_prev, tuple):
            for d in tensor_recv_prev:
                paddle.distributed.recv(
                    d, src=0, group=_hcg.recv_prev_group, use_calc_stream=True)
        else:
            paddle.distributed.recv(
                tensor_recv_prev,
                src=0,
                group=_hcg.recv_prev_group,
                use_calc_stream=True)

    if tensor_send_next is not None:
        if isinstance(tensor_send_next, tuple):
            for d in tensor_send_next:
                paddle.distributed.wait(d, use_calc_stream=True)
                paddle.distributed.send(
                    d, dst=1, group=_hcg.send_next_group, use_calc_stream=False)
        else:
            paddle.distributed.wait(tensor_send_next, use_calc_stream=True)
            paddle.distributed.send(
                tensor_send_next,
                dst=1,
                group=_hcg.send_next_group,
                use_calc_stream=False)

    if tensor_recv_next is not None:
        if isinstance(tensor_recv_next, tuple):
            for d in tensor_recv_next:
                paddle.distributed.recv(
                    d, src=1, group=_hcg.recv_next_group, use_calc_stream=True)
        else:
            paddle.distributed.recv(
                tensor_recv_next,
                src=1,
                group=_hcg.recv_next_group,
                use_calc_stream=True)

    return tensor_recv_prev, tensor_recv_next


def recv_forward():
    if _hcg.is_first_stage:
        input_tensor = None
    else:
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
