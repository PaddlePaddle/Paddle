# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle import framework
from paddle.distributed.communication.batch_isend_irecv import (
    _with_batch_p2p_guard,
)
from paddle.distributed.communication.group import (
    _get_global_group,
    _warn_cur_rank_not_in_group,
)

from ...utils import timer_helper as timer
from .utils import number_2_dtype, paddle_2_number

_hcg = None
_use_cache = False
_enable_partial_send_recv = True
_timers = None


def initialize_p2p_groups(
    hcg, use_cache=True, enable_partial_send_recv=True, enable_timer=False
):
    global _hcg, _use_cache, _enable_partial_send_recv, _timers
    _hcg = hcg
    _use_cache = use_cache
    _enable_partial_send_recv = enable_partial_send_recv
    if enable_timer:
        _timers = timer.get_timers()


class SendRecvMeta:
    """Mainly used to help p2p communication context information"""

    def __init__(self):
        self.send_shape_message = None
        self.send_dtype_message = None

        self.recv_shape_message = None
        self.recv_dtype_message = None
        self.recv_stop_gradient = None

        self.has_send_meta = False
        self.has_recv_meta = False

    def _recv_shape_dtype(self, group):
        # recv len(shape)
        dims = paddle.to_tensor([0])
        src_rank = _hcg._get_p2p_prev_rank()

        paddle.distributed.recv(dims, src=src_rank, group=group)
        dims = dims.item()

        # recv shape
        shape = paddle.to_tensor([0] * dims)
        paddle.distributed.recv(shape, src=src_rank, group=group)

        # recv dtype
        dtype = paddle.to_tensor([0])
        paddle.distributed.recv(dtype, src=src_rank, group=group)

        # recv stop_gradient
        stop_grad = paddle.to_tensor([0])
        paddle.distributed.recv(stop_grad, src=src_rank, group=group)
        return shape.tolist(), dtype.item(), stop_grad.item()

    def recv_meta(self, group):
        tensor_type = paddle.to_tensor([0])
        src_rank = _hcg._get_p2p_prev_rank()

        paddle.distributed.recv(tensor_type, src=src_rank, group=group)
        tensor_type = tensor_type.item()

        if tensor_type == 0:
            shape, dtype, stop_grad = self._recv_shape_dtype(group)
            self.recv_shape_message = shape
            self.recv_dtype_message = dtype
            self.recv_stop_gradient = bool(stop_grad)

        elif tensor_type == 1:
            num = paddle.to_tensor([0])
            paddle.distributed.recv(num, src=src_rank, group=group)
            num = num.item()
            shapes = []
            dtypes = []
            stop_grads = []
            for i in range(num):
                shape, dtype, stop_grad = self._recv_shape_dtype(group)
                shapes.append(shape)
                dtypes.append(dtype)
                stop_grads.append(bool(stop_grad))

            self.recv_shape_message = tuple(shapes)
            self.recv_dtype_message = tuple(dtypes)
            self.recv_stop_gradient = tuple(stop_grads)

    def _send_dims_shape_dtype(self, tensor, group):
        # send len(shape)
        dims = paddle.to_tensor([len(tensor.shape)])
        dst_rank = _hcg._get_p2p_next_rank()

        paddle.distributed.send(dims, dst=dst_rank, group=group)

        # send shape
        shape = paddle.to_tensor(tensor.shape)
        paddle.distributed.send(shape, dst=dst_rank, group=group)

        # send dtype
        dtype = paddle.to_tensor([paddle_2_number(tensor.dtype)])
        paddle.distributed.send(dtype, dst=dst_rank, group=group)

        # send trainable
        stop_grad = paddle.to_tensor([int(tensor.stop_gradient)])
        paddle.distributed.send(stop_grad, dst=dst_rank, group=group)

    def send_meta(self, tensor, group):
        dst_rank = _hcg._get_p2p_next_rank()

        if isinstance(tensor, (paddle.Tensor, framework.core.eager.Tensor)):
            tensor_type = paddle.to_tensor([0])
            # send tensor type
            paddle.distributed.send(tensor_type, dst=dst_rank, group=group)

            self._send_dims_shape_dtype(tensor, group)
        elif isinstance(tensor, tuple):
            tensor_type = paddle.to_tensor([1])
            # send tensor type
            paddle.distributed.send(tensor_type, dst=dst_rank, group=group)

            nums = paddle.to_tensor([len(tensor)])
            paddle.distributed.send(nums, dst=dst_rank, group=group)

            for d in tensor:
                assert isinstance(
                    d, (paddle.Tensor, framework.core.eager.Tensor)
                )
                self._send_dims_shape_dtype(d, group=group)

    def set_send_message(self, tensor):
        if isinstance(tensor, (paddle.Tensor, framework.core.eager.Tensor)):
            self.send_shape_message = tensor.shape
            self.send_dtype_message = paddle_2_number(tensor.dtype)
        elif isinstance(tensor, tuple):
            self.send_shape_message = tuple(
                [d.shape for d in tensor if not d.stop_gradient]
            )
            self.send_dtype_message = tuple(
                [
                    paddle_2_number(d.dtype)
                    for d in tensor
                    if not d.stop_gradient
                ]
            )


_send_recv_meta = SendRecvMeta()


def _is_valid_send_recv_partial(tensor, mp_degree):
    if not _enable_partial_send_recv:
        return False
    tensor_numel = np.prod(tensor.shape)
    assert tensor_numel > 0, "can't send/recv zero element"
    return mp_degree > 1 and tensor_numel % mp_degree == 0


def _send_on_calc_stream(tensor, group, dst, nranks=1, rank_id=0):
    assert (
        group is not None
    ), "Group should be an instance for _send_on_calc_stream."
    dst_rank_in_group = group.get_group_rank(dst)
    if _is_valid_send_recv_partial(tensor, nranks):
        return group.process_group.send_partial_on_calc_stream(
            tensor, dst_rank_in_group, nranks, rank_id
        )
    else:
        return group.process_group.send_on_calc_stream(
            tensor, dst_rank_in_group
        )


def _recv_on_calc_stream(tensor, group, src, nranks=1, rank_id=0):
    assert (
        group is not None
    ), "Group should be an instance for _recv_on_calc_stream."
    src_rank_in_group = group.get_group_rank(src)
    if _is_valid_send_recv_partial(tensor, nranks):
        return group.process_group.recv_partial_on_calc_stream(
            tensor, src_rank_in_group, nranks, rank_id
        )
    else:
        return group.process_group.recv_on_calc_stream(
            tensor, src_rank_in_group
        )


class P2PonCalcStream:
    def __init__(self, op, tensor, peer, group, nranks=1, rank_id=0):
        """
        Args:
            op (function): The function to be executed on the calc stream.
            tensor (Tensor): The tensor to be sent or received.
            peer (int): The peer rank.
            group (Group): The process group to p2p.
            nranks (int): The number of ranks in model parallel group.
            rank_id (int): The rank id in the model parallel group.
        """
        if op not in [_send_on_calc_stream, _recv_on_calc_stream]:
            raise RuntimeError(
                "Invalid ``op`` function. Expected ``op`` "
                "to be of type ``_send_on_calc_stream`` or "
                "``_recv_on_calc_stream``."
            )
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group
        self.nranks = nranks
        self.rank_id = rank_id


def _partial_allgather_op(
    tensor, group, use_calc_stream, ring_id, nranks, rank_id
):
    group = (
        paddle.distributed.collective._get_default_group()
        if group is None
        else group
    )
    comm_op = (
        group.process_group.all_gather_partial_on_calc_stream
        if use_calc_stream
        else group.process_group.all_gather_partial
    )
    return comm_op(tensor, tensor, nranks, rank_id)


def allgather_partial(
    tensor, nranks=1, rank_id=0, group=None, use_calc_stream=True
):
    if not _is_valid_send_recv_partial(tensor, nranks):
        return tensor
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    return _partial_allgather_op(
        tensor, group, use_calc_stream, ring_id, nranks, rank_id
    )


def batch_send_recv_on_calc_stream(p2p_op_list):
    group = p2p_op_list[0].group
    if _warn_cur_rank_not_in_group(group):
        return
    group = _get_global_group() if group is None else group
    backend = group.backend
    with _with_batch_p2p_guard(backend):
        for p2p_op in p2p_op_list:
            op = p2p_op.op
            tensor = p2p_op.tensor
            peer = p2p_op.peer
            comm_group = p2p_op.group
            nranks = p2p_op.nranks
            rank_id = p2p_op.rank_id
            op(tensor, comm_group, peer, nranks, rank_id)


def _process_p2p_tuple_or_tensor(
    tensors, p2p_func, pp_rank, pp_group, mp_degree=1, mp_rank=0
):
    ops = []
    if isinstance(tensors, tuple):
        for tensor in tensors:
            op = P2PonCalcStream(
                p2p_func, tensor, pp_rank, pp_group, mp_degree, mp_rank
            )
            ops.append(op)
    else:
        op = P2PonCalcStream(
            p2p_func, tensors, pp_rank, pp_group, mp_degree, mp_rank
        )
        ops.append(op)
    return ops


def _p2p_helper(
    tensor_send_next, tensor_send_prev, recv_prev, recv_next, sync_recv=True
):
    global _hcg

    tensor_recv_prev = None
    tensor_recv_next = None

    # send / recv message
    recv_shape_msg = _send_recv_meta.recv_shape_message
    recv_dtype_msg = _send_recv_meta.recv_dtype_message
    recv_stop_gradient = _send_recv_meta.recv_stop_gradient

    send_shape_msg = _send_recv_meta.send_shape_message
    send_dtype_msg = _send_recv_meta.send_dtype_message

    # model parallel message
    mp_group = _hcg.get_model_parallel_group()
    mp_degree = _hcg.get_model_parallel_world_size()
    mp_rank = _hcg.get_model_parallel_rank()

    if recv_prev:
        if isinstance(recv_shape_msg, tuple):
            tensor_recv_prev = []
            for idx, shape in enumerate(recv_shape_msg):
                tmp = paddle.empty(
                    shape=shape, dtype=number_2_dtype(recv_dtype_msg[idx])
                )
                tmp.stop_gradient = recv_stop_gradient[idx]
                tensor_recv_prev.append(tmp)
            tensor_recv_prev = tuple(tensor_recv_prev)
        else:
            tensor_recv_prev = paddle.empty(
                shape=recv_shape_msg, dtype=number_2_dtype(recv_dtype_msg)
            )
            tensor_recv_prev.stop_gradient = recv_stop_gradient

    if recv_next:
        if isinstance(send_shape_msg, tuple):
            tensor_recv_next = []
            for idx, shape in enumerate(send_shape_msg):
                tensor_recv_next.append(
                    paddle.empty(
                        shape=shape, dtype=number_2_dtype(send_dtype_msg[idx])
                    )
                )
            tensor_recv_next = tuple(tensor_recv_next)
        else:
            tensor_recv_next = paddle.empty(
                shape=send_shape_msg, dtype=number_2_dtype(send_dtype_msg)
            )

    ops = []
    pipe_group = _hcg.get_pipe_parallel_group()

    # start to p2p communicate
    if tensor_send_prev is not None:
        src_rank = _hcg._get_p2p_prev_rank()
        ops.extend(
            _process_p2p_tuple_or_tensor(
                tensor_send_prev,
                _send_on_calc_stream,
                src_rank,
                pipe_group,
                mp_degree,
                mp_rank,
            )
        )
    if tensor_recv_prev is not None:
        dst_rank = _hcg._get_p2p_prev_rank()
        ops.extend(
            _process_p2p_tuple_or_tensor(
                tensor_recv_prev,
                _recv_on_calc_stream,
                dst_rank,
                pipe_group,
                mp_degree,
                mp_rank,
            )
        )
    if tensor_send_next is not None:
        src_rank = _hcg._get_p2p_next_rank()
        ops.extend(
            _process_p2p_tuple_or_tensor(
                tensor_send_next,
                _send_on_calc_stream,
                src_rank,
                pipe_group,
                mp_degree,
                mp_rank,
            )
        )

    if tensor_recv_next is not None:
        dst_rank = _hcg._get_p2p_next_rank()
        ops.extend(
            _process_p2p_tuple_or_tensor(
                tensor_recv_next,
                _recv_on_calc_stream,
                dst_rank,
                pipe_group,
                mp_degree,
                mp_rank,
            )
        )
    if len(ops) > 0:
        batch_send_recv_on_calc_stream(ops)

    tensors_for_all_gather = []
    if tensor_recv_prev is not None:
        if isinstance(tensor_recv_prev, tuple):
            for d in tensor_recv_prev:
                tensors_for_all_gather.append(d)
        else:
            tensors_for_all_gather.append(tensor_recv_prev)
    if tensor_recv_next is not None:
        if isinstance(tensor_recv_next, tuple):
            for d in tensor_recv_next:
                tensors_for_all_gather.append(d)
        else:
            tensors_for_all_gather.append(tensor_recv_next)

    for tensor in tensors_for_all_gather:
        allgather_partial(
            tensor,
            nranks=mp_degree,
            rank_id=mp_rank,
            group=mp_group,
            use_calc_stream=True,
        )

    return tensor_recv_prev, tensor_recv_next


def recv_forward(pp_first_stage, sync_recv=True):
    global _timers
    if _timers is not None:
        _timers("recv_forward").start()
    if pp_first_stage:
        input_tensor = None
    else:
        if not _send_recv_meta.has_recv_meta:
            _send_recv_meta.recv_meta(_hcg.get_pipe_parallel_group())
            _send_recv_meta.has_recv_meta = _use_cache

        input_tensor, _ = _p2p_helper(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            sync_recv=sync_recv,
        )
    if _timers is not None:
        _timers("recv_forward").stop()
    return input_tensor


def recv_backward(pp_last_stage, sync_recv=True):
    global _timers
    if _timers is not None:
        _timers("recv_backward").start()
    if pp_last_stage:
        output_tensor_grad = None
    else:
        _, output_tensor_grad = _p2p_helper(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            sync_recv=sync_recv,
        )
    if _timers is not None:
        _timers("recv_backward").stop()
    return output_tensor_grad


def send_forward(output_tensor, pp_last_stage):
    global _timers
    if _timers is not None:
        _timers("send_forward").start()
    if not pp_last_stage:
        if not _send_recv_meta.has_send_meta:
            _send_recv_meta.set_send_message(output_tensor)
            _send_recv_meta.send_meta(
                output_tensor, _hcg.get_pipe_parallel_group()
            )
            _send_recv_meta.has_send_meta = _use_cache

        _p2p_helper(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
        )
    if _timers is not None:
        _timers("send_forward").stop()


def send_backward(input_tensor_grad, pp_first_stage):
    global _timers
    if _timers is not None:
        _timers("send_backward").start()
    if not pp_first_stage:
        _p2p_helper(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
        )
    if _timers is not None:
        _timers("send_backward").stop()


def send_forward_recv_backward(output_tensor, pp_last_stage):
    global _timers
    if _timers is not None:
        _timers("send_forward_recv_backward").start()
    if pp_last_stage:
        output_tensor_grad = None
    else:
        _, output_tensor_grad = _p2p_helper(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
        )
    if _timers is not None:
        _timers("send_forward_recv_backward").stop()
    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad, pp_first_stage):
    global _timers
    if _timers is not None:
        _timers("send_backward_recv_forward").start()
    if pp_first_stage:
        input_tensor = None
    else:
        input_tensor, _ = _p2p_helper(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False,
        )
    if _timers is not None:
        _timers("send_backward_recv_forward").stop()
    return input_tensor


def send_forward_backward_recv_forward_backward(
    output_tensor, input_tensor_grad, recv_prev, recv_next
):
    # always have to send dytpe info to downstream
    global _timers
    if _timers is not None:
        _timers("send_forward_backward_recv_forward_backward").start()
    if not _send_recv_meta.has_send_meta:
        _send_recv_meta.set_send_message(output_tensor)
        _send_recv_meta.send_meta(output_tensor, _hcg.get_pipe_parallel_group())
        _send_recv_meta.has_send_meta = _use_cache
    if recv_prev and not _send_recv_meta.has_recv_meta:
        _send_recv_meta.recv_meta(_hcg.get_pipe_parallel_group())
        _send_recv_meta.has_recv_meta = _use_cache
    input_tensor, output_tensor_grad = _p2p_helper(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        sync_recv=False,
    )
    if _timers is not None:
        _timers("send_forward_backward_recv_forward_backward").stop()
    return input_tensor, output_tensor_grad


def send_forward_recv_forward(output_tensor, recv_prev):
    # always have to send dytpe info to downstream
    global _timers
    if _timers is not None:
        _timers("send_forward_recv_forward").start()
    if not _send_recv_meta.has_send_meta:
        _send_recv_meta.set_send_message(output_tensor)
        _send_recv_meta.send_meta(output_tensor, _hcg.get_pipe_parallel_group())
        _send_recv_meta.has_send_meta = _use_cache
    if recv_prev and not _send_recv_meta.has_recv_meta:
        _send_recv_meta.recv_meta(_hcg.get_pipe_parallel_group())
        _send_recv_meta.has_recv_meta = _use_cache

    input_tensor, _ = _p2p_helper(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        sync_recv=False,
    )
    if _timers is not None:
        _timers("send_forward_recv_forward").stop()
    return input_tensor


def send_backward_recv_backward(input_tensor_grad, recv_next):
    global _timers
    if _timers is not None:
        _timers("send_backward_recv_backward").start()
    _, output_tensor_grad = _p2p_helper(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        sync_recv=False,
    )
    if _timers is not None:
        _timers("send_backward_recv_backward").stop()
    return output_tensor_grad
