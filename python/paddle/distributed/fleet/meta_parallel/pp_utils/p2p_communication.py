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

import os

import numpy as np

import paddle
from paddle.distributed.communication.batch_isend_irecv import (
    _coalescing_manager,
)
from paddle.distributed.communication.group import (
    _get_global_group,
    _warn_cur_rank_not_in_group,
)
from paddle.framework.recall_error import check_naninf
from paddle.utils import strtobool

from ...utils import timer_helper as timer
from .utils import number_2_dtype, paddle_2_number

_hcg = None
# _use_cache = False
_enable_partial_send_recv = True
_timers = None

_sync_send = os.environ.get("PADDLE_P2P_SYNC_SEND", "0")
_sync_send = _sync_send.lower() in ['1', 'true']


def initialize_p2p_groups(
    hcg, enable_partial_send_recv=True, enable_timer=False
):
    global _hcg, _enable_partial_send_recv, _timers
    _hcg = hcg
    _enable_partial_send_recv = enable_partial_send_recv
    if enable_timer:
        _timers = timer.get_timers()


class SendRecvMeta:
    """Mainly used to help p2p communication context information"""

    def __init__(self):
        self.init_or_erase_meta()

    def init_or_erase_meta(self):
        self.send_shape_message = None
        self.send_dtype_message = None

        self.recv_shape_message = None
        self.recv_dtype_message = None
        self.recv_stop_gradient = None

        self.has_send_meta = False
        self.has_recv_meta = False

    def recv_meta(self, group, reverse=False):
        if reverse:
            src_rank = _hcg._get_p2p_next_rank()
        else:
            src_rank = _hcg._get_p2p_prev_rank()

        data_numel = paddle.empty([1], dtype="int64")
        paddle.distributed.recv(data_numel, src=src_rank, group=group)
        data_numel = data_numel.item()

        data = paddle.empty([data_numel], dtype="int64")

        paddle.distributed.recv(data, src=src_rank, group=group)
        data = data.numpy().tolist()
        # parse data
        tensor_type = data.pop(0)

        if tensor_type == 1:
            tensor_num = data.pop(0)
        else:
            tensor_num = 1

        shapes = []
        dtypes = []
        stop_grads = []

        for _ in range(tensor_num):
            shape_len = data.pop(0)
            shape = data[:shape_len]
            data = data[shape_len:]
            dtype_number = data.pop(0)
            stop_gradient = bool(data.pop(0))

            shapes.append(shape)
            dtypes.append(dtype_number)
            stop_grads.append(stop_gradient)

        assert (
            len(data) == 0
        ), f"send data must be parsed zero, now it is {data}"

        if tensor_type == 0:
            self.recv_shape_message = shapes[0]
            self.recv_dtype_message = dtypes[0]
            self.recv_stop_gradient = stop_grads[0]
        else:
            self.recv_shape_message = tuple(shapes)
            self.recv_dtype_message = tuple(dtypes)
            self.recv_stop_gradient = tuple(stop_grads)

    def send_meta(self, tensor, group, reverse=False):
        if reverse:
            dst_rank = _hcg._get_p2p_prev_rank()
        else:
            dst_rank = _hcg._get_p2p_next_rank()

        if isinstance(tensor, paddle.Tensor):
            tensor_type = 0
            tensors_to_send = [tensor]
        elif isinstance(tensor, tuple):
            tensor_type = 1
            tensors_to_send = list(tensor)
        else:
            raise TypeError(
                "tensor must be paddle.Tensor or Tuple of paddle.Tensor"
            )

        # prepare data to send
        data = [tensor_type]

        if tensor_type == 1:
            data.append(len(tensors_to_send))

        for t in tensors_to_send:
            assert isinstance(t, paddle.Tensor)
            data.extend(
                [
                    len(t.shape),
                    *t.shape,
                    paddle_2_number(t.dtype),
                    int(t.stop_gradient),
                ]
            )

        data_tensor = paddle.to_tensor(data).astype("int64")
        data_numel = np.prod(data_tensor.shape)

        paddle.distributed.send(
            paddle.to_tensor(data_numel).astype("int64"),
            dst=dst_rank,
            group=group,
        )
        paddle.distributed.send(data_tensor, dst=dst_rank, group=group)

    def _obtain_send_message(self, tensor):
        if isinstance(tensor, paddle.Tensor):
            return tensor.shape, paddle_2_number(tensor.dtype)
        else:
            shapes = []
            dtypes = []
            for d in tensor:
                assert isinstance(d, paddle.Tensor)
                if d.stop_gradient:
                    continue
                shape, dtype = self._obtain_send_message(d)
                shapes.append(shape)
                dtypes.append(dtype)
            return tuple(shapes), tuple(dtypes)

    def set_send_message(self, tensor):
        (
            self.send_shape_message,
            self.send_dtype_message,
        ) = self._obtain_send_message(tensor)

    def check_send_message(self, tensor):
        if self.send_shape_message is None or self.send_dtype_message is None:
            return
        actual_shape, actual_dtype = self._obtain_send_message(tensor)
        assert (
            self.send_shape_message == actual_shape
        ), f"send_shape_message: {self.send_shape_message}, actual_shape: {actual_shape}"
        assert (
            self.send_dtype_message == actual_dtype
        ), f"send_dtype_message: {self.send_dtype_message}, actual_dtype: {actual_dtype}"

    def __repr__(self):
        return f"send_shape_message: {self.send_shape_message}, send_dtype_message: {self.send_dtype_message}, recv_shape_message: {self.recv_shape_message}, recv_dtype_message: {self.recv_dtype_message}, recv_stop_gradient: {self.recv_stop_gradient}"


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

    need_check = strtobool(os.getenv('FLAGS_pp_check_naninf', '0'))
    if need_check:
        for p2p_op in p2p_op_list:
            if p2p_op.op == _send_on_calc_stream:
                err_msg = check_naninf(p2p_op.tensor)
                if err_msg is not None:
                    raise ValueError(
                        f"{err_msg}. Tensor contains inf or nan values at rank {paddle.distributed.get_rank()}"
                    )

    group = _get_global_group() if group is None else group
    backend = group.backend
    tasks = []
    with _coalescing_manager(group, tasks):
        for p2p_op in p2p_op_list:
            op = p2p_op.op
            tensor = p2p_op.tensor
            peer = p2p_op.peer
            comm_group = p2p_op.group
            nranks = p2p_op.nranks
            rank_id = p2p_op.rank_id
            op(tensor, comm_group, peer, nranks, rank_id)


def _batch_p2p_tuple_or_tensor(
    tensors, p2p_func, pp_rank, pp_group, mp_degree=1, mp_rank=0
):
    if not isinstance(tensors, tuple):
        tensors = (tensors,)
    ops = [
        P2PonCalcStream(p2p_func, tensor, pp_rank, pp_group, mp_degree, mp_rank)
        for tensor in tensors
    ]
    return ops


def _batched_p2p_ops(
    tensor_send_prev, tensor_recv_prev, tensor_send_next, tensor_recv_next, hcg
):
    ops = []
    pipe_group = hcg.get_pipe_parallel_group()
    mp_degree = hcg.get_model_parallel_world_size()
    mp_rank = hcg.get_model_parallel_rank()
    mp_group = hcg.get_model_parallel_group()

    # start to p2p communicate
    if not _sync_send:
        if tensor_send_prev is not None:
            src_rank = hcg._get_p2p_prev_rank()
            ops.extend(
                _batch_p2p_tuple_or_tensor(
                    tensor_send_prev,
                    _send_on_calc_stream,
                    src_rank,
                    pipe_group,
                    mp_degree,
                    mp_rank,
                )
            )
        if tensor_recv_prev is not None:
            dst_rank = hcg._get_p2p_prev_rank()
            ops.extend(
                _batch_p2p_tuple_or_tensor(
                    tensor_recv_prev,
                    _recv_on_calc_stream,
                    dst_rank,
                    pipe_group,
                    mp_degree,
                    mp_rank,
                )
            )
        if tensor_send_next is not None:
            src_rank = hcg._get_p2p_next_rank()
            ops.extend(
                _batch_p2p_tuple_or_tensor(
                    tensor_send_next,
                    _send_on_calc_stream,
                    src_rank,
                    pipe_group,
                    mp_degree,
                    mp_rank,
                )
            )
        if tensor_recv_next is not None:
            dst_rank = hcg._get_p2p_next_rank()
            ops.extend(
                _batch_p2p_tuple_or_tensor(
                    tensor_recv_next,
                    _recv_on_calc_stream,
                    dst_rank,
                    pipe_group,
                    mp_degree,
                    mp_rank,
                )
            )
    else:
        # Some devices(NPU for example) do not support asynchronized send op, So the order is
        # recv_prev -> send_next -> recv_next -> send_prev
        # When using this order, the environment variable
        # 'PADDLE_P2P_SYNC_SEND' should be set True
        if tensor_recv_prev is not None:
            dst_rank = hcg._get_p2p_prev_rank()
            ops.extend(
                _batch_p2p_tuple_or_tensor(
                    tensor_recv_prev,
                    _recv_on_calc_stream,
                    dst_rank,
                    pipe_group,
                    mp_degree,
                    mp_rank,
                )
            )
        if tensor_send_next is not None:
            src_rank = hcg._get_p2p_next_rank()
            ops.extend(
                _batch_p2p_tuple_or_tensor(
                    tensor_send_next,
                    _send_on_calc_stream,
                    src_rank,
                    pipe_group,
                    mp_degree,
                    mp_rank,
                )
            )
        if tensor_recv_next is not None:
            dst_rank = hcg._get_p2p_next_rank()
            ops.extend(
                _batch_p2p_tuple_or_tensor(
                    tensor_recv_next,
                    _recv_on_calc_stream,
                    dst_rank,
                    pipe_group,
                    mp_degree,
                    mp_rank,
                )
            )
        if tensor_send_prev is not None:
            src_rank = hcg._get_p2p_prev_rank()
            ops.extend(
                _batch_p2p_tuple_or_tensor(
                    tensor_send_prev,
                    _send_on_calc_stream,
                    src_rank,
                    pipe_group,
                    mp_degree,
                    mp_rank,
                )
            )

    if len(ops) > 0:
        batch_send_recv_on_calc_stream(ops)
        if strtobool(os.getenv('FLAGS_p2p_device_synchronize', '0')):
            paddle.device.cuda.synchronize()

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


def _p2p_ops_tuple_or_tensor(tensors, p2p_func, pp_rank, pp_group):
    if not isinstance(tensors, tuple):
        tensors = (tensors,)

    need_check = strtobool(os.getenv('FLAGS_pp_check_naninf', '0'))
    if need_check:
        if p2p_func == paddle.distributed.isend:
            for t in tensors:
                err_msg = check_naninf(t)
                if err_msg is not None:
                    raise ValueError(
                        f"{err_msg}. Tensor contains inf or nan values at rank {paddle.distributed.get_rank()}"
                    )

    reqs = []
    for tensor in tensors:
        reqs.append(p2p_func(tensor, pp_rank, pp_group))
    return reqs


def _p2p_ops(
    tensor_send_prev, tensor_recv_prev, tensor_send_next, tensor_recv_next, hcg
):
    reqs = []
    group = hcg.get_pipe_parallel_group()
    if hcg.get_stage_id() % 2 == 0:
        if tensor_send_next is not None:
            reqs.extend(
                _p2p_ops_tuple_or_tensor(
                    tensor_send_next,
                    paddle.distributed.isend,
                    hcg._get_p2p_next_rank(),
                    group,
                )
            )
        if tensor_recv_prev is not None:
            reqs.extend(
                _p2p_ops_tuple_or_tensor(
                    tensor_recv_prev,
                    paddle.distributed.irecv,
                    hcg._get_p2p_prev_rank(),
                    group,
                )
            )

        if tensor_send_prev is not None:
            reqs.extend(
                _p2p_ops_tuple_or_tensor(
                    tensor_send_prev,
                    paddle.distributed.isend,
                    _hcg._get_p2p_prev_rank(),
                    group,
                )
            )

        if tensor_recv_next is not None:
            reqs.extend(
                _p2p_ops_tuple_or_tensor(
                    tensor_recv_next,
                    paddle.distributed.irecv,
                    hcg._get_p2p_next_rank(),
                    group,
                )
            )
    else:
        if tensor_recv_prev is not None:
            reqs.extend(
                _p2p_ops_tuple_or_tensor(
                    tensor_recv_prev,
                    paddle.distributed.irecv,
                    hcg._get_p2p_prev_rank(),
                    group,
                )
            )
        if tensor_send_next is not None:
            reqs.extend(
                _p2p_ops_tuple_or_tensor(
                    tensor_send_next,
                    paddle.distributed.isend,
                    hcg._get_p2p_next_rank(),
                    group,
                )
            )
        if tensor_recv_next is not None:
            reqs.extend(
                _p2p_ops_tuple_or_tensor(
                    tensor_recv_next,
                    paddle.distributed.irecv,
                    hcg._get_p2p_next_rank(),
                    group,
                )
            )
        if tensor_send_prev is not None:
            reqs.extend(
                _p2p_ops_tuple_or_tensor(
                    tensor_send_prev,
                    paddle.distributed.isend,
                    hcg._get_p2p_prev_rank(),
                    group,
                )
            )
    return reqs


def _p2p_helper(
    tensor_send_next,
    tensor_send_prev,
    recv_prev,
    recv_next,
    sync_recv=True,
    send_recv_meta=None,
    batch_p2p_comm=True,
    wait_on_reqs=True,
    dynamic_shape=False,
):
    global _hcg

    tensor_recv_prev = None
    tensor_recv_next = None

    # send / recv message
    assert send_recv_meta is not None, "send_recv_meta should not be None"
    recv_shape_msg = send_recv_meta.recv_shape_message
    recv_dtype_msg = send_recv_meta.recv_dtype_message
    recv_stop_gradient = send_recv_meta.recv_stop_gradient

    send_shape_msg = send_recv_meta.send_shape_message
    send_dtype_msg = send_recv_meta.send_dtype_message

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
        if dynamic_shape:
            send_shape_msg = send_recv_meta.recv_shape_message
            send_dtype_msg = send_recv_meta.recv_dtype_message
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

    p2p_func = _batched_p2p_ops if batch_p2p_comm else _p2p_ops
    reqs = p2p_func(
        tensor_send_prev,
        tensor_recv_prev,
        tensor_send_next,
        tensor_recv_next,
        _hcg,
    )

    # NOTE(shenliang03): batch_p2p_comm no need wait because of using calculate stream
    if wait_on_reqs and not batch_p2p_comm and len(reqs) > 0:
        for req in reqs:
            req.wait()
        reqs = None

    return tensor_recv_prev, tensor_recv_next, reqs


class P2pHelper:
    def __init__(self, use_cache=True, dynamic_shape=False):
        self._send_recv_meta = SendRecvMeta()
        self._use_cache = use_cache
        self._dynamic_shape = dynamic_shape

        if dynamic_shape:
            self._send_recv_meta_list = []
            self._dynamic_cnt = 0

    def _send_meta(self, output_tensor, skip_check_meta=False, reverse=False):
        if not self._dynamic_shape:
            if not self._send_recv_meta.has_send_meta:
                self._send_recv_meta.set_send_message(output_tensor)
                self._send_recv_meta.send_meta(
                    output_tensor,
                    _hcg.get_pipe_parallel_group(),
                    reverse=reverse,
                )
                self._send_recv_meta.has_send_meta = self._use_cache
            elif not skip_check_meta:
                self._send_recv_meta.check_send_message(output_tensor)
        else:
            if len(self._send_recv_meta_list) <= self._dynamic_cnt:
                meta = SendRecvMeta()
                meta.set_send_message(output_tensor)
                meta.send_meta(
                    output_tensor,
                    _hcg.get_pipe_parallel_group(),
                    reverse=reverse,
                )
                meta.has_send_meta = self._use_cache
                self._send_recv_meta_list.append(meta)
                self._send_recv_meta = meta
            elif not skip_check_meta:
                meta = self._send_recv_meta_list[self._dynamic_cnt]
                meta.check_send_message(output_tensor)
                self._send_recv_meta = meta

    def _recv_meta(self, reverse=False):
        if not self._dynamic_shape:
            if not self._send_recv_meta.has_recv_meta:
                self._send_recv_meta.recv_meta(
                    _hcg.get_pipe_parallel_group(), reverse=reverse
                )
                self._send_recv_meta.has_recv_meta = self._use_cache
        else:
            if len(self._send_recv_meta_list) <= self._dynamic_cnt:
                meta = SendRecvMeta()
                meta.recv_meta(_hcg.get_pipe_parallel_group(), reverse=reverse)
                meta.has_recv_meta = self._use_cache
                self._send_recv_meta_list.append(meta)
                self._send_recv_meta = meta
            elif not self._send_recv_meta_list[self._dynamic_cnt].has_recv_meta:
                meta = self._send_recv_meta_list[self._dynamic_cnt]
                meta.recv_meta(_hcg.get_pipe_parallel_group(), reverse=reverse)
                meta.has_recv_meta = self._use_cache
                self._send_recv_meta = meta
            else:
                self._send_recv_meta = self._send_recv_meta_list[
                    self._dynamic_cnt
                ]

    def clear_meta_cache(self):
        self._send_recv_meta.init_or_erase_meta()

    def recv_forward(self, pp_first_stage, sync_recv=True, batch_p2p_comm=True):
        global _timers
        if _timers is not None:
            _timers("recv_forward").start()
        if pp_first_stage:
            input_tensor = None
        else:
            self._recv_meta()

            input_tensor, _, _ = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                sync_recv=sync_recv,
                send_recv_meta=self._send_recv_meta,
                batch_p2p_comm=batch_p2p_comm,
            )
            if self._dynamic_shape:
                self._dynamic_cnt += 1

        if _timers is not None:
            _timers("recv_forward").stop()
        return input_tensor

    def recv_backward(
        self,
        pp_last_stage,
        sync_recv=True,
        batch_p2p_comm=True,
    ):
        global _timers
        if _timers is not None:
            _timers("recv_backward").start()

        assert (
            not self._dynamic_shape
        ), "p2p_helper.recv_backward function doesn't support dynamic_shape now"

        if pp_last_stage:
            output_tensor_grad = None
        else:
            _, output_tensor_grad, _ = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                sync_recv=sync_recv,
                send_recv_meta=self._send_recv_meta,
                batch_p2p_comm=batch_p2p_comm,
            )

        if _timers is not None:
            _timers("recv_backward").stop()

        return output_tensor_grad

    def send_forward(
        self,
        output_tensor,
        pp_last_stage,
        batch_p2p_comm=True,
        skip_check_meta=False,
    ):
        global _timers
        if _timers is not None:
            _timers("send_forward").start()

        assert (
            not self._dynamic_shape
        ), "p2p_helper.send_forward function doesn't support dynamic_shape now"

        if not pp_last_stage:
            self._send_meta(output_tensor, skip_check_meta=skip_check_meta)
            _p2p_helper(
                tensor_send_next=output_tensor,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                send_recv_meta=self._send_recv_meta,
                batch_p2p_comm=batch_p2p_comm,
            )
        if _timers is not None:
            _timers("send_forward").stop()

    def send_backward(
        self, input_tensor_grad, pp_first_stage, batch_p2p_comm=True
    ):
        global _timers
        if _timers is not None:
            _timers("send_backward").start()

        assert (
            not self._dynamic_shape
        ), "p2p_helper.send_backward function doesn't support dynamic_shape now"

        if not pp_first_stage:
            _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=input_tensor_grad,
                recv_prev=False,
                recv_next=False,
                send_recv_meta=self._send_recv_meta,
                batch_p2p_comm=batch_p2p_comm,
            )
        if _timers is not None:
            _timers("send_backward").stop()

    def send_forward_recv_backward(
        self, output_tensor, pp_last_stage, batch_p2p_comm=True
    ):
        global _timers
        if _timers is not None:
            _timers("send_forward_recv_backward").start()

        assert (
            not self._dynamic_shape
        ), "p2p_helper.send_forward_recv_backward function doesn't support dynamic_shape now"

        if pp_last_stage:
            output_tensor_grad = None
        else:
            _, output_tensor_grad, _ = _p2p_helper(
                tensor_send_next=output_tensor,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                send_recv_meta=self._send_recv_meta,
                batch_p2p_comm=batch_p2p_comm,
            )
        if _timers is not None:
            _timers("send_forward_recv_backward").stop()
        return output_tensor_grad

    def send_backward_recv_forward(
        self, input_tensor_grad, pp_first_stage, batch_p2p_comm=True
    ):
        global _timers
        if _timers is not None:
            _timers("send_backward_recv_forward").start()

        assert (
            not self._dynamic_shape
        ), "p2p_helper.send_backward_recv_forward function doesn't support dynamic_shape now"

        if pp_first_stage:
            input_tensor = None
        else:
            input_tensor, _, _ = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=input_tensor_grad,
                recv_prev=True,
                recv_next=False,
                send_recv_meta=self._send_recv_meta,
                batch_p2p_comm=batch_p2p_comm,
            )
        if _timers is not None:
            _timers("send_backward_recv_forward").stop()
        return input_tensor

    def send_forward_backward_recv_forward_backward(
        self,
        output_tensor,
        input_tensor_grad,
        recv_prev,
        recv_next,
        batch_p2p_comm=True,
        skip_check_meta=False,
    ):
        # always have to send dtype info to downstream
        global _timers
        if _timers is not None:
            _timers("send_forward_backward_recv_forward_backward").start()

        assert (
            not self._dynamic_shape
        ), "p2p_helper.send_forward_backward_recv_forward_backward function doesn't support dynamic_shape now"

        if output_tensor is not None:
            self._send_meta(output_tensor, skip_check_meta=skip_check_meta)
        if recv_prev:
            self._recv_meta()

        input_tensor, output_tensor_grad, _ = _p2p_helper(
            tensor_send_next=output_tensor,
            tensor_send_prev=input_tensor_grad,
            recv_prev=recv_prev,
            recv_next=recv_next,
            sync_recv=False,
            send_recv_meta=self._send_recv_meta,
            batch_p2p_comm=batch_p2p_comm,
        )
        if _timers is not None:
            _timers("send_forward_backward_recv_forward_backward").stop()
        return input_tensor, output_tensor_grad

    def send_forward_recv_forward(
        self,
        output_tensor,
        recv_prev,
        batch_p2p_comm=True,
        overlap_p2p_comm=False,
        skip_check_meta=False,
    ):
        # always have to send dtype info to downstream
        global _timers
        if _timers is not None:
            _timers("send_forward_recv_forward").start()

        need_increase_cnt = False

        if output_tensor is not None:
            self._send_meta(
                output_tensor,
                skip_check_meta=skip_check_meta,
            )
            need_increase_cnt = True

        if recv_prev:
            self._recv_meta()
            need_increase_cnt = True

        input_tensor, _, wait_handles = _p2p_helper(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            sync_recv=False,
            send_recv_meta=self._send_recv_meta,
            batch_p2p_comm=batch_p2p_comm,
            wait_on_reqs=(not overlap_p2p_comm),
        )
        if _timers is not None:
            _timers("send_forward_recv_forward").stop()

        if self._dynamic_shape and need_increase_cnt:
            self._dynamic_cnt += 1

        if overlap_p2p_comm:
            return input_tensor, wait_handles
        return input_tensor

    def send_backward_recv_backward(
        self,
        input_tensor_grad,
        recv_next,
        batch_p2p_comm=True,
        overlap_p2p_comm=False,
    ):
        global _timers
        if _timers is not None:
            _timers("send_backward_recv_backward").start()

        if self._dynamic_shape:
            need_increase_cnt = False
            if input_tensor_grad is not None:
                self._send_meta(input_tensor_grad, reverse=True)
                need_increase_cnt = True

            if recv_next:
                self._recv_meta(reverse=True)
                need_increase_cnt = True

        _, output_tensor_grad, wait_handles = _p2p_helper(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=recv_next,
            sync_recv=False,
            send_recv_meta=self._send_recv_meta,
            batch_p2p_comm=batch_p2p_comm,
            wait_on_reqs=(not overlap_p2p_comm),
            dynamic_shape=self._dynamic_shape,
        )
        if _timers is not None:
            _timers("send_backward_recv_backward").stop()

        if self._dynamic_shape and need_increase_cnt:
            self._dynamic_cnt += 1

        if overlap_p2p_comm:
            return output_tensor_grad, wait_handles
        return output_tensor_grad

    def __repr__(self):
        debug_str = f"using cache: {self._use_cache} \n"
        debug_str += repr(self._send_recv_meta)
        return debug_str
