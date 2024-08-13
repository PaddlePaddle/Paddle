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

import os

import numpy as np

import paddle
from paddle import framework

from ...utils import timer_helper as timer
from ...utils.log_util import logger
from .utils import number_2_dtype, paddle_2_number

_hcg = None
_enable_partial_send_recv = True
_timers = None

_xpu_comm_group_started = False

_sync_send = os.environ.get("PADDLE_P2P_SYNC_SEND", "0")
_sync_send = _sync_send.lower() in ['1', 'true']


def _xpu_comm_group_start():
    if not paddle.is_compiled_with_xpu():
        return
    global _xpu_comm_group_started
    assert not _xpu_comm_group_started
    framework.core.ProcessGroupBKCL.group_start()
    _xpu_comm_group_started = True


def _xpu_comm_group_end():
    if not paddle.is_compiled_with_xpu():
        return
    global _xpu_comm_group_started
    if _xpu_comm_group_started:
        framework.core.ProcessGroupBKCL.group_end()
        _xpu_comm_group_started = False


def initialize_p2p_groups(
    hcg, enable_partial_send_recv=True, enable_timer=False
):
    global _hcg, _enable_partial_send_recv, _timers
    _hcg = hcg
    _enable_partial_send_recv = enable_partial_send_recv
    if enable_timer:
        _timers = timer.get_timers()
    (
        send_next_group,
        send_prev_group,
        recv_next_group,
        recv_prev_group,
    ) = _hcg.get_p2p_groups()

    debug_str = (
        f"P2pInfo: send_next_group: {send_next_group!r}, send_prev_group: {send_prev_group!r}, "
        f"recv_next_group: {recv_next_group!r}, recv_prev_group: {recv_prev_group!r}"
    )
    logger.info(debug_str)


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

        if isinstance(tensor, paddle.Tensor):
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
                assert isinstance(d, paddle.Tensor)
                self._send_dims_shape_dtype(d, group=group)

    def set_send_message(self, tensor):
        if isinstance(tensor, paddle.Tensor):
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


def _is_valid_send_recv_partial(tensor, mp_degree):
    if not _enable_partial_send_recv:
        return False
    tensor_numel = np.prod(tensor.shape)
    assert tensor_numel != 0, "can't send/recv zero element"
    return mp_degree > 1 and tensor_numel % mp_degree == 0


def _partial_send_op(
    tensor, group, use_calc_stream, ring_id, dst, nranks, rank_id
):
    dst_rank_in_group = dst if group is None else group.get_group_rank(dst)
    if framework.in_dynamic_mode():
        group = (
            paddle.distributed.collective._get_default_group()
            if group is None
            else group
        )
        comm_op = (
            group.process_group.send_partial_on_calc_stream
            if use_calc_stream
            else group.process_group.send_partial
        )
        return comm_op(tensor, dst_rank_in_group, nranks, rank_id)


def send_partial(
    tensor, dst=0, nranks=1, rank_id=0, group=None, use_calc_stream=True
):
    # dst: local rank in group
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    dst_rank = (
        _hcg._get_p2p_next_rank() if dst == 1 else _hcg._get_p2p_prev_rank()
    )

    if _is_valid_send_recv_partial(tensor, nranks):
        return _partial_send_op(
            tensor, group, use_calc_stream, ring_id, dst_rank, nranks, rank_id
        )
    else:
        send_op = paddle.distributed.isend
        return send_op(tensor.detach(), dst=dst_rank, group=group)


def _partial_recv_op(
    tensor, group, use_calc_stream, ring_id, src, nranks, rank_id
):
    src_rank_in_group = src if group is None else group.get_group_rank(src)
    group = (
        paddle.distributed.collective._get_default_group()
        if group is None
        else group
    )
    comm_op = (
        group.process_group.recv_partial_on_calc_stream
        if use_calc_stream
        else group.process_group.recv_partial
    )
    return comm_op(tensor, src_rank_in_group, nranks, rank_id)


def recv_partial(
    tensor, src=0, nranks=1, rank_id=0, group=None, use_calc_stream=True
):
    # src: local rank in group
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    src_rank = (
        _hcg._get_p2p_prev_rank() if src == 0 else _hcg._get_p2p_next_rank()
    )

    if _is_valid_send_recv_partial(tensor, nranks):
        return _partial_recv_op(
            tensor, group, use_calc_stream, ring_id, src_rank, nranks, rank_id
        )
    else:
        if use_calc_stream:
            recv_op = paddle.distributed.recv
        elif framework.in_dynamic_mode():
            recv_op = paddle.distributed.irecv
        return recv_op(tensor.detach(), src=src_rank, group=group)


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


def _p2p_helper(
    tensor_send_next,
    tensor_send_prev,
    recv_prev,
    recv_next,
    sync_recv=True,
    send_recv_meta=None,
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

    # TODO(Yuang Liu): use batch_isend_irecv replace all these comm ops
    tasks = []
    # start to p2p communicate

    if _sync_send:
        # Some devices(NPU for example) do not support asynchronized send op, So the order is
        # recv_prev -> send_next -> recv_next -> send_prev
        # When using this order, the environment variable
        # 'PADDLE_P2P_SYNC_SEND' should be set True
        if tensor_recv_prev is not None:
            if isinstance(tensor_recv_prev, tuple):
                for d in tensor_recv_prev:
                    task = recv_partial(
                        d,
                        src=0,
                        nranks=mp_degree,
                        rank_id=mp_rank,
                        group=_hcg.recv_prev_group,
                        use_calc_stream=sync_recv,
                    )
                    if sync_recv:
                        allgather_partial(
                            d,
                            nranks=mp_degree,
                            rank_id=mp_rank,
                            group=mp_group,
                            use_calc_stream=True,
                        )
                    else:
                        tasks.append(task)
            else:
                task = recv_partial(
                    tensor_recv_prev,
                    src=0,
                    nranks=mp_degree,
                    rank_id=mp_rank,
                    group=_hcg.recv_prev_group,
                    use_calc_stream=sync_recv,
                )

                if sync_recv:
                    allgather_partial(
                        tensor_recv_prev,
                        nranks=mp_degree,
                        rank_id=mp_rank,
                        group=mp_group,
                        use_calc_stream=True,
                    )
                else:
                    tasks.append(task)

        if tensor_send_next is not None:
            if isinstance(tensor_send_next, tuple):
                for d in tensor_send_next:
                    paddle.distributed.wait(d, use_calc_stream=True)
                    send_partial(
                        d,
                        dst=1,
                        nranks=mp_degree,
                        rank_id=mp_rank,
                        group=_hcg.send_next_group,
                        use_calc_stream=False,
                    )
            else:
                paddle.distributed.wait(tensor_send_next, use_calc_stream=True)
                send_partial(
                    tensor_send_next,
                    dst=1,
                    nranks=mp_degree,
                    rank_id=mp_rank,
                    group=_hcg.send_next_group,
                    use_calc_stream=False,
                )

        if tensor_recv_next is not None:
            if isinstance(tensor_recv_next, tuple):
                for d in tensor_recv_next:
                    task = recv_partial(
                        d,
                        src=1,
                        nranks=mp_degree,
                        rank_id=mp_rank,
                        group=_hcg.recv_next_group,
                        use_calc_stream=sync_recv,
                    )

                    if sync_recv:
                        allgather_partial(
                            d,
                            nranks=mp_degree,
                            rank_id=mp_rank,
                            group=mp_group,
                            use_calc_stream=True,
                        )
                    else:
                        tasks.append(task)

            else:
                task = recv_partial(
                    tensor_recv_next,
                    src=1,
                    nranks=mp_degree,
                    rank_id=mp_rank,
                    group=_hcg.recv_next_group,
                    use_calc_stream=sync_recv,
                )
                if sync_recv:
                    allgather_partial(
                        tensor_recv_next,
                        nranks=mp_degree,
                        rank_id=mp_rank,
                        group=mp_group,
                        use_calc_stream=True,
                    )
                else:
                    tasks.append(task)

        if tensor_send_prev is not None:
            if isinstance(tensor_send_prev, tuple):
                for d in tensor_send_prev:
                    paddle.distributed.wait(d, use_calc_stream=True)
                    send_partial(
                        d,
                        dst=0,
                        nranks=mp_degree,
                        rank_id=mp_rank,
                        group=_hcg.send_prev_group,
                        use_calc_stream=False,
                    )
            else:
                paddle.distributed.wait(tensor_send_prev, use_calc_stream=True)
                send_partial(
                    tensor_send_prev,
                    dst=0,
                    nranks=mp_degree,
                    rank_id=mp_rank,
                    group=_hcg.send_prev_group,
                    use_calc_stream=False,
                )
    else:
        _xpu_comm_group_start()
        if tensor_send_prev is not None:
            if isinstance(tensor_send_prev, tuple):
                for d in tensor_send_prev:
                    paddle.distributed.wait(d, use_calc_stream=True)
                    send_partial(
                        d,
                        dst=0,
                        nranks=mp_degree,
                        rank_id=mp_rank,
                        group=_hcg.send_prev_group,
                        use_calc_stream=False,
                    )
            else:
                paddle.distributed.wait(tensor_send_prev, use_calc_stream=True)
                send_partial(
                    tensor_send_prev,
                    dst=0,
                    nranks=mp_degree,
                    rank_id=mp_rank,
                    group=_hcg.send_prev_group,
                    use_calc_stream=False,
                )

        if tensor_recv_prev is not None:
            if isinstance(tensor_recv_prev, tuple):
                for d in tensor_recv_prev:
                    task = recv_partial(
                        d,
                        src=0,
                        nranks=mp_degree,
                        rank_id=mp_rank,
                        group=_hcg.recv_prev_group,
                        use_calc_stream=sync_recv,
                    )
                    if sync_recv:
                        _xpu_comm_group_end()
                        allgather_partial(
                            d,
                            nranks=mp_degree,
                            rank_id=mp_rank,
                            group=mp_group,
                            use_calc_stream=True,
                        )
                    else:
                        tasks.append(task)
            else:
                task = recv_partial(
                    tensor_recv_prev,
                    src=0,
                    nranks=mp_degree,
                    rank_id=mp_rank,
                    group=_hcg.recv_prev_group,
                    use_calc_stream=sync_recv,
                )

                if sync_recv:
                    _xpu_comm_group_end()
                    allgather_partial(
                        tensor_recv_prev,
                        nranks=mp_degree,
                        rank_id=mp_rank,
                        group=mp_group,
                        use_calc_stream=True,
                    )
                else:
                    tasks.append(task)

        if tensor_send_next is not None:
            if isinstance(tensor_send_next, tuple):
                for d in tensor_send_next:
                    paddle.distributed.wait(d, use_calc_stream=True)
                    send_partial(
                        d,
                        dst=1,
                        nranks=mp_degree,
                        rank_id=mp_rank,
                        group=_hcg.send_next_group,
                        use_calc_stream=False,
                    )
            else:
                paddle.distributed.wait(tensor_send_next, use_calc_stream=True)
                send_partial(
                    tensor_send_next,
                    dst=1,
                    nranks=mp_degree,
                    rank_id=mp_rank,
                    group=_hcg.send_next_group,
                    use_calc_stream=False,
                )

        if tensor_recv_next is not None:
            if isinstance(tensor_recv_next, tuple):
                for d in tensor_recv_next:
                    task = recv_partial(
                        d,
                        src=1,
                        nranks=mp_degree,
                        rank_id=mp_rank,
                        group=_hcg.recv_next_group,
                        use_calc_stream=sync_recv,
                    )

                    if sync_recv:
                        _xpu_comm_group_end()
                        allgather_partial(
                            d,
                            nranks=mp_degree,
                            rank_id=mp_rank,
                            group=mp_group,
                            use_calc_stream=True,
                        )
                    else:
                        tasks.append(task)

            else:
                task = recv_partial(
                    tensor_recv_next,
                    src=1,
                    nranks=mp_degree,
                    rank_id=mp_rank,
                    group=_hcg.recv_next_group,
                    use_calc_stream=sync_recv,
                )
                if sync_recv:
                    _xpu_comm_group_end()
                    allgather_partial(
                        tensor_recv_next,
                        nranks=mp_degree,
                        rank_id=mp_rank,
                        group=mp_group,
                        use_calc_stream=True,
                    )
                else:
                    tasks.append(task)
        _xpu_comm_group_end()
    if not sync_recv:
        if framework.in_dynamic_mode():
            # wait irecv tasks in eager dygraph mode with new comm library
            for task in tasks:
                assert task is not None
                task.wait()

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


class P2pHelper:
    def __init__(self, use_cache=True):
        self._send_recv_meta = SendRecvMeta()
        self._use_cache = use_cache

    def _send_meta(self, output_tensor, skip_check_meta=False):
        if not self._send_recv_meta.has_send_meta:
            self._send_recv_meta.set_send_message(output_tensor)
            self._send_recv_meta.send_meta(
                output_tensor, _hcg.get_pipe_parallel_group()
            )
            self._send_recv_meta.has_send_meta = self._use_cache

    def _recv_meta(self):
        if not self._send_recv_meta.has_recv_meta:
            self._send_recv_meta.recv_meta(_hcg.get_pipe_parallel_group())
            self._send_recv_meta.has_recv_meta = self._use_cache

    def recv_forward(self, pp_first_stage, sync_recv=True):
        global _timers
        if _timers is not None:
            _timers("recv_forward").start()
        if pp_first_stage:
            input_tensor = None
        else:
            self._recv_meta()

            input_tensor, _ = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                sync_recv=sync_recv,
                send_recv_meta=self._send_recv_meta,
            )
        if _timers is not None:
            _timers("recv_forward").stop()
        return input_tensor

    def recv_backward(self, pp_last_stage, sync_recv=True):
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
                send_recv_meta=self._send_recv_meta,
            )
        if _timers is not None:
            _timers("recv_backward").stop()
        return output_tensor_grad

    def send_forward(self, output_tensor, pp_last_stage, skip_check_meta=False):
        global _timers
        if _timers is not None:
            _timers("send_forward").start()
        if not pp_last_stage:
            self._send_meta(output_tensor, skip_check_meta=skip_check_meta)

            _p2p_helper(
                tensor_send_next=output_tensor,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                send_recv_meta=self._send_recv_meta,
            )
        if _timers is not None:
            _timers("send_forward").stop()

    def send_backward(self, input_tensor_grad, pp_first_stage):
        global _timers
        if _timers is not None:
            _timers("send_backward").start()
        if not pp_first_stage:
            _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=input_tensor_grad,
                recv_prev=False,
                recv_next=False,
                send_recv_meta=self._send_recv_meta,
            )
        if _timers is not None:
            _timers("send_backward").stop()

    def send_forward_recv_backward(self, output_tensor, pp_last_stage):
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
                send_recv_meta=self._send_recv_meta,
            )
        if _timers is not None:
            _timers("send_forward_recv_backward").stop()
        return output_tensor_grad

    def send_backward_recv_forward(self, input_tensor_grad, pp_first_stage):
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
                send_recv_meta=self._send_recv_meta,
            )
        if _timers is not None:
            _timers("send_backward_recv_forward").stop()
        return input_tensor

    def send_forward_backward_recv_forward_backward(
        self, output_tensor, input_tensor_grad, recv_prev, recv_next
    ):
        # always have to send dtype info to downstream
        global _timers
        if _timers is not None:
            _timers("send_forward_backward_recv_forward_backward").start()

        self._send_meta(output_tensor)
        if recv_prev:
            self._recv_meta()
        input_tensor, output_tensor_grad = _p2p_helper(
            tensor_send_next=output_tensor,
            tensor_send_prev=input_tensor_grad,
            recv_prev=recv_prev,
            recv_next=recv_next,
            sync_recv=False,
            send_recv_meta=self._send_recv_meta,
        )
        if _timers is not None:
            _timers("send_forward_backward_recv_forward_backward").stop()
        return input_tensor, output_tensor_grad

    def send_forward_recv_forward(self, output_tensor, recv_prev):
        # always have to send dtype info to downstream
        global _timers
        if _timers is not None:
            _timers("send_forward_recv_forward").start()

        if output_tensor is not None:
            self._send_meta(output_tensor)

        if recv_prev:
            self._recv_meta()

        input_tensor, _ = _p2p_helper(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            sync_recv=False,
            send_recv_meta=self._send_recv_meta,
        )
        if _timers is not None:
            _timers("send_forward_recv_forward").stop()
        return input_tensor

    def send_backward_recv_backward(self, input_tensor_grad, recv_next):
        global _timers
        if _timers is not None:
            _timers("send_backward_recv_backward").start()
        _, output_tensor_grad = _p2p_helper(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=recv_next,
            sync_recv=False,
            send_recv_meta=self._send_recv_meta,
        )
        if _timers is not None:
            _timers("send_backward_recv_backward").stop()
        return output_tensor_grad
