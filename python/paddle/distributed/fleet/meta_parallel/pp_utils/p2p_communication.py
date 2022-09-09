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
from .utils import paddle_2_number, number_2_dtype
from ...utils.log_util import logger
import numpy as np
from paddle import _C_ops, _legacy_C_ops
import paddle.fluid.core as core
from paddle.fluid.framework import _in_legacy_dygraph, _non_static_mode, in_dygraph_mode

_hcg = None
_use_cache = False


def initialize_p2p_groups(hcg, use_cache=True):
    global _hcg, _use_cache
    _hcg = hcg
    _use_cache = use_cache
    send_next_group, send_prev_group, recv_next_group, recv_prev_group = _hcg.get_p2p_groups(
    )

    debug_str = "P2pInfo: send_next_group: %s, send_prev_group: %s, " \
                    "recv_next_group: %s, recv_prev_group: %s" % (repr(send_next_group),
                    repr(send_prev_group),repr(recv_next_group), repr(recv_prev_group))
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
        return shape.numpy().tolist(), dtype.item(), stop_grad.item()

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
        dims = paddle.to_tensor(len(tensor.shape))
        dst_rank = _hcg._get_p2p_next_rank()

        paddle.distributed.send(dims, dst=dst_rank, group=group)

        # send shape
        shape = paddle.to_tensor(tensor.shape)
        paddle.distributed.send(shape, dst=dst_rank, group=group)

        # send dtype
        dtype = paddle.to_tensor(paddle_2_number(tensor.dtype))
        paddle.distributed.send(dtype, dst=dst_rank, group=group)

        # send trainable
        stop_grad = paddle.to_tensor(int(tensor.stop_gradient))
        paddle.distributed.send(stop_grad, dst=dst_rank, group=group)

    def send_meta(self, tensor, group):
        dst_rank = _hcg._get_p2p_next_rank()

        if isinstance(tensor, (paddle.Tensor, core.eager.Tensor)):
            tensor_type = paddle.to_tensor([0])
            # send tensor type
            paddle.distributed.send(tensor_type, dst=dst_rank, group=group)

            self._send_dims_shape_dtype(tensor, group)
        elif isinstance(tensor, tuple):
            tensor_type = paddle.to_tensor([1])
            # send tensor type
            paddle.distributed.send(tensor_type, dst=dst_rank, group=group)

            nums = paddle.to_tensor(len(tensor))
            paddle.distributed.send(nums, dst=dst_rank, group=group)

            for d in tensor:
                assert isinstance(d, (paddle.Tensor, core.eager.Tensor))
                self._send_dims_shape_dtype(d, group=group)

    def set_send_message(self, tensor):
        if isinstance(tensor, (paddle.Tensor, core.eager.Tensor)):
            self.send_shape_message = tensor.shape
            self.send_dtype_message = paddle_2_number(tensor.dtype)
        elif isinstance(tensor, tuple):
            self.send_shape_message = tuple(
                [d.shape for d in tensor if not d.stop_gradient])
            self.send_dtype_message = tuple(
                [paddle_2_number(d.dtype) for d in tensor])


_send_recv_meta = SendRecvMeta()


def _is_valid_send_recv_partial(tensor, mp_degree):

    tensor_numel = np.prod(tensor.shape)
    assert tensor_numel != 0, "can't send/recv zero element"
    return mp_degree > 1 and tensor_numel % mp_degree == 0


def _partial_send_op(tensor, group, use_calc_stream, ring_id, dst, nranks,
                     rank_id):
    dst_rank_in_group = dst if group is None else group.get_group_rank(dst)
    if _in_legacy_dygraph():
        return _legacy_C_ops.partial_send(tensor.detach(), 'use_calc_stream',
                                          use_calc_stream, 'ring_id', ring_id,
                                          'peer', dst_rank_in_group, 'num',
                                          nranks, 'id', rank_id)
    elif in_dygraph_mode():
        group = paddle.distributed.collective._get_default_group(
        ) if group is None else group
        return group.process_group.send_partial(tensor, dst_rank_in_group,
                                                nranks, rank_id)


def send_partial(tensor,
                 dst=0,
                 nranks=1,
                 rank_id=0,
                 group=None,
                 use_calc_stream=True):
    # dst: local rank in group
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    dst_rank = _hcg._get_p2p_next_rank(
    ) if dst == 1 else _hcg._get_p2p_prev_rank()

    if _is_valid_send_recv_partial(tensor, nranks):
        return _partial_send_op(tensor, group, use_calc_stream, ring_id,
                                dst_rank, nranks, rank_id)
    else:
        if _in_legacy_dygraph():
            send_op = paddle.distributed.send
        elif in_dygraph_mode():
            send_op = paddle.distributed.isend
        return send_op(tensor.detach(), dst=dst_rank, group=group)


def _partial_recv_op(tensor, group, use_calc_stream, ring_id, src, nranks,
                     rank_id):
    src_rank_in_group = src if group is None else group.get_group_rank(src)
    if _in_legacy_dygraph():
        return _legacy_C_ops.partial_recv(tensor.detach(), 'use_calc_stream',
                                          use_calc_stream, 'ring_id', ring_id,
                                          'peer', src_rank_in_group, 'num',
                                          nranks, 'id', rank_id, 'dtype',
                                          tensor.dtype, 'out_shape',
                                          tensor.shape)
    elif in_dygraph_mode():
        group = paddle.distributed.collective._get_default_group(
        ) if group is None else group
        return group.process_group.recv_partial(tensor, src_rank_in_group,
                                                nranks, rank_id)


def recv_partial(tensor,
                 src=0,
                 nranks=1,
                 rank_id=0,
                 group=None,
                 use_calc_stream=True):
    # src: local rank in group
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    src_rank = _hcg._get_p2p_prev_rank(
    ) if src == 0 else _hcg._get_p2p_next_rank()

    if _is_valid_send_recv_partial(tensor, nranks):
        return _partial_recv_op(tensor, group, use_calc_stream, ring_id,
                                src_rank, nranks, rank_id)
    else:
        if _in_legacy_dygraph():
            recv_op = paddle.distributed.recv
        elif in_dygraph_mode():
            recv_op = paddle.distributed.irecv
        return recv_op(tensor.detach(), src=src_rank, group=group)


def _partial_allgather_op(tensor, group, use_calc_stream, ring_id, nranks,
                          rank_id):
    if _in_legacy_dygraph():
        return _legacy_C_ops.partial_allgather_(tensor.detach(),
                                                'use_calc_stream',
                                                use_calc_stream, 'ring_id',
                                                ring_id, 'nranks', nranks,
                                                'rank', rank_id)
    elif in_dygraph_mode():
        group = paddle.distributed.collective._get_default_group(
        ) if group is None else group
        return group.process_group.all_gather_partial(tensor, tensor, nranks,
                                                      rank_id)


def allgather_partial(tensor,
                      nranks=1,
                      rank_id=0,
                      group=None,
                      use_calc_stream=True):
    if not _is_valid_send_recv_partial(tensor, nranks):
        return None
    if group is not None and not group.is_member():
        return None
    ring_id = 0 if group is None else group.id

    return _partial_allgather_op(tensor, group, use_calc_stream, ring_id,
                                 nranks, rank_id)


def _p2p_helper(tensor_send_next, tensor_send_prev, recv_prev, recv_next):
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
                tmp = paddle.empty(shape=shape,
                                   dtype=number_2_dtype(recv_dtype_msg[idx]))
                tmp.stop_gradient = recv_stop_gradient[idx]
                tensor_recv_prev.append(tmp)
            tensor_recv_prev = tuple(tensor_recv_prev)
        else:

            tensor_recv_prev = paddle.empty(
                shape=recv_shape_msg, dtype=number_2_dtype(recv_dtype_msg))
            tensor_recv_prev.stop_gradient = recv_stop_gradient

    if recv_next:
        if isinstance(send_shape_msg, tuple):
            tensor_recv_next = []
            for idx, shape in enumerate(send_shape_msg):
                tensor_recv_next.append(
                    paddle.empty(shape=shape,
                                 dtype=number_2_dtype(send_dtype_msg[idx])))
            tensor_recv_next = tuple(tensor_recv_next)
        else:
            tensor_recv_next = paddle.empty(
                shape=send_shape_msg, dtype=number_2_dtype(send_dtype_msg))

    # TODO(Yuang Liu): use batch_isend_irecv replace all these comm ops
    tasks = []
    # start to p2p communicate
    if tensor_send_prev is not None:
        if isinstance(tensor_send_prev, tuple):
            for d in tensor_send_prev:
                if _in_legacy_dygraph():
                    paddle.distributed.wait(d, use_calc_stream=True)
                tasks.append(
                    send_partial(d,
                                 dst=0,
                                 nranks=mp_degree,
                                 rank_id=mp_rank,
                                 group=_hcg.send_prev_group,
                                 use_calc_stream=False))
        else:
            if _in_legacy_dygraph():
                paddle.distributed.wait(tensor_send_prev, use_calc_stream=True)
            tasks.append(
                send_partial(tensor_send_prev,
                             dst=0,
                             nranks=mp_degree,
                             rank_id=mp_rank,
                             group=_hcg.send_prev_group,
                             use_calc_stream=False))

    if tensor_recv_prev is not None:
        if isinstance(tensor_recv_prev, tuple):
            for d in tensor_recv_prev:
                tasks.append(
                    recv_partial(d,
                                 src=0,
                                 nranks=mp_degree,
                                 rank_id=mp_rank,
                                 group=_hcg.recv_prev_group,
                                 use_calc_stream=True))
        else:
            tasks.append(
                recv_partial(tensor_recv_prev,
                             src=0,
                             nranks=mp_degree,
                             rank_id=mp_rank,
                             group=_hcg.recv_prev_group,
                             use_calc_stream=True))

    if tensor_send_next is not None:
        if isinstance(tensor_send_next, tuple):
            for d in tensor_send_next:
                if _in_legacy_dygraph():
                    paddle.distributed.wait(d, use_calc_stream=True)
                tasks.append(
                    send_partial(d,
                                 dst=1,
                                 nranks=mp_degree,
                                 rank_id=mp_rank,
                                 group=_hcg.send_next_group,
                                 use_calc_stream=False))
        else:
            if _in_legacy_dygraph():
                paddle.distributed.wait(tensor_send_next, use_calc_stream=True)
            tasks.append(
                send_partial(tensor_send_next,
                             dst=1,
                             nranks=mp_degree,
                             rank_id=mp_rank,
                             group=_hcg.send_next_group,
                             use_calc_stream=False))

    if tensor_recv_next is not None:
        if isinstance(tensor_recv_next, tuple):
            for d in tensor_recv_next:
                tasks.append(
                    recv_partial(d,
                                 src=1,
                                 nranks=mp_degree,
                                 rank_id=mp_rank,
                                 group=_hcg.recv_next_group,
                                 use_calc_stream=True))

        else:
            tasks.append(
                recv_partial(tensor_recv_next,
                             src=1,
                             nranks=mp_degree,
                             rank_id=mp_rank,
                             group=_hcg.recv_next_group,
                             use_calc_stream=True))

    if in_dygraph_mode():
        # wait isend/irecv tasks in eager dygraph mode with new comm library
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

    tasks = []
    for tensor in tensors_for_all_gather:
        tasks.append(
            allgather_partial(tensor,
                              nranks=mp_degree,
                              rank_id=mp_rank,
                              group=mp_group,
                              use_calc_stream=True))

    for task in tasks:
        # wait partial all gather tasks
        if task is not None:
            task.wait()

    return tensor_recv_prev, tensor_recv_next


def recv_forward(pp_first_stage):
    if pp_first_stage:
        input_tensor = None
    else:
        if not _send_recv_meta.has_recv_meta:
            _send_recv_meta.recv_meta(_hcg.recv_prev_group)
            _send_recv_meta.has_recv_meta = _use_cache

        input_tensor, _ = _p2p_helper(tensor_send_next=None,
                                      tensor_send_prev=None,
                                      recv_prev=True,
                                      recv_next=False)
    return input_tensor


def recv_backward(pp_last_stage):
    if pp_last_stage:
        output_tensor_grad = None
    else:
        _, output_tensor_grad = _p2p_helper(tensor_send_next=None,
                                            tensor_send_prev=None,
                                            recv_prev=False,
                                            recv_next=True)
    return output_tensor_grad


def send_forward(output_tensor, pp_last_stage):
    if not pp_last_stage:
        if not _send_recv_meta.has_send_meta:
            _send_recv_meta.set_send_message(output_tensor)
            _send_recv_meta.send_meta(output_tensor, _hcg.send_next_group)
            _send_recv_meta.has_send_meta = _use_cache

        _p2p_helper(tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=False)


def send_backward(input_tensor_grad, pp_first_stage):
    if not pp_first_stage:
        _p2p_helper(tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad,
                    recv_prev=False,
                    recv_next=False)


def send_forward_recv_backward(output_tensor, pp_last_stage):
    if pp_last_stage:
        output_tensor_grad = None
    else:
        _, output_tensor_grad = _p2p_helper(tensor_send_next=output_tensor,
                                            tensor_send_prev=None,
                                            recv_prev=False,
                                            recv_next=True)
    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad, pp_first_stage):
    if pp_first_stage:
        input_tensor = None
    else:
        input_tensor, _ = _p2p_helper(tensor_send_next=None,
                                      tensor_send_prev=input_tensor_grad,
                                      recv_prev=True,
                                      recv_next=False)
    return input_tensor


def send_forward_backward_recv_forward_backward(output_tensor,
                                                input_tensor_grad, recv_prev,
                                                recv_next):
    # always have to send dytpe info to downstream
    if not _send_recv_meta.has_send_meta:
        _send_recv_meta.set_send_message(output_tensor)
        _send_recv_meta.send_meta(output_tensor, _hcg.send_next_group)
        _send_recv_meta.has_send_meta = _use_cache
    if recv_prev and not _send_recv_meta.has_recv_meta:
        _send_recv_meta.recv_meta(_hcg.recv_prev_group)
        _send_recv_meta.has_recv_meta = _use_cache
    input_tensor, output_tensor_grad = _p2p_helper(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next)
    return input_tensor, output_tensor_grad


def send_forward_recv_forward(output_tensor, recv_prev):
    # always have to send dytpe info to downstream
    if not _send_recv_meta.has_send_meta:
        _send_recv_meta.set_send_message(output_tensor)
        _send_recv_meta.send_meta(output_tensor, _hcg.send_next_group)
        _send_recv_meta.has_send_meta = _use_cache
    if recv_prev and not _send_recv_meta.has_recv_meta:
        _send_recv_meta.recv_meta(_hcg.recv_prev_group)
        _send_recv_meta.has_recv_meta = _use_cache

    input_tensor, _ = _p2p_helper(tensor_send_next=output_tensor,
                                  tensor_send_prev=None,
                                  recv_prev=recv_prev,
                                  recv_next=False)

    return input_tensor


def send_backward_recv_backward(input_tensor_grad, recv_next):
    _, output_tensor_grad = _p2p_helper(tensor_send_next=None,
                                        tensor_send_prev=input_tensor_grad,
                                        recv_prev=False,
                                        recv_next=recv_next)
    return output_tensor_grad
