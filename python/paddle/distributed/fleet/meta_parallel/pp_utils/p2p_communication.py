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

import collections
import distutils.util
import os
from enum import IntEnum

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


class SendRecvPackType(IntEnum):
    TENSOR = 0
    TENSOR_LIST_OR_TENSOR_TUPLE = 1
    DICT_WITH_STR_TENSOR_PAIR = 2


class SendRecvMeta:
    """Mainly used to help p2p communication context information"""

    def __init__(self):
        self.init_or_erase_meta()

    def init_or_erase_meta(self):
        self.send_pack_type = None
        self.send_shape_message = None
        self.send_dtype_message = None
        self.send_keys_names = None  # valid in transmission of dict. Only contains keys of the Tensor with `stop_gradient == False`
        self.send_all_keys_names = None  # valid in transmission of dict. Contains the keys for all Tensors, regardless of whether the Tensor's `stop_gradient` property is True or False.

        self.recv_pack_type = None
        self.recv_shape_message = None
        self.recv_dtype_message = None
        self.recv_stop_gradient = None
        self.recv_keys_names = None  # valid in transmission of dict. Contains keys of all received Tensors

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

    def _recv_key(self, group):
        src_rank = _hcg._get_p2p_prev_rank()

        # recv shape
        buf_shape = paddle.to_tensor([0])
        paddle.distributed.recv(buf_shape, src=src_rank, group=group)

        # recv buf
        recv_buf = paddle.empty(shape=buf_shape, dtype=paddle.uint8)
        paddle.distributed.recv(recv_buf, src=src_rank, group=group)

        key = self._deserilize_to_string(recv_buf)
        return key

    def recv_meta(self, group):
        tensor_type = paddle.to_tensor([0])
        src_rank = _hcg._get_p2p_prev_rank()

        paddle.distributed.recv(tensor_type, src=src_rank, group=group)
        tensor_type = tensor_type.item()

        if tensor_type == SendRecvPackType.TENSOR:
            shape, dtype, stop_grad = self._recv_shape_dtype(group)
            self.recv_shape_message = shape
            self.recv_dtype_message = dtype
            self.recv_stop_gradient = bool(stop_grad)
            self.recv_pack_type = SendRecvPackType.TENSOR

        elif tensor_type == SendRecvPackType.TENSOR_LIST_OR_TENSOR_TUPLE:
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
            self.recv_pack_type = SendRecvPackType.TENSOR_LIST_OR_TENSOR_TUPLE

        elif tensor_type == SendRecvPackType.DICT_WITH_STR_TENSOR_PAIR:
            num_kv = paddle.to_tensor([0])
            paddle.distributed.recv(num_kv, src=src_rank, group=group)
            num_kv = num_kv.item()

            key_names = []
            shapes = []
            dtypes = []
            stop_grads = []
            for i in range(num_kv):
                key = self._recv_key(group)
                shape, dtype, stop_grad = self._recv_shape_dtype(group)
                key_names.append(key)
                shapes.append(shape)
                dtypes.append(dtype)
                stop_grads.append(bool(stop_grad))

            self.recv_keys_names = tuple(key_names)
            self.recv_shape_message = tuple(shapes)
            self.recv_dtype_message = tuple(dtypes)
            self.recv_stop_gradient = tuple(stop_grads)
            self.recv_pack_type = SendRecvPackType.DICT_WITH_STR_TENSOR_PAIR

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

    # NOTE: Only support the key is the `str` type
    def _send_key(self, key, group):
        # encode string and serialize it into buffer
        buf = self._serialize_from_string(key)

        dst_rank = _hcg._get_p2p_next_rank()
        # send shape
        assert len(buf.shape) == 1
        buf_shape = paddle.to_tensor(buf.shape)
        paddle.distributed.send(buf_shape, dst=dst_rank, group=group)

        # send buf
        paddle.distributed.send(buf, dst=dst_rank, group=group)

    def send_meta(self, tensor, group):
        dst_rank = _hcg._get_p2p_next_rank()

        if isinstance(tensor, (paddle.Tensor, framework.core.eager.Tensor)):
            tensor_type = paddle.to_tensor([int(SendRecvPackType.TENSOR)])
            # send tensor type
            paddle.distributed.send(tensor_type, dst=dst_rank, group=group)

            self._send_dims_shape_dtype(tensor, group)
        elif isinstance(tensor, tuple):
            tensor_type = paddle.to_tensor(
                [int(SendRecvPackType.TENSOR_LIST_OR_TENSOR_TUPLE)]
            )
            # send tensor type
            paddle.distributed.send(tensor_type, dst=dst_rank, group=group)

            nums = paddle.to_tensor([len(tensor)])
            paddle.distributed.send(nums, dst=dst_rank, group=group)

            for d in tensor:
                assert isinstance(
                    d, (paddle.Tensor, framework.core.eager.Tensor)
                )
                self._send_dims_shape_dtype(d, group=group)
        elif isinstance(tensor, (dict, collections.OrderedDict)):
            tensor_type = paddle.to_tensor(
                [int(SendRecvPackType.DICT_WITH_STR_TENSOR_PAIR)]
            )
            # send tensor type
            paddle.distributed.send(tensor_type, dst=dst_rank, group=group)

            # send numbers of key-value pair
            keys_nums = paddle.to_tensor([len(self.send_all_keys_names)])
            assert set(self.send_all_keys_names) == set(tensor.keys())
            paddle.distributed.send(keys_nums, dst=dst_rank, group=group)

            for k in self.send_all_keys_names:
                assert isinstance(k, str)
                assert isinstance(
                    tensor[k], (paddle.Tensor, framework.core.eager.Tensor)
                )

                # send key
                self._send_key(k, group=group)

                # send value's dim_shape and dtype
                self._send_dims_shape_dtype(tensor[k], group=group)

    def _obtain_send_message(self, tensor):
        if isinstance(tensor, (paddle.Tensor, framework.core.eager.Tensor)):
            return (
                SendRecvPackType.TENSOR,
                tensor.shape,
                paddle_2_number(tensor.dtype),
                None,
                None,
            )
        elif isinstance(tensor, (dict, collections.OrderedDict)):
            shapes = []
            dtypes = []
            keys_names = []
            keys_all_names = []
            for k, v in tensor.items():
                assert isinstance(
                    v, (paddle.Tensor, framework.core.eager.Tensor)
                )
                if v.stop_gradient:
                    keys_all_names.append(k)
                    continue
                _, shape, dtype, _, _ = self._obtain_send_message(v)
                shapes.append(shape)
                dtypes.append(dtype)
                keys_names.append(k)
                keys_all_names.append(k)
            return (
                SendRecvPackType.DICT_WITH_STR_TENSOR_PAIR,
                shapes,
                dtypes,
                tuple(keys_names),
                tuple(keys_all_names),
            )
        else:
            shapes = []
            dtypes = []
            for d in tensor:
                assert isinstance(
                    d, (paddle.Tensor, framework.core.eager.Tensor)
                )
                if d.stop_gradient:
                    continue
                _, shape, dtype, _, _ = self._obtain_send_message(d)
                shapes.append(shape)
                dtypes.append(dtype)
            return (
                SendRecvPackType.TENSOR_LIST_OR_TENSOR_TUPLE,
                tuple(shapes),
                tuple(dtypes),
                None,
                None,
            )

    def set_send_message(self, tensor):
        (
            self.send_pack_type,
            self.send_shape_message,
            self.send_dtype_message,
            self.send_keys_names,
            self.send_all_keys_names,
        ) = self._obtain_send_message(tensor)

    def check_send_message(self, tensor):
        if self.send_shape_message is None or self.send_dtype_message is None:
            return
        (
            actual_send_pack_type,
            actual_shape,
            actual_dtype,
            actual_send_keys_names,
            actual_send_all_keys_names,
        ) = self._obtain_send_message(tensor)

        assert (
            self.send_pack_type == actual_send_pack_type
        ), f"send_pack_type: {self.send_pack_type}, actual_send_pack_type: {actual_send_pack_type}"
        assert (
            self.send_shape_message == actual_shape
        ), f"send_shape_message: {self.send_shape_message}, actual_shape: {actual_shape}"
        assert (
            self.send_dtype_message == actual_dtype
        ), f"send_dtype_message: {self.send_dtype_message}, actual_dtype: {actual_dtype}"
        assert (
            self.send_keys_names == actual_send_keys_names
        ), f"send_keys_names: {self.send_keys_names}, actual_send_keys_names: {actual_send_keys_names}"
        assert (
            self.send_all_keys_names == actual_send_all_keys_names
        ), f"send_all_keys_names: {self.send_all_keys_names}, actual_send_all_keys_names: {actual_send_all_keys_names}"

    def _serialize_from_string(
        self, string, encoding='utf-32', errors='strict'
    ):
        string_encoded = string.encode(encoding=encoding, errors=errors)
        buf = np.frombuffer(string_encoded, dtype=np.uint8)
        buf = paddle.to_tensor(buf, dtype=paddle.uint8)
        return buf

    def _deserilize_to_string(self, buf, encoding='utf-32', errors='strict'):
        buf = buf.numpy()
        string = buf.tobytes("C").decode(encoding=encoding, errors=errors)
        return string

    def __repr__(self):
        return f"send_pack_type: {self.send_pack_type}, send_shape_message: {self.send_shape_message}, send_dtype_message: {self.send_dtype_message}, send_keys_names: {self.send_keys_names}, send_all_keys_names: {self.send_all_keys_names}, recv_pack_type: {self.recv_pack_type}, recv_shape_message: {self.recv_shape_message}, recv_dtype_message: {self.recv_dtype_message}, recv_stop_gradient: {self.recv_stop_gradient}, recv_keys_names: {self.recv_keys_names}"


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
        if distutils.util.strtobool(
            os.getenv('FLAGS_p2p_device_synchronize', '0')
        ):
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


'''
NOTE(MarioLulab): This wrapper is used to convert Dict[str, Tensor] into tuple[Tensor] before calling `Send` and `Recv` ops
'''


def _p2p_ops_wrapper(_p2p_ops):
    def wrapper(
        send_recv_meta,
        tensor_send_prev,
        tensor_recv_prev,
        tensor_send_next,
        tensor_recv_next,
        _hcg,
    ):
        if (
            send_recv_meta.recv_pack_type
            == SendRecvPackType.DICT_WITH_STR_TENSOR_PAIR
        ):
            if tensor_send_prev is not None:
                assert isinstance(
                    tensor_send_prev, (dict, collections.OrderedDict)
                )
                tensor_send_prev_keys = set(tensor_send_prev.keys())
                # NOTE: tensor_send_prev.keys() will not equal to send_recv_meta.recv_keys_names only if tensor with `stop_gradient == True` exists.
                assert tensor_send_prev_keys.issubset(
                    set(send_recv_meta.recv_keys_names)
                ), "`tensor_send_prev.keys()` should be a subset of `send_recv_meta.recv_keys_names`"
                tensor_send_prev = tuple(
                    [
                        tensor_send_prev[key]
                        for key in send_recv_meta.recv_keys_names
                        if key in tensor_send_prev_keys
                    ]
                )

            if tensor_recv_prev is not None:
                assert isinstance(
                    tensor_recv_prev, (dict, collections.OrderedDict)
                )
                assert set(send_recv_meta.recv_keys_names) == set(
                    tensor_recv_prev.keys()
                ), "`tensor_recv_prev.keys()` should be equal to `send_recv_meta.recv_keys_names`"
                # NOTE: `recv` operation is an in-place operation, so the received data will be written into original `tensor_recv_prev`
                tensor_recv_prev = tuple(
                    [
                        tensor_recv_prev[key]
                        for key in send_recv_meta.recv_keys_names
                    ]
                )

        if (
            send_recv_meta.send_pack_type
            == SendRecvPackType.DICT_WITH_STR_TENSOR_PAIR
        ):
            if tensor_send_next is not None:
                assert isinstance(
                    tensor_send_next, (dict, collections.OrderedDict)
                )

                send_meta_all_keys = set(send_recv_meta.send_all_keys_names)
                tensor_send_next_keys = set(tensor_send_next.keys())
                assert (
                    send_meta_all_keys == tensor_send_next_keys
                ), "`tensor_send_next.keys()` should be equal to `send_recv_meta.send_all_keys_names`"

                tensor_send_next = tuple(
                    [
                        tensor_send_next[key]
                        for key in send_recv_meta.send_all_keys_names
                    ]
                )

            if tensor_recv_next is not None:
                assert isinstance(
                    tensor_recv_next, (dict, collections.OrderedDict)
                )
                assert set(send_recv_meta.send_keys_names) == set(
                    tensor_recv_next.keys()
                ), "`tensor_recv_next.keys()` should be equal to `send_recv_meta.send_keys_names`"
                # NOTE: `recv` operation is an in-place operation, so the received data will be written into original `tensor_recv_next`
                tensor_recv_next = tuple(
                    [
                        tensor_recv_next[key]
                        for key in send_recv_meta.send_keys_names
                    ]
                )

        # call actual p2p ops
        return _p2p_ops(
            tensor_send_prev,
            tensor_recv_prev,
            tensor_send_next,
            tensor_recv_next,
            _hcg,
        )

    return wrapper


def _p2p_helper(
    tensor_send_next,
    tensor_send_prev,
    recv_prev,
    recv_next,
    sync_recv=True,
    send_recv_meta=None,
    batch_p2p_comm=True,
    wait_on_reqs=True,
):
    global _hcg

    tensor_recv_prev = None
    tensor_recv_next = None

    # send / recv message
    assert send_recv_meta is not None, "send_recv_meta should not be None"
    recv_pack_type = send_recv_meta.recv_pack_type
    recv_shape_msg = send_recv_meta.recv_shape_message
    recv_dtype_msg = send_recv_meta.recv_dtype_message
    recv_stop_gradient = send_recv_meta.recv_stop_gradient
    recv_keys_names = send_recv_meta.recv_keys_names

    send_pack_type = send_recv_meta.send_pack_type
    send_shape_msg = send_recv_meta.send_shape_message
    send_dtype_msg = send_recv_meta.send_dtype_message
    send_keys_names = send_recv_meta.send_keys_names
    send_all_keys_names = send_recv_meta.send_all_keys_names

    # model parallel message
    mp_group = _hcg.get_model_parallel_group()
    mp_degree = _hcg.get_model_parallel_world_size()
    mp_rank = _hcg.get_model_parallel_rank()

    if recv_prev:
        if recv_pack_type == SendRecvPackType.TENSOR_LIST_OR_TENSOR_TUPLE:
            tensor_recv_prev = []
            for idx, shape in enumerate(recv_shape_msg):
                tmp = paddle.empty(
                    shape=shape, dtype=number_2_dtype(recv_dtype_msg[idx])
                )
                tmp.stop_gradient = recv_stop_gradient[idx]
                tensor_recv_prev.append(tmp)
            tensor_recv_prev = tuple(tensor_recv_prev)
        elif recv_pack_type == SendRecvPackType.DICT_WITH_STR_TENSOR_PAIR:
            tensor_recv_prev = {}
            for idx, key in enumerate(recv_keys_names):
                tmp = paddle.empty(
                    shape=recv_shape_msg[idx],
                    dtype=number_2_dtype(recv_dtype_msg[idx]),
                )
                tmp.stop_gradient = recv_stop_gradient[idx]
                tensor_recv_prev[key] = tmp
        else:
            tensor_recv_prev = paddle.empty(
                shape=recv_shape_msg, dtype=number_2_dtype(recv_dtype_msg)
            )
            tensor_recv_prev.stop_gradient = recv_stop_gradient

    if recv_next:
        if send_pack_type == SendRecvPackType.TENSOR_LIST_OR_TENSOR_TUPLE:
            tensor_recv_next = []
            for idx, shape in enumerate(send_shape_msg):
                tensor_recv_next.append(
                    paddle.empty(
                        shape=shape, dtype=number_2_dtype(send_dtype_msg[idx])
                    )
                )
            tensor_recv_next = tuple(tensor_recv_next)
        elif send_pack_type == SendRecvPackType.DICT_WITH_STR_TENSOR_PAIR:
            tensor_recv_next = {}
            for idx, key in enumerate(send_keys_names):
                tensor_recv_next[key] = paddle.empty(
                    shape=send_shape_msg[idx],
                    dtype=number_2_dtype(send_dtype_msg[idx]),
                )
        else:
            tensor_recv_next = paddle.empty(
                shape=send_shape_msg, dtype=number_2_dtype(send_dtype_msg)
            )

    p2p_func = (
        _p2p_ops_wrapper(_batched_p2p_ops)
        if batch_p2p_comm
        else _p2p_ops_wrapper(_p2p_ops)
    )
    reqs = p2p_func(
        send_recv_meta,
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
        elif not skip_check_meta:
            self._send_recv_meta.check_send_message(output_tensor)

    def _recv_meta(self):
        if not self._send_recv_meta.has_recv_meta:
            self._send_recv_meta.recv_meta(_hcg.get_pipe_parallel_group())
            self._send_recv_meta.has_recv_meta = self._use_cache

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
        if _timers is not None:
            _timers("recv_forward").stop()
        return input_tensor

    def recv_backward(self, pp_last_stage, sync_recv=True, batch_p2p_comm=True):
        global _timers
        if _timers is not None:
            _timers("recv_backward").start()
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
    ):
        # always have to send dtype info to downstream
        global _timers
        if _timers is not None:
            _timers("send_forward_backward_recv_forward_backward").start()

        if output_tensor is not None:
            self._send_meta(output_tensor)
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
    ):
        # always have to send dtype info to downstream
        global _timers
        if _timers is not None:
            _timers("send_forward_recv_forward").start()

        if output_tensor is not None:
            self._send_meta(output_tensor)

        if recv_prev:
            self._recv_meta()

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
        _, output_tensor_grad, wait_handles = _p2p_helper(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=recv_next,
            sync_recv=False,
            send_recv_meta=self._send_recv_meta,
            batch_p2p_comm=batch_p2p_comm,
            wait_on_reqs=(not overlap_p2p_comm),
        )
        if _timers is not None:
            _timers("send_backward_recv_backward").stop()

        if overlap_p2p_comm:
            return output_tensor_grad, wait_handles
        return output_tensor_grad

    def __repr__(self):
        debug_str = f"using cache: {self._use_cache} \n"
        debug_str += repr(self._send_recv_meta)
        return debug_str
