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
# limitations under the License.

from __future__ import annotations

import paddle
from paddle.distributed.communication.batch_isend_irecv import (
    P2POp,
)

from .p2p_communication import SendRecvMeta
from .utils import number_2_dtype


class BatchCommHelper:
    # NOTE(zhangyuqin1998): Tensors to be sent or received must have a
    # consistent shape and data type throughout the entire pipeline.
    def __init__(self, use_cache=True):
        self._send_recv_meta = SendRecvMeta()
        self._use_cache = use_cache

    def clear_meta_cache(self):
        self._send_recv_meta.init_or_erase_meta()

    def _send_meta(self, tensors, group, broadcast=False):
        self._send_recv_meta.set_send_message(tensors)
        self._send_recv_meta.send_meta(tensors, group, broadcast=broadcast)
        self._send_recv_meta.recv_shape_message = (
            self._send_recv_meta.send_shape_message
        )
        self._send_recv_meta.recv_dtype_message = (
            self._send_recv_meta.send_dtype_message
        )

    def _recv_meta(self, group, broadcast=False):
        self._send_recv_meta.recv_meta(group, broadcast=broadcast)

    def _build_from_meta(self):
        shape_message = self._send_recv_meta.recv_shape_message
        dtype_message = self._send_recv_meta.recv_dtype_message
        stop_gradient = self._send_recv_meta.recv_stop_gradient
        assert (shape_message is not None) and (
            dtype_message is not None
        ), "Failed to build from meta."

        res = []
        if isinstance(shape_message, tuple):
            for idx, shape in enumerate(shape_message):
                tmp = paddle.empty(
                    shape=shape, dtype=number_2_dtype(dtype_message[idx])
                )
                tmp.stop_gradient = (
                    stop_gradient[idx] if stop_gradient is not None else False
                )
                res.append(tmp)
        else:
            tmp = paddle.empty(
                shape=shape_message, dtype=number_2_dtype(dtype_message)
            )
            tmp.stop_gradient = stop_gradient
            res.append(tmp)
        return res

    def recv_meta_from_head(self, group, need_recv_meta):
        if not need_recv_meta:
            return
        self._recv_meta(group, broadcast=True)

    def append_irecv(self, ops, src, group):
        tensors = self._build_from_meta()
        for tensor in tensors:
            if tensor is not None:
                ops.append(P2POp(paddle.distributed.irecv, tensor, src, group))
        return tensors

    def append_isend(self, ops, tensors, dst, group, need_broadcast_meta=False):
        if need_broadcast_meta:
            self._send_meta(tensors, group, broadcast=True)
        for tensor in tensors:
            if tensor is not None:
                ops.append(P2POp(paddle.distributed.isend, tensor, dst, group))
