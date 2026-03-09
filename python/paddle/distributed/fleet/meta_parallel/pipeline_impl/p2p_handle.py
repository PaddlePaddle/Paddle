#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Callable

import paddle


@dataclass
class P2PAsyncHandle:
    # funcs
    forward_handle_wait_fn: Callable
    forward_async_comm_fn: Callable
    backward_handle_wait_fn: Callable
    backward_async_comm_fn: Callable

    # outputs
    next_forward_virtual_pp_rank = None
    input_tensor = None
    out_fwd_wait_handles = None
    next_backward_virtual_pp_rank = None
    output_tensor_grad = None
    recv_next = None
    out_bwd_wait_handles = None

    def forward_handle_wait(self):
        self.forward_handle_wait_fn()

    def forward_async_comm(self, output_tensor):
        (
            self.next_forward_virtual_pp_rank,
            self.input_tensor,
            self.out_fwd_wait_handles,
        ) = self.forward_async_comm_fn(output_tensor=output_tensor)

    def backward_handle_wait(self):
        self.backward_handle_wait_fn()

    def backward_async_comm(self, input_tensor_grad):
        (
            self.next_backward_virtual_pp_rank,
            self.output_tensor_grad,
            self.recv_next,
            self.out_bwd_wait_handles,
        ) = self.backward_async_comm_fn(input_tensor_grad=input_tensor_grad)


class OffloadQueue(queue.Queue):
    def __init__(self, maxsize=0):
        super().__init__(maxsize=maxsize)

    def put(self, tensor, *args, **kwargs):
        if isinstance(tensor, paddle.Tensor):
            # Only offload to CPU if the tensor is not None
            if tensor is not None and tensor._is_initialized():
                tensor = tensor.cpu()
        super().put(tensor, *args, **kwargs)

    def get(self, *args, **kwargs):
        tensor = super().get(*args, **kwargs)
        if isinstance(tensor, paddle.Tensor):
            # Only move to GPU if the tensor is not None
            if tensor is not None and tensor._is_initialized():
                tensor = tensor.cuda()
        return tensor
