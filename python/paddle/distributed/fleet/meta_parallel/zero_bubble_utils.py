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

# The file has been adapted from DeepSeek DualPipe project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DualPipe/blob/main/LICENSE


import queue
from functools import partial

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.autograd import PyLayer


class WeightGradStore:
    enabled = False
    cache = []
    funcs_queue = queue.Queue()

    @classmethod
    def put(cls, func) -> None:
        cls.cache.append(func)

    @classmethod
    def flush(cls) -> None:
        cls.funcs_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls) -> None:
        assert not cls.funcs_queue.empty(), "Pop empty queue."
        funcs = cls.funcs_queue.get()
        for func in funcs:
            func()

    @classmethod
    def clear(cls) -> None:
        cls.cache = []
        cls.funcs_queue = queue.Queue()


class RecomputeStore:
    """Pending selective-recompute forwards, runnable inside a p2p window.

    Companion to WeightGradStore, for when the deferred dW is not enough to fill
    an exposed p2p window. A selective-recompute span replays its forward from
    inputs saved during the original forward and never reads the incoming
    activation gradient, so it can run any time before its backward reaches it.

    Scope is deliberately narrow: the scheduler only ever runs early the
    recompute of **the chunk whose backward comes next**. That keeps it simple
    and makes the memory cost structural rather than tunable -- at most one
    chunk's discarded activations are resident early, and they are consumed by
    the very next backward. No looking further ahead: if the next backward has no
    spans (an EmptyLayer chunk), nothing runs and the window stays as it was.

    Spans are grouped by ``(virtual_pp_rank, micro_id)``, which is the only key
    that lets the scheduler name a chunk: forward order and backward order differ
    under interleaving, so "the most recent forward" is not "the next backward".
    """

    enabled = False
    # (virtual_pp_rank, micro_id) -> {id(span): span}
    groups = {}
    _open_key = None

    @classmethod
    def begin_chunk(cls, key) -> None:
        """Start recording the forward chunk identified by `key`."""
        cls._open_key = key

    @classmethod
    def end_chunk(cls) -> None:
        cls._open_key = None

    @classmethod
    def put(cls, span) -> None:
        if cls._open_key is None:
            # Registered outside a scheduler-tracked forward, e.g. a plain
            # single-card run. Nothing can name it, so leave it to its own hook.
            return
        cls.groups.setdefault(cls._open_key, {})[id(span)] = span

    @classmethod
    def drop(cls, span) -> None:
        """Called by a span that ran on its own, so `pending` stays honest."""
        key = id(span)
        emptied = None
        for group_key, group in cls.groups.items():
            if group.pop(key, None) is not None:
                emptied = group_key if not group else None
                break
        if emptied is not None:
            # Deleting inside the loop above would invalidate the iterator.
            del cls.groups[emptied]

    @classmethod
    def pending(cls, key) -> int:
        return len(cls.groups.get(key, ()))

    @classmethod
    def run(cls, key) -> int:
        """Recompute the spans of chunk `key`. Returns how many ran."""
        group = cls.groups.pop(key, None)
        if not group:
            return 0
        ran = 0
        while group:
            group.popitem()[1].run_recompute_now()
            ran += 1
        return ran

    @classmethod
    def clear(cls) -> None:
        cls.groups = {}
        cls._open_key = None


class EventStore:
    event = None

    @classmethod
    def set(cls, event) -> None:
        cls.event = event


def fold_init_dims(tensor):
    # NOTE(zhangyuqin1998): Reshape a rank-3 tensor from P x M x N to (P * M) x N,
    # to keep weight_grad in a correct rank. See phi::FoldInitDims.
    if tensor.ndim == 3:
        tensor = paddle.reshape(tensor, [-1, tensor.shape[-1]])
    return tensor


def grad_weight_fn(input, weight, out_grad, inplace_update_grad=True):
    if weight.stop_gradient:
        return
    with paddle.no_grad():
        weight_grad = paddle.matmul(
            x=fold_init_dims(input),
            y=fold_init_dims(out_grad),
            transpose_x=True,
            transpose_y=False,
        )

        if hasattr(weight, "main_grad"):
            if weight.main_grad is None:
                weight.main_grad = paddle.base.framework.core.eager.Tensor(
                    value=weight_grad.cast(paddle.float32).value(),
                    place=weight_grad.place,
                    name="main_grad@" + weight.name,
                )
            else:
                weight.main_grad.add_(weight_grad)
            weight_grad._clear_data()
        else:
            if weight.grad is None:
                weight.grad = paddle.zeros_like(weight, dtype=weight.dtype)
            weight.grad = paddle.add(weight.grad, weight_grad)


class SplitBWMatmul(PyLayer):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        out = F.linear(x=input, weight=weight, bias=bias)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        input, weight, bias = ctx.saved_tensor()

        if WeightGradStore.enabled:
            WeightGradStore.put(
                partial(grad_weight_fn, input, weight, out_grad)
            )
        else:
            grad_weight_fn(input, weight, out_grad)

        input_grad = None
        if not input.stop_gradient:
            input_grad = paddle.matmul(
                x=out_grad, y=weight, transpose_x=False, transpose_y=True
            )
        if bias is not None:
            bias_grad = None
            if not bias.stop_gradient:
                bias_grad = paddle.sum(fold_init_dims(out_grad), axis=0)
            return input_grad, None, bias_grad
        else:
            return input_grad, None


class SplitBWLinear(nn.Linear):
    def forward(self, input):
        return SplitBWMatmul.apply(input, self.weight, bias=self.bias)
