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
