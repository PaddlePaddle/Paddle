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
from .graphs import CUDAGraph

class CUDAGraphContext:
    def __init__(self, layer, num_warmup_steps):
        self.step = 0
        self.layer = layer
        self.forward_graph = CUDAGraph()
        self.backward_graph = CUDAGraph()
        self.num_warmup_steps = num_warmup_steps

def detach(x):
    if isinstance(x, paddle.Tensor):
        x_detached = x.detach()
        x_detached.stop_gradient = x.stop_gradient
        return x_detached
    else:
        return x

def get_grad(x):
    if isinstance(x, paddle.Tensor):
        return x.grad
    else:
        return x

class _CUDAGraphedLayer(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, context, *args):
        args = [detach(x) for x in args]

        if context.step < context.num_warmup_steps:
            with paddle.enable_grad():
                y = context.layer(*args)
            ctx.save_for_backward(context, args, y)
            return y.detach()

        elif context.step == context.num_warmup_steps:
            context.args_static = args
            context.forward_graph.capture_begin()
            with paddle.enable_grad():
                y = context.layer(*context.args_static)
            context.forward_graph.capture_end()

            context.forward_graph.replay()
            context.y_static = y

            ctx.save_for_backward(context, context.args_static, y)
            return y.detach()
        else:
            for x_staic, x in zip(context.args_static, args):
                if isinstance(x_staic, paddle.Tensor):
                    x_staic.copy_(x, True)

            context.forward_graph.replay()
            y = context.y_static

            ctx.save_for_backward(context, context.args_static, y)
            return y.detach()

    @staticmethod
    def backward(ctx, dy):
        context, args, y = ctx.saved_tensor()

        if context.step < context.num_warmup_steps:
            y.backward(dy)
        elif context.step == context.num_warmup_steps:
            context.dy_static = dy
            context.backward_graph.capture_begin()
            context.y_static.backward(context.dy_static)
            context.backward_graph.capture_end()
            context.backward_graph.replay()
        else:
            context.dy_static.copy_(dy, True)
            context.backward_graph.replay()

        def get_grad(x): return x.grad if isinstance(x, paddle.Tensor) else x
        args_grad = tuple(get_grad(x) for x in args)
        context.step += 1
        return args_grad

class CUDAGraphedLayer(paddle.nn.Layer):
    def __init__(self, layer: paddle.nn.Layer, num_warmup_steps=3):
        super().__init__()
        self.context = CUDAGraphContext(layer, num_warmup_steps)

    def forward(self, *args):
        return _CUDAGraphedLayer.apply(self.context, *args)
