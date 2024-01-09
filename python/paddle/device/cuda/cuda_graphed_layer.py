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

import logging
from collections import deque
from enum import Enum

import paddle

from .graphs import CUDAGraph


def recurive_apply(function, input_var):
    if isinstance(input_var, list):
        return [recurive_apply(function, item) for item in input_var]
    elif isinstance(input_var, tuple):
        return tuple(recurive_apply(function, item) for item in input_var)
    elif isinstance(input_var, dict):
        return {
            key: recurive_apply(function, value)
            for key, value in input_var.items()
        }
    else:
        return function(input_var)


def detach_tensor(tensor):
    # Detach an individual tensor and preserve its 'stop_gradient' property
    if isinstance(tensor, paddle.Tensor):
        detached_tensor = tensor.detach()
        detached_tensor.stop_gradient = tensor.stop_gradient
        return detached_tensor
    return tensor


def recurive_flatten_args(target):
    ret = []

    def append(arg):
        if isinstance(arg, paddle.Tensor):
            if not arg.stop_gradient:
                ret.append(arg)

    recurive_apply(append, target)
    return ret


detach = lambda x: recurive_apply(detach_tensor, x)


def get_grad_tensor(x):
    """Returns the gradient of a Paddle Tensor if it's a tensor; otherwise, returns the input."""
    if isinstance(x, paddle.Tensor):
        if x.stop_gradient:
            return None
        else:
            return x.grad
    return None


# CUDA Graph with Static Input and Output
class CUDAGraphWithStaticInputOutput:
    def __init__(self, num_warmup_steps):
        self.num_warmup_steps = num_warmup_steps
        self.graph = CUDAGraph()

        self.has_recorded = False
        self.args_static = None
        self.kwargs_static = None
        self.output_static = None

    def record(self, f, *args, **kwargs):
        self.args_static = args
        self.kwargs_static = kwargs

        self.graph.capture_begin()
        self.output_static = f(*self.args_static, **self.kwargs_static)
        self.graph.capture_end()
        self.graph.replay()

        self.has_recorded = True

        return self.output_static

    def replay(self, *args, **kwargs):
        if not self.has_recorded:
            raise RuntimeError("Graph should be recorded first")

        for x_staic, x in zip(self.args_static, args):
            if isinstance(x_staic, paddle.Tensor):
                x_staic.copy_(x, True)

        for x_staic, x in zip(self.kwargs_static.values(), kwargs.values()):
            if isinstance(x_staic, paddle.Tensor):
                x_staic.copy_(x, True)

        self.graph.replay()
        return self.output_static

    def save(self, name):
        logging.info(f"save graph to {name}")
        self.graph.print_to_dot_files(name)


# CUDA Graph Layer Status Enumeration


class CUDAGraphLayerStatus(Enum):
    """Enum to represent the status of a CUDA Graph Layer."""

    WARMUP = 1
    RECORD = 2
    WAIT_BACKWARD_RECORD = 3
    CUDAGRAPH = 4


class CUDAGraphContext:
    """
    Manages the context for CUDA graph execution in layers. This includes handling
    the state of CUDA graph layers, managing forward and backward graphs, and
    tracking the execution steps.
    """

    def __init__(self, layer, num_warmup_steps):
        """
        Initializes the CUDA graph context.
        :param layer: The layer to be used in the CUDA graph.
        :param num_warmup_steps: Number of warmup steps before recording starts.
        """
        self.layer = layer
        self.num_warmup_steps = num_warmup_steps
        self._step = 0
        self.forward_graph = CUDAGraphWithStaticInputOutput(
            self.num_warmup_steps
        )
        self.backward_graph = CUDAGraphWithStaticInputOutput(
            self.num_warmup_steps
        )
        self.recorded_grad_tesor = None

        # Queue for saved tensors to support 1f1b/interleved scheduler, assuming FIFO order
        self.queue = deque()
        self.status = CUDAGraphLayerStatus.WARMUP

    def queue_push(self, args):
        self.queue.append(args)

    def queue_pop(self):
        return self.queue.popleft()

    def warmup_step(self):
        self._step += 1
        if self._step == self.num_warmup_steps:
            self.status = CUDAGraphLayerStatus.RECORD

    def record_step(self):
        self.status = CUDAGraphLayerStatus.WAIT_BACKWARD_RECORD

    def backward_record_step(self):
        self.status = CUDAGraphLayerStatus.CUDAGRAPH

    def is_warmup_step(self):
        return (self.status == CUDAGraphLayerStatus.WARMUP) or (
            self.status == CUDAGraphLayerStatus.WAIT_BACKWARD_RECORD
        )

    def is_record_step(self):
        return self.status == CUDAGraphLayerStatus.RECORD

    def is_cuda_graph_step(self):
        return self.status == CUDAGraphLayerStatus.CUDAGRAPH


class _CUDAGraphedLayer(paddle.autograd.PyLayer):
    """
    A custom layer that integrates CUDA Graph recording and execution into PaddlePaddle's autograd system.
    It handles forward and backward operations differently based on the CUDA graph layer status.

    Input of the Layer: paddle.Tensor or List/Tuple of paddle.Tensor
    Output of the Layer: paddle.Tensor or List/Tuple of paddle.Tensor

    """

    @staticmethod
    def forward(ctx, context, arg_tuple, require_grads, *grad_inputs):
        """
        Handles the forward pass of the layer. It operates differently based on the
        context's status: warmup, recording, or CUDA graph step.
        """
        args, kwargs = arg_tuple
        # Detach all inputs from the computational graph
        args = detach(args)
        kwargs = detach(kwargs)

        detached_grad_inputs = [
            *recurive_flatten_args(args),
            *recurive_flatten_args(tuple(kwargs.values())),
        ]

        if context.is_warmup_step():
            # In warmup step, perform the operation with gradient tracking
            with paddle.enable_grad():
                y = context.layer(*args, **kwargs)

            context.queue_push(
                (
                    CUDAGraphLayerStatus.WARMUP,
                    require_grads,
                    detached_grad_inputs,
                    y,
                )
            )
            context.warmup_step()
        elif context.is_record_step():
            # In record step, record the forward pass in CUDA graph
            print(f"{id(context)} FW (cudagraph-record)".center(100, "-"))

            def forward(*args, **kwargs):
                with paddle.enable_grad():
                    return context.layer(*args, **kwargs)

            y = context.forward_graph.record(forward, *args, **kwargs)

            context.queue_push(
                (
                    CUDAGraphLayerStatus.RECORD,
                    require_grads,
                    detached_grad_inputs,
                    y,
                )
            )
            context.record_step()
        else:
            # In CUDA graph step, replay the recorded graph
            y = context.forward_graph.replay(*args, **kwargs)
            context.queue_push((CUDAGraphLayerStatus.CUDAGRAPH, None, None, y))

        ctx.save_for_backward(context)
        return detach(y)

    @staticmethod
    def backward(ctx, *dys):
        """
        Handles the backward pass of the layer. Similar to forward, it handles
        backward based on the context's status: warmup, recording, or CUDA graph step.
        """
        (context,) = ctx.saved_tensor()

        status, require_grads, detached_grad_inputs, ys = context.queue_pop()

        # [TODO] when there is multiple output tensor, we support only one y that allows backward
        y, dy = None, None
        if isinstance(ys, paddle.Tensor):
            y, dy = ys, dys[0]
        elif isinstance(ys, (list, tuple)):
            for v, dv in zip(ys, dys):
                if isinstance(v, paddle.Tensor) and (not v.stop_gradient):
                    y, dy = v, dv
                    break
        assert isinstance(y, paddle.Tensor) and isinstance(dy, paddle.Tensor)

        if status == CUDAGraphLayerStatus.WARMUP:
            # In warmup step, perform standard backward operation
            y.backward(dy)
        elif status == CUDAGraphLayerStatus.RECORD:
            # In record step, record the backward pass in CUDA graph
            print(f"{id(context)} BW (cudagraph-record)".center(100, "-"))

            def backward(y, dy):
                y.backward(dy)

            context.backward_graph.record(backward, y, dy)
            context.backward_record_step()
        elif status == CUDAGraphLayerStatus.CUDAGRAPH:
            # In CUDA graph step, replay the recorded graph for backward pass
            context.backward_graph.replay(y, dy)
            args_grad = context.recorded_grad_tesor
        else:
            raise RuntimeError("Unknown cuda graph status")

        if (
            status == CUDAGraphLayerStatus.WARMUP
            or status == CUDAGraphLayerStatus.RECORD
        ):
            args_grad = []
            for require_grad, detached_x in zip(
                require_grads, detached_grad_inputs
            ):
                if require_grad:
                    if detached_x.grad is None:
                        args_grad.append(paddle.zeros(detached_x.shape))
                    else:
                        args_grad.append(detached_x.grad)
                else:
                    args_grad.append(None)

        if status == CUDAGraphLayerStatus.RECORD:
            # preserve the grad and we can get the output of these grad when replay
            context.recorded_grad_tesor = args_grad
        elif status == CUDAGraphLayerStatus.CUDAGRAPH:
            args_grad = context.recorded_grad_tesor

        return tuple(args_grad)


class CUDAGraphedLayer(paddle.nn.Layer):
    """
    CUDAGraphedLayer: A PaddlePaddle Layer to convert an eager mode model to utilize CUDA Graphs.

    CUDA Graphs provide a way to capture kernel-level operations of a model and play
    them back efficiently, allowing for potential speedups in repetitive computations,
    such as those during training iterations. This layer is a wrapper that enables
    the usage of CUDA Graphs with PaddlePaddle models.

    Overview:
    - The layer encapsulates another layer (the model to be converted).
    - During the first few (num_warmup_steps) iterations, the layer operates in
      eager mode without any CUDA Graphs.
    - After the warmup steps, the layer captures the forward and backward computations
      and replays them using CUDA Graphs in subsequent iterations.

    Usage:
        model = Model()
        graphed_model = CUDAGraphedLayer(model)

    Parameters:
    - layer (paddle.nn.Layer): The PaddlePaddle model/layer to be converted.
    - num_warmup_steps (int): The number of iterations before the CUDA Graph
      capture begins. Default is 3.

    Notes:
    - Restrictions:
        * CPU-GPU Synchronization: Operations that synchronize the CPU with the GPU, like device to host transfers, are not allowed.
        * CPU Work: Any operations on the CPU within the captured graph are not recorded.
        * Memory Address (Pointer) Consistency: Replays consistently read from and write to identical virtual memory addresses.
        * Dynamic Operations:
            - Control Flow: Dynamic control flows, especially those based on CPU data like if/else statements, are prohibited.
            - Tensor Shapes: Dynamic tensor shapes are not supported.

    - Allowed Operations:
        * CUDA RNG Operations: CUDA-based Random Number Generation operations are allowed.
    """

    def __init__(self, layer: paddle.nn.Layer, num_warmup_steps=3):
        super().__init__()
        self.context = CUDAGraphContext(layer, num_warmup_steps)
        self.add_sublayer(f"Graphed {type(layer).__name__}", layer)

    def forward(self, *args, **kwargs):
        # We collect them into a list of tensor that required grad
        grad_inputs = [
            *recurive_flatten_args(args),
            *recurive_flatten_args(tuple(kwargs.values())),
        ]
        require_grads = [not x.stop_gradient for x in grad_inputs]
        return _CUDAGraphedLayer.apply(
            self.context, (args, kwargs), require_grads, *grad_inputs
        )

    def is_warmup_step(self):
        return self.context.is_warmup_step()

    def is_record_step(self):
        return self.context.is_record_step()

    def is_cuda_graph_step(self):
        return self.context.is_cuda_graph_step()
