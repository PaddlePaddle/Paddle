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

from collections import deque
from enum import Enum

import paddle

from .graphs import CUDAGraph

# Tensor manipulation functions


def clone(x):
    """Clones a Paddle Tensor, preserving the stop_gradient property."""
    if isinstance(x, paddle.Tensor):
        c = x.clone()
        c.stop_gradient = x.stop_gradient
        return c
    return x


def detach(input_var):
    """
    Detaches a Paddle Tensor or collection of Tensors from the computation graph.
    It preserves the 'stop_gradient' property for individual tensors.

    Args:
        input_var (paddle.Tensor | list | tuple | dict): A Paddle Tensor or a collection
        of Tensors (list, tuple, or dictionary) to be detached from the computation graph.

    Returns:
        A single detached Tensor or a collection of detached Tensors (matching the input type)
        with the 'stop_gradient' property preserved.
    """

    def detach_tensor(tensor):
        # Detach an individual tensor and preserve its 'stop_gradient' property
        if isinstance(tensor, paddle.Tensor):
            detached_tensor = tensor.detach()
            detached_tensor.stop_gradient = tensor.stop_gradient
            return detached_tensor
        return tensor

    # Process input based on its type (Tensor, list, tuple, or dict)
    if isinstance(input_var, paddle.Tensor):
        return detach_tensor(input_var)
    elif isinstance(input_var, list):
        return [detach_tensor(item) for item in input_var]
    elif isinstance(input_var, tuple):
        return tuple(detach_tensor(item) for item in input_var)
    elif isinstance(input_var, dict):
        return {key: detach_tensor(value) for key, value in input_var.items()}

    # Return the input as-is if it's not a Tensor or collection of Tensors
    return input_var


def get_grad(x):
    """Returns the gradient of a Paddle Tensor if it's a tensor; otherwise, returns the input."""
    if isinstance(x, paddle.Tensor):
        return x.grad
    return x


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
        print(f"save graph to {name}")
        self.graph.print_to_dot_files(name)


# CUDA Graph Layer Status Enumeration


class CUDAGraphLayerStatus(Enum):
    """Enum to represent the status of a CUDA Graph Layer."""

    WARMUP = 1
    RECORD = 2
    CUDAGRAPH = 3


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
        self.graph_index = -1
        self._step = 0
        self.forward_graph = []
        self.backward_graph = []

        # Queue for saved tensors to support virtual pipeline, assuming FIFO order
        self.saved_tensor = deque()
        self.status = CUDAGraphLayerStatus.WARMUP

    def append_graph(self):
        new_forward_graph = CUDAGraphWithStaticInputOutput(
            self.num_warmup_steps
        )
        new_backward_graph = CUDAGraphWithStaticInputOutput(
            self.num_warmup_steps
        )
        self.forward_graph.append(new_forward_graph)
        self.backward_graph.append(new_backward_graph)
        self.graph_index += 1

    def current_forward_graph(self):
        return self.forward_graph[self.graph_index]

    def current_backward_graph(self):
        return self.backward_graph[self.graph_index]

    def reset_graph_index(self):
        self.graph_index = 0

    def queue_push(self, args):
        self.saved_tensor.append(args)

    def queue_pop(self):
        return self.saved_tensor.popleft()

    def next_graph(self):
        self.graph_index += 1

    def update_status(self):
        if self._step < self.num_warmup_steps:
            self.status = CUDAGraphLayerStatus.WARMUP
        elif self._step == self.num_warmup_steps:
            self.status = CUDAGraphLayerStatus.RECORD
        else:
            self.status = CUDAGraphLayerStatus.CUDAGRAPH

    def step(self):
        self._step += 1
        self.update_status()

    def is_warmup_step(self):
        return self.status == CUDAGraphLayerStatus.WARMUP

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
    def forward(ctx, context, *args, **kwargs):
        """
        Handles the forward pass of the layer. It operates differently based on the
        context's status: warmup, recording, or CUDA graph step.
        """
        # Detach all inputs from the computational graph
        args = detach(args)
        kwargs = detach(kwargs)
        if context.is_warmup_step():
            # In warmup step, perform the operation with gradient tracking
            with paddle.enable_grad():
                y = context.layer(*args, **kwargs)
            context.queue_push((args, y, None))

        elif context.is_record_step():
            # In record step, record the forward pass in CUDA graph
            print(f"{id(context)} FW (cudagraph-record)".center(100, "-"))
            context.append_graph()
            g = context.current_forward_graph()

            def forward(*args, **kwargs):
                with paddle.enable_grad():
                    return context.layer(*args, **kwargs)

            y = g.record(forward, *args, **kwargs)
            context.queue_push((args, y, context.current_backward_graph()))

        else:
            # In CUDA graph step, replay the recorded graph
            # print(f"{id(context)} FW (cudagraph)".center(100, "-"))
            g = context.current_forward_graph()
            y = g.replay(*args, **kwargs)
            context.queue_push(
                (g.args_static, y, context.current_backward_graph())
            )
            context.next_graph()

        ctx.save_for_backward(context)
        return detach(y)

    @staticmethod
    def backward(ctx, dy):
        """
        Handles the backward pass of the layer. Similar to forward, it handles
        backward based on the context's status: warmup, recording, or CUDA graph step.
        """
        (context,) = ctx.saved_tensor()

        args, input_var, g = context.queue_pop()

        y = None
        # Check the type of input_var and select an appropriate tensor for the backward operation
        if isinstance(input_var, paddle.Tensor):
            # Directly use the tensor if input_var itself is a tensor and eligible for gradient computation
            y = input_var
        elif isinstance(input_var, (list, tuple)):
            # Iterate through the list or tuple to find an eligible tensor for the backward operation
            for v in input_var:
                if isinstance(v, paddle.Tensor) and (not v.stop_gradient):
                    y = v  # Select the first eligible tensor
                    break  # Exit the loop once an eligible tensor is found

        assert isinstance(y, paddle.Tensor)

        if context.is_warmup_step():
            # In warmup step, perform standard backward operation
            y.backward(dy)
        elif context.is_record_step():
            # In record step, record the backward pass in CUDA graph
            print(f"{id(context)} BW (cudagraph-record)".center(100, "-"))

            def backward(y, dy):
                y.backward(dy)

            g.record(backward, y, dy)
        else:
            # In CUDA graph step, replay the recorded graph for backward pass
            # print(f"{id(context)} BW (cudagraph)".center(100, "-"))
            g.replay(y, dy)

        args_grad = tuple(get_grad(x) for x in args)
        # Step the global state if the queue is empty
        if len(context.saved_tensor) == 0:
            if context.is_record_step() or context.is_cuda_graph_step():
                context.reset_graph_index()
            context.step()

        return args_grad


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

        # self.layers = layer.layers

        # Delegator: bind special methods of `layer` to `CUDAGraphedLayer`
        # The special method should be binded to Type instead of self
        # my_methods = dir(self)
        # is_special = lambda name: name.startswith("__") and name.endswith("__")
        # for name in dir(layer):
        #     if is_special(name) and (name not in my_methods):
        #         setattr(CUDAGraphedLayer, name, getattr(layer, name))

    def forward(self, *args, **kwargs):
        return _CUDAGraphedLayer.apply(self.context, *args, **kwargs)

    def is_warmup_step(self):
        return self.context.is_warmup_step()

    def is_record_step(self):
        return self.context.is_record_step()

    def is_cuda_graph_step(self):
        return self.context.is_cuda_graph_step()

    # def __getattr__(self, attr):
    # """Delegate attribute access to the inner object."""
    # return getattr(self.context.layer, attr)
