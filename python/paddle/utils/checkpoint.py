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
from typing import Any, Iterable, List, Tuple
from paddle.fluid import core
from paddle.autograd import PyLayer
from paddle.fluid import framework

# import warnings
# from typing import Any, Iterable, List, Tuple


def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[core.VarBase, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, core.VarBase):
                out.append(inp)
                continue

            x = inp.detach()
            x.stop_gradient = inp.stop_gradient
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__)


class CheckpointFunction(PyLayer):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        # check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # ctx.had_autocast_in_fwd = torch.is_autocast_enabled()

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if paddle.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with paddle.no_grad():
            outputs = run_function(*args)

        return outputs

    @staticmethod
    def backward(ctx, *args):
        print("here!!!")
        # if not torch.autograd._is_checkpoint_valid():
        #     raise RuntimeError(
        #         "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
        #         " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
        #         " argument.")
        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensor()

        # Restore inputs
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        # rng_devices = []
        # if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
        #     rng_devices = ctx.fwd_gpu_devices
        # with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
        #     if ctx.preserve_rng_state:
        #         torch.set_rng_state(ctx.fwd_cpu_state)
        #         if ctx.had_cuda_in_fwd:
        #             set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
        detached_inputs = detach_variable(tuple(inputs))
        # with torch.enable_grad(), torch.cuda.amp.autocast(ctx.had_autocast_in_fwd):

        # paddle.enable_grad()
        tracer = framework._dygraph_tracer()
        tracer._has_grad = True

        outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, core.VarBase):
            outputs = (outputs, )
        # enable_grad
        assert len(outputs) == len(args)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if isinstance(outputs[i],
                          core.VarBase) and not outputs[i].stop_gradient:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError("none of output has requires_grad=True,"
                               " this checkpoint() is not necessary")
        paddle.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(
            inp.grad if isinstance(inp, core.VarBase) else None
            for inp in detached_inputs)

        return (None, None) + grads


def checkpoint(function, *args, **kwargs):
    r"""Checkpoint a model or part of the model

    Checkpointing works by trading compute for memory. Rather than storing all
    intermediate activations of the entire computation graph for computing
    backward, the checkpointed part does **not** save intermediate activations,
    and instead recomputes them in backward pass. It can be applied on any part
    of a model.

    Specifically, in the forward pass, :attr:`function` will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. Instead, the forward pass saves the inputs tuple and the
    :attr:`function` parameter. In the backwards pass, the saved inputs and
    :attr:`function` is retrieved, and the forward pass is computed on
    :attr:`function` again, now tracking the intermediate activations, and then
    the gradients are calculated using these activation values.

    The output of :attr:`function` can contain non-Tensor values and gradient
    recording is only performed for the Tensor values. Note that if the output
    consists of nested structures (ex: custom objects, lists, dicts etc.)
    consisting of Tensors, these Tensors nested in custom structures will not
    be considered as part of autograd.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning::
        If :attr:`function` invocation during backward does anything different
        than the one during forward, e.g., due to some global variable, the
        checkpointed version won't be equivalent, and unfortunately it can't be
        detected.

    .. warning::
        If checkpointed segment contains tensors detached from the computational
        graph by `detach()` or `torch.no_grad()`, the backward pass will raise an
        error. This is because `checkpoint` makes all the outputs require
        gradients which causes issues when a tensor is defined to have no
        gradient in the model. To circumvent this, detach the tensors outside of
        the `checkpoint` function.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients. At least one of the outputs needs to have
        :code:`requires_grad=True` as well.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(
            arg for arg in kwargs))

    return CheckpointFunction.apply(function, preserve, *args)


def checkpoint_sequential(functions, segments, input, **kwargs):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a model in various segments
    and checkpoint each segment. All segments except the last will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. The inputs of each checkpointed segment will be saved for
    re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    .. warning:
        Since PyTorch 1.4, it allows only one Tensor as the input and
        intermediate outputs, just like :class:`torch.nn.Sequential`.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or
            functions (comprising the model) to run sequentially.
        segments: Number of chunks to create in the model
        input: A Tensor that is input to :attr:`functions`
        preserve_rng_state(bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_sequential(model, chunks, input_var)
    """
    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(
            arg for arg in kwargs))

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input)
            return input

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        input = checkpoint(
            run_function(start, end, functions),
            input,
            preserve_rng_state=preserve)
    return run_function(end + 1, len(functions) - 1, functions)(input)
