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

import contextlib
import copy
import inspect
import weakref

import paddle
from paddle import framework
from paddle.autograd import PyLayer
from paddle.base.framework import EagerParamBase
from paddle.distributed.fleet.meta_parallel.parallel_layers.random import (
    get_rng_state_tracker,
)
from paddle.framework import core, in_dynamic_mode

from ..utils.log_util import logger

__all__ = []


def _varbase_help(param):
    state = copy.deepcopy(param.__dict__)
    new_param = EagerParamBase(
        shape=param.shape, dtype=param.dtype, name=param.name, **state
    )
    param._share_buffer_to(new_param)
    return new_param


def detach_variable(inputs):
    out = []
    for inp in inputs:
        if not isinstance(inp, core.eager.Tensor) and (
            type(inp) is not tuple or not isinstance(inp[0], core.eager.Tensor)
        ):
            # the inp is not a tensor or not a tuple of tensors
            out.append(inp)
            continue

        if isinstance(inp, EagerParamBase):
            out.append(_varbase_help(inp))
            continue

        if type(inp) is tuple:
            detach_inp = []
            for i in inp:
                # detach all tensors in the tuple
                assert isinstance(i, core.eager.Tensor)

                if isinstance(i, EagerParamBase):
                    detach_inp.append(_varbase_help(i))
                else:
                    tmp_i = i.detach()
                    tmp_i.stop_gradient = i.stop_gradient
                    detach_inp.append(tmp_i)

            out.append(tuple(detach_inp))
            continue

        x = inp.detach()
        x.stop_gradient = inp.stop_gradient
        out.append(x)
    return tuple(out)


def check_recompute_necessary(inputs):
    necessary_for_each_input = []
    for input_ in inputs:
        if isinstance(input_, (core.eager.Tensor, paddle.Tensor)):
            necessary_for_each_input.append(input_.stop_gradient)
        elif type(input_) is tuple:
            for i in input_:
                # traverse all tensors in the tuple
                if isinstance(i, (core.eager.Tensor, paddle.Tensor)):
                    necessary_for_each_input.append(i.stop_gradient)
    if all(necessary_for_each_input):
        logger.warning(
            "[Recompute]: None of the inputs to current recompute block need grad, "
            "therefore there is NO need to recompute this block in backward !"
        )


@contextlib.contextmanager
def switch_rng_state_tracker(rng_state, tracker):
    orig_rng_state = paddle.get_rng_state()
    orig_rng_tracker = get_rng_state_tracker().get_states_tracker()
    paddle.set_rng_state(rng_state)
    get_rng_state_tracker().set_states_tracker(tracker)
    try:
        yield
    finally:
        paddle.set_rng_state(orig_rng_state)
        get_rng_state_tracker().set_states_tracker(orig_rng_tracker)


class RecomputeFunction(PyLayer):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args, **kwargs):
        # store for recomputing
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.kwargs = kwargs

        # NOTE the number of outputs of backward() should be equal to the number of tensors in forward()'s input
        # the order of tensors in backward()'s output should be the same as tensors in forward()'s input
        # None tensor inputs will be filtered in backward inputs.

        # save input for backward
        ctx.inputs = []
        ctx.tensor_indices = []
        ctx.duplicate_tensor = [False for _ in range(len(args))]
        tensor_inputs = []
        for i, arg in enumerate(args):
            if paddle.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            elif type(arg) is tuple:
                is_tensors = [paddle.is_tensor(a) for a in arg]
                if all(is_tensors):
                    # the tuple is a tuple of tensors
                    tensors_stop_gradient = [a.stop_gradient for a in arg]
                    if not all(tensors_stop_gradient) and any(
                        tensors_stop_gradient
                    ):
                        # tensors in the tuple have different stop_gradient value, which pylayer doesn't support
                        raise ValueError(
                            "Recompute receive a tuple containing tensor holds different stop gradient."
                        )
                    tensor_inputs.append(arg)
                    ctx.tensor_indices.append(i)
                    # Mark the tuple is a tuple of tensors
                    ctx.duplicate_tensor[i] = True
                    ctx.inputs.append(None)
                elif any(is_tensors):
                    # the tuple contains tensors and non-tensor values
                    raise ValueError(
                        "Recompute receive a tuple containing tensor and non-tensor at same time."
                    )
                else:
                    ctx.inputs.append(arg)
            else:
                ctx.inputs.append(arg)
        ctx.save_for_backward(*tensor_inputs)

        # NOTE recompute with restore RNG only support one scenario where one process for one cuda gpu.
        # one process with multiple gpu and mix-gpu-cpu scenarios are not support
        if ctx.preserve_rng_state:
            ctx.fw_rng_state = paddle.get_rng_state()
            ctx.fwd_rng_state_tracker = (
                get_rng_state_tracker().get_states_tracker()
            )

        # TODO support AMP
        tracer = framework._dygraph_tracer()
        ctx.is_fw_autocast = (
            False if tracer._amp_level == core.AmpLevel.O0 else True
        )
        if tracer._amp_level == core.AmpLevel.O2:
            ctx.amp_level = 'O2'
        elif tracer._amp_level in (core.AmpLevel.O1, core.AmpLevel.O0):
            ctx.amp_level = 'O1'
        else:
            raise ValueError(f"unsupported amp level: {tracer._amp_level}")

        if tracer._amp_dtype == 'float16':
            ctx.amp_dtype = 'float16'
        elif tracer._amp_dtype in ('bfloat16', 'float32'):
            ctx.amp_dtype = 'bfloat16'
        else:
            raise ValueError(f"unsupported amp dtype: {tracer._amp_dtype}")

        ctx.amp_white_list, ctx.amp_black_list = tracer._get_amp_op_list()

        with paddle.no_grad():
            outputs = run_function(*args, **kwargs)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        with paddle.base.dygraph.guard():
            # TODO need to check the recompute calling is valid or not

            # Restore inputs
            inputs = list(ctx.inputs)
            tensor_indices = ctx.tensor_indices
            duplicate_tensor = ctx.duplicate_tensor
            tensors = ctx.saved_tensor()
            for i, idx in enumerate(tensor_indices):
                inputs[idx] = tensors[i]

            # paddle.enable_grad()
            tracer = framework._dygraph_tracer()
            tracer._has_grad = True

            # NOTE support AMP
            # need restore auto_cast state as well as w/b list
            if ctx.preserve_rng_state:
                with switch_rng_state_tracker(
                    ctx.fw_rng_state, ctx.fwd_rng_state_tracker
                ):
                    with paddle.amp.auto_cast(
                        enable=ctx.is_fw_autocast,
                        custom_white_list=ctx.amp_white_list,
                        custom_black_list=ctx.amp_black_list,
                        level=ctx.amp_level,
                        dtype=ctx.amp_dtype,
                    ):
                        detached_inputs = detach_variable(tuple(inputs))
                        outputs = ctx.run_function(
                            *detached_inputs, **ctx.kwargs
                        )
            else:
                with paddle.amp.auto_cast(
                    enable=ctx.is_fw_autocast,
                    custom_white_list=ctx.amp_white_list,
                    custom_black_list=ctx.amp_black_list,
                    level=ctx.amp_level,
                    dtype=ctx.amp_dtype,
                ):
                    detached_inputs = detach_variable(tuple(inputs))
                    outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)

            if isinstance(outputs, core.eager.Tensor):
                outputs = (outputs,)
            assert len(outputs) == len(args)

            # run backward() with only tensor that requires grad
            forward_outputs_with_grad = []
            # NOTE In Transformer-like network, if user put the attention mask into the recompute segment output,
            # pylayer will force the stop_gradient of attention mask to be False, which will make the number of
            # tensor that need grad does not match.
            # the following backward_inputs_with_grad is used to avoid this case.
            backward_inputs_with_grad = []
            for i in range(len(outputs)):
                if (
                    isinstance(outputs[i], core.eager.Tensor)
                    and not outputs[i].stop_gradient
                ):
                    forward_outputs_with_grad.append(outputs[i])
                    backward_inputs_with_grad.append(args[i])

            if len(forward_outputs_with_grad) == 0:
                raise RuntimeError(
                    "none of output has requires_grad=True, this recompute() is not necessary"
                )

            # actually backward
            with paddle.amp.auto_cast(enable=False):
                paddle.autograd.backward(
                    forward_outputs_with_grad, backward_inputs_with_grad
                )

            grads = []
            for idx, inp in enumerate(detached_inputs):
                if isinstance(inp, core.eager.Tensor):
                    grads.append(inp._grad_ivar())
                elif type(inp) is tuple and duplicate_tensor[idx]:
                    # input is a tuple and is a tuple of tensors
                    if all(i.stop_gradient for i in inp):
                        # all tensors in the tuple doesn't need grad, only return a None for the whole tuple
                        grads.append(None)
                    else:
                        # all tensors in the tuple need grad, should return a tuple of grads
                        grads.append(tuple(i._grad_ivar() for i in inp))

            if in_dynamic_mode():
                grads = tuple(grads)
            else:
                grads = list(grads)
            return grads


def _recompute_without_reentrant(
    function, preserve_rng_state=True, *args, **kwargs
):
    """
    recompute without reentrant, that means use hook to implement the recompute function rather than re-entrant autograd.
    """

    if preserve_rng_state:
        cur_device = paddle.get_device()
        if 'gpu:' in cur_device:
            fw_cuda_rng_state = paddle.get_cuda_rng_state()
        elif 'xpu:' in cur_device:
            fw_cuda_rng_state = paddle.get_rng_state()
        elif (
            cur_device.split(':')[0]
            in paddle.device.get_all_custom_device_type()
        ):
            fw_cuda_rng_state = paddle.get_rng_state(cur_device)
        else:
            raise RuntimeError(
                f"Recompute with RNG preserve is not support current device: {cur_device}."
            )
        fwd_cuda_rng_state_tracker = (
            get_rng_state_tracker().get_states_tracker()
        )
    tracer = framework._dygraph_tracer()
    is_fw_autocast = False if tracer._amp_level == core.AmpLevel.O0 else True
    if tracer._amp_level == core.AmpLevel.O2:
        amp_level = 'O2'
    elif tracer._amp_level in (core.AmpLevel.O1, core.AmpLevel.O0):
        amp_level = 'O1'

    if tracer._amp_dtype == 'float16':
        amp_dtype = 'float16'
    elif tracer._amp_dtype in ('bfloat16', 'float32'):
        amp_dtype = 'bfloat16'

    amp_white_list, amp_black_list = tracer._get_amp_op_list()

    class Intermediate_Holder:
        pass

    storage = weakref.WeakKeyDictionary()
    holder_list = []

    def pack(x):
        res = Intermediate_Holder()
        holder_list.append(weakref.ref(res))
        return res

    def unpack(x):
        unpack_counter = 0
        if len(storage) == 0:

            def inner_pack(inner_x):
                nonlocal unpack_counter
                unpack_counter += 1

                if holder_list[unpack_counter - 1]() is None:
                    return

                tmp_tensor = core.eager.Tensor(
                    inner_x.dtype,
                    inner_x.shape,
                    inner_x.name + "cpy",
                    core.VarDesc.VarType.LOD_TENSOR,
                    inner_x.persistable,
                )
                inner_x._share_buffer_to(tmp_tensor)
                storage[holder_list[unpack_counter - 1]()] = tmp_tensor
                return

            def inner_unpack(inner_x):
                raise Exception("An unexpected backward called on a tensor!")

            if preserve_rng_state:
                with switch_rng_state_tracker(
                    fw_cuda_rng_state, fwd_cuda_rng_state_tracker
                ):
                    with paddle.set_grad_enabled(True):
                        with paddle.amp.auto_cast(
                            enable=is_fw_autocast,
                            custom_white_list=amp_white_list,
                            custom_black_list=amp_black_list,
                            level=amp_level,
                            dtype=amp_dtype,
                        ):
                            with paddle.autograd.saved_tensors_hooks(
                                inner_pack, inner_unpack
                            ):
                                unused_outputs = function(*args, **kwargs)
            else:
                with paddle.set_grad_enabled(True), paddle.amp.auto_cast(
                    enable=is_fw_autocast,
                    custom_white_list=amp_white_list,
                    custom_black_list=amp_black_list,
                    level=amp_level,
                    dtype=amp_dtype,
                ), paddle.autograd.saved_tensors_hooks(
                    inner_pack, inner_unpack
                ):
                    unused_outputs = function(*args, **kwargs)

        if x not in storage:
            raise Exception(
                "Not supported to retrieve a tensor saved by autograd multiple times that is no need to recompute."
            )

        return storage[x]

    with paddle.autograd.saved_tensors_hooks(pack, unpack):
        outputs = function(*args, **kwargs)

    return outputs


def recompute(function, *args, **kwargs):
    """
    recompute intermediate activations to save then memory.

    Parameters:
        function(paddle.nn.Layer): layer of sequence of layers that describes part of forward pass of the model
              whose intermediate activations will be released to save memory in forward stage and will be recomputed
              in backward stage for gradient calculation.
        *args(Tensor): inputs to the function.
        **kwargs(Dict): Kwargs should only contain two kinds of key-value params, the one is part of function's key-value params,
                        and the other contains 'preserve_rng_state' and 'use_reentrant'. the key-value pair of preserve_rng_state,
                        which is used to indicate whether to save the forward rng. If it is True, then the last forward rng value
                        will be restored when the forward recalculation of backpropagation is performed, its default value is True.
                        the key-value pair of use_reentrant is used to indicate which implementation of recompute you will be used.
                        'use_reentrant=True' means to use the PyLayer implementation of recompute, 'use_reentrant=False' means to
                        use the Hook implementation of recompute, its default value is True.
    Returns:
        Output of function on args.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED, env:GPU)
            >>> import paddle
            >>> from paddle.distributed.fleet.utils import recompute
            >>> import random
            >>> paddle.seed(2023)
            >>> def get_fc_block(block_idx, input_size, is_last=False):
            ...     block_name = "block_" + str(block_idx)
            ...     block = paddle.nn.Sequential(
            ...         (block_name + "_fc_0", paddle.nn.Linear(input_size, input_size, bias_attr=False)),
            ...         (block_name + "_dropout", paddle.nn.Dropout(p=0.5)),
            ...         (block_name + "_relu_1", paddle.nn.ReLU()),
            ...         (block_name + "_fc_1", paddle.nn.Linear(input_size, input_size, bias_attr=False)),
            ...         (block_name + "_relu_2", paddle.nn.ReLU()),
            ...     )
            ...     if is_last:
            ...         block.add_sublayer(
            ...             block_name + "_fc_2",
            ...             paddle.nn.Linear(
            ...                 input_size, 1, bias_attr=False
            ...             )
            ...         )
            ...     else:
            ...         block.add_sublayer(
            ...             block_name + "_fc_2",
            ...             paddle.nn.Linear(input_size, input_size, bias_attr=False)
            ...         )
            ...     return block

            >>> class Naive_fc_net(paddle.nn.Layer):
            ...     def __init__(self, input_size=10,
            ...                 recompute_blocks=[1, 3],
            ...                 recompute_kwargs={}):
            ...         super().__init__()
            ...         self.recompute_blocks = recompute_blocks
            ...         self.recompute_kwargs = recompute_kwargs
            ...         self.runfunc0 = get_fc_block(0, input_size, is_last=False)
            ...         self.runfunc1 = get_fc_block(1, input_size, is_last=False)
            ...         self.runfunc2 = get_fc_block(2, input_size, is_last=False)
            ...         self.runfunc3 = get_fc_block(3, input_size, is_last=False)
            ...         self.runfunc4 = get_fc_block(4, input_size, is_last=True)
            ...         self.total_func = [self.runfunc0, self.runfunc1, self.runfunc2, self.runfunc3, self.runfunc4]
            ...     def forward(self, inputs):
            ...         nums = len(self.total_func)
            ...         for i in range(nums):
            ...             if i in self.recompute_blocks:
            ...                 inputs = recompute(self.total_func[i], inputs, **{"preserve_rng_state": True})
            ...             else:
            ...                 inputs = self.total_func[i](inputs)
            ...         return inputs

            >>> def run_model(cuda_state, recompute_block=[], recompute_kwargs={}):
            ...     gen = paddle.seed(10)
            ...     gen.manual_seed(10)
            ...     random.seed(10)
            ...     if cuda_state:
            ...         paddle.set_cuda_rng_state(cuda_state)
            ...     batch_size, input_size = 1, 10
            ...     model = Naive_fc_net(
            ...         input_size,
            ...         recompute_blocks=recompute_block,
            ...         recompute_kwargs=recompute_kwargs)
            ...     optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
            ...     loss_ = []
            ...     param_ = []
            ...     grad_ = []
            ...     for _ in range(5):
            ...         x = paddle.rand(shape=[batch_size, input_size], dtype="float32")
            ...         y_pred = model(x)
            ...         loss = y_pred.mean()
            ...         loss_.append(loss.item())
            ...         loss.backward()
            ...         optimizer.step()
            ...         param_.append(model.parameters()[9])
            ...         grad_.append(model.parameters()[3]._grad_ivar())
            ...         optimizer.clear_grad()
            ...     return loss_, param_, grad_

            >>> cuda_state = paddle.get_cuda_rng_state()
            >>> # without recompute
            >>> loss_ref, param_ref, grad_ref = run_model(
            ...     cuda_state, recompute_block=[]
            ... )

            >>> loss, param, grad = run_model(cuda_state, recompute_block=[1, 2])
            >>> print("normal_loss: {}, recompute_loss: {}".format(loss_ref, loss))
            >>> # The result of the recompute_loss should be the same as the normal_loss.
            normal_loss: [0.0018744759727269411, 0.0, 0.035971127450466156, 0.0, 0.0], recompute_loss: [0.0018744759727269411, 0.0, 0.035971127450466156, 0.0, 0.0]

    """
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)

    # whether to use reentrant method to implement recompute
    use_reentrant = kwargs.pop('use_reentrant', True)

    if not in_dynamic_mode():
        from paddle.distributed.auto_parallel.interface import (
            recompute as static_auto_recompute,
        )

        return static_auto_recompute(function)(*args, **kwargs)

    if framework._dygraph_tracer()._has_grad:
        check_args = list(args)
        check_args.extend(list(kwargs.values()))
        check_recompute_necessary(check_args)

    if use_reentrant:
        input_args = []
        # rearrange `position-args + keyword-args` into `position-args`
        if isinstance(function, paddle.nn.Layer):
            dyfunc_sig = inspect.signature(function.forward)
        else:
            dyfunc_sig = inspect.signature(function)

        bound_args = dyfunc_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for arg, param in zip(
            bound_args.arguments.values(), dyfunc_sig.parameters.values()
        ):
            if param.kind == param.VAR_POSITIONAL:
                input_args.extend(arg)
            elif param.kind in (
                param.POSITIONAL_ONLY,
                param.POSITIONAL_OR_KEYWORD,
            ):
                input_args.append(arg)
            elif param.kind == param.VAR_KEYWORD:
                input_args.extend(arg.values())
            elif param.kind == param.KEYWORD_ONLY:
                raise ValueError(
                    "Currently, keyword-only arguments are not supported when you want to send kwargs(dict parameter) to function with use_reentrant=True."
                )
            else:
                raise ValueError("Unknown parameter kind.")

        return RecomputeFunction.apply(function, preserve, *input_args)
    else:
        return _recompute_without_reentrant(function, preserve, *args, **kwargs)


def recompute_sequential(ctx, functions, *args, **kwargs):
    """
    recompute intermediate activations to save the memory for 'Sequential' models. use 'ctx' to transmit some context params, it is similar to 'recompute_hybrid' API.

    Parameters:
        ctx(dict): include 'segments' and  'preserve_rng_state' keys, the key 'segments' (int, default 1), represents the number of chunks to create in the model,
                   the key 'preserve_rng_state' (bool, optional, default=True) indicate whether to save the forward rng. If it is True, then the last forward rng value will be
                   restored when the forward recalculation of backpropagation is performed.
        functions(paddle.nn.Sequential): layer of sequence of layers that describes part of forward pass of the model
              whose intermediate activations will be released to save memory in forward stage and will be recomputed
              in backward stage for gradient calculation.
        *args(Tensor): inputs(tuple) to the function.
        **kwargs(Dict): inputs(dict) to the function.

    Returns:
        Output of function on args and kwargs.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle
            >>> from paddle.incubate.distributed.fleet import recompute_sequential
            >>> input = paddle.ones(shape=[8, 10])
            >>> model = paddle.nn.Sequential(paddle.nn.Linear(10, 10), paddle.nn.Linear(10, 2))
            >>> output = recompute_sequential({'segments' : 1}, model, input)

    """
    segments = ctx.get('segments', 1)
    preserve_rng_state = ctx.get('preserve_rng_state', True)

    def _run_func(begin, end, funcs):
        def do_run(input):
            for i in range(begin, end + 1):
                input = funcs[i](input)
            return input

        return do_run

    if isinstance(functions, paddle.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments

    end = -1
    for begin in range(0, segment_size * (segments - 1), segment_size):
        end = begin + segment_size - 1
        args = recompute(
            _run_func(begin, end, functions),
            *args,
            preserve_rng_state=preserve_rng_state,
            **kwargs,
        )
    return _run_func(end + 1, len(functions) - 1, functions)(args)
