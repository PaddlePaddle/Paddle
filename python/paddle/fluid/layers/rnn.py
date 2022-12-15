# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import sys
from functools import partial, reduce
import warnings


import paddle
from paddle.utils import deprecated
from . import nn
from . import tensor
from . import control_flow
from . import utils
from . import sequence_lod
from .utils import *
from .. import core
from ..framework import default_main_program
from ..data_feeder import convert_dtype
from ..layer_helper import LayerHelper
from ..framework import _non_static_mode
from ..param_attr import ParamAttr
from ..data_feeder import check_variable_and_dtype, check_type, check_dtype

from collections.abc import Sequence

__all__ = [
    'dynamic_decode',
]


class ArrayWrapper:
    def __init__(self, x):
        self.array = [x]

    def append(self, x):
        self.array.append(x)
        return self

    def __getitem__(self, item):
        return self.array.__getitem__(item)


def _dynamic_decode_imperative(
    decoder,
    inits=None,
    max_step_num=None,
    output_time_major=False,
    impute_finished=False,
    is_test=False,
    return_length=False,
    **kwargs
):
    def _maybe_copy(state, new_state, step_mask):
        # TODO: use where_op
        state_dtype = state.dtype
        if convert_dtype(state_dtype) in ["bool"]:
            state = tensor.cast(state, dtype="float32")
            new_state = tensor.cast(new_state, dtype="float32")
        if step_mask.dtype != state.dtype:
            step_mask = tensor.cast(step_mask, dtype=state.dtype)
            # otherwise, renamed bool gradients of would be summed up leading
            # to sum(bool) error.
            step_mask.stop_gradient = True
        new_state = paddle.tensor.math._multiply_with_axis(
            state, step_mask, axis=0
        ) - paddle.tensor.math._multiply_with_axis(
            new_state, (step_mask - 1), axis=0
        )
        if convert_dtype(state_dtype) in ["bool"]:
            new_state = tensor.cast(new_state, dtype=state_dtype)
        return new_state

    initial_inputs, initial_states, initial_finished = decoder.initialize(inits)
    inputs, states, finished = (
        initial_inputs,
        initial_states,
        initial_finished,
    )
    cond = paddle.logical_not((paddle.all(initial_finished)))
    sequence_lengths = tensor.cast(paddle.zeros_like(initial_finished), "int64")
    outputs = None

    step_idx = 0
    step_idx_tensor = tensor.fill_constant(
        shape=[1], dtype="int64", value=step_idx
    )
    while cond.numpy():
        (step_outputs, next_states, next_inputs, next_finished) = decoder.step(
            step_idx_tensor, inputs, states, **kwargs
        )
        if not decoder.tracks_own_finished:
            # BeamSearchDecoder would track it own finished, since
            # beams would be reordered and the finished status of each
            # entry might change. Otherwise, perform logical OR which
            # would not change the already finished.
            next_finished = paddle.logical_or(next_finished, finished)
            # To confirm states.finished/finished be consistent with
            # next_finished.
            tensor.assign(next_finished, finished)
            next_sequence_lengths = paddle.add(
                sequence_lengths,
                tensor.cast(
                    paddle.logical_not(finished), sequence_lengths.dtype
                ),
            )
            if impute_finished:  # rectify the states for the finished.
                next_states = map_structure(
                    lambda x, y: _maybe_copy(x, y, finished),
                    states,
                    next_states,
                )
        else:
            warnings.warn(
                "`next_states` has no `lengths` attribute, the returned `sequence_lengths` would be all zeros."
            ) if not hasattr(next_states, "lengths") else None
            next_sequence_lengths = getattr(
                next_states, "lengths", sequence_lengths
            )

        outputs = (
            map_structure(lambda x: ArrayWrapper(x), step_outputs)
            if step_idx == 0
            else map_structure(
                lambda x, x_array: x_array.append(x), step_outputs, outputs
            )
        )
        inputs, states, finished, sequence_lengths = (
            next_inputs,
            next_states,
            next_finished,
            next_sequence_lengths,
        )

        paddle.increment(x=step_idx_tensor, value=1.0)
        step_idx += 1

        cond = paddle.logical_not(paddle.all(finished))
        if max_step_num is not None and step_idx > max_step_num:
            break

    final_outputs = map_structure(
        lambda x: paddle.stack(x.array, axis=0), outputs
    )
    final_states = states

    try:
        final_outputs, final_states = decoder.finalize(
            final_outputs, final_states, sequence_lengths
        )
    except NotImplementedError:
        pass

    if not output_time_major:
        final_outputs = map_structure(
            lambda x: paddle.transpose(
                x, [1, 0] + list(range(2, len(x.shape)))
            ),
            final_outputs,
        )

    return (
        (final_outputs, final_states, sequence_lengths)
        if return_length
        else (final_outputs, final_states)
    )


def _dynamic_decode_declarative(
    decoder,
    inits=None,
    max_step_num=None,
    output_time_major=False,
    impute_finished=False,
    is_test=False,
    return_length=False,
    **kwargs
):
    initial_inputs, initial_states, initial_finished = decoder.initialize(inits)
    global_inputs, global_states, global_finished = (
        initial_inputs,
        initial_states,
        initial_finished,
    )
    global_finished.stop_gradient = True
    step_idx = tensor.fill_constant(shape=[1], dtype="int64", value=0)

    cond = paddle.logical_not((paddle.all(initial_finished)))
    if max_step_num is not None:
        max_step_num = tensor.fill_constant(
            shape=[1], dtype="int64", value=max_step_num
        )
    while_op = paddle.static.nn.control_flow.While(cond, is_test=is_test)

    sequence_lengths = tensor.cast(paddle.zeros_like(initial_finished), "int64")
    sequence_lengths.stop_gradient = True

    if is_test:
        # for test, reuse inputs and states variables to save memory
        inputs = map_structure(lambda x: x, initial_inputs)
        states = map_structure(lambda x: x, initial_states)
    else:
        # inputs and states of all steps must be saved for backward and training
        inputs_arrays = map_structure(
            lambda x: paddle.tensor.array_write(x, step_idx), initial_inputs
        )
        states_arrays = map_structure(
            lambda x: paddle.tensor.array_write(x, step_idx), initial_states
        )

    def _maybe_copy(state, new_state, step_mask):
        # TODO: use where_op
        state_dtype = state.dtype
        if convert_dtype(state_dtype) in ["bool"]:
            state = tensor.cast(state, dtype="float32")
            new_state = tensor.cast(new_state, dtype="float32")
        if step_mask.dtype != state.dtype:
            step_mask = tensor.cast(step_mask, dtype=state.dtype)
            # otherwise, renamed bool gradients of would be summed up leading
            # to sum(bool) error.
            step_mask.stop_gradient = True
        new_state = paddle.tensor.math._multiply_with_axis(
            state, step_mask, axis=0
        ) - paddle.tensor.math._multiply_with_axis(
            new_state, (step_mask - 1), axis=0
        )
        if convert_dtype(state_dtype) in ["bool"]:
            new_state = tensor.cast(new_state, dtype=state_dtype)
        return new_state

    def _transpose_batch_time(x):
        return paddle.transpose(x, [1, 0] + list(range(2, len(x.shape))))

    def _create_array_out_of_while(dtype):
        current_block_idx = default_main_program().current_block_idx
        default_main_program().current_block_idx = (
            default_main_program().current_block().parent_idx
        )
        tensor_array = paddle.tensor.create_array(dtype)
        default_main_program().current_block_idx = current_block_idx
        return tensor_array

    # While
    with while_op.block():
        if not is_test:
            inputs = map_structure(
                lambda array: paddle.tensor.array_read(array, step_idx),
                inputs_arrays,
            )
            states = map_structure(
                lambda array: paddle.tensor.array_read(array, step_idx),
                states_arrays,
            )
        (outputs, next_states, next_inputs, next_finished) = decoder.step(
            step_idx, inputs, states, **kwargs
        )
        if not decoder.tracks_own_finished:
            # BeamSearchDecoder would track it own finished, since beams would
            # be reordered and the finished status of each entry might change.
            # Otherwise, perform logical OR which would not change the already
            # finished.
            next_finished = paddle.logical_or(next_finished, global_finished)
            next_sequence_lengths = paddle.add(
                sequence_lengths,
                tensor.cast(
                    paddle.logical_not(global_finished),
                    sequence_lengths.dtype,
                ),
            )
            if impute_finished:  # rectify the states for the finished.
                next_states = map_structure(
                    lambda x, y: _maybe_copy(x, y, global_finished),
                    states,
                    next_states,
                )
        else:
            warnings.warn(
                "`next_states` has no `lengths` attribute, the returned `sequence_lengths` would be all zeros."
            ) if not hasattr(next_states, "lengths") else None
            next_sequence_lengths = getattr(
                next_states, "lengths", sequence_lengths
            )

        # create tensor array in global block after dtype[s] of outputs can be got
        outputs_arrays = map_structure(
            lambda x: _create_array_out_of_while(x.dtype), outputs
        )

        map_structure(
            lambda x, x_array: paddle.tensor.array_write(
                x, i=step_idx, array=x_array
            ),
            outputs,
            outputs_arrays,
        )

        paddle.increment(x=step_idx, value=1.0)
        # update the global_finished first, since it might be also in states of
        # decoder, which otherwise would write a stale finished status to array
        tensor.assign(next_finished, global_finished)
        tensor.assign(next_sequence_lengths, sequence_lengths)
        if is_test:
            map_structure(tensor.assign, next_inputs, global_inputs)
            map_structure(tensor.assign, next_states, global_states)
        else:
            map_structure(
                lambda x, x_array: paddle.tensor.array_write(
                    x, i=step_idx, array=x_array
                ),
                next_inputs,
                inputs_arrays,
            )
            map_structure(
                lambda x, x_array: paddle.tensor.array_write(
                    x, i=step_idx, array=x_array
                ),
                next_states,
                states_arrays,
            )
        if max_step_num is not None:
            paddle.logical_and(
                paddle.logical_not(paddle.all(global_finished)),
                paddle.less_equal(step_idx, max_step_num),
                cond,
            )
        else:
            paddle.logical_not(paddle.all(global_finished), cond)

    final_outputs = map_structure(
        lambda array: tensor.tensor_array_to_tensor(
            array, axis=0, use_stack=True
        )[0],
        outputs_arrays,
    )
    if is_test:
        final_states = global_states
    else:
        final_states = map_structure(
            lambda array: paddle.tensor.array_read(array, step_idx),
            states_arrays,
        )

    try:
        final_outputs, final_states = decoder.finalize(
            final_outputs, final_states, sequence_lengths
        )
    except NotImplementedError:
        pass

    if not output_time_major:
        final_outputs = map_structure(_transpose_batch_time, final_outputs)

    return (
        (final_outputs, final_states, sequence_lengths)
        if return_length
        else (final_outputs, final_states)
    )


def dynamic_decode(
    decoder,
    inits=None,
    max_step_num=None,
    output_time_major=False,
    impute_finished=False,
    is_test=False,
    return_length=False,
    **kwargs
):
    r"""
    Dynamic decoding performs :code:`decoder.step()` repeatedly until the returned
    Tensor indicating finished status contains all True values or the number of
    decoding step reaches to :attr:`max_step_num`.

    :code:`decoder.initialize()` would be called once before the decoding loop.
    If the `decoder` has implemented `finalize` method, :code:`decoder.finalize()`
    would be called once after the decoding loop.

    Parameters:
        decoder(Decoder): An instance of `Decoder`.
        inits(object, optional): Argument passed to `decoder.initialize`.
            Default `None`.
        max_step_num(int, optional): The maximum number of steps. If not provided,
            decode until the decoder is fully done, or in other words, the returned
            Tensor by :code:`decoder.step()` indicating finished status contains
            all True. Default `None`.
        output_time_major(bool, optional): Indicate the data layout of Tensor included
            in the final outputs(the first returned value of this method). If
            attr:`False`, the data layout would be batch major with shape
            `[batch_size, seq_len, ...]`.  If attr:`True`, the data layout would
            be time major with shape `[seq_len, batch_size, ...]`. Default: `False`.
        impute_finished(bool, optional): If `True` and `decoder.tracks_own_finished`
            is False, then states get copied through for batch entries which are
            marked as finished, which differs with the unfinished using the new states
            returned by :code:`decoder.step()` and ensures that the final states have
            the correct values. Otherwise, states wouldn't be copied through when
            finished. If the returned `final_states` is needed, it should be set as
            True, which causes some slowdown. Default `False`.
        is_test(bool, optional): A flag indicating whether to use test mode. In
            test mode, it is more memory saving. Default `False`.
        return_length(bool, optional):  A flag indicating whether to return an
            extra Tensor variable in the output tuple, which stores the actual
            lengths of all decoded sequences. Default `False`.
        **kwargs: Additional keyword arguments. Arguments passed to `decoder.step`.

    Returns:

        - final_outputs (Tensor, nested structure of Tensor), each Tensor in :code:`final_outputs` is the stacked of all decoding steps' outputs, which might be revised
            by :code:`decoder.finalize()` if the decoder has implemented finalize.
            And :code:`final_outputs` has the same structure and data types as the :code:`outputs`
            returned by :code:`decoder.step()`

        - final_states (Tensor, nested structure of Tensor), :code:`final_states` is the counterpart at last time step of initial states \
            returned by :code:`decoder.initialize()` , thus has the same structure
            with it and has tensors with same shapes and data types.

        - sequence_lengths (Tensor), stores the actual lengths of all decoded sequences.
            sequence_lengths is provided only if :code:`return_length` is True.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.nn import BeamSearchDecoder, dynamic_decode
            from paddle.nn import GRUCell, Linear, Embedding
            trg_embeder = Embedding(100, 32)
            output_layer = Linear(32, 32)
            decoder_cell = GRUCell(input_size=32, hidden_size=32)
            decoder = BeamSearchDecoder(decoder_cell,
                                        start_token=0,
                                        end_token=1,
                                        beam_size=4,
                                        embedding_fn=trg_embeder,
                                        output_fn=output_layer)
            encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
            outputs = dynamic_decode(decoder=decoder,
                                    inits=decoder_cell.get_initial_states(encoder_output),
                                    max_step_num=10)
    """
    if _non_static_mode():
        return _dynamic_decode_imperative(
            decoder,
            inits,
            max_step_num,
            output_time_major,
            impute_finished,
            is_test,
            return_length,
            **kwargs
        )
    else:
        return _dynamic_decode_declarative(
            decoder,
            inits,
            max_step_num,
            output_time_major,
            impute_finished,
            is_test,
            return_length,
            **kwargs
        )
