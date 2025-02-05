# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import warnings
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, overload

import numpy as np

import paddle
from paddle.common_ops_import import default_main_program
from paddle.framework import in_dynamic_mode

from ..base.data_feeder import convert_dtype

if TYPE_CHECKING:
    from collections.abc import Callable

    from paddle import Tensor
    from paddle.nn import Embedding, Layer, RNNCellBase


__all__ = []


class ArrayWrapper:
    def __init__(self, x):
        self.array = [x]

    def append(self, x):
        self.array.append(x)
        return self

    def __getitem__(self, item):
        return self.array.__getitem__(item)


class Decoder:
    """
    Decoder is the base class for any decoder instance used in `dynamic_decode`.
    It provides interface for output generation for one time step, which can be
    used to generate sequences.

    The key abstraction provided by Decoder is:

    1. :code:`(initial_input, initial_state, finished) = initialize(inits)` ,
    which generates the input and state for the first decoding step, and gives the
    initial status telling whether each sequence in the batch is finished.
    It would be called once before the decoding iterations.

    2. :code:`(output, next_state, next_input, finished) = step(time, input, state)` ,
    which transforms the input and state to the output and new state, generates
    input for the next decoding step, and emits the flag indicating finished status.
    It is the main part for each decoding iteration.

    3. :code:`(final_outputs, final_state) = finalize(outputs, final_state, sequence_lengths)` ,
    which revises the outputs(stack of all time steps' output) and final state(state from the
    last decoding step) to get the counterpart for special usage.
    Not necessary to be implemented if no need to revise the stacked outputs and
    state from the last decoding step. If implemented, it would be called after
    the decoding iterations.

    Decoder is more general compared to RNNCell, since the returned `next_input`
    and `finished` make it can determine the input and when to finish by itself
    when used in dynamic decoding. Decoder always wraps a RNNCell instance though
    not necessary.
    """

    def initialize(self, inits):
        r"""
        Called once before the decoding iterations.

        Parameters:
            inits: Argument provided by the caller.

        Returns:
            tuple: A tuple( :code:`(initial_inputs, initial_states, finished)` ). \
                `initial_inputs` and `initial_states` both are a (possibly nested \
                structure of) tensor variable[s], and `finished` is a tensor with \
                bool data type.
        """
        raise NotImplementedError

    def step(self, time, inputs, states, **kwargs):
        r"""
        Called per step of decoding.

        Parameters:
            time(Tensor): A Tensor with shape :math:`[1]` provided by the caller.
                The data type is int64.
            inputs(Tensor): A (possibly nested structure of) tensor variable[s].
            states(Tensor): A (possibly nested structure of) tensor variable[s].
            **kwargs: Additional keyword arguments, provided by the caller.

        Returns:
            tuple: A tuple( :code:(outputs, next_states, next_inputs, finished)` ). \
                `next_inputs` and `next_states` both are a (possibly nested \
                structure of) tensor variable[s], and the structure, shape and \
                data type must be same as the counterpart from input arguments. \
                `outputs` is a (possibly nested structure of) tensor variable[s]. \
                `finished` is a Tensor with bool data type.
        """
        raise NotImplementedError

    def finalize(self, outputs, final_states, sequence_lengths):
        r"""
        Called once after the decoding iterations if implemented.

        Parameters:
            outputs(Tensor): A (possibly nested structure of) tensor variable[s].
                The structure and data type is same as `output_dtype`.
                The tensor stacks all time steps' output thus has shape
                :math:`[time\_step, batch\_size, ...]` , which is done by the caller.
            final_states(Tensor): A (possibly nested structure of) tensor variable[s].
                It is the `next_states` returned by `decoder.step` at last decoding step,
                thus has the same structure, shape and data type with states at any time
                step.

        Returns:
            tuple: A tuple( :code:`(final_outputs, final_states)` ). \
                `final_outputs` and `final_states` both are a (possibly nested \
                structure of) tensor variable[s].
        """
        raise NotImplementedError

    @property
    def tracks_own_finished(self):
        """
        Describes whether the Decoder keeps track of finished states by itself.

        `decoder.step()` would emit a bool `finished` value at each decoding
        step. The emitted `finished` can be used to determine whether every
        batch entries is finished directly, or it can be combined with the
        finished tracker keeped in `dynamic_decode` by performing a logical OR
        to take the already finished into account.

        If `False`, the latter would be took when performing `dynamic_decode`,
        which is the default. Otherwise, the former would be took, which uses
        the finished value emitted by the decoder as all batch entry finished
        status directly, and it is the case when batch entries might be
        reordered such as beams in BeamSearchDecoder.

        Returns:
            bool: A python bool `False`.
        """
        return False


class BeamSearchDecoder(Decoder):
    """
    Decoder with beam search decoding strategy. It wraps a cell to get probabilities,
    and follows a beam search step to calculate scores and select candidate
    token ids for each decoding step.

    Please refer to `Beam search <https://en.wikipedia.org/wiki/Beam_search>`_
    for more details.

    Note:
        When decoding with beam search, the `inputs` and `states` of cell
        would be tiled to `beam_size` (unsqueeze and tile), resulting to shapes like
        `[batch_size * beam_size, ...]` , which is built into `BeamSearchDecoder` and
        done automatically. Thus any other tensor with shape `[batch_size, ...]` used
        in `cell.call` needs to be tiled manually first, which can be completed by using
        :code:`BeamSearchDecoder.tile_beam_merge_with_batch` . The most common case
        for this is the encoder output in attention mechanism.

    Parameters:
        cell (RNNCellBase): An instance of `RNNCellBase` or object with the same interface.
        start_token (int): The start token id.
        end_token (int): The end token id.
        beam_size (int): The beam width used in beam search.
        embedding_fn (optional): A callable to apply to selected candidate ids.
            Mostly it is an embedding layer to transform ids to embeddings,
            and the returned value acts as the `input` argument for `cell.call`.
            If not provided, the id to embedding transformation must be built into
            `cell.call`. Default None.
        output_fn (optional): A callable to apply to the cell's output prior to
            calculate scores and select candidate token ids. Default None.

    Returns:
        BeamSearchDecoder: An instance of decoder which can be used in \
            `paddle.nn.dynamic_decode` to implement decoding.

    Examples:

        .. code-block:: python

            >>> import numpy as np
            >>> import paddle
            >>> from paddle.nn import BeamSearchDecoder, dynamic_decode
            >>> from paddle.nn import GRUCell, Linear, Embedding
            >>> trg_embeder = Embedding(100, 32)
            >>> output_layer = Linear(32, 32)
            >>> decoder_cell = GRUCell(input_size=32, hidden_size=32)
            >>> decoder = BeamSearchDecoder(decoder_cell,
            ...                             start_token=0,
            ...                             end_token=1,
            ...                             beam_size=4,
            ...                             embedding_fn=trg_embeder,
            ...                             output_fn=output_layer)
            ...
    """

    cell: RNNCellBase
    embedding_fn: Embedding | Callable[..., Any] | None
    output_fn: Layer | Callable[..., Any] | None
    start_token: int
    end_token: int
    beam_size: int

    def __init__(
        self,
        cell: RNNCellBase,
        start_token: int,
        end_token: int,
        beam_size: int,
        embedding_fn: Embedding | Callable[..., Any] | None = None,
        output_fn: Layer | Callable[..., Any] | None = None,
    ) -> None:
        """
        Constructor of BeamSearchDecoder.

        Parameters:
            cell(RNNCellBase): An instance of `RNNCellBase` or object with the same interface.
            start_token(int): The start token id.
            end_token(int): The end token id.
            beam_size(int): The beam width used in beam search.
            embedding_fn(optional): A callable to apply to selected candidate ids.
                Mostly it is an embedding layer to transform ids to embeddings,
                and the returned value acts as the `input` argument for `cell.call`.
                If not provided, the id to embedding transformation must be built into
                `cell.call`. Default None.
            output_fn(optional): A callable to apply to the cell's output prior to
                calculate scores and select candidate token ids. Default None.
        """
        self.cell = cell
        self.embedding_fn = embedding_fn
        self.output_fn = output_fn
        self.start_token = start_token
        self.end_token = end_token
        self.beam_size = beam_size

    @staticmethod
    def tile_beam_merge_with_batch(x: Tensor, beam_size: int) -> Tensor:
        r"""
        Tile the batch dimension of a tensor. Specifically, this function takes
        a tensor t shaped `[batch_size, s0, s1, ...]` composed of minibatch
        entries `t[0], ..., t[batch_size - 1]` and tiles it to have a shape
        `[batch_size * beam_size, s0, s1, ...]` composed of minibatch entries
        `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated
        `beam_size` times.

        Parameters:
            x(Tensor): A tensor with shape `[batch_size, ...]`. The data type
                should be float32, float64, int32, int64 or bool.
            beam_size(int): The beam width used in beam search.

        Returns:
            Tensor: A tensor with shape `[batch_size * beam_size, ...]`, whose \
                data type is same as `x`.
        """
        x = paddle.unsqueeze(x, [1])  # [batch_size, 1, ...]
        expand_times = [1] * len(x.shape)
        expand_times[1] = beam_size
        x = paddle.tile(x, expand_times)  # [batch_size, beam_size, ...]
        x = paddle.transpose(
            x, [*list(range(2, len(x.shape))), 0, 1]
        )  # [..., batch_size, beam_size]
        # use 0 to copy to avoid wrong shape
        x = paddle.reshape(
            x, shape=[0] * (len(x.shape) - 2) + [-1]
        )  # [..., batch_size * beam_size]
        x = paddle.transpose(
            x, [len(x.shape) - 1, *list(range(0, len(x.shape) - 1))]
        )  # [batch_size * beam_size, ...]
        return x

    def _split_batch_beams(self, x):
        r"""
        Reshape a tensor with shape `[batch_size * beam_size, ...]` to a new
        tensor with shape `[batch_size, beam_size, ...]`.

        Parameters:
            x(Tensor): A tensor with shape `[batch_size * beam_size, ...]`. The
                data type should be float32, float64, int32, int64 or bool.

        Returns:
            Tensor: A tensor with shape `[batch_size, beam_size, ...]`, whose \
                data type is same as `x`.
        """
        # TODO: avoid fake shape in compile-time like tile_beam_merge_with_batch
        return paddle.reshape(x, shape=[-1, self.beam_size, *list(x.shape[1:])])

    def _merge_batch_beams(self, x):
        r"""
        Reshape a tensor with shape `[batch_size, beam_size, ...]` to a new
        tensor with shape `[batch_size * beam_size, ...]`.

        Parameters:
            x(Tensor): A tensor with shape `[batch_size, beam_size, ...]`. The
                data type should be float32, float64, int32, int64 or bool.

        Returns:
            Tensor: A tensor with shape `[batch_size * beam_size, ...]`, whose \
                data type is same as `x`.
        """
        # TODO: avoid fake shape in compile-time like tile_beam_merge_with_batch
        return paddle.reshape(x, shape=[-1, *list(x.shape[2:])])

    def _expand_to_beam_size(self, x):
        r"""
        This function takes a tensor t shaped `[batch_size, s0, s1, ...]` composed
        of minibatch entries `t[0], ..., t[batch_size - 1]` and tiles it to have a
        shape `[batch_size, beam_size, s0, s1, ...]` composed of minibatch entries
        `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated
        `beam_size` times.

        Parameters:
            x(Tensor): A tensor with shape `[batch_size, ...]`, The data type
                should be float32, float64, int32, int64 or bool.

        Returns:
            Tensor: A tensor with shape `[batch_size, beam_size, ...]`, whose \
                data type is same as `x`.
        """
        x = paddle.unsqueeze(x, [1])
        expand_times = [1] * len(x.shape)
        expand_times[1] = self.beam_size
        x = paddle.tile(x, expand_times)
        return x

    def _mask_probs(self, probs, finished):
        r"""
        Mask log probabilities. It forces finished beams to allocate all probability
        mass to eos and unfinished beams to remain unchanged.

        Parameters:
            probs(Tensor): A tensor with shape `[batch_size, beam_size, vocab_size]`,
                representing the log probabilities. Its data type should be float32 or float64.
            finished(Tensor): A tensor with shape `[batch_size, beam_size]`,
                representing the finished status for all beams. Its data type
                should be bool.

        Returns:
            Tensor: A tensor with the same shape and data type as `x`, \
                where unfinished beams stay unchanged and finished beams are \
                replaced with a tensor with all probability on the EOS token.
        """
        # TODO: use where_op
        finished = paddle.cast(finished, dtype=probs.dtype)

        probs = paddle.multiply(
            paddle.tile(
                paddle.unsqueeze(finished, [2]), [1, 1, self.vocab_size]
            ),
            self.noend_mask_tensor,
        ) - paddle.multiply(probs, (finished - 1).unsqueeze([2]))

        return probs

    def _gather(self, x, indices, batch_size):
        r"""
        Gather from the tensor `x` using `indices`.

        Parameters:
            x(Tensor): A tensor with shape `[batch_size, beam_size, ...]`.
            indices(Tensor): A `int64` tensor with shape `[batch_size, beam_size]`,
                representing the indices that we use to gather.
            batch_size(Tensor): A tensor with shape `[1]`. Its data type should
                be int32 or int64.

        Returns:
            Tensor: A tensor with the same shape and data type as `x`, \
                representing the gathered tensor.
        """
        # TODO: compatibility of int32 and int64
        batch_size = (
            paddle.cast(batch_size, indices.dtype)
            if batch_size.dtype != indices.dtype
            else batch_size
        )
        batch_size.stop_gradient = True  # TODO: remove this
        batch_pos = paddle.tile(
            paddle.unsqueeze(
                paddle.arange(0, batch_size, 1, dtype=indices.dtype), [1]
            ),
            [1, self.beam_size],
        )
        topk_coordinates = paddle.stack([batch_pos, indices], axis=2)
        topk_coordinates.stop_gradient = True
        return paddle.gather_nd(x, topk_coordinates)

    class OutputWrapper(NamedTuple):
        """
        The structure for the returned value `outputs` of `decoder.step`.
        A namedtuple includes scores, predicted_ids, parent_ids as fields.
        """

        scores: Tensor
        predicted_ids: Tensor
        parent_ids: Tensor

    class StateWrapper(NamedTuple):
        """
        The structure for the argument `states` of `decoder.step`.
        A namedtuple includes cell_states, log_probs, finished, lengths as fields.
        """

        cell_states: Tensor
        log_probs: Tensor
        finished: Tensor
        lengths: Tensor

    def initialize(
        self, initial_cell_states: Tensor
    ) -> tuple[Tensor, StateWrapper, Tensor]:
        r"""
        Initialize the BeamSearchDecoder.

        Parameters:
            initial_cell_states(Tensor): A (possibly nested structure of)
                tensor variable[s]. An argument provided by the caller.

        Returns:
            tuple: A tuple( :code:`(initial_inputs, initial_states, finished)` ). \
                `initial_inputs` is a tensor t filled by `start_token` with shape \
                `[batch_size, beam_size]` when `embedding_fn` is None, or the \
                returned value of `embedding_fn(t)` when `embedding_fn` is provided. \
                `initial_states` is a nested structure(namedtuple including cell_states, \
                log_probs, finished, lengths as fields) of tensor variables, where \
                `log_probs, finished, lengths` all has a tensor value shaped \
                `[batch_size, beam_size]` with data type `float32, bool, int64`. \
                cell_states has a value with the same structure as the input \
                argument `initial_cell_states` but with tiled shape `[batch_size, beam_size, ...]`. \
                `finished` is a `bool` tensor filled by False with shape `[batch_size, beam_size]`.
        """
        self.kinf = 1e9
        state = paddle.utils.flatten(initial_cell_states)[0]
        self.batch_size = paddle.shape(state)[0]

        self.start_token_tensor = paddle.full(
            shape=[1], dtype="int64", fill_value=self.start_token
        )
        self.end_token_tensor = paddle.full(
            shape=[1], dtype="int64", fill_value=self.end_token
        )

        init_cell_states = paddle.utils.map_structure(
            self._expand_to_beam_size, initial_cell_states
        )
        init_inputs = paddle.full(
            shape=[self.batch_size, self.beam_size],
            fill_value=self.start_token_tensor,
            dtype=self.start_token_tensor.dtype,
        )
        log_probs = paddle.tile(
            paddle.assign(
                np.array(
                    [[0.0] + [-self.kinf] * (self.beam_size - 1)],
                    dtype="float32",
                )
            ),
            [self.batch_size, 1],
        )
        if paddle.get_default_dtype() == "float64":
            log_probs = paddle.cast(log_probs, "float64")

        init_finished = paddle.full(
            shape=[paddle.shape(state)[0], self.beam_size],
            fill_value=False,
            dtype="bool",
        )

        init_lengths = paddle.zeros_like(init_inputs)
        init_inputs = (
            self.embedding_fn(init_inputs) if self.embedding_fn else init_inputs
        )
        return (
            init_inputs,
            self.StateWrapper(
                init_cell_states, log_probs, init_finished, init_lengths
            ),
            init_finished,
        )

    def _beam_search_step(self, time, logits, next_cell_states, beam_state):
        r"""
        Calculate scores and select candidate token ids.

        Parameters:
            time(Tensor): An `int64` tensor with shape `[1]` provided by the caller,
                representing the current time step number of decoding.
            logits(Tensor): A tensor with shape `[batch_size, beam_size, vocab_size]`,
                representing the logits at the current time step. Its data type is float32.
            next_cell_states(Tensor): A (possibly nested structure of) tensor variable[s].
                It has the same structure, shape and data type as the `cell_states` of
                `initial_states` returned by `initialize()`. It represents the next state
                from the cell.
            beam_state(Tensor): A structure of tensor variables.
                It is same as the `initial_states` returned by `initialize()` for
                the first decoding step and `beam_search_state` returned by
                `step()` for the others.

        Returns:
            tuple: A tuple( :code:`(beam_search_output, beam_search_state)` ). \
                `beam_search_output` is a namedtuple(including scores, predicted_ids, \
                parent_ids as fields) of tensor variables, where \
                `scores, predicted_ids, parent_ids` all has a tensor value shaped \
                `[batch_size, beam_size]` with data type `float32, int64, int64`.
                `beam_search_state` has the same structure, shape and data type \
                as the input argument `beam_state`.

        """
        self.vocab_size = logits.shape[-1]
        self.vocab_size_tensor = paddle.full(
            shape=[1], dtype="int64", fill_value=self.vocab_size
        )
        noend_array = [-self.kinf] * self.vocab_size
        noend_array[self.end_token] = 0

        self.noend_mask_tensor = paddle.assign(np.array(noend_array, "float32"))
        if paddle.get_default_dtype() == "float64":
            self.noend_mask_tensor = paddle.cast(
                self.noend_mask_tensor, "float64"
            )

        step_log_probs = paddle.log(paddle.nn.functional.softmax(logits))
        step_log_probs = self._mask_probs(step_log_probs, beam_state.finished)

        log_probs = paddle.add(
            step_log_probs, beam_state.log_probs.unsqueeze([2])
        )

        # TODO: length penalty
        scores = log_probs
        scores = paddle.reshape(scores, [-1, self.beam_size * self.vocab_size])
        # TODO: add grad for topk then this beam search can be used to train
        topk_scores, topk_indices = paddle.topk(x=scores, k=self.beam_size)
        beam_indices = paddle.floor_divide(topk_indices, self.vocab_size_tensor)
        token_indices = paddle.remainder(topk_indices, self.vocab_size_tensor)
        next_log_probs = self._gather(
            paddle.reshape(log_probs, [-1, self.beam_size * self.vocab_size]),
            topk_indices,
            self.batch_size,
        )
        next_cell_states = paddle.utils.map_structure(
            lambda x: self._gather(x, beam_indices, self.batch_size),
            next_cell_states,
        )
        next_finished = self._gather(
            beam_state.finished, beam_indices, self.batch_size
        )
        next_lengths = self._gather(
            beam_state.lengths, beam_indices, self.batch_size
        )
        next_lengths = next_lengths + paddle.cast(
            paddle.logical_not(next_finished), beam_state.lengths.dtype
        )
        next_finished = paddle.logical_or(
            next_finished,
            paddle.equal(token_indices, self.end_token_tensor),
        )

        beam_search_output = self.OutputWrapper(
            topk_scores, token_indices, beam_indices
        )
        beam_search_state = self.StateWrapper(
            next_cell_states, next_log_probs, next_finished, next_lengths
        )
        return beam_search_output, beam_search_state

    def step(
        self, time: Tensor, inputs: Tensor, states: Tensor, **kwargs: Any
    ) -> tuple[OutputWrapper, StateWrapper, Tensor, Tensor]:
        r"""
        Perform a beam search decoding step, which uses `cell` to get probabilities,
        and follows a beam search step to calculate scores and select candidate
        token ids.

        Parameters:
            time(Tensor): An `int64` tensor with shape `[1]` provided by the caller,
                representing the current time step number of decoding.
            inputs(Tensor): A tensor variable. It is same as `initial_inputs`
                returned by `initialize()` for the first decoding step and
                `next_inputs` returned by `step()` for the others.
            states(Tensor): A structure of tensor variables.
                It is same as the `initial_states` returned by `initialize()` for
                the first decoding step and `beam_search_state` returned by
                `step()` for the others.
            **kwargs: Additional keyword arguments, provided by the caller.

        Returns:
            tuple: A tuple( :code:`(beam_search_output, beam_search_state, next_inputs, finished)` ). \
                `beam_search_state` and `next_inputs` have the same structure, \
                shape and data type as the input arguments `states` and `inputs` separately. \
                `beam_search_output` is a namedtuple(including scores, predicted_ids, \
                parent_ids as fields) of tensor variables, where \
                `scores, predicted_ids, parent_ids` all has a tensor value shaped \
                `[batch_size, beam_size]` with data type `float32, int64, int64`. \
                `finished` is a `bool` tensor with shape `[batch_size, beam_size]`.
        """
        inputs = paddle.utils.map_structure(self._merge_batch_beams, inputs)
        cell_states = paddle.utils.map_structure(
            self._merge_batch_beams, states.cell_states
        )
        cell_outputs, next_cell_states = self.cell(
            inputs, cell_states, **kwargs
        )
        cell_outputs = paddle.utils.map_structure(
            self._split_batch_beams, cell_outputs
        )
        next_cell_states = paddle.utils.map_structure(
            self._split_batch_beams, next_cell_states
        )

        if self.output_fn is not None:
            cell_outputs = self.output_fn(cell_outputs)

        beam_search_output, beam_search_state = self._beam_search_step(
            time=time,
            logits=cell_outputs,
            next_cell_states=next_cell_states,
            beam_state=states,
        )
        finished = beam_search_state.finished
        sample_ids = beam_search_output.predicted_ids
        sample_ids.stop_gradient = True
        next_inputs = (
            self.embedding_fn(sample_ids) if self.embedding_fn else sample_ids
        )

        return (beam_search_output, beam_search_state, next_inputs, finished)

    def finalize(
        self, outputs: Tensor, final_states: Tensor, sequence_lengths: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""
        Use `gather_tree` to backtrace along the beam search tree and construct
        the full predicted sequences.

        Parameters:
            outputs(Tensor): A structure(namedtuple) of tensor variables,
                The structure and data type is same as `output_dtype`.
                The tensor stacks all time steps' output thus has shape
                `[time_step, batch_size, ...]`, which is done by the caller.
            final_states(Tensor): A structure(namedtuple) of tensor variables.
                It is the `next_states` returned by `decoder.step` at last
                decoding step, thus has the same structure, shape and data type
                with states at any time step.
            sequence_lengths(Tensor): An `int64` tensor shaped `[batch_size, beam_size]`.
                It contains sequence lengths for each beam determined during
                decoding.

        Returns:
            tuple: A tuple( :code:`(predicted_ids, final_states)` ). \
                `predicted_ids` is an `int64` tensor shaped \
                `[time_step, batch_size, beam_size]`. `final_states` is the same \
                as the input argument `final_states`.
        """
        predicted_ids = paddle.nn.functional.gather_tree(
            outputs.predicted_ids, outputs.parent_ids
        )
        predicted_ids = paddle.cast(
            predicted_ids, dtype=outputs.predicted_ids.dtype
        )
        # TODO: use FinalBeamSearchDecoderOutput as output
        return predicted_ids, final_states

    @property
    def tracks_own_finished(self) -> Literal[True]:
        """
        BeamSearchDecoder reorders its beams and their finished state. Thus it
        conflicts with `dynamic_decode` function's tracking of finished states.
        Setting this property to true to avoid early stopping of decoding due
        to mismanagement of the finished state.

        Returns:
            bool: A python bool `True`.
        """
        return True


def _dynamic_decode_imperative(
    decoder,
    inits=None,
    max_step_num=None,
    output_time_major=False,
    impute_finished=False,
    is_test=False,
    return_length=False,
    **kwargs,
):
    def _maybe_copy(state, new_state, step_mask):
        # TODO: use where_op
        state_dtype = state.dtype
        if convert_dtype(state_dtype) in ["bool"]:
            state = paddle.cast(state, dtype="float32")
            new_state = paddle.cast(new_state, dtype="float32")
        if step_mask.dtype != state.dtype:
            step_mask = paddle.cast(step_mask, dtype=state.dtype)
            # otherwise, renamed bool gradients of would be summed up leading
            # to sum(bool) error.
        step_mask = step_mask.unsqueeze([1])
        step_mask.stop_gradient = True
        new_state = paddle.multiply(state, step_mask) - paddle.multiply(
            new_state, (step_mask - 1)
        )
        if convert_dtype(state_dtype) in ["bool"]:
            new_state = paddle.cast(new_state, dtype=state_dtype)
        return new_state

    initial_inputs, initial_states, initial_finished = decoder.initialize(inits)
    inputs, states, finished = (
        initial_inputs,
        initial_states,
        initial_finished,
    )
    cond = paddle.logical_not(paddle.all(initial_finished))
    sequence_lengths = paddle.cast(paddle.zeros_like(initial_finished), "int64")
    outputs = None

    step_idx = 0
    step_idx_tensor = paddle.full(shape=[1], fill_value=step_idx, dtype="int64")
    while np.array(cond).item():
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
            paddle.assign(next_finished, finished)
            next_sequence_lengths = paddle.add(
                sequence_lengths,
                paddle.cast(
                    paddle.logical_not(finished), sequence_lengths.dtype
                ),
            )
            if impute_finished:  # rectify the states for the finished.
                next_states = paddle.utils.map_structure(
                    lambda x, y: _maybe_copy(x, y, finished),
                    states,
                    next_states,
                )
        else:
            (
                warnings.warn(
                    "`next_states` has no `lengths` attribute, the returned `sequence_lengths` would be all zeros."
                )
                if not hasattr(next_states, "lengths")
                else None
            )
            next_sequence_lengths = getattr(
                next_states, "lengths", sequence_lengths
            )

        outputs = (
            paddle.utils.map_structure(lambda x: ArrayWrapper(x), step_outputs)
            if step_idx == 0
            else paddle.utils.map_structure(
                lambda x, x_array: x_array.append(x), step_outputs, outputs
            )
        )
        inputs, states, finished, sequence_lengths = (
            next_inputs,
            next_states,
            next_finished,
            next_sequence_lengths,
        )

        step_idx_tensor = paddle.increment(x=step_idx_tensor, value=1.0)
        step_idx += 1

        cond = paddle.logical_not(paddle.all(finished))
        if max_step_num is not None and step_idx > max_step_num:
            break

    final_outputs = paddle.utils.map_structure(
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
        final_outputs = paddle.utils.map_structure(
            lambda x: paddle.transpose(
                x, [1, 0, *list(range(2, len(x.shape)))]
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
    **kwargs,
):
    initial_inputs, initial_states, initial_finished = decoder.initialize(inits)
    global_inputs, global_states, global_finished = (
        initial_inputs,
        initial_states,
        initial_finished,
    )
    global_finished.stop_gradient = True
    step_idx = paddle.full(shape=[1], fill_value=0, dtype="int64")

    cond = paddle.logical_not(paddle.all(initial_finished))
    if max_step_num is not None:
        max_step_num = paddle.full(
            shape=[1], fill_value=max_step_num, dtype="int64"
        )

    while_op = paddle.static.nn.control_flow.While(cond, is_test=is_test)

    sequence_lengths = paddle.cast(paddle.zeros_like(initial_finished), "int64")
    sequence_lengths.stop_gradient = True

    if is_test:
        # for test, reuse inputs and states variables to save memory
        inputs = paddle.utils.map_structure(lambda x: x, initial_inputs)
        states = paddle.utils.map_structure(lambda x: x, initial_states)
    else:
        # inputs and states of all steps must be saved for backward and training
        inputs_arrays = paddle.utils.map_structure(
            lambda x: paddle.tensor.array.array_write(x, step_idx),
            initial_inputs,
        )
        states_arrays = paddle.utils.map_structure(
            lambda x: paddle.tensor.array.array_write(x, step_idx),
            initial_states,
        )

    def _maybe_copy(state, new_state, step_mask):
        # TODO: use where_op
        state_dtype = state.dtype
        if convert_dtype(state_dtype) in ["bool"]:
            state = paddle.cast(state, dtype="float32")
            new_state = paddle.cast(new_state, dtype="float32")
        if step_mask.dtype != state.dtype:
            step_mask = paddle.cast(step_mask, dtype=state.dtype)
            # otherwise, renamed bool gradients of would be summed up leading
            # to sum(bool) error.
        step_mask = step_mask.unsqueeze([1])
        step_mask.stop_gradient = True
        new_state = paddle.multiply(state, step_mask) - paddle.multiply(
            new_state, (step_mask - 1)
        )
        if convert_dtype(state_dtype) in ["bool"]:
            new_state = paddle.cast(new_state, dtype=state_dtype)
        return new_state

    def _transpose_batch_time(x):
        return paddle.transpose(x, [1, 0, *list(range(2, len(x.shape)))])

    def _create_array_out_of_while(dtype):
        current_block_idx = default_main_program().current_block_idx
        default_main_program().current_block_idx = (
            default_main_program().current_block().parent_idx
        )
        tensor_array = paddle.tensor.array.create_array(dtype)
        default_main_program().current_block_idx = current_block_idx
        return tensor_array

    # While
    with while_op.block():
        if not is_test:
            inputs = paddle.utils.map_structure(
                lambda array: paddle.tensor.array.array_read(array, step_idx),
                inputs_arrays,
            )
            states = paddle.utils.map_structure(
                lambda array: paddle.tensor.array.array_read(array, step_idx),
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
                paddle.cast(
                    paddle.logical_not(global_finished),
                    sequence_lengths.dtype,
                ),
            )
            if impute_finished:  # rectify the states for the finished.
                next_states = paddle.utils.map_structure(
                    lambda x, y: _maybe_copy(x, y, global_finished),
                    states,
                    next_states,
                )
        else:
            (
                warnings.warn(
                    "`next_states` has no `lengths` attribute, the returned `sequence_lengths` would be all zeros."
                )
                if not hasattr(next_states, "lengths")
                else None
            )
            next_sequence_lengths = getattr(
                next_states, "lengths", sequence_lengths
            )

        # create tensor array in global block after dtype[s] of outputs can be got
        outputs_arrays = paddle.utils.map_structure(
            lambda x: _create_array_out_of_while(x.dtype), outputs
        )

        paddle.utils.map_structure(
            lambda x, x_array: paddle.tensor.array.array_write(
                x, i=step_idx, array=x_array
            ),
            outputs,
            outputs_arrays,
        )
        step_idx = paddle.increment(x=step_idx, value=1.0)
        # update the global_finished first, since it might be also in states of
        # decoder, which otherwise would write a stale finished status to array
        paddle.assign(next_finished, global_finished)
        paddle.assign(next_sequence_lengths, sequence_lengths)
        if is_test:
            paddle.utils.map_structure(
                paddle.assign, next_inputs, global_inputs
            )
            paddle.utils.map_structure(
                paddle.assign, next_states, global_states
            )
        else:
            paddle.utils.map_structure(
                lambda x, x_array: paddle.tensor.array.array_write(
                    x, i=step_idx, array=x_array
                ),
                next_inputs,
                inputs_arrays,
            )
            paddle.utils.map_structure(
                lambda x, x_array: paddle.tensor.array.array_write(
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

    final_outputs = paddle.utils.map_structure(
        lambda array: paddle.tensor.manipulation.tensor_array_to_tensor(
            array, axis=0, use_stack=True
        )[0],
        outputs_arrays,
    )
    if is_test:
        final_states = global_states
    else:
        final_states = paddle.utils.map_structure(
            lambda array: paddle.tensor.array.array_read(array, step_idx),
            states_arrays,
        )

    try:
        final_outputs, final_states = decoder.finalize(
            final_outputs, final_states, sequence_lengths
        )
    except NotImplementedError:
        pass

    if not output_time_major:
        final_outputs = paddle.utils.map_structure(
            _transpose_batch_time, final_outputs
        )

    return (
        (final_outputs, final_states, sequence_lengths)
        if return_length
        else (final_outputs, final_states)
    )


def _dynamic_decode_pir_declarative(
    decoder,
    inits=None,
    max_step_num=None,
    output_time_major=False,
    impute_finished=False,
    is_test=False,
    return_length=False,
    **kwargs,
):
    initial_inputs, initial_states, initial_finished = decoder.initialize(inits)
    global_inputs, global_states, global_finished = (
        initial_inputs,
        initial_states,
        initial_finished,
    )
    global_finished.stop_gradient = True
    step_idx = paddle.full(shape=[1], fill_value=0, dtype="int64")

    cond = paddle.logical_not(paddle.all(initial_finished))
    a = paddle.to_tensor(1)
    b = paddle.to_tensor(5)
    cond1 = paddle.less_than(a, b)

    if max_step_num is not None:
        max_step_num = paddle.full(
            shape=[1], fill_value=max_step_num, dtype="int64"
        )

    while_op = paddle.static.nn.control_flow.While(cond, is_test=is_test)

    sequence_lengths = paddle.cast(paddle.zeros_like(initial_finished), "int64")
    sequence_lengths.stop_gradient = True

    if is_test:
        # for test, reuse inputs and states variables to save memory
        inputs = paddle.utils.map_structure(lambda x: x, initial_inputs)
        states = paddle.utils.map_structure(lambda x: x, initial_states)
    else:
        # inputs and states of all steps must be saved for backward and training
        inputs_arrays = paddle.utils.map_structure(
            lambda x: paddle.tensor.array.array_write(x, step_idx),
            initial_inputs,
        )
        states_arrays = paddle.utils.map_structure(
            lambda x: paddle.tensor.array.array_write(x, step_idx),
            initial_states,
        )

    def _maybe_copy(state, new_state, step_mask):
        # TODO: use where_op
        state_dtype = state.dtype
        if convert_dtype(state_dtype) in ["bool"]:
            state = paddle.cast(state, dtype="float32")
            new_state = paddle.cast(new_state, dtype="float32")
        if step_mask.dtype != state.dtype:
            step_mask = paddle.cast(step_mask, dtype=state.dtype)
            # otherwise, renamed bool gradients of would be summed up leading
            # to sum(bool) error.
        step_mask = step_mask.unsqueeze([1])
        step_mask.stop_gradient = True
        new_state = paddle.multiply(state, step_mask) - paddle.multiply(
            new_state, (step_mask - 1)
        )
        if convert_dtype(state_dtype) in ["bool"]:
            new_state = paddle.cast(new_state, dtype=state_dtype)
        return new_state

    def _transpose_batch_time(x):
        return paddle.transpose(x, [1, 0, *list(range(2, len(x.shape)))])

    from paddle.pir.core import default_main_program

    def _create_array_out_of_while(dtype):
        with paddle.base.program_guard(default_main_program()):
            prev_insertion_point = paddle.pir.get_current_insertion_point()
            paddle.pir.set_insertion_point(
                default_main_program().global_block().back()
            )
            tensor_array = paddle.tensor.array.create_array(dtype)
            paddle.pir.set_insertion_point(prev_insertion_point)
        return tensor_array

    # While
    with while_op.block():
        a = paddle.increment(x=a)

        paddle.assign(paddle.less_than(x=a, y=b), cond1)
        if not is_test:
            inputs = paddle.utils.map_structure(
                lambda array: paddle.tensor.array.array_read(array, step_idx),
                inputs_arrays,
            )
            states = paddle.utils.map_structure(
                lambda array: paddle.tensor.array.array_read(array, step_idx),
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
                paddle.cast(
                    paddle.logical_not(global_finished),
                    sequence_lengths.dtype,
                ),
            )
            if impute_finished:  # rectify the states for the finished.
                next_states = paddle.utils.map_structure(
                    lambda x, y: _maybe_copy(x, y, global_finished),
                    states,
                    next_states,
                )
        else:
            (
                warnings.warn(
                    "`next_states` has no `lengths` attribute, the returned `sequence_lengths` would be all zeros."
                )
                if not hasattr(next_states, "lengths")
                else None
            )
            next_sequence_lengths = getattr(
                next_states, "lengths", sequence_lengths
            )

        # # create tensor array in global block after dtype[s] of outputs can be got
        outputs_arrays = paddle.utils.map_structure(
            lambda x: _create_array_out_of_while(x.dtype), outputs
        )

        paddle.utils.map_structure(
            lambda x, x_array: paddle.tensor.array.array_write(
                x, i=step_idx, array=x_array
            ),
            outputs,
            outputs_arrays,
        )

        step_idx = paddle.increment(x=step_idx, value=1.0)
        # update the global_finished first, since it might be also in states of
        # decoder, which otherwise would write a stale finished status to array
        paddle.assign(next_finished, global_finished)
        paddle.assign(next_sequence_lengths, sequence_lengths)
        if is_test:
            paddle.utils.map_structure(
                paddle.assign, next_inputs, global_inputs
            )
            paddle.utils.map_structure(
                paddle.assign, next_states, global_states
            )
        else:
            paddle.utils.map_structure(
                lambda x, x_array: paddle.tensor.array.array_write(
                    x, i=step_idx, array=x_array
                ),
                next_inputs,
                inputs_arrays,
            )
            paddle.utils.map_structure(
                lambda x, x_array: paddle.tensor.array.array_write(
                    x, i=step_idx, array=x_array
                ),
                next_states,
                states_arrays,
            )
        if max_step_num is not None:
            cond_tmp = paddle.logical_and(
                paddle.less_equal(step_idx, max_step_num),
                paddle.logical_not(paddle.all(global_finished)),
            )
            paddle.assign(cond_tmp, cond)
        else:
            paddle.assign(paddle.logical_not(paddle.all(global_finished)), cond)

    final_outputs = paddle.utils.map_structure(
        lambda array: paddle.tensor.manipulation.tensor_array_to_tensor(
            array, axis=0, use_stack=True
        )[0],
        outputs_arrays,
    )
    if is_test:
        final_states = global_states
    else:
        final_states = paddle.utils.map_structure(
            lambda array: paddle.tensor.array.array_read(array, step_idx),
            states_arrays,
        )

    try:
        final_outputs, final_states = decoder.finalize(
            final_outputs, final_states, sequence_lengths
        )
    except NotImplementedError as e:
        pass

    if not output_time_major:
        final_outputs = paddle.utils.map_structure(
            _transpose_batch_time, final_outputs
        )

    return (
        (final_outputs, final_states, sequence_lengths)
        if return_length
        else (final_outputs, final_states)
    )


@overload
def dynamic_decode(
    decoder: Decoder,
    inits: object | None = ...,
    max_step_num: int | None = ...,
    output_time_major: bool = ...,
    impute_finished: bool = ...,
    is_test: bool = ...,
    return_length: Literal[False] = ...,
    **kwargs: Any,
) -> tuple[Tensor, BeamSearchDecoder.StateWrapper]: ...


@overload
def dynamic_decode(
    decoder: Decoder,
    inits: object | None = ...,
    max_step_num: int | None = ...,
    output_time_major: bool = ...,
    impute_finished: bool = ...,
    is_test: bool = ...,
    return_length: Literal[True] = ...,
    **kwargs: Any,
) -> tuple[Tensor, BeamSearchDecoder.StateWrapper, Tensor]: ...


@overload
def dynamic_decode(
    decoder: Decoder,
    inits: object | None = ...,
    max_step_num: int | None = ...,
    output_time_major: bool = ...,
    impute_finished: bool = ...,
    is_test: bool = ...,
    return_length: bool = ...,
    **kwargs: Any,
) -> (
    tuple[Tensor, BeamSearchDecoder.StateWrapper]
    | tuple[Tensor, BeamSearchDecoder.StateWrapper, Tensor]
): ...


def dynamic_decode(
    decoder,
    inits=None,
    max_step_num=None,
    output_time_major=False,
    impute_finished=False,
    is_test=False,
    return_length=False,
    **kwargs,
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

            >>> import paddle
            >>> from paddle.nn import BeamSearchDecoder, dynamic_decode
            >>> from paddle.nn import GRUCell, Linear, Embedding
            >>> trg_embeder = Embedding(100, 32)
            >>> output_layer = Linear(32, 32)
            >>> decoder_cell = GRUCell(input_size=32, hidden_size=32)
            >>> decoder = BeamSearchDecoder(decoder_cell,
            ...                             start_token=0,
            ...                             end_token=1,
            ...                             beam_size=4,
            ...                             embedding_fn=trg_embeder,
            ...                             output_fn=output_layer)
            >>> encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
            >>> outputs = dynamic_decode(decoder=decoder,
            ...                          inits=decoder_cell.get_initial_states(encoder_output),
            ...                          max_step_num=10)
            >>> print(outputs[0].shape)
            [4, 11, 4]
    """
    if in_dynamic_mode():
        return _dynamic_decode_imperative(
            decoder,
            inits,
            max_step_num,
            output_time_major,
            impute_finished,
            is_test,
            return_length,
            **kwargs,
        )
    elif paddle.framework.in_pir_mode():
        return _dynamic_decode_pir_declarative(
            decoder,
            inits,
            max_step_num,
            output_time_major,
            impute_finished,
            is_test,
            return_length,
            **kwargs,
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
            **kwargs,
        )
