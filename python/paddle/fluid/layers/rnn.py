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

from __future__ import print_function

from functools import partial

from . import nn
from . import tensor
from . import control_flow
from . import utils
from .utils import *

__all__ = [
    'RNNCell',
    'GRUCell',
    'LSTMCell',
    'Decoder',
    'BeamSearchDecoder',
    'dynamic_rnn',
    'dynamic_decode',
]


class RNNCell(object):
    """
    RNNCell is the base class for abstraction representing the calculations
    mapping the input and state to the output and new state. It is suitable to
    and mostly used in RNN.
    """

    def call(self, inputs, states, **kwargs):
        """
        Every cell must implement this method to do the calculations mapping the
        inputs and states to the output and new states.

        Both inputs and states can be nested structure(list|tuple|namedtuple|dict)
        of tensor variable to be more flexible.

        Args:
            inputs: tensor variable or nested structure of tensor variables.
            states: The hidden size used in the cell.
            **kwargs: Additional keyword arguments. The caller of cell can pass
                through these arguments transparently making the more flexible. 
        
        Returns:
            tuple: outputs and new_states pair. outputs and new_states both \
                can be nested structure of tensor variables. new_states must \
                have the same structure with states.

        """
        raise NotImplementedError("RNNCell must implent the call function.")

    def __call__(self, inputs, states, **kwargs):
        return self.call(inputs, states, **kwargs)

    def get_initial_states(self,
                           batch_ref,
                           shape=None,
                           dtype=None,
                           init_value=0):
        """
        Generate initialized states according to provided shape, date type and
        value.

        Args:
            batch_ref: tensor variable or nested structure of tensor variables.
                The first dimension of the tensor will be used as batch size to
                initialize states.
            shape: nested structure of shape(shape is represented as a list/tuple
                of integer). -1(for batch size) will be automatically inserted if
                shape is not started with it. If None, property `state_shape` will
                be used. The default value is None.
            shape: nested structure of data type(float32, float64, int32, int64). 
                The structure should be same as the structure of shape, while if
                all tensor variables' data type in state structure is same.
                started with it. If None and property `state_shape` is not available,
                float32 will be used as the data type. The default value is None.
            init_value: A float value used to initialize states.
        
        Returns:
            nested structure: tensor variables packed in the same structure provided \
                by shape, representing the initialized states.
        """
        # TODO: use inputs and batch_size
        batch_ref = flatten(batch_ref)[0]

        def _is_shape_sequence(seq):
            """For shape, list/tuple of integer is the finest-grained objection"""
            if (isinstance(seq, list) or isinstance(seq, tuple)):
                if reduce(lambda flag, x: isinstance(x, int) and flag, seq,
                          True):
                    return False
            # TODO: Add check for the illegal
            if isinstance(seq, dict):
                return True
            return (isinstance(seq, collections.Sequence) and
                    not isinstance(seq, six.string_types))

        class Shape(object):
            def __init__(self, shape):
                self.shape = shape if shape[0] == -1 else ([-1] + list(shape))

        # nested structure of shapes
        states_shapes = self.state_shape if shape is None else shape
        is_sequence_ori = utils.is_sequence
        utils.is_sequence = _is_shape_sequence
        states_shapes = map_structure(lambda shape: Shape(shape), states_shapes)
        utils.is_sequence = is_sequence_ori

        # nested structure of dtypes
        try:
            states_dtypes = self.state_dtype if dtype is None else dtype
        except NotImplementedError:  # use fp32 as default
            states_dtypes = "float32"
        if len(flatten(states_dtypes)) == 1:
            dtype = flatten(states_dtypes)[0]
            states_dtypes = map_structure(lambda shape: dtype, states_shapes)

        init_states = map_structure(
            lambda shape, dtype: tensor.fill_constant_batch_size_like(
                input=batch_ref,
                shape=shape.shape,
                dtype=dtype,
                value=init_value), states_shapes, states_dtypes)
        return init_states

    @property
    def state_dtype(self):
        """
        nested structure of shape(shape is represented as a list/tuple). 
        Used to initialize states. Not necessary to be implemented if states
        are not initialized by `get_initial_states` or provide shape in when
        using `get_initial_states`.
        """
        raise NotImplementedError

    @property
    def state_shape(self):
        """
        nested structure of shape(shape is represented as a list/tuple). 
        Used to initialize states. Not necessary to be implemented.
        """
        raise NotImplementedError


class GRUCell(RNNCell):
    """
    A wrapper for BasicGRUUnit to be adapted to cell.
    """

    def __init__(self,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 dtype="float32",
                 name="GRUCell"):
        self.hidden_size = hidden_size
        from .. import contrib  # TODO: resolve recurrent import
        self.gru_unit = contrib.layers.rnn_impl.BasicGRUUnit(
            name, hidden_size, param_attr, bias_attr, gate_activation,
            activation, dtype)

    def call(self, inputs, states):
        new_hidden = self.gru_unit(inputs, states)
        return new_hidden, new_hidden

    @property
    def state_shape(self):
        return [self.hidden_size]


class LSTMCell(RNNCell):
    """
    A wrapper for BasicLSTMUnit to be adapted to cell.
    """

    def __init__(self,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 dtype="float32",
                 name="LSTMCell"):
        self.hidden_size = hidden_size
        from .. import contrib  # TODO: resolve recurrent import
        self.lstm_unit = contrib.layers.rnn_impl.BasicLSTMUnit(
            name, hidden_size, param_attr, bias_attr, gate_activation,
            activation, forget_bias, dtype)

    def call(self, inputs, states):
        pre_hidden, pre_cell = states
        new_hidden, new_cell = self.lstm_unit(inputs, pre_hidden, pre_cell)
        return new_hidden, [new_hidden, new_cell]

    @property
    def state_shape(self):
        """[hidden shape, cell shape]"""
        return [[self.hidden_size], [self.hidden_size]]


def dynamic_rnn(cell,
                inputs,
                initial_states=None,
                sequence_length=None,
                time_major=False,
                is_reverse=False,
                **kwargs):
    """
    """

    def _maybe_copy(state, new_state, step_mask):
        # TODO: use where_op
        new_state = nn.elementwise_mul(
            new_state, step_mask, axis=0) - nn.elementwise_mul(
                state, (step_mask - 1), axis=0)
        return new_state

    def _transpose_batch_time(x):
        return nn.transpose(x, [1, 0] + range(2, len(x.shape)))

    def _switch_grad(x, stop=False):
        x.stop_gradient = stop
        return x

    if initial_states is None:
        initial_states = cell.get_initial_states(batch_ref=inputs)
    initial_states = map_structure(_switch_grad, initial_states)

    if not time_major:
        inputs = map_structure(_transpose_batch_time, inputs)

    if sequence_length:
        max_seq_len = nn.shape(flatten(inputs)[0])[0]
        mask = nn.sequence_mask(
            sequence_length,
            maxlen=max_seq_len,
            dtype=flatten(initial_states)[0].dtype)
        mask = nn.transpose(mask, [1, 0])
    if is_reverse:
        inputs = map_structure(lambda x: tensor.reverse(x, axis=[0]), inputs)
        mask = tensor.reverse(mask, axis=[0]) if sequence_length else None

    # StaticRNN
    rnn = control_flow.StaticRNN()
    with rnn.step():
        inputs = map_structure(rnn.step_input, inputs)
        states = map_structure(rnn.memory, initial_states)
        copy_states = map_structure(lambda x: x, states)
        outputs, new_states = cell.call(inputs, copy_states, **kwargs)
        assert_same_structure(states, new_states)
        if sequence_length:
            step_mask = rnn.step_input(mask)
            new_states = map_structure(
                partial(
                    _maybe_copy, step_mask=step_mask), states, new_states)

        map_structure(rnn.update_memory, states, new_states)
        flat_outputs = flatten(outputs)
        map_structure(rnn.step_output, outputs)
        map_structure(rnn.step_output, new_states)

    rnn_out = rnn()
    final_outputs = rnn_out[:len(flat_outputs)]
    final_outputs = pack_sequence_as(outputs, final_outputs)
    final_states = map_structure(lambda x: x[-1], rnn_out[len(flat_outputs):])
    final_states = pack_sequence_as(new_states, final_states)

    if is_reverse:
        final_outputs = map_structure(lambda x: tensor.reverse(x, axis=[0]),
                                      final_outputs)

    if not time_major:
        final_outputs = map_structure(_transpose_batch_time, final_outputs)

    return (final_outputs, final_states)


class Decoder(object):
    @property
    def output_dtype(self):
        """A (possibly nested tuple of...) dtype[s]."""
        raise NotImplementedError

    def initialize(self, inits):
        """Called before any decoding iterations."""
        raise NotImplementedError

    def step(self, time, inputs, state):
        """Called per step of decoding (but only once for dynamic decoding)."""
        raise NotImplementedError

    def finalize(self, outputs, final_state, sequence_lengths):
        raise NotImplementedError


class BeamSearchDecoder(Decoder):
    def __init__(self,
                 cell,
                 start_token,
                 end_token,
                 beam_size,
                 vocab_size,
                 embedding_fn=None,
                 output_fn=None):
        self.cell = cell
        self.embedding_fn = embedding_fn
        self.output_fn = output_fn
        self.start_token = start_token
        self.end_token = end_token
        self.beam_size = beam_size
        self.vocab_size = vocab_size

    @staticmethod
    def tile_beam_merge_with_batch(x, beam_size):
        x = nn.unsqueeze(x, [1])  # [batch_size, 1, ...]
        expand_times = [1] * len(x.shape)
        expand_times[1] = beam_size
        x = nn.expand(x, expand_times)  # [batch_size, beam_size, ...]
        x = nn.transpose(
            x, range(2, len(x.shape)) + [0, 1])  # [..., batch_size, beam_size]
        # use 0 to copy to avoid wrong shape
        x = nn.reshape(
            x, shape=[0] *
            (len(x.shape) - 2) + [-1])  # [..., batch_size * beam_size]
        x = nn.transpose(x, [len(x.shape) - 1] + range(
            0, len(x.shape) - 1))  # [batch_size * beam_size, ...]
        return x

    def _split_batch_beams(self, x):
        # TODO: avoid fake shape in compile-time like tile_beam_merge_with_batch
        return nn.reshape(x, shape=(-1, self.beam_size) + x.shape[1:])

    def _merge_batch_beams(self, x):
        # TODO: avoid fake shape in compile-time like tile_beam_merge_with_batch
        return nn.reshape(x, shape=(-1, ) + x.shape[2:])

    def _expand_to_beam_size(self, x):
        x = nn.unsqueeze(x, [1])
        expand_times = [1] * len(x.shape)
        expand_times[1] = self.beam_size
        x = nn.expand(x, expand_times)
        return x

    def _mask_probs(self, probs, finished):
        # TODO: use where_op
        finished = tensor.cast(finished, dtype=probs.dtype)
        probs = nn.elementwise_mul(
            nn.expand(nn.unsqueeze(finished, [2]), [1, 1, self.vocab_size]),
            self.noend_mask_tensor,
            axis=-1) - nn.elementwise_mul(
                probs, (finished - 1), axis=0)
        return probs

    def _score(self, batch_size, range_size):
        pass

    def _gather(self, x, indices, batch_size):
        batch_size = tensor.cast(
            batch_size,
            indices.dtype) if batch_size.dtype != indices.dtype else batch_size
        batch_pos = nn.expand(
            nn.unsqueeze(
                tensor.range(
                    0, batch_size, 1, dtype=indices.dtype), [1]),
            [1, self.beam_size])
        topk_coordinates = nn.stack([batch_pos, indices], axis=2)
        return nn.gather_nd(x, topk_coordinates)

    class OutputWrapper(
            collections.namedtuple("OutputWrapper",
                                   ("scores", "predicted_ids", "parent_ids"))):
        pass

    class StateWrapper(
            collections.namedtuple(
                "StateWrapper",
                ("cell_states", "log_probs", "finished", "lengths"))):
        pass

    def initialize(self, initial_cell_states):
        kinf = 1e9
        state = flatten(initial_cell_states)[0]
        self.batch_size = nn.shape(state)[0]

        self.start_token_tensor = tensor.fill_constant(
            shape=[1], dtype="int64", value=self.start_token)
        self.end_token_tensor = tensor.fill_constant(
            shape=[1], dtype="int64", value=self.end_token)
        self.vocab_size_tensor = tensor.fill_constant(
            shape=[1], dtype="int64", value=self.vocab_size)

        noend_array = [-kinf] * self.vocab_size
        noend_array[self.end_token] = 0
        self.noend_mask_tensor = tensor.assign(np.array(noend_array, "float32"))
        init_cell_states = map_structure(self._expand_to_beam_size,
                                         initial_cell_states)
        # TODO: use fill_constant when support variable shape
        init_inputs = nn.expand(
            nn.unsqueeze(
                nn.expand(self.start_token_tensor, [self.batch_size]), [1]),
            [1, self.beam_size])
        log_probs = nn.expand(
            tensor.assign(
                np.array(
                    [[0.] + [-kinf] * (self.beam_size - 1)], dtype="float32")),
            [self.batch_size, 1])
        init_finished = tensor.fill_constant_batch_size_like(
            input=state,
            shape=[-1, self.beam_size],
            dtype="bool",
            value=False,
            force_cpu=True)
        init_lengths = tensor.zeros_like(init_inputs)
        init_inputs = nn.unsqueeze(init_inputs, [2])
        init_inputs = self.embedding_fn(
            init_inputs) if self.embedding_fn else init_inputs
        return init_inputs, self.StateWrapper(init_cell_states, log_probs,
                                              init_finished,
                                              init_lengths), init_finished

    def _beam_search_step(self, time, logits, next_cell_states, beam_state):
        step_log_probs = nn.log(nn.softmax(logits))
        step_log_probs = self._mask_probs(step_log_probs, beam_state.finished)
        log_probs = nn.elementwise_add(
            x=step_log_probs, y=beam_state.log_probs, axis=0)
        # TODO: length penalty
        scores = log_probs
        scores = nn.reshape(scores, [-1, self.beam_size * self.vocab_size])
        topk_scores, topk_indices = nn.topk(input=scores, k=self.beam_size)
        beam_indices = nn.elementwise_floordiv(topk_indices,
                                               self.vocab_size_tensor)
        token_indices = nn.elementwise_mod(topk_indices, self.vocab_size_tensor)
        next_log_probs = self._gather(
            nn.reshape(log_probs, [-1, self.beam_size * self.vocab_size]),
            topk_indices, self.batch_size)
        next_cell_states = map_structure(
            lambda x: self._gather(x, beam_indices, self.batch_size),
            next_cell_states)
        next_finished = self._gather(beam_state.finished, beam_indices,
                                     self.batch_size)
        next_lengths = self._gather(beam_state.lengths, beam_indices,
                                    self.batch_size)
        next_lengths = next_lengths + tensor.cast(
            nn.logical_not(next_finished), beam_state.lengths.dtype)
        next_finished = control_flow.logical_or(
            next_finished,
            control_flow.equal(token_indices, self.end_token_tensor))

        beam_search_output = self.OutputWrapper(topk_scores, token_indices,
                                                beam_indices)
        beam_search_state = self.StateWrapper(next_cell_states, next_log_probs,
                                              next_finished, next_lengths)
        return beam_search_output, beam_search_state

    def step(self, time, inputs, states, **kwargs):
        inputs = map_structure(self._merge_batch_beams, inputs)
        cell_states = map_structure(self._merge_batch_beams, states.cell_states)
        cell_outputs, next_cell_states = self.cell(inputs, cell_states,
                                                   **kwargs)
        cell_outputs = map_structure(self._split_batch_beams, cell_outputs)
        next_cell_states = map_structure(self._split_batch_beams,
                                         next_cell_states)

        if self.output_fn is not None:
            cell_outputs = self.output_fn(cell_outputs)

        beam_search_output, beam_search_state = self._beam_search_step(
            time=time,
            logits=cell_outputs,
            next_cell_states=next_cell_states,
            beam_state=states)
        finished = beam_search_state.finished
        sample_ids = beam_search_output.predicted_ids
        next_inputs = self.embedding_fn(nn.unsqueeze(
            sample_ids, [2])) if self.embedding_fn else sample_ids

        return (beam_search_output, beam_search_state, next_inputs, finished)

    def finalize(self, outputs, final_states, sequence_lengths):
        predicted_ids = nn.gather_tree(outputs.predicted_ids,
                                       outputs.parent_ids)
        return predicted_ids, final_states

    @property
    def output_dtype(self):
        return self.OutputWrapper(
            scores="float32", predicted_ids="int64", parent_ids="int64")


def dynamic_decode(decoder,
                   inits=None,
                   max_step_num=None,
                   output_time_major=False,
                   **kwargs):
    initial_inputs, initial_states, initial_finished = decoder.initialize(inits)
    global_inputs, global_states, global_finished = (
        initial_inputs, initial_states, initial_finished)

    step_idx = tensor.fill_constant(shape=[1], dtype="int64", value=0)
    cond = control_flow.logical_not((nn.reduce_all(initial_finished)))
    if max_step_num is not None:
        max_step_num = tensor.fill_constant(
            shape=[1], dtype="int64", value=max_step_num)
    while_op = control_flow.While(cond)

    inputs = map_structure(lambda x: x, initial_inputs)
    states = map_structure(lambda x: x, initial_states)
    outputs_arrays = map_structure(
        lambda dtype: control_flow.create_array(dtype), decoder.output_dtype)
    sequence_lengths = tensor.cast(tensor.zeros_like(initial_finished), "int64")

    def _maybe_copy(state, new_state, step_mask):
        # TODO: use where_op
        new_state = nn.elementwise_mul(
            new_state, step_mask, axis=0) - nn.elementwise_mul(
                state, (step_mask - 1), axis=0)
        return new_state

    def _transpose_batch_time(x):
        return nn.transpose(x, [1, 0] + range(2, len(x.shape)))

    # While
    with while_op.block():
        (outputs, next_states, next_inputs,
         next_finished) = decoder.step(step_idx, inputs, states, **kwargs)
        next_sequence_lengths = nn.elementwise_add(
            sequence_lengths,
            tensor.cast(
                control_flow.logical_not(global_finished),
                sequence_lengths.dtype))

        map_structure(
            lambda x, x_array: control_flow.array_write(
                x, i=step_idx, array=x_array), outputs, outputs_arrays)
        control_flow.increment(x=step_idx, value=1.0, in_place=True)
        map_structure(tensor.assign, next_inputs, global_inputs)
        map_structure(tensor.assign, next_states, global_states)
        tensor.assign(next_finished, global_finished)
        tensor.assign(next_sequence_lengths, sequence_lengths)
        if max_step_num is not None:
            control_flow.logical_and(
                control_flow.logical_not(nn.reduce_all(next_finished)),
                control_flow.less_equal(step_idx, max_step_num), cond)
        else:
            control_flow.logical_not(nn.reduce_all(next_finished), cond)

    final_outputs = map_structure(
        lambda array: tensor.tensor_array_to_tensor(
            array, axis=0, use_stack=True)[0], outputs_arrays)
    final_states = global_states

    try:
        final_outputs, final_states = decoder.finalize(
            final_outputs, global_states, sequence_lengths)
    except NotImplementedError:
        pass

    if not output_time_major:
        final_outputs = map_structure(_transpose_batch_time, final_outputs)

    return final_outputs, final_states
