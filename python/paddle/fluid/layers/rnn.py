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

from functools import partial, reduce

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
    'rnn',
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

        To be more flexible, both inputs and states can be a tensor variable or
        a nested structure (list|tuple|namedtuple|dict) of tensor variable, that
        is, a (possibly nested structure of) tensor variable[s].

        Parameters:
            inputs: A (possibly nested structure of) tensor variable[s].
            states: A (possibly nested structure of) tensor variable[s].
            **kwargs: Additional keyword arguments, provided by the caller. 
        
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
        Generate initialized states according to provided shape, data type and
        value.

        Parameters:
            batch_ref: A (possibly nested structure of) tensor variable[s].
                The first dimension of the tensor will be used as batch size to
                initialize states.
            shape: A (possiblely nested structure of) shape[s], where a shape is
                represented as a list/tuple of integer). -1(for batch size) will
                beautomatically inserted if shape is not started with it. If None,
                property `state_shape` will be used. The default value is None.
            dtype: A (possiblely nested structure of) data type[s]. The structure
                must be same as that of `shape`, except when all tensors' in states
                has the same data type, a single data type can be used. If None and
                property `cell.state_shape` is not available, float32 will be used
                as the data type. The default value is None.
            init_value: A float value used to initialize states.
        
        Returns:
            Variable: tensor variable[s] packed in the same structure provided \
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
    def state_shape(self):
        """
        Used to initialize states.
        A (possiblely nested structure of) shape[s], where a shape is represented
        as a list/tuple of integers (-1 for batch size would be automatically
        inserted into a shape if shape is not started with it). 
        Not necessary to be implemented if states are not initialized by
        `get_initial_states` or the `shape` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError

    @property
    def state_dtype(self):
        """
        Used to initialize states.
        A (possiblely nested structure of) data types[s]. The structure must be
        same as that of `shape`, except when all tensors' in states has the same
        data type, a signle data type can be used.
        Not necessary to be implemented if states are not initialized
        by `get_initial_states` or the `dtype` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError


class GRUCell(RNNCell):
    """
    Gated Recurrent Unit cell. It is a wrapper for 
    `fluid.contrib.layers.rnn_impl.BasicGRUUnit` to make it adapt to RNNCell.

    The formula used is as follow:

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \\tilde{h_t}

    For more details, please refer to  `Learning Phrase Representations using
    RNN Encoder Decoder for Statistical Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_

    Examples:

        .. code-block:: python

            import paddle.fluid.layers as layers
            cell = layers.GRUCell(hidden_size=256)
    """

    def __init__(self,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 dtype="float32",
                 name="GRUCell"):
        """
        Constructor of GRUCell.

        Parameters:
            hidden_size (int): The hidden size in the GRU cell.
            param_attr(ParamAttr, optional): The parameter attribute for the learnable
                weight matrix. Default: None.
            bias_attr (ParamAttr, optional): The parameter attribute for the bias
                of GRU. Default: None.
            gate_activation (function, optional): The activation function for :math:`act_g`.
                Default: `fluid.layers.sigmoid`.
            activation (function, optional): The activation function for :math:`act_c`.
                Default: `fluid.layers.tanh`.
            dtype(string, optional): The data type used in this cell. Default float32.
            name(string, optional) : The name scope used to identify parameters and biases.
        """
        self.hidden_size = hidden_size
        from .. import contrib  # TODO: resolve recurrent import
        self.gru_unit = contrib.layers.rnn_impl.BasicGRUUnit(
            name, hidden_size, param_attr, bias_attr, gate_activation,
            activation, dtype)

    def call(self, inputs, states):
        """
        Perform calculations of GRU.

        Parameters:
            inputs(Variable): A tensor with shape `[batch_size, input_size]`,
                corresponding to :math:`x_t` in the formula. The data type
                should be float32.
            states(Variable): A tensor with shape `[batch_size, hidden_size]`.
                corresponding to :math:`h_{t-1}` in the formula. The data type
                should be float32.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` and \
                `new_states` is the same tensor shaped `[batch_size, hidden_size]`, \
                corresponding to :math:`h_t` in the formula. The data type of the \
                tensor is same as that of `states`.        
        """
        new_hidden = self.gru_unit(inputs, states)
        return new_hidden, new_hidden

    @property
    def state_shape(self):
        """
        The `state_shape` of GRUCell is a shape `[hidden_size]` (-1 for batch
        size would be automatically inserted into shape). The shape corresponds
        to :math:`h_{t-1}`.
        """
        return [self.hidden_size]


class LSTMCell(RNNCell):
    """
    Long-Short Term Memory cell. It is a wrapper for 
    `fluid.contrib.layers.rnn_impl.BasicLSTMUnit` to make it adapt to RNNCell.

    The formula used is as follow:

    .. math::

        i_{t} & = act_g(W_{x_{i}}x_{t} + W_{h_{i}}h_{t-1} + b_{i})

        f_{t} & = act_g(W_{x_{f}}x_{t} + W_{h_{f}}h_{t-1} + b_{f} + forget\\_bias)

        c_{t} & = f_{t}c_{t-1} + i_{t} act_c (W_{x_{c}}x_{t} + W_{h_{c}}h_{t-1} + b_{c})

        o_{t} & = act_g(W_{x_{o}}x_{t} + W_{h_{o}}h_{t-1} + b_{o})

        h_{t} & = o_{t} act_c (c_{t})
    
    For more details, please refer to `RECURRENT NEURAL NETWORK REGULARIZATION <http://arxiv.org/abs/1409.2329>`_

    Examples:

        .. code-block:: python

            import paddle.fluid.layers as layers
            cell = layers.LSTMCell(hidden_size=256)
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
        """
        Constructor of LSTMCell.

        Parameters:
            hidden_size (int): The hidden size in the LSTM cell.
            param_attr(ParamAttr, optional): The parameter attribute for the learnable
                weight matrix. Default: None.
            bias_attr (ParamAttr, optional): The parameter attribute for the bias
                of LSTM. Default: None.
            gate_activation (function, optional): The activation function for :math:`act_g`.
                Default: 'fluid.layers.sigmoid'.
            activation (function, optional): The activation function for :math:`act_h`.
                Default: 'fluid.layers.tanh'.
            forget_bias(float, optional): forget bias used when computing forget gate.
                Default 1.0
            dtype(string, optional): The data type used in this cell. Default float32.
            name(string, optional) : The name scope used to identify parameters and biases.
        """
        self.hidden_size = hidden_size
        from .. import contrib  # TODO: resolve recurrent import
        self.lstm_unit = contrib.layers.rnn_impl.BasicLSTMUnit(
            name, hidden_size, param_attr, bias_attr, gate_activation,
            activation, forget_bias, dtype)

    def call(self, inputs, states):
        """
        Perform calculations of LSTM.

        Parameters:
            inputs(Variable): A tensor with shape `[batch_size, input_size]`,
                corresponding to :math:`x_t` in the formula. The data type
                should be float32.
            states(Variable): A list of containing two tensers, each shaped
                `[batch_size, hidden_size]`, corresponding to :math:`h_{t-1}, c_{t-1}`
                in the formula. The data type should be float32.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` is \
                a tensor with shape `[batch_size, hidden_size]`, corresponding \
                to :math:`h_{t}` in the formula; `new_states` is a list containing \
                two tenser variables shaped `[batch_size, hidden_size]`, corresponding \
                to :math:`h_{t}, c_{t}` in the formula. The data type of these \
                tensors all is same as that of `states`.
        """
        pre_hidden, pre_cell = states
        new_hidden, new_cell = self.lstm_unit(inputs, pre_hidden, pre_cell)
        return new_hidden, [new_hidden, new_cell]

    @property
    def state_shape(self):
        """
        The `state_shape` of LSTMCell is a list with two shapes: `[[hidden_size], [hidden_size]]`
        (-1 for batch size would be automatically inserted into shape). These two
        shapes correspond to :math:`h_{t-1}` and :math:`c_{t-1}` separately.
        """
        return [[self.hidden_size], [self.hidden_size]]


def rnn(cell,
        inputs,
        initial_states=None,
        sequence_length=None,
        time_major=False,
        is_reverse=False,
        **kwargs):
    """
    rnn creates a recurrent neural network specified by RNNCell `cell`,
    which performs :code:`cell.call()` repeatedly until reachs to the maximum
    length of `inputs`.

    Parameters:
        cell(RNNCell): An instance of `RNNCell`.
        inputs(Variable): A (possibly nested structure of) tensor variable[s]. 
            The shape of tensor should be `[batch_size, sequence_length, ...]`
            for `time_major == False` or `[sequence_length, batch_size, ...]`
            for `time_major == True`. It represents the inputs to be unrolled
            in RNN.
        initial_states(Variable, optional): A (possibly nested structure of)
            tensor variable[s], representing the initial state for RNN. 
            If not provided, `cell.get_initial_states` would be used to produce
            the initial state. Default None.
        sequence_length(Variable, optional): A tensor with shape `[batch_size]`.
            It stores real length of each instance, thus enables users to extract
            the last valid state when past a batch element's sequence length for
            correctness. If not provided, the padddings would be treated same as
            non-padding inputs. Default None.
        time_major(bool, optional): Indicate the data layout of Tensor included
            in `input` and `output` tensors. If `False`, the data layout would
            be batch major with shape `[batch_size, sequence_length, ...]`.  If
            `True`, the data layout would be time major with shape
            `[sequence_length, batch_size, ...]`. Default: `False`.
        is_reverse(bool, optional): Indicate whether to calculate in the reverse
            order of input sequences. Default: `False`.
        **kwargs: Additional keyword arguments. Arguments passed to `cell.call`. 

    Returns:
        tuple: A tuple( :code:`(final_outputs, final_states)` ) including the final \
            outputs and states, both are Tensor or nested structure of Tensor. \
            `final_outputs` has the same structure and data types as \
            the returned `outputs` of :code:`cell.call` , and each Tenser in `final_outputs` \
            stacks all time steps' counterpart in `outputs` thus has shape `[batch_size, sequence_length, ...]` \
            for `time_major == False` or `[sequence_length, batch_size, ...]` for `time_major == True`. \
            `final_states` is the counterpart at last time step of initial states, \
            thus has the same structure with it and has tensors with same shapes \
            and data types.
            

    Examples:

        .. code-block:: python
            
            import paddle.fluid as fluid

            inputs = fluid.data(name="inputs",
                                shape=[-1, 32, 128],
                                dtype="float32")
            cell = fluid.layers.GRUCell(hidden_size=128)
            outputs = fluid.layers.rnn(cell=cell, inputs=inputs)
    """

    def _maybe_copy(state, new_state, step_mask):
        # TODO: use where_op
        new_state = nn.elementwise_mul(
            new_state, step_mask, axis=0) - nn.elementwise_mul(
                state, (step_mask - 1), axis=0)
        return new_state

    def _transpose_batch_time(x):
        return nn.transpose(x, [1, 0] + list(range(2, len(x.shape))))

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
    """
    Decoder is the base class for any decoder instance used in `dynamic_decode`.
    It provides interface for output generation for one time step, which can be
    used to generate sequences. 

    The key abstraction provided by Decoder is:

    1. :code:`(initial_input, initial_state, finished) = initialize(inits)` ,
    which generates the input and state for the first decoding step, and gives the
    inintial status telling whether each sequence in the batch is finished.
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
        """
        Called once before the decoding iterations.

        Parameters:
            inits: Argument provided by the caller.

        Returns:
            tuple: A tuple( :code:(initial_inputs, initial_states, finished)` ). \
                `initial_inputs` and `initial_states` both are a (possibly nested \
                structure of) tensor variable[s], and `finished` is a tensor with \
                bool data type.
        """
        raise NotImplementedError

    def step(self, time, inputs, states):
        """
        Called per step of decoding. 

        Parameters:
            time(Variable): A Tensor with shape :math:`[1]` provided by the caller.
                The data type is int64.
            inputs(Variable): A (possibly nested structure of) tensor variable[s].
            states(Variable): A (possibly nested structure of) tensor variable[s].
        
        Returns:
            tuple: A tuple( :code:(outputs, next_states, next_inputs, finished)` ). \
                `next_inputs` and `next_states` both are a (possibly nested \
                structure of) tensor variable[s], and the structure, shape and \
                data type must be same as the counterpart from input arguments. \
                `outputs` is a (possibly nested structure of) tensor variable[s]. \
                `finished` is a Tensor with bool data type.
        """
        raise NotImplementedError

    @property
    def output_dtype(self):
        """
        A (possiblely nested structure of) data type[s]. The structure must be
        same as `outputs` returned by `decoder.step`.
        """
        raise NotImplementedError

    def finalize(self, outputs, final_states, sequence_lengths):
        """
        Called once after the decoding iterations if implemented.

        Parameters:
            outputs(Variable): A (possibly nested structure of) tensor variable[s].
                The structure and data type is same as `output_dtype`.
                The tensor stacks all time steps' output thus has shape 
                :math:`[time\_step, batch\_size, ...]` , which is done by the caller. 
            final_states(Variable): A (possibly nested structure of) tensor variable[s].
                It is the `next_states` returned by `decoder.step` at last decoding step,
                thus has the same structrue, shape and data type with states at any time
                step.

        Returns:
            tuple: A tuple( :code:`(final_outputs, final_states)` ). \
                `final_outputs` and `final_states` both are a (possibly nested \
                structure of) tensor variable[s].
        """
        raise NotImplementedError


class BeamSearchDecoder(Decoder):
    """
    Decoder with beam search decoding strategy. It wraps a cell to get probabilities,
    and follows a beam search step to calculate scores and select candidate
    token ids for each decoding step.

    Please refer to `Beam search <https://en.wikipedia.org/wiki/Beam_search>`_
    for more details.

    **NOTE** When decoding with beam search, the `inputs` and `states` of cell
    would be tiled to `beam_size` (unsqueeze and tile), resulting to shapes like
    `[batch_size * beam_size, ...]` , which is built into `BeamSearchDecoder` and
    done automatically. Thus any other tensor with shape `[batch_size, ...]` used
    in `cell.call` needs to be tiled manually first, which can be completed by using
    :code:`BeamSearchDecoder.tile_beam_merge_with_batch` . The most common case
    for this is the encoder output in attention mechanism.


    Examples:

        .. code-block:: python
            
            import paddle.fluid as fluid
            from paddle.fluid.layers import GRUCell, BeamSearchDecoder

            trg_embeder = lambda x: fluid.embedding(
                x, size=[10000, 128], param_attr=fluid.ParamAttr(name="trg_embedding"))
            output_layer = lambda x: layers.fc(x,
                                            size=10000,
                                            num_flatten_dims=len(x.shape) - 1,
                                            param_attr=fluid.ParamAttr(name=
                                                                        "output_w"),
                                            bias_attr=False)
            decoder_cell = GRUCell(hidden_size=128)
            decoder = BeamSearchDecoder(decoder_cell,
                                        start_token=0,
                                        end_token=1,
                                        beam_size=4,
                                        embedding_fn=trg_embeder,
                                        output_fn=output_layer)
    """

    def __init__(self,
                 cell,
                 start_token,
                 end_token,
                 beam_size,
                 embedding_fn=None,
                 output_fn=None):
        """
        Constructor of BeamSearchDecoder.

        Parameters:
            cell(RNNCell): An instance of `RNNCell` or object with the same interface.
            start_token(int): The start token id.
            end_token(int): The end token id.
            beam_size(int): The beam width used in beam search.
            embedding_fn(optional): A callable to apply to selected candidate ids. 
                Mostly it is an embedding layer to transform ids to embeddings,
                and the returned value acts as the `input` argument for `cell.call`.
                **Note that fluid.embedding should be used here rather than
                fluid.layers.embedding, since shape of ids is [batch_size, beam_size].
                when using fluid.layers.embedding, must unsqueeze in embedding_fn.**
                If not provided, the id to embedding transfomation must be built into
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
    def tile_beam_merge_with_batch(x, beam_size):
        """
        Tile the batch dimension of a tensor. Specifically, this function takes
        a tensor t shaped `[batch_size, s0, s1, ...]` composed of minibatch 
        entries `t[0], ..., t[batch_size - 1]` and tiles it to have a shape
        `[batch_size * beam_size, s0, s1, ...]` composed of minibatch entries
        `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated
        `beam_size` times.

        Parameters:
            x(Variable): A tenosr with shape `[batch_size, ...]`. The data type
                should be float32, float64, int32, int64 or bool.
            beam_size(int): The beam width used in beam search.

        Returns:
            Variable: A tensor with shape `[batch_size * beam_size, ...]`, whose \
                data type is same as `x`.
        """
        x = nn.unsqueeze(x, [1])  # [batch_size, 1, ...]
        expand_times = [1] * len(x.shape)
        expand_times[1] = beam_size
        x = nn.expand(x, expand_times)  # [batch_size, beam_size, ...]
        x = nn.transpose(x, list(range(2, len(x.shape))) +
                         [0, 1])  # [..., batch_size, beam_size]
        # use 0 to copy to avoid wrong shape
        x = nn.reshape(
            x, shape=[0] *
            (len(x.shape) - 2) + [-1])  # [..., batch_size * beam_size]
        x = nn.transpose(
            x, [len(x.shape) - 1] +
            list(range(0, len(x.shape) - 1)))  # [batch_size * beam_size, ...]
        return x

    def _split_batch_beams(self, x):
        """
        Reshape a tensor with shape `[batch_size * beam_size, ...]` to a new
        tensor with shape `[batch_size, beam_size, ...]`. 

        Parameters:
            x(Variable): A tenosr with shape `[batch_size * beam_size, ...]`. The
                data type should be float32, float64, int32, int64 or bool.

        Returns:
            Variable: A tensor with shape `[batch_size, beam_size, ...]`, whose \
                data type is same as `x`.     
        """
        # TODO: avoid fake shape in compile-time like tile_beam_merge_with_batch
        return nn.reshape(x, shape=(-1, self.beam_size) + x.shape[1:])

    def _merge_batch_beams(self, x):
        """
        Reshape a tensor with shape `[batch_size, beam_size, ...]` to a new
        tensor with shape `[batch_size * beam_size, ...]`. 

        Parameters:
            x(Variable): A tenosr with shape `[batch_size, beam_size, ...]`. The
                data type should be float32, float64, int32, int64 or bool.

        Returns:
            Variable: A tensor with shape `[batch_size * beam_size, ...]`, whose \
                data type is same as `x`.     
        """
        # TODO: avoid fake shape in compile-time like tile_beam_merge_with_batch
        return nn.reshape(x, shape=(-1, ) + x.shape[2:])

    def _expand_to_beam_size(self, x):
        """
        This function takes a tensor t shaped `[batch_size, s0, s1, ...]` composed
        of minibatch entries `t[0], ..., t[batch_size - 1]` and tiles it to have a
        shape `[batch_size, beam_size, s0, s1, ...]` composed of minibatch entries
        `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated
        `beam_size` times.

        Parameters:
            probs(Variable): A tensor with shape `[batch_size, ...]`, representing
                the log probabilities. Its data type should be float32.
            finished(Variable): A tensor with shape `[batch_size, beam_size]`,
                representing the finished status for all beams. Its data type
                should be bool.

        Returns:
            Variable: A tensor with shape `[batch_size, beam_size, ...]`, whose \
                data type is same as `x`.
        """
        x = nn.unsqueeze(x, [1])
        expand_times = [1] * len(x.shape)
        expand_times[1] = self.beam_size
        x = nn.expand(x, expand_times)
        return x

    def _mask_probs(self, probs, finished):
        """
        Mask log probabilities. It forces finished beams to allocate all probability
        mass to eos and unfinished beams to remain unchanged.

        Parameters:
            probs(Variable): A tensor with shape `[batch_size, beam_size, vocab_size]`,
                representing the log probabilities. Its data type should be float32.
            finished(Variable): A tensor with shape `[batch_size, beam_size]`,
                representing the finished status for all beams. Its data type
                should be bool.

        Returns:
            Variable: A tensor with the same shape and data type as `x`, \
                where unfinished beams stay unchanged and finished beams are \
                replaced with a tensor with all probability on the EOS token.
        """
        # TODO: use where_op
        finished = tensor.cast(finished, dtype=probs.dtype)
        probs = nn.elementwise_mul(
            nn.expand(nn.unsqueeze(finished, [2]), [1, 1, self.vocab_size]),
            self.noend_mask_tensor,
            axis=-1) - nn.elementwise_mul(
                probs, (finished - 1), axis=0)
        return probs

    def _gather(self, x, indices, batch_size):
        """
        Gather from the tensor `x` using `indices`.

        Parameters:
            x(Variable): A tensor with shape `[batch_size, beam_size, ...]`.
            indices(Variable): A `int64` tensor with shape `[batch_size, beam_size]`,
                representing the indices that we use to gather.
            batch_size(Variable): A tensor with shape `[1]`. Its data type should
                be int32 or int64.

        Returns:
            Variable: A tensor with the same shape and data type as `x`, \
                representing the gathered tensor.
        """
        # TODO: compatibility of int32 and int64
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
        """
        The structure for the returned value `outputs` of `decoder.step`.
        A namedtuple includes scores, predicted_ids, parent_ids as fields.
        """
        pass

    class StateWrapper(
            collections.namedtuple(
                "StateWrapper",
                ("cell_states", "log_probs", "finished", "lengths"))):
        """
        The structure for the argument `states` of `decoder.step`.
        A namedtuple includes cell_states, log_probs, finished, lengths as fields.
        """
        pass

    def initialize(self, initial_cell_states):
        """
        Initialize the BeamSearchDecoder.

        Parameters:
            initial_cell_states(Variable): A (possibly nested structure of)
                tensor variable[s]. An argument provided by the caller.

        Returns:
            tuple: A tuple( :code:`(initial_inputs, initial_states, finished)` ). \
                `initial_inputs` is a tensor t filled by `start_token` with shape \
                `[batch_size, beam_size, 1]` when `embedding_fn` is None, or the \
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
        state = flatten(initial_cell_states)[0]
        self.batch_size = nn.shape(state)[0]

        self.start_token_tensor = tensor.fill_constant(
            shape=[1], dtype="int64", value=self.start_token)
        self.end_token_tensor = tensor.fill_constant(
            shape=[1], dtype="int64", value=self.end_token)

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
                    [[0.] + [-self.kinf] * (self.beam_size - 1)],
                    dtype="float32")), [self.batch_size, 1])
        # TODO: remove the restriction of force_cpu
        init_finished = tensor.fill_constant_batch_size_like(
            input=state,
            shape=[-1, self.beam_size],
            dtype="bool",
            value=False,
            force_cpu=True)
        init_lengths = tensor.zeros_like(init_inputs)
        init_inputs = self.embedding_fn(
            init_inputs) if self.embedding_fn else init_inputs
        return init_inputs, self.StateWrapper(init_cell_states, log_probs,
                                              init_finished,
                                              init_lengths), init_finished

    def _beam_search_step(self, time, logits, next_cell_states, beam_state):
        """
        Calculate scores and select candidate token ids.

        Parameters:
            time(Variable): An `int64` tensor with shape `[1]` provided by the caller,
                representing the current time step number of decoding.
            logits(Variable): A tensor with shape `[batch_size, beam_size, vocab_size]`,
                representing the logits at the current time step. Its data type is float32.
            next_cell_states(Variable): A (possibly nested structure of) tensor variable[s].
                It has the same structure, shape and data type as the `cell_states` of 
                `initial_states` returned by `initialize()`. It represents the next state 
                from the cell.
            beam_state(Variable): A structure of tensor variables.
                It is same as the `initial_states` returned by `initialize()` for
                the first decoding step and `beam_search_state` returned by
                `initialize()` for the others.
        
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
        self.vocab_size_tensor = tensor.fill_constant(
            shape=[1], dtype="int64", value=self.vocab_size)
        noend_array = [-self.kinf] * self.vocab_size
        noend_array[self.end_token] = 0
        self.noend_mask_tensor = tensor.assign(np.array(noend_array, "float32"))

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
        """
        Perform a beam search decoding step, which uses `cell` to get probabilities,
        and follows a beam search step to calculate scores and select candidate
        token ids.

        Parameters:
            time(Variable): An `int64` tensor with shape `[1]` provided by the caller,
                representing the current time step number of decoding.
            inputs(Variable): A tensor variable. It is same as `initial_inputs`
                returned by `initialize()` for the first decoding step and
                `next_inputs` returned by `step()` for the others.
            states(Variable): A structure of tensor variables.
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
        next_inputs = self.embedding_fn(
            sample_ids) if self.embedding_fn else sample_ids

        return (beam_search_output, beam_search_state, next_inputs, finished)

    def finalize(self, outputs, final_states, sequence_lengths):
        """
        Use `gather_tree` to backtrace along the beam search tree and construct
        the full predicted sequences.

        Parameters:
            outputs(Variable): A structure(namedtuple) of tensor variables,
                The structure and data type is same as `output_dtype`.
                The tensor stacks all time steps' output thus has shape 
                `[time_step, batch_size, ...]`, which is done by the caller. 
            final_states(Variable): A structure(namedtuple) of tensor variables.
                It is the `next_states` returned by `decoder.step` at last
                decoding step, thus has the same structrue, shape and data type
                with states at any time step.
            sequence_lengths(Variable): An `int64` tensor shaped `[batch_size, beam_size]`.
                It contains sequence lengths for each beam determined during
                decoding.

        Returns:
            tuple: A tuple( :code:`(predicted_ids, final_states)` ). \
                `predicted_ids` is an `int64` tensor shaped \
                `[time_step, batch_size, beam_size]`. `final_states` is the same \
                as the input argument `final_states`.
        """
        predicted_ids = nn.gather_tree(outputs.predicted_ids,
                                       outputs.parent_ids)
        # TODO: use FinalBeamSearchDecoderOutput as output
        return predicted_ids, final_states

    @property
    def output_dtype(self):
        """
        The nested structure of data types for beam search output. It is a namedtuple
        including scores, predicted_ids, parent_ids as fields.
        """
        return self.OutputWrapper(
            scores="float32", predicted_ids="int64", parent_ids="int64")


def dynamic_decode(decoder,
                   inits=None,
                   max_step_num=None,
                   output_time_major=False,
                   **kwargs):
    """
    Dynamic decoding performs :code:`decoder.step()` repeatedly until the returned
    Tensor indicating finished status contains all True values or the number of
    decoding step reachs to :attr:`max_step_num`.

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
            all True). Default `None`.
        output_time_major(bool, optional): Indicate the data layout of Tensor included
            in the final outpus(the first returned value of this method). If
            attr:`False`, the data layout would be batch major with shape
            `[batch_size, seq_len, ...]`.  If attr:`True`, the data layout would
            be time major with shape `[seq_len, batch_size, ...]`. Default: `False`.
        **kwargs: Additional keyword arguments. Arguments passed to `decoder.step`. 

    Returns:
        tuple: A tuple( :code:`(final_outputs, final_states)` ) including the final \
            outputs and states, both are Tensor or nested structure of Tensor. \
            `final_outputs` has the same structure and data types as \
            :code:`decoder.output_dtype` , and each Tenser in `final_outputs` \
            is the stacked of all decoding steps' outputs, which might be revised \
            by :code:`decoder.finalize` . `final_states` is the counterpart \
            at last time step of initial states returned by :code:`decoder.initialize` , \
            thus has the same structure with it and has tensors with same shapes \
            and data types.
            

    Examples:

        .. code-block:: python
            
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            from paddle.fluid.layers import GRUCell, BeamSearchDecoder, dynamic_decode

            encoder_output = fluid.data(name="encoder_output",
                                    shape=[-1, 32, 128],
                                    dtype="float32")
            trg_embeder = lambda x: fluid.embedding(
                x, size=[10000, 128], param_attr=fluid.ParamAttr(name="trg_embedding"))
            output_layer = lambda x: layers.fc(x,
                                            size=10000,
                                            num_flatten_dims=len(x.shape) - 1,
                                            param_attr=fluid.ParamAttr(name=
                                                                        "output_w"),
                                            bias_attr=False)
            decoder_cell = GRUCell(hidden_size=128)
            decoder = BeamSearchDecoder(decoder_cell,
                                        start_token=0,
                                        end_token=1,
                                        beam_size=4,
                                        embedding_fn=trg_embeder,
                                        output_fn=output_layer)

            outputs = dynamic_decode(
                decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output))
    """
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
        return nn.transpose(x, [1, 0] + list(range(2, len(x.shape))))

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
