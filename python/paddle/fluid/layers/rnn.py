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
    'RNNCell',
    'GRUCell',
    'LSTMCell',
    'rnn',
    'birnn',
    'dynamic_decode',
    'dynamic_lstm',
    'dynamic_lstmp',
    'gru_unit',
    'lstm_unit',
    'lstm',
]


class RNNCell:
    """
        :api_attr: Static Graph

    RNNCell is the base class for abstraction representing the calculations
    mapping the input and state to the output and new state. It is suitable to
    and mostly used in RNN.
    """

    def call(self, inputs, states, **kwargs):
        r"""
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

    def get_initial_states(
        self,
        batch_ref,
        shape=None,
        dtype='float32',
        init_value=0,
        batch_dim_idx=0,
    ):
        r"""
        Generate initialized states according to provided shape, data type and
        value.

        Parameters:
            batch_ref: A (possibly nested structure of) tensor variable[s].
                The first dimension of the tensor will be used as batch size to
                initialize states.
            shape: A (possibly nested structure of) shape[s], where a shape is
                represented as a list/tuple of integer). -1(for batch size) will
                beautomatically inserted if shape is not started with it. If None,
                property `state_shape` will be used. The default value is None.
            dtype: A (possibly nested structure of) data type[s]. The structure
                must be same as that of `shape`, except when all tensors' in states
                has the same data type, a single data type can be used. If
                property `cell.state_shape` is not available, float32 will be used
                as the data type. The default value is float32.
            init_value: A float value used to initialize states.
            batch_dim_idx: An integer indicating which dimension of the tensor in
                inputs represents batch size.  The default value is 0.

        Returns:
            Variable: tensor variable[s] packed in the same structure provided \
                by shape, representing the initialized states.
        """
        check_variable_and_dtype(
            batch_ref,
            'batch_ref',
            ['float32', 'float64', 'int32', 'int64'],
            'RNNCell',
        )
        check_type(shape, 'shape', (list, tuple, type(None), int), 'RNNCell')
        if isinstance(shape, (list, tuple)):
            shapes = map_structure(lambda x: x, shape)
            if isinstance(shape, list):
                for i, _shape in enumerate(shapes):
                    check_type(_shape, 'shapes[' + str(i) + ']', int, 'RNNCell')
            else:
                check_type(shapes, 'shapes', int, 'RNNCell')
        check_dtype(dtype, 'dtype', ['float32', 'float64'], 'RNNCell')

        # TODO: use inputs and batch_size
        batch_ref = flatten(batch_ref)[0]

        def _is_shape_sequence(seq):
            """For shape, list/tuple of integer is the finest-grained objection"""
            if isinstance(seq, list) or isinstance(seq, tuple):
                if reduce(
                    lambda flag, x: isinstance(x, int) and flag, seq, True
                ):
                    return False
            # TODO: Add check for the illegal
            if isinstance(seq, dict):
                return True
            return isinstance(seq, Sequence) and not isinstance(seq, str)

        class Shape:
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
                value=init_value,
                input_dim_idx=batch_dim_idx,
            ),
            states_shapes,
            states_dtypes,
        )
        return init_states

    @property
    def state_shape(self):
        """
        Abstract method (property).
        Used to initialize states.
        A (possibly nested structure of) shape[s], where a shape is represented
        as a list/tuple of integers (-1 for batch size would be automatically
        inserted into a shape if shape is not started with it).
        Not necessary to be implemented if states are not initialized by
        `get_initial_states` or the `shape` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_shape` in the used cell."
        )

    @property
    def state_dtype(self):
        """
        Abstract method (property).
        Used to initialize states.
        A (possibly nested structure of) data types[s]. The structure must be
        same as that of `shape`, except when all tensors' in states has the same
        data type, a single data type can be used.
        Not necessary to be implemented if states are not initialized
        by `get_initial_states` or the `dtype` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_dtype` in the used cell."
        )


class GRUCell(RNNCell):
    r"""
        :api_attr: Static Graph

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

    def __init__(
        self,
        hidden_size,
        param_attr=None,
        bias_attr=None,
        gate_activation=None,
        activation=None,
        dtype="float32",
        name="GRUCell",
    ):
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
        check_type(hidden_size, 'hidden_size', (int), 'GRUCell')
        check_dtype(dtype, 'dtype', ['float32', 'float64'], 'GRUCell')
        self.hidden_size = hidden_size
        from .. import contrib  # TODO: resolve recurrent import

        self.gru_unit = contrib.layers.rnn_impl.BasicGRUUnit(
            name,
            hidden_size,
            param_attr,
            bias_attr,
            gate_activation,
            activation,
            dtype,
        )

    def call(self, inputs, states):
        r"""
        Perform calculations of GRU.

        Parameters:
            inputs(Variable): A tensor with shape `[batch_size, input_size]`,
                corresponding to :math:`x_t` in the formula. The data type
                should be float32 or float64.
            states(Variable): A tensor with shape `[batch_size, hidden_size]`.
                corresponding to :math:`h_{t-1}` in the formula. The data type
                should be float32 or float64.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` and \
                `new_states` is the same tensor shaped `[batch_size, hidden_size]`, \
                corresponding to :math:`h_t` in the formula. The data type of the \
                tensor is same as that of `states`.
        """

        check_variable_and_dtype(
            inputs, 'inputs', ['float32', 'float64'], 'GRUCell'
        )
        check_variable_and_dtype(
            states, 'states', ['float32', 'float64'], 'GRUCell'
        )
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
    r"""
        :api_attr: Static Graph

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

    def __init__(
        self,
        hidden_size,
        param_attr=None,
        bias_attr=None,
        gate_activation=None,
        activation=None,
        forget_bias=1.0,
        dtype="float32",
        name="LSTMCell",
    ):
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

        check_type(hidden_size, 'hidden_size', (int), 'LSTMCell')
        check_dtype(dtype, 'dtype', ['float32', 'float64'], 'LSTMCell')
        self.hidden_size = hidden_size
        from .. import contrib  # TODO: resolve recurrent import

        self.lstm_unit = contrib.layers.rnn_impl.BasicLSTMUnit(
            name,
            hidden_size,
            param_attr,
            bias_attr,
            gate_activation,
            activation,
            forget_bias,
            dtype,
        )

    def call(self, inputs, states):
        r"""
        Perform calculations of LSTM.

        Parameters:
            inputs(Variable): A tensor with shape `[batch_size, input_size]`,
                corresponding to :math:`x_t` in the formula. The data type
                should be float32 or float64.
            states(Variable): A list of containing two tensors, each shaped
                `[batch_size, hidden_size]`, corresponding to :math:`h_{t-1}, c_{t-1}`
                in the formula. The data type should be float32 or float64.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` is \
                a tensor with shape `[batch_size, hidden_size]`, corresponding \
                to :math:`h_{t}` in the formula; `new_states` is a list containing \
                two tenser variables shaped `[batch_size, hidden_size]`, corresponding \
                to :math:`h_{t}, c_{t}` in the formula. The data type of these \
                tensors all is same as that of `states`.
        """

        check_variable_and_dtype(
            inputs, 'inputs', ['float32', 'float64'], 'LSTMCell'
        )
        check_type(states, 'states', list, 'LSTMCell')
        if isinstance(states, list):
            for i, state in enumerate(states):
                check_variable_and_dtype(
                    state,
                    'state[' + str(i) + ']',
                    ['float32', 'float64'],
                    'LSTMCell',
                )

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


def rnn(
    cell,
    inputs,
    initial_states=None,
    sequence_length=None,
    time_major=False,
    is_reverse=False,
    **kwargs
):
    """
    rnn creates a recurrent neural network specified by RNNCell `cell`,
    which performs :code:`cell.call()` (for dygraph mode :code:`cell.forward`)
    repeatedly until reaches to the maximum length of `inputs`.

    Arguments:
        cell(RNNCellBase): An instance of `RNNCellBase`.
        inputs(Tensor): the input sequences.
            If time_major is True, the shape is
            `[time_steps, batch_size, input_size]`
            else the shape is `[batch_size, time_steps, input_size]`.
        initial_states(Tensor|tuple|list, optional): the initial state of the
            rnn cell. Tensor or a possibly nested structure of tensors. If not
            provided, `cell.get_initial_states` would be called to produce
            the initial state. Defaults to None.
        sequence_length (Tensor, optional): shape `[batch_size]`, dtype: int64
            or int32. The valid lengths of input sequences. Defaults to None.
            If `sequence_length` is not None, the inputs are treated as
            padded sequences. In each input sequence, elements whose time step
            index are not less than the valid length are treated as paddings.
        time_major (bool): Whether the first dimension of the input means the
            time steps. Defaults to False.
        is_reverse (bool, optional): Indicate whether to calculate in the reverse
            order of input sequences. Defaults to False.
        **kwargs: Additional keyword arguments to pass to `forward` of the cell.

    Returns:
        (outputs, final_states)
        outputs (Tensor|list|tuple): the output sequence. Tensor or nested
            structure of Tensors.
            If `time_major` is True, the shape of each tensor in outpus is
            `[time_steps, batch_size, hidden_size]`, else
            `[batch_size, time_steps, hidden_size]`.
        final_states (Tensor|list|tuple): final states. A (possibly nested structure of)
            tensor[s], representing the final state for RNN. It has the same
            structure of intial state. Each tensor in final states has the same
            shape and dtype as the corresponding tensor in initial states.


    Examples:

        .. code-block:: python

            import paddle
            paddle.disable_static()

            cell = paddle.nn.SimpleRNNCell(16, 32)

            inputs = paddle.rand((4, 23, 16))
            prev_h = paddle.randn((4, 32))
            outputs, final_states = paddle.fluid.layers.rnn(cell, inputs, prev_h)

    """
    if _non_static_mode():
        return _rnn_dynamic_graph(
            cell,
            inputs,
            initial_states,
            sequence_length,
            time_major,
            is_reverse,
            **kwargs
        )
    else:
        return _rnn_static_graph(
            cell,
            inputs,
            initial_states,
            sequence_length,
            time_major,
            is_reverse,
            **kwargs
        )


class ArrayWrapper:
    def __init__(self, x):
        self.array = [x]

    def append(self, x):
        self.array.append(x)
        return self

    def __getitem__(self, item):
        return self.array.__getitem__(item)


def _maybe_copy(state, new_state, step_mask):
    """update rnn state or just pass the old state through"""
    new_state = paddle.tensor.math._multiply_with_axis(
        new_state, step_mask, axis=0
    ) + paddle.tensor.math._multiply_with_axis(state, (1 - step_mask), axis=0)
    return new_state


def _transpose_batch_time(x):
    perm = [1, 0] + list(range(2, len(x.shape)))
    return paddle.transpose(x, perm)


def _rnn_dynamic_graph(
    cell,
    inputs,
    initial_states=None,
    sequence_length=None,
    time_major=False,
    is_reverse=False,
    **kwargs
):
    time_step_index = 0 if time_major else 1
    flat_inputs = flatten(inputs)
    time_steps = flat_inputs[0].shape[time_step_index]

    if initial_states is None:
        initial_states = cell.get_initial_states(
            batch_ref=inputs, batch_dim_idx=1 if time_major else 0
        )

    if not time_major:
        inputs = map_structure(_transpose_batch_time, inputs)

    if sequence_length is not None:
        mask = sequence_lod.sequence_mask(
            sequence_length, maxlen=time_steps, dtype=inputs.dtype
        )
        mask = paddle.transpose(mask, [1, 0])

    if is_reverse:
        inputs = map_structure(lambda x: paddle.reverse(x, axis=[0]), inputs)
        mask = (
            paddle.reverse(mask, axis=[0])
            if sequence_length is not None
            else None
        )

    states = initial_states
    outputs = []
    for i in range(time_steps):
        step_inputs = map_structure(lambda x: x[i], inputs)
        step_outputs, new_states = cell(step_inputs, states, **kwargs)
        if sequence_length is not None:
            new_states = map_structure(
                partial(_maybe_copy, step_mask=mask[i]), states, new_states
            )
        states = new_states
        outputs = (
            map_structure(lambda x: ArrayWrapper(x), step_outputs)
            if i == 0
            else map_structure(
                lambda x, x_array: x_array.append(x), step_outputs, outputs
            )
        )

    final_outputs = map_structure(
        lambda x: paddle.stack(x.array, axis=time_step_index), outputs
    )

    if is_reverse:
        final_outputs = map_structure(
            lambda x: paddle.reverse(x, axis=time_step_index), final_outputs
        )

    final_states = new_states
    return final_outputs, final_states


def _rnn_static_graph(
    cell,
    inputs,
    initial_states=None,
    sequence_length=None,
    time_major=False,
    is_reverse=False,
    **kwargs
):
    check_type(inputs, 'inputs', (Variable, list, tuple), 'rnn')
    if isinstance(inputs, (list, tuple)):
        for i, input_x in enumerate(inputs):
            check_variable_and_dtype(
                input_x, 'inputs[' + str(i) + ']', ['float32', 'float64'], 'rnn'
            )
    check_type(
        initial_states,
        'initial_states',
        (Variable, list, tuple, type(None)),
        'rnn',
    )

    check_type(
        sequence_length, 'sequence_length', (Variable, type(None)), 'rnn'
    )

    def _switch_grad(x, stop=False):
        x.stop_gradient = stop
        return x

    if initial_states is None:
        initial_states = cell.get_initial_states(
            batch_ref=inputs, batch_dim_idx=1 if time_major else 0
        )
    initial_states = map_structure(_switch_grad, initial_states)

    if not time_major:
        inputs = map_structure(_transpose_batch_time, inputs)

    if sequence_length:
        max_seq_len = paddle.shape(flatten(inputs)[0])[0]
        mask = sequence_lod.sequence_mask(
            sequence_length,
            maxlen=max_seq_len,
            dtype=flatten(initial_states)[0].dtype,
        )
        mask = paddle.transpose(mask, [1, 0])
    if is_reverse:
        inputs = map_structure(lambda x: paddle.reverse(x, axis=[0]), inputs)
        mask = paddle.reverse(mask, axis=[0]) if sequence_length else None

    # StaticRNN
    rnn = control_flow.StaticRNN()
    with rnn.step():
        inputs = map_structure(rnn.step_input, inputs)
        states = map_structure(rnn.memory, initial_states)
        copy_states = map_structure(lambda x: x, states)
        outputs, new_states = cell(inputs, copy_states, **kwargs)
        assert_same_structure(states, new_states)
        if sequence_length:
            step_mask = rnn.step_input(mask)
            new_states = map_structure(
                partial(_maybe_copy, step_mask=step_mask), states, new_states
            )

        map_structure(rnn.update_memory, states, new_states)
        flat_outputs = flatten(outputs)
        map_structure(rnn.step_output, outputs)
        map_structure(rnn.step_output, new_states)

    rnn_out = rnn()
    final_outputs = rnn_out[: len(flat_outputs)]
    final_outputs = pack_sequence_as(outputs, final_outputs)
    final_states = map_structure(lambda x: x[-1], rnn_out[len(flat_outputs) :])
    final_states = pack_sequence_as(new_states, final_states)

    if is_reverse:
        final_outputs = map_structure(
            lambda x: paddle.reverse(x, axis=[0]), final_outputs
        )

    if not time_major:
        final_outputs = map_structure(_transpose_batch_time, final_outputs)

    return (final_outputs, final_states)


def birnn(
    cell_fw,
    cell_bw,
    inputs,
    initial_states=None,
    sequence_length=None,
    time_major=False,
    **kwargs
):
    """
    birnn creates a bidirectional recurrent neural network specified by
    RNNCell `cell_fw` and `cell_bw`, which performs :code:`cell.call()`
    (for dygraph mode :code:`cell.forward`) repeatedly until reaches to
    the maximum length of `inputs` and then concat the outputs for both RNNs
    along the last axis.

    Arguments:
        cell_fw(RNNCellBase): An instance of `RNNCellBase`.
        cell_bw(RNNCellBase): An instance of `RNNCellBase`.
        inputs(Tensor): the input sequences.
            If time_major is True, the shape is
            `[time_steps, batch_size, input_size]`
            else the shape is `[batch_size, time_steps, input_size]`.
        initial_states(tuple, optional): A tuple of initial states of
            `cell_fw` and `cell_bw`.
            If not provided, `cell.get_initial_states` would be called to
            produce initial state for each cell. Defaults to None.
        sequence_length (Tensor, optional): shape `[batch_size]`, dtype: int64
            or int32. The valid lengths of input sequences. Defaults to None.
            If `sequence_length` is not None, the inputs are treated as
            padded sequences. In each input sequence, elements whose time step
            index are not less than the valid length are treated as paddings.
        time_major (bool): Whether the first dimension of the input means the
            time steps. Defaults to False.
        **kwargs: Additional keyword arguments to pass to `forward` of each cell.

    Returns:
        (outputs, final_states)
        outputs (Tensor): the outputs of the bidirectional RNN. It is the
            concatenation of the outputs from the forward RNN and backward
            RNN along the last axis.
            If time major is True, the shape is `[time_steps, batch_size, size]`,
            else the shape is `[batch_size, time_steps, size]`, where size is
            `cell_fw.hidden_size + cell_bw.hidden_size`.
        final_states (tuple): A tuple of the final states of the forward
            cell and backward cell.

    Examples:

        .. code-block:: python

            import paddle
            paddle.disable_static()

            cell_fw = paddle.nn.LSTMCell(16, 32)
            cell_bw = paddle.nn.LSTMCell(16, 32)

            inputs = paddle.rand((4, 23, 16))
            hf, cf = paddle.rand((4, 32)), paddle.rand((4, 32))
            hb, cb = paddle.rand((4, 32)), paddle.rand((4, 32))
            initial_states = ((hf, cf), (hb, cb))
            outputs, final_states = paddle.fluid.layers.birnn(
                cell_fw, cell_bw, inputs, initial_states)

    """
    if initial_states is None:
        states_fw = cell_fw.get_initial_states(
            batch_ref=inputs, batch_dim_idx=1 if time_major else 0
        )
        states_bw = cell_fw.get_initial_states(
            batch_ref=inputs, batch_dim_idx=1 if time_major else 0
        )
    else:
        states_fw, states_bw = initial_states
    outputs_fw, states_fw = rnn(
        cell_fw,
        inputs,
        states_fw,
        sequence_length,
        time_major=time_major,
        **kwargs
    )

    outputs_bw, states_bw = rnn(
        cell_bw,
        inputs,
        states_bw,
        sequence_length,
        time_major=time_major,
        is_reverse=True,
        **kwargs
    )

    outputs = map_structure(
        lambda x, y: tensor.concat([x, y], -1), outputs_fw, outputs_bw
    )

    final_states = (states_fw, states_bw)
    return outputs, final_states


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
            lambda x: control_flow.array_write(x, step_idx), initial_inputs
        )
        states_arrays = map_structure(
            lambda x: control_flow.array_write(x, step_idx), initial_states
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
                lambda array: control_flow.array_read(array, step_idx),
                inputs_arrays,
            )
            states = map_structure(
                lambda array: control_flow.array_read(array, step_idx),
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
            lambda x, x_array: control_flow.array_write(
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
                lambda x, x_array: control_flow.array_write(
                    x, i=step_idx, array=x_array
                ),
                next_inputs,
                inputs_arrays,
            )
            map_structure(
                lambda x, x_array: control_flow.array_write(
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
            lambda array: control_flow.array_read(array, step_idx),
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


def dynamic_lstm(
    input,
    size,
    h_0=None,
    c_0=None,
    param_attr=None,
    bias_attr=None,
    use_peepholes=True,
    is_reverse=False,
    gate_activation='sigmoid',
    cell_activation='tanh',
    candidate_activation='tanh',
    dtype='float32',
    name=None,
):
    r"""
	:api_attr: Static Graph

    **Note**:
        1. This OP only supports LoDTensor as inputs. If you need to deal with Tensor, please use :ref:`api_fluid_layers_lstm` .
        2. In order to improve efficiency, users must first map the input of dimension [T, hidden_size] to input of [T, 4 * hidden_size], and then pass it to this OP.

    The implementation of this OP include diagonal/peephole connections.
    Please refer to `Gers, F. A., & Schmidhuber, J. (2000) <ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf>`_ .
    If you do not need peephole connections, please set use_peepholes to False .

    This OP computes each timestep as follows:

    .. math::
      i_t = \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + b_{x_i} + b_{h_i})
    .. math::
      f_t = \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + b_{x_f} + b_{h_f})
    .. math::
      o_t = \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + b_{x_o} + b_{h_o})
    .. math::
      \widetilde{c_t} = tanh(W_{cx}x_t + W_{ch}h_{t-1} + b{x_c} + b_{h_c})
    .. math::
      c_t = f_t \odot c_{t-1} + i_t \odot \widetilde{c_t}
    .. math::
      h_t = o_t \odot tanh(c_t)

    The symbolic meanings in the formula are as follows:

    - :math:`x_{t}` represents the input at timestep :math:`t`
    - :math:`h_{t}` represents the hidden state at timestep :math:`t`
    - :math:`h_{t-1}, c_{t-1}` represent the hidden state and cell state at timestep :math:`t-1` , respectively
    - :math:`\widetilde{c_t}` represents the candidate cell state
    - :math:`i_t` , :math:`f_t` and :math:`o_t` represent input gate, forget gate, output gate, respectively
    - :math:`W` represents weight (e.g., :math:`W_{ix}` is the weight of a linear transformation of input :math:`x_{t}` when calculating input gate :math:`i_t` )
    - :math:`b` represents bias (e.g., :math:`b_{i}` is the bias of input gate)
    - :math:`\sigma` represents nonlinear activation function for gate, default sigmoid
    - :math:`\odot` represents the Hadamard product of a matrix, i.e. multiplying the elements of the same position for two matrices with the same dimension to get another matrix with the same dimension

    Parameters:
        input ( :ref:`api_guide_Variable_en` ): LSTM input tensor, multi-dimensional LODTensor of shape :math:`[T, 4*hidden\_size]` . Data type is float32 or float64.
        size (int): must be 4 * hidden_size.
        h_0( :ref:`api_guide_Variable_en` , optional): The initial hidden state of the LSTM, multi-dimensional Tensor of shape :math:`[batch\_size, hidden\_size]` .
                       Data type is float32 or float64. If set to None, it will be a vector of all 0. Default: None.
        c_0( :ref:`api_guide_Variable_en` , optional): The initial hidden state of the LSTM, multi-dimensional Tensor of shape :math:`[batch\_size, hidden\_size]` .
                       Data type is float32 or float64. If set to None, it will be a vector of all 0. `h_0` and `c_0` can be None but only at the same time. Default: None.
        param_attr(ParamAttr, optional): Parameter attribute of weight. If it is None, the default weight parameter attribute is used. Please refer to ref:`api_fluid_ParamAttr' .
                              If the user needs to set this parameter, the dimension must be :math:`[hidden\_size, 4*hidden\_size]` . Default: None.

                              - Weights = :math:`\{ W_{cr},W_{ir},W_{fr},W_{or} \}` , the shape is [hidden_size, 4*hidden_size].

        bias_attr (ParamAttr, optional): The bias attribute for the learnable bias
                              weights, which contains two parts, input-hidden
                              bias weights and peephole connections weights if
                              setting `use_peepholes` to `True`.
                              Please refer to ref:`api_fluid_ParamAttr' . Default: None.

                              1. `use_peepholes = False`
                                 - Biases = {:math:`b_c, b_i, b_f, b_o`}.
                                 - The shape is [1, 4*hidden_size].
                              2. `use_peepholes = True`
                                 - Biases = { :math:`b_c, b_i, b_f, b_o, W_{ic}, \
                                                 W_{fc}, W_{oc}`}.
                                 - The shape is [1, 7*hidden_size].

        use_peepholes (bool, optional): Whether to use peephole connection or not. Default: True.
        is_reverse (bool, optional): Whether to calculate reverse LSTM. Default: False.
        gate_activation (str, optional): The activation for input gate, forget gate and output gate. Default: "sigmoid".
        cell_activation (str, optional): The activation for cell output. Default: "tanh".
        candidate_activation (str, optional): The activation for candidate hidden state. Default: "tanh".
        dtype (str, optional): Data type, can be "float32" or "float64". Default: "float32".
        name (str, optional): A name for this layer. Please refer to :ref:`api_guide_Name` . Default: None.

    Returns:
        tuple ( :ref:`api_guide_Variable` , :ref:`api_guide_Variable` ) :

            The hidden state and cell state of LSTM

                - hidden: LoDTensor with shape of :math:`[T, hidden\_size]` , and its lod and dtype is the same as the input.
                - cell: LoDTensor with shape of :math:`[T, hidden\_size]` , and its lod and dtype is the same as the input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            emb_dim = 256
            vocab_size = 10000
            hidden_dim = 512

            data = fluid.data(name='x', shape=[None], dtype='int64', lod_level=1)
            emb = fluid.embedding(input=data, size=[vocab_size, emb_dim], is_sparse=True)

            forward_proj = fluid.layers.fc(input=emb, size=hidden_dim * 4,
                                           bias_attr=False)

            forward, cell = fluid.layers.dynamic_lstm(
                input=forward_proj, size=hidden_dim * 4, use_peepholes=False)
            forward.shape  # (-1, 512)
            cell.shape  # (-1, 512)
    """
    assert (
        _non_static_mode() is not True
    ), "please use lstm instead of dynamic_lstm in dygraph mode!"
    assert (
        bias_attr is not False
    ), "bias_attr should not be False in dynamic_lstm."

    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'dynamic_lstm'
    )

    check_type(h_0, 'h_0', (Variable, type(None)), 'dynamic_lstm')
    if isinstance(h_0, Variable):
        check_variable_and_dtype(
            h_0, 'h_0', ['float32', 'float64'], 'dynamic_lstm'
        )

    check_type(c_0, 'c_0', (Variable, type(None)), 'dynamic_lstm')
    if isinstance(c_0, Variable):
        check_variable_and_dtype(
            c_0, 'c_0', ['float32', 'float64'], 'dynamic_lstm'
        )

    helper = LayerHelper('lstm', **locals())
    size = size // 4
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 4 * size], dtype=dtype
    )
    bias_size = [1, 7 * size]
    if not use_peepholes:
        bias_size[1] = 4 * size
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True
    )

    hidden = helper.create_variable_for_type_inference(dtype)
    cell = helper.create_variable_for_type_inference(dtype)
    batch_gate = helper.create_variable_for_type_inference(dtype)
    batch_cell_pre_act = helper.create_variable_for_type_inference(dtype)
    inputs = {'Input': input, 'Weight': weight, 'Bias': bias}
    batch_size = input.shape[0]
    if h_0:
        assert h_0.shape == (batch_size, size), (
            'The shape of h0 should be (batch_size, %d)' % size
        )
        inputs['H0'] = h_0
    if c_0:
        assert c_0.shape == (batch_size, size), (
            'The shape of c0 should be (batch_size, %d)' % size
        )
        inputs['C0'] = c_0

    helper.append_op(
        type='lstm',
        inputs=inputs,
        outputs={
            'Hidden': hidden,
            'Cell': cell,
            'BatchGate': batch_gate,
            'BatchCellPreAct': batch_cell_pre_act,
        },
        attrs={
            'use_peepholes': use_peepholes,
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'cell_activation': cell_activation,
            'candidate_activation': candidate_activation,
        },
    )
    return hidden, cell


@deprecated(
    since='2.0.0',
    update_to='paddle.nn.LSTM',
    reason="This API may occur CUDNN errors.",
)
def lstm(
    input,
    init_h,
    init_c,
    max_len,
    hidden_size,
    num_layers,
    dropout_prob=0.0,
    is_bidirec=False,
    is_test=False,
    name=None,
    default_initializer=None,
    seed=-1,
):
    r"""
	:api_attr: Static Graph

    **Note**:
        This OP only supports running on GPU devices.

    This OP implements LSTM operation - `Hochreiter, S., & Schmidhuber, J. (1997) <http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf>`_ .

    The implementation of this OP does not include diagonal/peephole connections.
    Please refer to `Gers, F. A., & Schmidhuber, J. (2000) <ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf>`_ .
    If you need peephole connections, please use :ref:`api_fluid_layers_dynamic_lstm` .

    This OP computes each timestep as follows:

    .. math::
      i_t = \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + b_{x_i} + b_{h_i})
    .. math::
      f_t = \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + b_{x_f} + b_{h_f})
    .. math::
      o_t = \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + b_{x_o} + b_{h_o})
    .. math::
      \widetilde{c_t} = tanh(W_{cx}x_t + W_{ch}h_{t-1} + b{x_c} + b_{h_c})
    .. math::
      c_t = f_t \odot c_{t-1} + i_t \odot \widetilde{c_t}
    .. math::
      h_t = o_t \odot tanh(c_t)

    The symbolic meanings in the formula are as follows:

    - :math:`x_{t}` represents the input at timestep :math:`t`
    - :math:`h_{t}` represents the hidden state at timestep :math:`t`
    - :math:`h_{t-1}, c_{t-1}` represent the hidden state and cell state at timestep :math:`t-1` , respectively
    - :math:`\widetilde{c_t}` represents the candidate cell state
    - :math:`i_t` , :math:`f_t` and :math:`o_t` represent input gate, forget gate, output gate, respectively
    - :math:`W` represents weight (e.g., :math:`W_{ix}` is the weight of a linear transformation of input :math:`x_{t}` when calculating input gate :math:`i_t` )
    - :math:`b` represents bias (e.g., :math:`b_{i}` is the bias of input gate)
    - :math:`\sigma` represents nonlinear activation function for gate, default sigmoid
    - :math:`\odot` represents the Hadamard product of a matrix, i.e. multiplying the elements of the same position for two matrices with the same dimension to get another matrix with the same dimension

    Parameters:
        input ( :ref:`api_guide_Variable_en` ): LSTM input tensor, 3-D Tensor of shape :math:`[batch\_size, seq\_len, input\_dim]` . Data type is float32 or float64
        init_h( :ref:`api_guide_Variable_en` ): The initial hidden state of the LSTM, 3-D Tensor of shape :math:`[num\_layers, batch\_size, hidden\_size]` .
                       If is_bidirec = True, shape should be :math:`[num\_layers*2, batch\_size, hidden\_size]` . Data type is float32 or float64.
        max_len (int): This parameter has no effect and will be discarded.
        init_c( :ref:`api_guide_Variable_en` ): The initial cell state of the LSTM, 3-D Tensor of shape :math:`[num\_layers, batch\_size, hidden\_size]` .
                       If is_bidirec = True, shape should be :math:`[num\_layers*2, batch\_size, hidden\_size]` . Data type is float32 or float64.
        hidden_size (int): hidden size of the LSTM.
        num_layers (int): total layers number of the LSTM.
        dropout_prob(float, optional): dropout prob, dropout ONLY work between rnn layers, NOT between time steps
                             There is NO dropout work on rnn output of the last RNN layers.
                             Default: 0.0.
        is_bidirec (bool, optional): If it is bidirectional. Default: False.
        is_test (bool, optional): If it is in test phrase. Default: False.
        name (str, optional): A name for this layer. If set None, the layer
                         will be named automatically. Default: None.
        default_initializer(Initializer, optional): Where use initializer to initialize the Weight
                         If set None, default initializer will be used. Default: None.
        seed(int, optional): Seed for dropout in LSTM, If it's -1, dropout will use random seed. Default: 1.


    Returns:
        tuple ( :ref:`api_guide_Variable_en` , :ref:`api_guide_Variable_en` , :ref:`api_guide_Variable_en` ) :

                        Three tensors, rnn_out, last_h, last_c:

                        - rnn_out is result of LSTM hidden, shape is :math:`[seq\_len, batch\_size, hidden\_size]` \
                          if is_bidirec set to True, shape will be :math:`[seq\_len, batch\_size, hidden\_size*2]`
                        - last_h is the hidden state of the last step of LSTM \
                          shape is :math:`[num\_layers, batch\_size, hidden\_size]` \
                          if is_bidirec set to True, shape will be :math:`[num\_layers*2, batch\_size, hidden\_size]`
                        - last_c(Tensor): the cell state of the last step of LSTM \
                          shape is :math:`[num\_layers, batch\_size, hidden\_size]` \
                          if is_bidirec set to True, shape will be :math:`[num\_layers*2, batch\_size, hidden\_size]`


    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            paddle.enable_static()

            emb_dim = 256
            vocab_size = 10000
            data = fluid.data(name='x', shape=[None, 100], dtype='int64')
            emb = fluid.embedding(input=data, size=[vocab_size, emb_dim], is_sparse=True)
            batch_size = 100
            dropout_prob = 0.2
            input_size = 100
            hidden_size = 150
            num_layers = 1
            max_len = 12
            init_h = layers.fill_constant( [num_layers, batch_size, hidden_size], 'float32', 0.0 )
            init_c = layers.fill_constant( [num_layers, batch_size, hidden_size], 'float32', 0.0 )
            rnn_out, last_h, last_c = layers.lstm( emb, init_h, init_c, \
                    max_len, hidden_size, num_layers, \
                    dropout_prob=dropout_prob)
            rnn_out.shape  # (-1, 100, 150)
            last_h.shape  # (1, 20, 150)
            last_c.shape  # (1, 20, 150)
    """

    helper = LayerHelper('cudnn_lstm', **locals())
    check_variable_and_dtype(input, 'input', ['float32', 'float64'], 'lstm')
    check_variable_and_dtype(init_h, 'init_h', ['float32', 'float64'], 'lstm')
    check_variable_and_dtype(init_c, 'init_c', ['float32', 'float64'], 'lstm')
    check_type(max_len, 'max_len', (int), 'lstm')
    check_type(hidden_size, 'hidden_size', (int), 'lstm')
    check_type(num_layers, 'num_layers', (int), 'lstm')
    dtype = input.dtype
    input_shape = list(input.shape)
    input_size = input_shape[-1]
    weight_size = 0
    num_dirrection = 2 if is_bidirec == True else 1

    for i in range(num_layers):
        if i == 0:
            input_weight_size = (input_size * hidden_size) * 4 * num_dirrection
        else:
            input_weight_size = (hidden_size * hidden_size) * 4 * num_dirrection
        hidden_weight_size = (hidden_size * hidden_size) * 4 * num_dirrection

        weight_size += input_weight_size + hidden_weight_size
        weight_size += hidden_size * 8 * num_dirrection

    weight = helper.create_parameter(
        attr=helper.param_attr,
        shape=[weight_size],
        dtype=dtype,
        default_initializer=default_initializer,
    )

    out = helper.create_variable_for_type_inference(dtype)
    last_h = helper.create_variable_for_type_inference(dtype)
    last_c = helper.create_variable_for_type_inference(dtype)
    reserve = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.UINT8, stop_gradient=True
    )
    state_out = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.UINT8, stop_gradient=True
    )
    state_out.persistable = True

    helper.append_op(
        type='cudnn_lstm',
        inputs={
            'Input': input,
            'InitH': init_h,
            'InitC': init_c,
            'W': weight,
        },
        outputs={
            'Out': out,
            'LastH': last_h,
            'LastC': last_c,
            'Reserve': reserve,
            'StateOut': state_out,
        },
        attrs={
            'is_bidirec': is_bidirec,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'is_test': is_test,
            'dropout_prob': dropout_prob,
            'seed': seed,
        },
    )
    return out, last_h, last_c


def dynamic_lstmp(
    input,
    size,
    proj_size,
    param_attr=None,
    bias_attr=None,
    use_peepholes=True,
    is_reverse=False,
    gate_activation='sigmoid',
    cell_activation='tanh',
    candidate_activation='tanh',
    proj_activation='tanh',
    dtype='float32',
    name=None,
    h_0=None,
    c_0=None,
    cell_clip=None,
    proj_clip=None,
):
    r"""
	:api_attr: Static Graph

    **Note**:
        1. In order to improve efficiency, users must first map the input of dimension [T, hidden_size] to input of [T, 4 * hidden_size], and then pass it to this OP.

    This OP implements the LSTMP (LSTM Projected) layer.
    The LSTMP layer has a separate linear mapping layer behind the LSTM layer. -- `Sak, H., Senior, A., & Beaufays, F. (2014) <https://ai.google/research/pubs/pub43905.pdf>`_ .

    Compared with the standard LSTM layer, LSTMP has an additional linear mapping layer,
    which is used to map from the original hidden state :math:`h_t` to the lower dimensional state :math:`r_t` .
    This reduces the total number of parameters and computational complexity, especially when the output unit is relatively large.

    The default implementation of the OP contains diagonal/peephole connections,
    please refer to `Gers, F. A., & Schmidhuber, J. (2000) <ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf>`_ .
    If you need to disable the peephole connections, set use_peepholes to False.

    This OP computes each timestep as follows:

    .. math::
      i_t = \sigma(W_{ix}x_{t} + W_{ir}r_{t-1} + W_{ic}c_{t-1} + b_i)
    .. math::
          f_t = \sigma(W_{fx}x_{t} + W_{fr}r_{t-1} + W_{fc}c_{t-1} + b_f)
    .. math::
          o_t = \sigma(W_{ox}x_{t} + W_{or}r_{t-1} + W_{oc}c_{t-1} + b_o)
    .. math::
          \widetilde{c_t} = act_g(W_{cx}x_t + W_{cr}r_{t-1} + b_c)
    .. math::
          c_t = f_t \odot c_{t-1} + i_t \odot \widetilde{c_t}
    .. math::
          h_t = o_t \odot act_h(c_t)
    .. math::
          r_t = \overline{act_h}(W_{rh}h_t)

    The symbolic meanings in the formula are as follows:

    - :math:`x_{t}` represents the input at timestep :math:`t`
    - :math:`h_{t}` represents the hidden state at timestep :math:`t`
    - :math:`r_{t}` : represents the state of the projected output of the hidden state :math:`h_{t}`
    - :math:`h_{t-1}, c_{t-1}, r_{t-1}` represent the hidden state, cell state and projected output at timestep :math:`t-1` , respectively
    - :math:`\widetilde{c_t}` represents the candidate cell state
    - :math:`i_t` , :math:`f_t` and :math:`o_t` represent input gate, forget gate, output gate, respectively
    - :math:`W` represents weight (e.g., :math:`W_{ix}` is the weight of a linear transformation of input :math:`x_{t}` when calculating input gate :math:`i_t` )
    - :math:`b` represents bias (e.g., :math:`b_{i}` is the bias of input gate)
    - :math:`\sigma` represents nonlinear activation function for gate, default sigmoid
    - :math:`\odot` represents the Hadamard product of a matrix, i.e. multiplying the elements of the same position for two matrices with the same dimension to get another matrix with the same dimension

    Parameters:
        input( :ref:`api_guide_Variable_en` ): The input of dynamic_lstmp layer, which supports
                         variable-time length input sequence.
                         It is a multi-dimensional LODTensor of shape :math:`[T, 4*hidden\_size]` . Data type is float32 or float64.
        size(int): must be 4 * hidden_size.
        proj_size(int): The size of projection output.
        param_attr(ParamAttr, optional): Parameter attribute of weight. If it is None, the default weight parameter attribute is used. Please refer to ref:`api_fluid_ParamAttr' .
                              If the user needs to set this parameter, the dimension must be :math:`[hidden\_size, 4*hidden\_size]` . Default: None.

                              - Weights = :math:`\{ W_{cr},W_{ir},W_{fr},W_{or} \}` , the shape is [P, 4*hidden_size] , where P is the projection size.
                              - Projection weight  = :math:`\{ W_{rh} \}` , the shape is [hidden_size, P].

        bias_attr (ParamAttr, optional): The bias attribute for the learnable bias
                              weights, which contains two parts, input-hidden
                              bias weights and peephole connections weights if
                              setting `use_peepholes` to `True`.
                              Please refer to ref:`api_fluid_ParamAttr' . Default: None.

                              1. `use_peepholes = False`
                                 - Biases = {:math:`b_c, b_i, b_f, b_o`}.
                                 - The shape is [1, 4*hidden_size].
                              2. `use_peepholes = True`
                                 - Biases = { :math:`b_c, b_i, b_f, b_o, W_{ic}, \
                                                 W_{fc}, W_{oc}`}.
                                 - The shape is [1, 7*hidden_size].

        use_peepholes (bool, optional): Whether to use peephole connection or not. Default True.
        is_reverse (bool, optional): Whether to calculate reverse LSTM. Default False.
        gate_activation (str, optional): The activation for input gate, forget gate and output gate. Default "sigmoid".
        cell_activation (str, optional): The activation for cell output. Default "tanh".
        candidate_activation (str, optional): The activation for candidate hidden state. Default "tanh".
        proj_activation(str, optional): The activation for projection output. Default "tanh".
        dtype (str, optional): Data type, can be "float32" or "float64". Default "float32".
        name (str, optional): A name for this layer. Please refer to :ref:`api_guide_Name` . Default: None.
        h_0( :ref:`api_guide_Variable` , optional): The initial hidden state is an optional input, default is zero.
                       This is a tensor with shape :math:`[batch\_size, P]` , where P is the projection size. Default: None.
        c_0( :ref:`api_guide_Variable` , optional): The initial cell state is an optional input, default is zero.
                       This is a tensor with shape :math:`[batch\_size, P]` , where P is the projection size.
                       `h_0` and `c_0` can be None but only at the same time. Default: None.
        cell_clip(float, optional): If not None, the cell state is clipped
                             by this value prior to the cell output activation. Default: None.
        proj_clip(float, optional): If `num_proj > 0` and `proj_clip` is
                            provided, then the projected values are clipped elementwise to within
                            `[-proj_clip, proj_clip]`. Default: None.

    Returns:
        tuple ( :ref:`api_guide_Variable` , :ref:`api_guide_Variable` ) :

                The hidden state and cell state of LSTMP

                - hidden: LoDTensor with shape of :math:`[T, P]` , and its lod and dtype is the same as the input.
                - cell: LoDTensor with shape of :math:`[T, hidden\_size]` , and its lod and dtype is the same as the input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            dict_dim, emb_dim = 128, 64
            data = fluid.data(name='sequence', shape=[None], dtype='int64', lod_level=1)
            emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim, proj_dim = 512, 256
            fc_out = fluid.layers.fc(input=emb, size=hidden_dim * 4,
                                    act=None, bias_attr=None)
            proj_out, last_c = fluid.layers.dynamic_lstmp(input=fc_out,
                                                    size=hidden_dim * 4,
                                                    proj_size=proj_dim,
                                                    use_peepholes=False,
                                                    is_reverse=True,
                                                    cell_activation="tanh",
                                                    proj_activation="tanh")
            proj_out.shape  # (-1, 256)
            last_c.shape  # (-1, 512)
    """

    assert (
        _non_static_mode() is not True
    ), "please use lstm instead of dynamic_lstmp in dygraph mode!"

    assert (
        bias_attr is not False
    ), "bias_attr should not be False in dynamic_lstmp."

    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'dynamic_lstmp'
    )

    check_type(h_0, 'h_0', (Variable, type(None)), 'dynamic_lstmp')
    if isinstance(h_0, Variable):
        check_variable_and_dtype(
            h_0, 'h_0', ['float32', 'float64'], 'dynamic_lstmp'
        )

    check_type(c_0, 'c_0', (Variable, type(None)), 'dynamic_lstmp')
    if isinstance(c_0, Variable):
        check_variable_and_dtype(
            c_0, 'c_0', ['float32', 'float64'], 'dynamic_lstmp'
        )

    helper = LayerHelper('lstmp', **locals())
    size = size // 4
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[proj_size, 4 * size], dtype=dtype
    )
    proj_weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, proj_size], dtype=dtype
    )
    bias_size = [1, 7 * size]
    if not use_peepholes:
        bias_size[1] = 4 * size
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True
    )

    projection = helper.create_variable_for_type_inference(dtype)
    cell = helper.create_variable_for_type_inference(dtype)
    ordered_proj0 = helper.create_variable_for_type_inference(dtype)
    batch_hidden = helper.create_variable_for_type_inference(dtype)
    batch_gate = helper.create_variable_for_type_inference(dtype)
    batch_cell_pre_act = helper.create_variable_for_type_inference(dtype)
    inputs = {
        'Input': input,
        'Weight': weight,
        'ProjWeight': proj_weight,
        'Bias': bias,
    }
    batch_size = input.shape[0]
    if h_0:
        assert h_0.shape == (batch_size, proj_size), (
            'The shape of h0 should be (batch_size, %d)' % proj_size
        )
        inputs['H0'] = h_0
    if c_0:
        assert c_0.shape == (batch_size, size), (
            'The shape of c0 should be (batch_size, %d)' % size
        )
        inputs['C0'] = c_0

    if cell_clip:
        assert cell_clip >= 0, "cell_clip should not be negative."
    if proj_clip:
        assert proj_clip >= 0, "proj_clip should not be negative."

    helper.append_op(
        type='lstmp',
        inputs=inputs,
        outputs={
            'Projection': projection,
            'Cell': cell,
            'BatchHidden': batch_hidden,
            'BatchGate': batch_gate,
            'BatchCellPreAct': batch_cell_pre_act,
        },
        attrs={
            'use_peepholes': use_peepholes,
            'cell_clip': cell_clip,
            'proj_clip': proj_clip,
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'cell_activation': cell_activation,
            'candidate_activation': candidate_activation,
            'proj_activation': proj_activation,
        },
    )
    return projection, cell


def gru_unit(
    input,
    hidden,
    size,
    param_attr=None,
    bias_attr=None,
    activation='tanh',
    gate_activation='sigmoid',
    origin_mode=False,
):
    r"""
	:api_attr: Static Graph

    Gated Recurrent Unit (GRU) RNN cell. This operator performs GRU calculations for
    one time step and it supports these two modes:

    If ``origin_mode`` is True, then the formula used is from paper
    `Learning Phrase Representations using RNN Encoder Decoder for Statistical
    Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_ .

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \\tilde{h_t}


    if ``origin_mode`` is False, then the formula used is from paper
    `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
    Modeling  <https://arxiv.org/pdf/1412.3555.pdf>`_

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = (1-u_t) \odot h_{t-1} + u_t \odot \\tilde{h_t}

    :math:`x_t` is the input of current time step, but it is not ``input`` .
    This operator does not include the calculations :math:`W_{ux}x_{t}, W_{rx}x_{t}, W_{cx}x_{t}` ,
    **Note** thus a fully-connect layer whose size is 3 times of GRU hidden size should
    be used before this operator, and the output should be used as ``input`` here.
    :math:`h_{t-1}` is the hidden state from previous time step.
    :math:`u_t` , :math:`r_t` , :math:`\\tilde{h_t}` and :math:`h_t` stand for
    update gate, reset gate, candidate hidden and hidden output separately.
    :math:`W_{uh}, b_u` , :math:`W_{rh}, b_r` and :math:`W_{ch}, b_c` stand for
    the weight matrix and bias used in update gate, reset gate, candidate hidden
    calculations. For implementation, the three weight matrix are merged into a
    tensor shaped :math:`[D, D \\times 3]` , the three bias are concatenated as
    a tensor shaped :math:`[1, D \\times 3]` , where :math:`D` stands for the
    hidden size; The data layout of weight tensor is: :math:`W_{uh}` and :math:`W_{rh}`
    are concatenated with shape :math:`[D, D  \\times 2]` lying on the first part,
    and :math:`W_{ch}` lying on the latter part with shape :math:`[D, D]` .


    Args:
        input(Variable): A 2D Tensor representing the input after linear projection
            after linear projection. Its shape should be :math:`[N, D \\times 3]` ,
            where :math:`N` stands for batch size, :math:`D` for the hidden size.
            The data type should be float32 or float64.
        hidden(Variable): A 2D Tensor representing the hidden state from previous step.
            Its shape should be :math:`[N, D]` , where :math:`N` stands for batch size,
            :math:`D` for the hidden size. The data type should be same as ``input`` .
        size(int): Indicate the hidden size.
        param_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        activation(str, optional): The activation function corresponding to
            :math:`act_c` in the formula. "sigmoid", "tanh", "relu" and "identity"
            are supported. Default "tanh".
        gate_activation(str, optional): The activation function corresponding to
            :math:`act_g` in the formula. "sigmoid", "tanh", "relu" and "identity"
            are supported. Default "sigmoid".

    Returns:
        tuple: The tuple contains three Tensor variables with the same data type \
            as ``input`` . They represent the hidden state for next time step ( :math:`h_t` ), \
            reset previous hidden state ( :math:`r_t \odot h_{t-1}` ), and the \
            concatenation of :math:`h_t, r_t, \\tilde{h_t}` . And they have shape \
            :math:`[N, D]` , :math:`[N, D]` , :math:`[N, D \times 3]` separately. \
            Usually only the hidden state for next time step ( :math:`h_t` ) is used \
            as output and state, the other two are intermediate results of calculations.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            dict_dim, emb_dim = 128, 64
            data = fluid.data(name='step_data', shape=[None], dtype='int64')
            emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim = 512
            x = fluid.layers.fc(input=emb, size=hidden_dim * 3)
            pre_hidden = fluid.data(
                name='pre_hidden', shape=[None, hidden_dim], dtype='float32')
            hidden = fluid.layers.gru_unit(
                input=x, hidden=pre_hidden, size=hidden_dim * 3)

    """
    check_variable_and_dtype(input, 'input', ['float32', 'float64'], 'gru_unit')
    check_variable_and_dtype(
        hidden, 'hidden', ['float32', 'float64'], 'gru_unit'
    )
    check_type(size, 'size', (int), 'gru_unit')
    activation_dict = dict(
        identity=0,
        sigmoid=1,
        tanh=2,
        relu=3,
    )
    activation = activation_dict[activation]
    gate_activation = activation_dict[gate_activation]

    helper = LayerHelper('gru_unit', **locals())
    dtype = helper.input_dtype()
    size = size // 3

    # create weight
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 3 * size], dtype=dtype
    )

    gate = helper.create_variable_for_type_inference(dtype)
    reset_hidden_pre = helper.create_variable_for_type_inference(dtype)
    updated_hidden = helper.create_variable_for_type_inference(dtype)
    inputs = {'Input': input, 'HiddenPrev': hidden, 'Weight': weight}
    # create bias
    if helper.bias_attr:
        bias_size = [1, 3 * size]
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True
        )
        inputs['Bias'] = bias

    helper.append_op(
        type='gru_unit',
        inputs=inputs,
        outputs={
            'Gate': gate,
            'ResetHiddenPrev': reset_hidden_pre,
            'Hidden': updated_hidden,
        },
        attrs={
            'activation': 2,  # tanh
            'gate_activation': 1,  # sigmoid
            'origin_mode': origin_mode,
        },
    )

    return updated_hidden, reset_hidden_pre, gate


def lstm_unit(
    x_t,
    hidden_t_prev,
    cell_t_prev,
    forget_bias=0.0,
    param_attr=None,
    bias_attr=None,
    name=None,
):
    r"""
	:api_attr: Static Graph

    Long-Short Term Memory (LSTM) RNN cell. This operator performs LSTM calculations for
    one time step, whose implementation is based on calculations described in `RECURRENT
    NEURAL NETWORK REGULARIZATION <http://arxiv.org/abs/1409.2329>`_  .

    We add forget_bias to the biases of the forget gate in order to
    reduce the scale of forgetting. The formula is as follows:

    .. math::

        i_{t} & = \sigma(W_{x_{i}}x_{t} + W_{h_{i}}h_{t-1} + b_{i})

        f_{t} & = \sigma(W_{x_{f}}x_{t} + W_{h_{f}}h_{t-1} + b_{f} + forget\\_bias)

        c_{t} & = f_{t}c_{t-1} + i_{t} tanh (W_{x_{c}}x_{t} + W_{h_{c}}h_{t-1} + b_{c})

        o_{t} & = \sigma(W_{x_{o}}x_{t} + W_{h_{o}}h_{t-1} + b_{o})

        h_{t} & = o_{t} tanh (c_{t})

    :math:`x_{t}` stands for ``x_t`` , corresponding to the input of current time step;
    :math:`h_{t-1}` and :math:`c_{t-1}` correspond to ``hidden_t_prev`` and ``cell_t_prev`` ,
    representing the output of from previous time step.
    :math:`i_{t}, f_{t}, c_{t}, o_{t}, h_{t}` are input gate, forget gate, cell, output gate
    and hidden calculation.

    Args:
        x_t(Variable): A 2D Tensor representing the input of current time step.
            Its shape should be :math:`[N, M]` , where :math:`N` stands for batch
            size, :math:`M` for the feature size of input. The data type should
            be float32 or float64.
        hidden_t_prev(Variable): A 2D Tensor representing the hidden value from
            previous step. Its shape should be :math:`[N, D]` , where :math:`N`
            stands for batch size, :math:`D` for the hidden size. The data type
            should be same as ``x_t`` .
        cell_t_prev(Variable): A 2D Tensor representing the cell value from
            previous step. It has the same shape and data type with ``hidden_t_prev`` .
        forget_bias (float, optional): :math:`forget\\_bias` added to the biases
            of the forget gate. Default 0.
        param_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        tuple: The tuple contains two Tensor variables with the same shape and \
            data type with ``hidden_t_prev`` , representing the hidden value and \
            cell value which correspond to :math:`h_{t}` and :math:`c_{t}` in \
            the formula.

    Raises:
        ValueError: Rank of x_t must be 2.
        ValueError: Rank of hidden_t_prev must be 2.
        ValueError: Rank of cell_t_prev must be 2.
        ValueError: The 1st dimensions of x_t, hidden_t_prev and cell_t_prev must be the same.
        ValueError: The 2nd dimensions of hidden_t_prev and cell_t_prev must be the same.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            dict_dim, emb_dim, hidden_dim = 128, 64, 512
            data = fluid.data(name='step_data', shape=[None], dtype='int64')
            x = fluid.embedding(input=data, size=[dict_dim, emb_dim])
            pre_hidden = fluid.data(
                name='pre_hidden', shape=[None, hidden_dim], dtype='float32')
            pre_cell = fluid.data(
                name='pre_cell', shape=[None, hidden_dim], dtype='float32')
            hidden = fluid.layers.lstm_unit(
                x_t=x,
                hidden_t_prev=pre_hidden,
                cell_t_prev=pre_cell)
    """
    helper = LayerHelper('lstm_unit', **locals())
    check_variable_and_dtype(x_t, 'x_t', ['float32', 'float64'], 'lstm_unit')
    check_variable_and_dtype(
        hidden_t_prev, 'hidden_t_prev', ['float32', 'float64'], 'lstm_unit'
    )
    check_variable_and_dtype(
        cell_t_prev, 'cell_t_prev', ['float32', 'float64'], 'lstm_unit'
    )
    if len(x_t.shape) != 2:
        raise ValueError("Rank of x_t must be 2.")

    if len(hidden_t_prev.shape) != 2:
        raise ValueError("Rank of hidden_t_prev must be 2.")

    if len(cell_t_prev.shape) != 2:
        raise ValueError("Rank of cell_t_prev must be 2.")

    if (
        x_t.shape[0] != hidden_t_prev.shape[0]
        or x_t.shape[0] != cell_t_prev.shape[0]
    ):
        raise ValueError(
            "The 1st dimensions of x_t, hidden_t_prev and "
            "cell_t_prev must be the same."
        )

    if hidden_t_prev.shape[1] != cell_t_prev.shape[1]:
        raise ValueError(
            "The 2nd dimensions of hidden_t_prev and "
            "cell_t_prev must be the same."
        )

    if bias_attr is None:
        bias_attr = ParamAttr()

    size = cell_t_prev.shape[1]
    concat_out = nn.concat(input=[x_t, hidden_t_prev], axis=1)
    fc_out = nn.fc(
        input=concat_out,
        size=4 * size,
        param_attr=param_attr,
        bias_attr=bias_attr,
    )
    dtype = x_t.dtype
    c = helper.create_variable_for_type_inference(dtype)
    h = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type='lstm_unit',
        inputs={"X": fc_out, "C_prev": cell_t_prev},
        outputs={"C": c, "H": h},
        attrs={"forget_bias": forget_bias},
    )

    return h, c
