#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import collections
import itertools
import six
import math
import sys
import warnings
from functools import partial, reduce

import paddle
from paddle import framework
from paddle.nn import functional as F
from paddle.nn import initializer as I
from paddle.fluid.dygraph import Layer, LayerList
from paddle.fluid.layers import utils
from paddle.fluid.layers.utils import map_structure, flatten, pack_sequence_as
from paddle.fluid.data_feeder import convert_dtype

__all__ = [
    'RNNCellBase',
    'SimpleRNNCell',
    'LSTMCell',
    'GRUCell',
    'RNN',
    'BiRNN',
    'SimpleRNN',
    'LSTM',
    'GRU',
]


def split_states(states, bidirectional=False, state_components=1):
    r"""
    Split states of RNN network into possibly nested list or tuple of
    states of each RNN cells of the RNN network.

    Arguments:
        states (Tensor|tuple|list): the concatenated states for RNN network.
            When `state_components` is 1, states in a Tensor with shape
            `(L*D, N, C)` where `L` is the number of layers of the RNN 
            network, `D` is the number of directions of the RNN network(1 
            for unidirectional RNNs and 2 for bidirectional RNNs), `N` is 
            the batch size of the input to the RNN network, `C` is the 
            hidden size of the RNN network. 

            When `state_components` is larger than 1, `states` is a tuple of 
            `state_components` Tensors that meet the requirements described 
            above. 
            
            For SimpleRNNs and GRUs, `state_components` is 1, and for LSTMs, 
            `state_components` is 2.
        bidirectional (bool): whether the state is of a bidirectional RNN 
            network. Defaults to False.
        state_components (int): the number of the components of the states. see
            `states` above. Defaults to 1.
    
    Returns:
        A nested list or tuple of RNN cell states. 
        If `bidirectional` is True, it can be indexed twice to get an RNN 
        cell state. The first index indicates the layer, the second index 
        indicates the direction.
        If `bidirectional` is False, it can be indexed once to get an RNN
        cell state. The index indicates the layer.
        Note that if `state_components` is larger than 1, an RNN cell state
        can be indexed one more time to get a tensor of shape(N, C), where 
        `N` is the batch size of the input to the RNN cell, and `C` is the
        hidden size of the RNN cell.
    """
    if state_components == 1:
        states = paddle.unstack(states)
        if not bidirectional:
            return states
        else:
            return list(zip(states[::2], states[1::2]))
    else:
        assert len(states) == state_components
        states = tuple([paddle.unstack(item) for item in states])
        if not bidirectional:
            return list(zip(*states))
        else:
            states = list(zip(*states))
            return list(zip(states[::2], states[1::2]))


def concat_states(states, bidirectional=False, state_components=1):
    r"""
    Concatenate a possibly nested list or tuple of RNN cell states into a 
    compact form.

    Arguments:
        states (list|tuple): a possibly nested list or tuple of RNN cell 
            states. 
            If `bidirectional` is True, it can be indexed twice to get an 
            RNN cell state. The first index indicates the layer, the second 
            index indicates the direction.
            If `bidirectional` is False, it can be indexed once to get an RNN
            cell state. The index indicates the layer.
            Note that if `state_components` is larger than 1, an RNN cell 
            state can be indexed one more time to get a tensor of shape(N, C), 
            where `N` is the batch size of the input to the RNN cell, and 
            `C` is the hidden size of the RNN cell. 
        bidirectional (bool): whether the state is of a bidirectional RNN 
            network. Defaults to False.
        state_components (int): the number of the components of the states. see
            `states` above. Defaults to 1.
    
    Returns:
        Concatenated states for RNN network.
        When `state_components` is 1, states in a Tensor with shape
        `(L\*D, N, C)` where `L` is the number of layers of the RNN 
        network, `D` is the number of directions of the RNN network(1 for 
        unidirectional RNNs and 2 for bidirectional RNNs), `N` is the batch 
        size of the input to the RNN network, `C` is the hidden size of the 
        RNN network.
        
    """
    if state_components == 1:
        return paddle.stack(flatten(states))
    else:
        states = flatten(states)
        componnets = []
        for i in range(state_components):
            componnets.append(states[i::state_components])
        return [paddle.stack(item) for item in componnets]


class RNNCellBase(Layer):
    r"""
    RNNCellBase is the base class for abstraction representing the calculations
    mapping the input and state to the output and new state. It is suitable to
    and mostly used in RNN.
    """

    def get_initial_states(self,
                           batch_ref,
                           shape=None,
                           dtype=None,
                           init_value=0,
                           batch_dim_idx=0):
        r"""
        Generate initialized states according to provided shape, data type and
        value.
        Arguments:
            batch_ref: A (possibly nested structure of) tensor variable[s].
                The first dimension of the tensor will be used as batch size to
                initialize states.
            shape: A (possibly nested structure of) shape[s], where a shape is
                represented as a list/tuple of integer). -1(for batch size) will
                beautomatically inserted if shape is not started with it. If None,
                property `state_shape` will be used. The default value is None.
            dtype: A (possibly nested structure of) data type[s]. The structure
                must be same as that of `shape`, except when all tensors' in states
                has the same data type, a single data type can be used. If None and
                property `cell.state_shape` is not available, float32 will be used
                as the data type. The default value is None.
            init_value: A float value used to initialize states.
            batch_dim_idx: An integer indicating which dimension of the tensor in
                inputs represents batch size.  The default value is 0.
        Returns:
            Variable: tensor variable[s] packed in the same structure provided \
                by shape, representing the initialized states.
        """
        # TODO: use inputs and batch_size
        batch_ref = flatten(batch_ref)[0]

        def _is_shape_sequence(seq):
            if sys.version_info < (3, ):
                integer_types = (
                    int,
                    long, )
            else:
                integer_types = (int, )
            """For shape, list/tuple of integer is the finest-grained objection"""
            if (isinstance(seq, list) or isinstance(seq, tuple)):
                if reduce(lambda flag, x: isinstance(x, integer_types) and flag,
                          seq, True):
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
            states_dtypes = framework.get_default_dtype()
        if len(flatten(states_dtypes)) == 1:
            dtype = flatten(states_dtypes)[0]
            states_dtypes = map_structure(lambda shape: dtype, states_shapes)

        init_states = map_structure(
            lambda shape, dtype: paddle.fluid.layers.fill_constant_batch_size_like(
                input=batch_ref,
                shape=shape.shape,
                dtype=dtype,
                value=init_value,
                input_dim_idx=batch_dim_idx), states_shapes, states_dtypes)
        return init_states

    @property
    def state_shape(self):
        r"""
        Abstract method (property).
        Used to initialize states.
        A (possiblely nested structure of) shape[s], where a shape is represented
        as a list/tuple of integers (-1 for batch size would be automatically
        inserted into a shape if shape is not started with it).
        Not necessary to be implemented if states are not initialized by
        `get_initial_states` or the `shape` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_shape` in the used cell.")

    @property
    def state_dtype(self):
        r"""
        Abstract method (property).
        Used to initialize states.
        A (possiblely nested structure of) data types[s]. The structure must be
        same as that of `shape`, except when all tensors' in states has the same
        data type, a signle data type can be used.
        Not necessary to be implemented if states are not initialized
        by `get_initial_states` or the `dtype` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_dtype` in the used cell.")


class SimpleRNNCell(RNNCellBase):
    r"""
    Elman RNN (SimpleRNN) cell.

    The formula used is as follows:

    .. math::
        h_{t} & = \mathrm{tanh}(W_{ih}x_{t} + b_{ih} + W_{hh}h{t-1} + b_{hh})
        y_{t} & = h_{t}
    
    where :math:`\sigma` is the sigmoid fucntion, and \* is the elemetwise 
    multiplication operator.

    Please refer to `Finding Structure in Time 
    <https://crl.ucsd.edu/~elman/Papers/fsit.pdf>`_ for more details.
    
    Arguments:
        input_size (int): The input size.
        hidden_size (int): The hidden size.
        nonlinearity (str): The activation in the SimpleRNN cell. It can be 
            `tanh` or `relu`. Defaults to `tanh`.
        weight_ih_attr(ParamAttr, optional): The parameter attribute for 
            `weight_ih`. Default: None.
        weight_hh_attr(ParamAttr, optional): The parameter attribute for 
            `weight_hh`. Default: None.
        bias_ih_attr (ParamAttr, optional): The parameter attribute for the 
            `bias_ih`. Default: None.
        bias_ih_attr (ParamAttr, optional): The parameter attribute for the 
            `bias_hh`. Default: None.
        name (str, optional): Name for the operation (optional, default is 
            None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python
            import paddle
            paddle.disable_static()

            x = paddle.randn((4, 16))
            prev_h = paddle.randn((4, 32))

            cell = paddle.nn.SimpleRNNCell(16, 32)
            y, h = cell(x, prev_h)

    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 nonlinearity="tanh",
                 weight_ih_attr=None,
                 weight_hh_attr=None,
                 bias_ih_attr=None,
                 bias_hh_attr=None,
                 name=None):
        super(SimpleRNNCell, self).__init__()
        std = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = self.create_parameter(
            (hidden_size, input_size),
            weight_ih_attr,
            default_initializer=I.Uniform(-std, std))
        self.weight_hh = self.create_parameter(
            (hidden_size, hidden_size),
            weight_hh_attr,
            default_initializer=I.Uniform(-std, std))
        self.bias_ih = self.create_parameter(
            (hidden_size, ),
            bias_ih_attr,
            is_bias=True,
            default_initializer=I.Uniform(-std, std))
        self.bias_hh = self.create_parameter(
            (hidden_size, ),
            bias_hh_attr,
            is_bias=True,
            default_initializer=I.Uniform(-std, std))

        self.input_size = input_size
        self.hidden_size = hidden_size
        if nonlinearity not in ["tanh", "relu"]:
            raise ValueError(
                "nonlinearity for SimpleRNNCell should be tanh or relu, "
                "but get {}".format(nonlinearity))
        self.nonlinearity = nonlinearity
        self._nonlinear_fn = paddle.tanh \
            if nonlinearity == "tanh" \
            else F.relu

    def forward(self, inputs, states=None):
        r"""
        Given the input and previous atate, compute the output and update state.

        Arguments:
            inputs (Tensor): shape `[batch_size, input_size]`, the input, 
                corresponding to :math:`x_t` in the formula.
            states (Tensor, optional): shape `[batch_size, hidden_size]`, the
                previous hidden state, corresponding to :math:`h_{t-1}` in the 
                formula. When states is None, zero state is used. Defaults to 
                None.
        Returns:
            (outputs, new_states)
            outputs (Tensor): shape `[batch_size, hidden_size]`, the output, 
                corresponding to :math:`h_{t}` in the formula.
            states (Tensor): shape `[batch_size, hidden_size]`, the new hidden 
                state, corresponding to :math:`h_{t}` in the formula.

        """
        if states is None:
            states = self.get_initial_states(inputs, self.state_shape)
        pre_h = states
        i2h = paddle.matmul(inputs, self.weight_ih, transpose_y=True)
        if self.bias_ih is not None:
            i2h += self.bias_ih
        h2h = paddle.matmul(pre_h, self.weight_hh, transpose_y=True)
        if self.bias_hh is not None:
            h2h += self.bias_hh
        h = self._nonlinear_fn(i2h + h2h)
        return h, h

    @property
    def state_shape(self):
        return (self.hidden_size, )


class LSTMCell(RNNCellBase):
    r"""
    Long-Short Term Memory(LSTM) RNN cell.

    The formula used is as follows:

    .. math::
        i_{t} & = \sigma(W_{ii}x_{t} + b_{ii} + W_{hi}h_{t-1} + b_{hi})
        f_{t} & = \sigma(W_{if}x_{t} + b_{if} + W_{hf}h_{t-1} + b_{hf})
        o_{t} & = \sigma(W_{io}x_{t} + b_{io} + W_{ho}h_{t-1} + b_{ho})
        \\widetilde{c}_{t} & = \\tanh (W_{ig}x_{t} + b_{ig} + W_{hg}h_{t-1} + b_{hg})
        c_{t} & = f_{t} \* c{t-1} + i{t} \* \\widetile{c}_{t}
        h_{t} & = o_{t} \* \\tanh(c_{t})
        y_{t} & = h_{t}

    where :math:`\sigma` is the sigmoid fucntion, and \* is the elemetwise 
    multiplication operator.

    Please refer to `An Empirical Exploration of Recurrent Network Architectures
    <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_ for more details.

    Arguments:
        input_size (int): The input size.
        hidden_size (int): The hidden size.
        weight_ih_attr(ParamAttr, optional): The parameter attribute for 
            `weight_ih`. Default: None.
        weight_hh_attr(ParamAttr, optional): The parameter attribute for 
            `weight_hh`. Default: None.
        bias_ih_attr (ParamAttr, optional): The parameter attribute for the 
            `bias_ih`. Default: None.
        bias_ih_attr (ParamAttr, optional): The parameter attribute for the 
            `bias_hh`. Default: None.
        name (str, optional): Name for the operation (optional, default is 
            None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

            import paddle
            paddle.disable_static()

            x = paddle.randn((4, 16))
            prev_h = paddle.randn((4, 32))

            cell = paddle.nn.LSTMCell(16, 32)
            y, h = cell(x, prev_h)

    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 weight_ih_attr=None,
                 weight_hh_attr=None,
                 bias_ih_attr=None,
                 bias_hh_attr=None,
                 name=None):
        super(LSTMCell, self).__init__()
        std = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = self.create_parameter(
            (4 * hidden_size, input_size),
            weight_ih_attr,
            default_initializer=I.Uniform(-std, std))
        self.weight_hh = self.create_parameter(
            (4 * hidden_size, hidden_size),
            weight_hh_attr,
            default_initializer=I.Uniform(-std, std))
        self.bias_ih = self.create_parameter(
            (4 * hidden_size, ),
            bias_ih_attr,
            is_bias=True,
            default_initializer=I.Uniform(-std, std))
        self.bias_hh = self.create_parameter(
            (4 * hidden_size, ),
            bias_hh_attr,
            is_bias=True,
            default_initializer=I.Uniform(-std, std))

        self.hidden_size = hidden_size
        self.input_size = input_size
        self._gate_activation = F.sigmoid
        self._activation = paddle.tanh

    def forward(self, inputs, states=None):
        r"""
        Given the input and previous atate, compute the output and update state.

        Arguments:
            inputs (Tensor): shape `[batch_size, input_size]`, the input, 
                corresponding to :math:`x_t` in the formula.
            states (tuple, optional): a tuple of two tensors, each of shape 
                `[batch_size, hidden_size]`, the previous hidden state, 
                corresponding to :math:`h_{t-1}, c_{t-1}` in the formula. 
                When states is None, zero state is used. Defaults to None.
        Returns:
            (outputs, new_states)
            outputs (Tensor): shape `[batch_size, hidden_size]`, the output, 
                corresponding to :math:`h_{t}` in the formula.
            states (tuple): a tuple of two tensors, each of shape 
                `[batch_size, hidden_size]`, the new hidden states,
                corresponding to :math:`h_{t}, c{t}` in the formula.

        """
        if states is None:
            states = self.get_initial_states(inputs, self.state_shape)
        pre_hidden, pre_cell = states
        gates = paddle.matmul(inputs, self.weight_ih, transpose_y=True)
        if self.bias_ih is not None:
            gates = gates + self.bias_ih
        gates += paddle.matmul(pre_hidden, self.weight_hh, transpose_y=True)
        if self.bias_hh is not None:
            gates = gates + self.bias_hh

        chunked_gates = paddle.split(gates, num_or_sections=4, axis=-1)

        i = self._gate_activation(chunked_gates[0])
        f = self._gate_activation(chunked_gates[1])
        o = self._gate_activation(chunked_gates[3])
        c = f * pre_cell + i * self._activation(chunked_gates[2])
        h = o * self._activation(c)

        return h, (h, c)

    @property
    def state_shape(self):
        r"""
        The `state_shape` of LSTMCell is a tuple with two shapes: 
        `((hidden_size, ), (hidden_size,))`. (-1 for batch size would be 
        automatically inserted into shape). These two shapes correspond 
        to :math:`h_{t-1}` and :math:`c_{t-1}` separately.
        """
        return ((self.hidden_size, ), (self.hidden_size, ))


class GRUCell(RNNCellBase):
    r"""
    Gated Recurrent Unit (GRU) RNN cell.

    The formula for GRU used is as follows:

    .. math::

        r_{t} & = \sigma(W_{ir}x_{t} + b_{ir} + W_{hr}x_{t} + b_{hr})
        z_{t} & = \sigma(W_{iz)x_{t} + b_{iz} + W_{hz}x_{t} + b_{hz})
        \\widetilde{h}_{t} & = \\tanh(W_{ic)x_{t} + b_{ic} + r_{t} \* (W_{hc}x_{t} + b{hc}))
        h_{t} & = z_{t} \* h_{t-1} + (1 - z_{t}) \* \\widetilde{h}_{t}
        y_{t} & = h_{t}
    
    where :math:`\sigma` is the sigmoid fucntion, and \* is the elemetwise 
    multiplication operator.

    Please refer to `An Empirical Exploration of Recurrent Network Architectures
    <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_ for more details.

    Parameters:
        input_size (int): The input size..
        hidden_size (int): The hidden size.
        weight_ih_attr(ParamAttr, optional): The parameter attribute for 
            `weight_ih`. Default: None.
        weight_hh_attr(ParamAttr, optional): The parameter attribute for 
            `weight_hh`. Default: None.
        bias_ih_attr (ParamAttr, optional): The parameter attribute for the 
            `bias_ih`. Default: None.
        bias_ih_attr (ParamAttr, optional): The parameter attribute for the 
            `bias_hh`. Default: None.
        name (str, optional): Name for the operation (optional, default is 
            None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

            import paddle
            paddle.disable_static()

            x = paddle.randn((4, 16))
            prev_h = paddle.randn((4, 32))

            cell = paddle.nn.GRUCell(16, 32)
            y, h = cell(x, prev_h)

    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 weight_ih_attr=None,
                 weight_hh_attr=None,
                 bias_ih_attr=None,
                 bias_hh_attr=None,
                 name=None):
        super(GRUCell, self).__init__()
        std = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = self.create_parameter(
            (3 * hidden_size, input_size),
            weight_ih_attr,
            default_initializer=I.Uniform(-std, std))
        self.weight_hh = self.create_parameter(
            (3 * hidden_size, hidden_size),
            weight_hh_attr,
            default_initializer=I.Uniform(-std, std))
        self.bias_ih = self.create_parameter(
            (3 * hidden_size, ),
            bias_ih_attr,
            is_bias=True,
            default_initializer=I.Uniform(-std, std))
        self.bias_hh = self.create_parameter(
            (3 * hidden_size, ),
            bias_hh_attr,
            is_bias=True,
            default_initializer=I.Uniform(-std, std))

        self.hidden_size = hidden_size
        self.input_size = input_size
        self._gate_activation = F.sigmoid
        self._activation = paddle.tanh

    def forward(self, inputs, states=None):
        r"""
        Performs single step GRU calculations.

        Parameters:
            inputs (Variable): A tensor with shape `[batch_size, input_size]`,
                corresponding to :math:`x_t` in the formula. The data type
                should be float32 or float64.
            states (Variable): A tensor with shape `[batch_size, hidden_size]`.
                corresponding to :math:`h_{t-1}` in the formula. The data type
                should be float32 or float64.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` and \
                `new_states` is the same tensor shaped `[batch_size, hidden_size]`, \
                corresponding to :math:`h_t` in the formula. The data type of the \
                tensor is same as that of `states`.        
        """
        if states is None:
            states = self.get_initial_states(inputs, self.state_shape)

        pre_hidden = states
        x_gates = paddle.matmul(inputs, self.weight_ih, transpose_y=True)
        if self.bias_ih is not None:
            x_gates = x_gates + self.bias_ih
        h_gates = paddle.matmul(pre_hidden, self.weight_hh, transpose_y=True)
        if self.bias_hh is not None:
            h_gates = h_gates + self.bias_hh

        x_r, x_z, x_c = paddle.split(x_gates, num_or_sections=3, axis=1)
        h_r, h_z, h_c = paddle.split(h_gates, num_or_sections=3, axis=1)

        r = self._gate_activation(x_r + h_r)
        z = self._gate_activation(x_z + h_z)
        c = self._activation(x_c + r * h_c)  # apply reset gate after mm
        h = (pre_hidden - c) * z + c

        return h, h

    @property
    def state_shape(self):
        r"""
        The `state_shape` of GRUCell is a shape `[hidden_size]` (-1 for batch
        size would be automatically inserted into shape). The shape corresponds
        to :math:`h_{t-1}`.
        """
        return (self.hidden_size, )


class RNN(Layer):
    r"""
    Wrapper for RNN, which creates a recurrent neural network specified with a
    RNN cell. It performs :code:`cell.forward()` repeatedly until reaches to 
    the maximum length of `inputs`.

    Arguments:
        cell(RNNCellBase): An instance of `RNNCell`.
        is_reverse (bool, optional): Indicate whether to calculate in the reverse
            order of input sequences. Defaults to False.
        time_major (bool, optional): Indicate the data layout of Tensor included
            in `input` and `output` tensors. If `False`, the data layout would
            be batch major with shape `[batch_size, time_steps, ...]`.  If
            `True`, the data layout would be time major with shape
            `[time_steps, batch_size, ...]`. Defaults to False.

    Inputs:
        inputs (Tensor): A (possibly nested structure of) tensor variable[s]. 
            The shape of tensor should be `[batch_size, time_steps, ...]`
            for `time_major == False` or `[time_steps, batch_size, ...]`
            for `time_major == True`. It represents the inputs to be unrolled
            in RNN.
        initial_states (Tensor|list|tuple, optional): A (possibly nested structure of)
            tensor[s], representing the initial state for the rnn cell. 
            If not provided, `cell.get_initial_states` would be used to produce
            the initial state. Defaults to None.
        sequence_length (Tensor, optional): shape `[batch_size]`, dtype: int64 
            or int32. The valid lengths of input sequences.
            If `sequence_length` is not None, the inputs are treated as 
            padded sequences. In each input sequence, elements whos time step 
            index are not less than the valid length are treated as paddings.
        **kwargs: Additional keyword arguments. Arguments passed to `cell.forward`. 

    Outputs:
        (outputs, final_states)
        outputs (Tensor|list|tuple): the output sequence. Tensor or nested 
            structure of Tensor.
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

            inputs = paddle.rand((4, 23, 16))
            prev_h = paddle.randn((4, 32))

            cell = paddle.nn.SimpleRNNCell(16, 32)
            rnn = paddle.RNN(cell)
            outputs, final_states = rnn(inputs, prev_h)

    """

    def __init__(self, cell, is_reverse=False, time_major=False):
        super(RNN, self).__init__()
        self.cell = cell
        if not hasattr(self.cell, "call"):
            # for non-dygraph mode, `rnn` api uses cell.call
            self.cell.call = self.cell.forward
        self.is_reverse = is_reverse
        self.time_major = time_major

    def forward(self, inputs, initial_states=None, sequence_length=None):
        if initial_states is None:
            initial_states = self.cell.get_initial_states(
                batch_ref=inputs,
                dtype=inputs.dtype,
                batch_dim_idx=self.batch_index)

        final_outputs, final_states = F.rnn(self.cell,
                                            inputs,
                                            initial_states=initial_states,
                                            sequence_length=sequence_length,
                                            time_major=self.time_major,
                                            is_reverse=self.is_reverse)
        return final_outputs, final_states


class BiRNN(Layer):
    r"""
    Wrapper for bidirectional RNN. It assembles two RNN cells by performing
    forward and backward RNN separately, and concat outputs.

    Parameters:
        cell_fw (RNNCellBase): A RNNCell instance used for forward RNN.
        cell_bw (RNNCellBase): A RNNCell instance used for backward RNN.
        time_major (bool): Whether the first dimension of the input means the
            time steps.

    Inputs:
        inputs (Tensor): A (possibly nested structure of) tensor variable[s]. 
            The shape of tensor should be `[batch_size, sequence_length, ...]`
            for `time_major == False` or `[sequence_length, batch_size, ...]`
            for `time_major == True`. It represents the inputs to be unrolled
            in both forward and backward RNN.
        initial_states (list|tuple, optional): A tuple of the initial states of 
            the forward cell and backward cell. 
            If not provided, `cell.get_initial_states` would be used to produce 
            the initial states. Defaults to None.
        sequence_length (Tensor, optional): shape `[batch_size]`, dtype: int64 
            or int32. The valid lengths of input sequences.
            If `sequence_length` is not None, the inputs are treated as 
            padded sequences. In each input sequence, elements whos time step 
            index are not less than the valid length are treated as paddings.
        **kwargs: Additional keyword arguments. Arguments passed to `cell.forward`.

    Outputs:
            outputs (Tensor): A (possibly nested structure of) tensor variable[s],
                the outputs of the bidirectional RNN. It is the concatenation 
                of the outputs for both the forward RNN and backward RNN along
                the last axis. 
                The shape of tensor should be `[batch_size, time_steps, ...]`
                for `time_major == False` or `[time_steps, batch_size, ...]`
                for `time_major == True`.
            final_states (tuple): A tuple of the final states of the forward 
                cell and backward cell. 

    Examples:
        .. code-block:: python

            import paddle
            paddle.disable_static()

            cell_fw = LSTMCell(16, 32)
            cell_bw = LSTMCell(16, 32)
            rnn = BidirectionalRNN(cell_fw, cell_bw)

            inputs = paddle.rand((2, 23, 16))
            outputs, final_states = rnn(inputs)

    """

    def __init__(self, cell_fw, cell_bw, time_major=False):
        super(BiRNN, self).__init__()
        self.cell_fw = cell_fw
        self.cell_bw = cell_bw
        for cell in [self.cell_fw, self.cell_bw]:
            if not hasattr(cell, "call"):
                # for non-dygraph mode, `rnn` api uses cell.call
                cell.call = cell.forward
        self.time_major = time_major

    def forward(self,
                inputs,
                initial_states=None,
                sequence_length=None,
                **kwargs):
        if isinstance(initial_states, (list, tuple)):
            assert len(initial_states) == 2, \
                "length of initial_states should be 2 when it is a list/tuple"
        else:
            initial_states = [initial_states, initial_states]

        outputs, final_states = F.birnn(self.cell_fw, self.cell_bw, inputs,
                                        initial_states, sequence_length,
                                        self.time_major)
        return outputs, final_states


class RNNMixin(LayerList):
    r"""
    A Mixin class for RNN networks. It provides forward method for SimpleRNN,
    LSTM and GRU.
    """

    def forward(self, inputs, initial_states=None, sequence_length=None):
        batch_index = 1 if self.time_major else 0
        dtype = inputs.dtype
        if initial_states is None:
            state_shape = (self.num_layers * self.num_directions, -1,
                           self.hidden_size)
            if self.state_components == 1:
                initial_states = paddle.fluid.layers.fill_constant_batch_size_like(
                    inputs, state_shape, dtype, 0, batch_index, 1)
            else:
                initial_states = tuple([
                    paddle.fluid.layers.fill_constant_batch_size_like(
                        inputs, state_shape, dtype, 0, batch_index, 1)
                    for _ in range(self.state_components)
                ])

        states = split_states(initial_states, self.num_directions == 2,
                              self.state_components)
        final_states = []

        for i, rnn_layer in enumerate(self):
            if i > 0:
                inputs = F.dropout(
                    inputs, self.dropout, mode="upscale_in_train")
            outputs, final_state = rnn_layer(inputs, states[i], sequence_length)
            final_states.append(final_state)
            inputs = outputs

        final_states = concat_states(final_states, self.num_directions == 2,
                                     self.state_components)
        return outputs, final_states


class SimpleRNN(RNNMixin):
    r"""
    Multilayer Elman network(SimpleRNN). It takes a sequence and an initial 
    state as inputs, and returns the output sequence and the final state.

    Each layer inside the SimpleRNN maps the input sequence and initial state 
    to the output sequence and final state in the following manner: at each 
    step, it takes step input(:math:`x_{t}`) and previous 
    state(:math:`h_{t-1}`) as inputs, and returns step output(:math:`y_{t}`)
    and new state(:math:`h_{t}`).

    .. math::

        h_{t} & = \mathrm{tanh}(W_{ih}x_{t} + b_{ih} + W_{hh}h{t-1} + b_{hh})
        y_{t} & = h_{t}
    
    where :math:`\sigma` is the sigmoid fucntion, and \* is the elemetwise 
    multiplication operator.

    Arguments:
        input_size (int): The input size for the first layer's cell.
        hidden_size (int): The hidden size for each layer's cell.
        num_layers (int): Number of layers. Defaults to 1.
        nonlinearity (str): The activation in each SimpleRNN cell. It can be 
            `tanh` or `relu`. Defaults to `tanh`.
        direction (str): The direction of the network. It can be "forward", 
            "backward" and "bidirectional". Defaults to "forward".
        dropout (float): The droput probability. Dropout is applied to the 
            input of each layer except for the first layer.
        time_major (bool): Whether the first dimension of the input means the
            time steps.
        weight_ih_attr (ParamAttr, optional): The parameter attribute for 
            `weight_ih` of each cell. Default: None.
        weight_hh_attr (ParamAttr, optional): The parameter attribute for 
            `weight_hh` of each cell. Default: None.
        bias_ih_attr (ParamAttr, optional): The parameter attribute for the 
            `bias_ih` of each cells. Default: None.
        bias_ih_attr (ParamAttr, optional): The parameter attribute for the 
            `bias_hh` of each cells. Default: None.
        name (str, optional): Name for the operation (optional, default is 
            None). For more information, please refer to :ref:`api_guide_Name`.

    Inputs:
        inputs (Tensor): the input sequence. 
            If `time_major` is True, the shape is `[time_steps, batch_size, input_size]`,
            else, the shape is `[batch_size, time_steps, hidden_size]`.
        initial_states (Tensor, optional): the initial state. The shape is
            `[num_lauers * num_directions, batch_size, hidden_size]`. 
            If initial_state is not given, zero initial states are used.
        sequence_length (Tensor, optional): shape `[batch_size]`, dtype: int64 
            or int32. The valid lengths of input sequences.
            If `sequence_length` is not None, the inputs are treated as 
            padded sequences. In each input sequence, elements whos time step 
            index are not less than the valid length are treated as paddings.

    Outputs:
        (outputs, final_states)
        outputs (Tensor): the output sequence. 
            If `time_major` is True, the shape is `[time_steps, batch_size, hidden_size]`,
            else, the shape is `[batch_size, time_steps, hidden_size]`.
        final_states (Tensor): final states. The shape is
            `[num_lauers * num_directions, batch_size, hidden_size]`.

    Examples:
        .. code-block:: python

            import paddle
            paddle.disable_static()

            rnn = paddle.nn.SimpleRNN(16, 32, 2)

            x = paddle.randn((4, 23, 16))
            prev_h = paddle.randn((2, 4, 32))
            y, h = rnn(x, prev_h)

    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 nonlinearity="tanh",
                 direction="forward",
                 dropout=0.,
                 time_major=False,
                 weight_ih_attr=None,
                 weight_hh_attr=None,
                 bias_ih_attr=None,
                 bias_hh_attr=None,
                 name=None):
        super(SimpleRNN, self).__init__()

        if direction in ["forward", "backward"]:
            is_reverse = direction == "backward"
            cell = SimpleRNNCell(input_size, hidden_size, nonlinearity,
                                 weight_ih_attr, weight_hh_attr, bias_ih_attr,
                                 bias_hh_attr)
            self.append(RNN(cell, is_reverse, time_major))
            for i in range(1, num_layers):
                cell = SimpleRNNCell(hidden_size, hidden_size, nonlinearity,
                                     weight_ih_attr, weight_hh_attr,
                                     bias_ih_attr, bias_hh_attr)
                self.append(RNN(cell, is_reverse, time_major))
        elif direction == "bidirectional":
            cell_fw = SimpleRNNCell(input_size, hidden_size, nonlinearity,
                                    weight_ih_attr, weight_hh_attr,
                                    bias_ih_attr, bias_hh_attr)
            cell_bw = SimpleRNNCell(input_size, hidden_size, nonlinearity,
                                    weight_ih_attr, weight_hh_attr,
                                    bias_ih_attr, bias_hh_attr)
            self.append(BiRNN(cell_fw, cell_bw, time_major))
            for i in range(1, num_layers):
                cell_fw = SimpleRNNCell(
                    2 * hidden_size, hidden_size, nonlinearity, weight_ih_attr,
                    weight_hh_attr, bias_ih_attr, bias_hh_attr)
                cell_bw = SimpleRNNCell(
                    2 * hidden_size, hidden_size, nonlinearity, weight_ih_attr,
                    weight_hh_attr, bias_ih_attr, bias_hh_attr)
                self.append(BiRNN(cell_fw, cell_bw, time_major))
        else:
            raise ValueError(
                "direction should be forward, backward or bidirectional, "
                "received direction = {}".format(direction))

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_directions = 2 if direction == "bidirectional" else 1
        self.time_major = time_major
        self.num_layers = num_layers
        self.state_components = 1


class LSTM(RNNMixin):
    r"""
    Multilayer LSTM. It takes a sequence and an initial state as inputs, and 
    returns the output sequence and the final state.

    Each layer inside the LSTM maps the input sequence and initial state 
    to the output sequence and final state in the following manner: at each 
    step, it takes step input(:math:`x_{t}`) and previous 
    state(:math:`h_{t-1}, c_{t-1}`) as inputs, and returns step 
    output(:math:`y_{t}`) and new state(:math:`h_{t}, c_{t}`).

    .. math::

        i_{t} & = \sigma(W_{ii}x_{t} + b_{ii} + W_{hi}h_{t-1} + b_{hi})
        f_{t} & = \sigma(W_{if}x_{t} + b_{if} + W_{hf}h_{t-1} + b_{hf})
        o_{t} & = \sigma(W_{io}x_{t} + b_{io} + W_{ho}h_{t-1} + b_{ho})
        \\widetilde{c}_{t} & = \\tanh (W_{ig}x_{t} + b_{ig} + W_{hg}h_{t-1} + b_{hg})
        c_{t} & = f_{t} \* c{t-1} + i{t} \* \\widetile{c}_{t}
        h_{t} & = o_{t} \* \\tanh(c_{t})
        y_{t} & = h_{t}

    where :math:`\sigma` is the sigmoid fucntion, and \* is the elemetwise 
    multiplication operator.

    Arguments:
        input_size (int): The input size for the first layer's cell.
        hidden_size (int): The hidden size for each layer's cell.
        num_layers (int): Number of layers. Defaults to 1.
        direction (str): The direction of the network. It can be "forward", 
            "backward" and "bidirectional". Defaults to "forward".
        dropout (float): The droput probability. Dropout is applied to the 
            input of each layer except for the first layer.
        time_major (bool): Whether the first dimension of the input means the
            time steps.
        weight_ih_attr (ParamAttr, optional): The parameter attribute for 
            `weight_ih` of each cell. Default: None.
        weight_hh_attr (ParamAttr, optional): The parameter attribute for 
            `weight_hh` of each cell. Default: None.
        bias_ih_attr (ParamAttr, optional): The parameter attribute for the 
            `bias_ih` of each cells. Default: None.
        bias_ih_attr (ParamAttr, optional): The parameter attribute for the 
            `bias_hh` of each cells. Default: None.
        name (str, optional): Name for the operation (optional, default is 
            None). For more information, please refer to :ref:`api_guide_Name`.

    Inputs:
        inputs (Tensor): the input sequence. 
            If `time_major` is True, the shape is `[time_steps, batch_size, input_size]`,
            else, the shape is `[batch_size, time_steps, hidden_size]`.
        initial_states (tuple, optional): the initial state, a tuple of (h, c), 
            the shape of each is `[num_lauers * num_directions, batch_size, hidden_size]`. 
            If initial_state is not given, zero initial states are used.
        sequence_length (Tensor, optional): shape `[batch_size]`, dtype: int64 
            or int32. The valid lengths of input sequences.
            If `sequence_length` is not None, the inputs are treated as 
            padded sequences. In each input sequence, elements whos time step 
            index are not less than the valid length are treated as paddings.

    Outputs:
        (outputs, final_states)
        outputs (Tensor): the output sequence. 
            If `time_major` is True, the shape is `[time_steps, batch_size, hidden_size]`,
            else, the shape is `[batch_size, time_steps, hidden_size]`.
        final_states (Tensor): the final state, a tuple of (h, c), 
            the shape of each is `[num_lauers * num_directions, batch_size, hidden_size]`.

    Examples:
        .. code-block:: python

            import paddle
            paddle.disable_static()

            rnn = paddle.nn.LSTM(16, 32, 2)

            x = paddle.randn((4, 23, 16))
            prev_h = paddle.randn((2, 4, 32))
            prev_c = paddle.randn((2, 4, 32))
            y, (h, c) = rnn(x, (prev_h, prev_c))

    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.,
                 time_major=False,
                 weight_ih_attr=None,
                 weight_hh_attr=None,
                 bias_ih_attr=None,
                 bias_hh_attr=None,
                 name=None):
        super(LSTM, self).__init__()

        if direction in ["forward", "backward"]:
            is_reverse = direction == "backward"
            cell = LSTMCell(input_size, hidden_size, weight_ih_attr,
                            weight_hh_attr, bias_ih_attr, bias_hh_attr)
            self.append(RNN(cell, is_reverse, time_major))
            for i in range(1, num_layers):
                cell = LSTMCell(hidden_size, hidden_size, weight_ih_attr,
                                weight_hh_attr, bias_ih_attr, bias_hh_attr)
                self.append(RNN(cell, is_reverse, time_major))
        elif direction == "bidirectional":
            cell_fw = LSTMCell(input_size, hidden_size, weight_ih_attr,
                               weight_hh_attr, bias_ih_attr, bias_hh_attr)
            cell_bw = LSTMCell(input_size, hidden_size, weight_ih_attr,
                               weight_hh_attr, bias_ih_attr, bias_hh_attr)
            self.append(BiRNN(cell_fw, cell_bw, time_major))
            for i in range(1, num_layers):
                cell_fw = LSTMCell(2 * hidden_size, hidden_size, weight_ih_attr,
                                   weight_hh_attr, bias_ih_attr, bias_hh_attr)
                cell_bw = LSTMCell(2 * hidden_size, hidden_size, weight_ih_attr,
                                   weight_hh_attr, bias_ih_attr, bias_hh_attr)
                self.append(BiRNN(cell_fw, cell_bw, time_major))
        else:
            raise ValueError(
                "direction should be forward, backward or bidirectional, "
                "received direction = {}".format(direction))

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_directions = 2 if direction == "bidirectional" else 1
        self.time_major = time_major
        self.num_layers = num_layers
        self.state_components = 2


class GRU(RNNMixin):
    r"""
    Multilayer GRU. It takes a sequence and an initial state as inputs, and 
    returns the output sequence and the final state.

    Each layer inside the GRU maps the input sequence and initial state 
    to the output sequence and final state in the following manner: at each 
    step, it takes step input(:math:`x_{t}`) and previous 
    state(:math:`h_{t-1}`) as inputs, and returns step output(:math:`y_{t}`) 
    and new state(:math:`h_{t}`).

    .. math::

        r_{t} & = \sigma(W_{ir}x_{t} + b_{ir} + W_{hr}x_{t} + b_{hr})
        z_{t} & = \sigma(W_{iz)x_{t} + b_{iz} + W_{hz}x_{t} + b_{hz})
        \\widetilde{h}_{t} & = \\tanh(W_{ic)x_{t} + b_{ic} + r_{t} \* (W_{hc}x_{t} + b{hc}))
        h_{t} & = z_{t} \* h_{t-1} + (1 - z_{t}) \* \\widetilde{h}_{t}
        y_{t} & = h_{t}

    where :math:`\sigma` is the sigmoid fucntion, and \* is the elemetwise 
    multiplication operator.

    Arguments:
        input_size (int): The input size for the first layer's cell.
        hidden_size (int): The hidden size for each layer's cell.
        num_layers (int): Number of layers. Defaults to 1.
        direction (str): The direction of the network. It can be "forward", 
            "backward" and "bidirectional". Defaults to "forward".
        dropout (float): The droput probability. Dropout is applied to the 
            input of each layer except for the first layer.
        time_major (bool): Whether the first dimension of the input means the
            time steps.
        weight_ih_attr (ParamAttr, optional): The parameter attribute for 
            `weight_ih` of each cell. Default: None.
        weight_hh_attr (ParamAttr, optional): The parameter attribute for 
            `weight_hh` of each cell. Default: None.
        bias_ih_attr (ParamAttr, optional): The parameter attribute for the 
            `bias_ih` of each cells. Default: None.
        bias_ih_attr (ParamAttr, optional): The parameter attribute for the 
            `bias_hh` of each cells. Default: None.
        name (str, optional): Name for the operation (optional, default is 
            None). For more information, please refer to :ref:`api_guide_Name`.

    Inputs:
        inputs (Tensor): the input sequence. 
            If `time_major` is True, the shape is `[time_steps, batch_size, input_size]`,
            else, the shape is `[batch_size, time_steps, hidden_size]`.
        initial_states (Tensor, optional): the initial state. The shape is
            `[num_lauers * num_directions, batch_size, hidden_size]`. 
            If initial_state is not given, zero initial states are used.
        sequence_length (Tensor, optional): shape `[batch_size]`, dtype: int64 
            or int32. The valid lengths of input sequences.
            If `sequence_length` is not None, the inputs are treated as 
            padded sequences. In each input sequence, elements whos time step 
            index are not less than the valid length are treated as paddings.

    Outputs:
        (outputs, final_states)
        outputs (Tensor): the output sequence. 
            If `time_major` is True, the shape is `[time_steps, batch_size, hidden_size]`,
            else, the shape is `[batch_size, time_steps, hidden_size]`.
        final_states (Tensor): final states. The shape is
            `[num_lauers * num_directions, batch_size, hidden_size]`.

    Examples:
        .. code-block:: python

            import paddle
            paddle.disable_static()

            rnn = paddle.nn.GRU(16, 32, 2)

            x = paddle.randn((4, 23, 16))
            prev_h = paddle.randn((2, 4, 32))
            y, h = rnn(x, prev_h)

    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.,
                 time_major=False,
                 weight_ih_attr=None,
                 weight_hh_attr=None,
                 bias_ih_attr=None,
                 bias_hh_attr=None,
                 name=None):
        super(GRU, self).__init__()

        if direction in ["forward", "backward"]:
            is_reverse = direction == "backward"
            cell = GRUCell(input_size, hidden_size, weight_ih_attr,
                           weight_hh_attr, bias_ih_attr, bias_hh_attr)
            self.append(RNN(cell, is_reverse, time_major))
            for i in range(1, num_layers):
                cell = GRUCell(hidden_size, hidden_size, weight_ih_attr,
                               weight_hh_attr, bias_ih_attr, bias_hh_attr)
                self.append(RNN(cell, is_reverse, time_major))
        elif direction == "bidirectional":
            cell_fw = GRUCell(input_size, hidden_size, weight_ih_attr,
                              weight_hh_attr, bias_ih_attr, bias_hh_attr)
            cell_bw = GRUCell(input_size, hidden_size, weight_ih_attr,
                              weight_hh_attr, bias_ih_attr, bias_hh_attr)
            self.append(BiRNN(cell_fw, cell_bw, time_major))
            for i in range(1, num_layers):
                cell_fw = GRUCell(2 * hidden_size, hidden_size, weight_ih_attr,
                                  weight_hh_attr, bias_ih_attr, bias_hh_attr)
                cell_bw = GRUCell(2 * hidden_size, hidden_size, weight_ih_attr,
                                  weight_hh_attr, bias_ih_attr, bias_hh_attr)
                self.append(BiRNN(cell_fw, cell_bw, time_major))
        else:
            raise ValueError(
                "direction should be forward, backward or bidirectional, "
                "received direction = {}".format(direction))

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_directions = 2 if direction == "bidirectional" else 1
        self.time_major = time_major
        self.num_layers = num_layers
        self.state_components = 1
