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
    'dynamic_lstm',
    'dynamic_lstmp',
    'dynamic_gru',
    'gru_unit',
    'lstm',
]


class ArrayWrapper:
    def __init__(self, x):
        self.array = [x]

    def append(self, x):
        self.array.append(x)
        return self

    def __getitem__(self, item):
        return self.array.__getitem__(item)


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

    **Note**:
        This OP only supports running on GPU devices.

    This OP implements LSTM operation - `Hochreiter, S., & Schmidhuber, J. (1997) <https://blog.xpgreat.com/file/lstm.pdf>`_ .

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


def dynamic_gru(
    input,
    size,
    param_attr=None,
    bias_attr=None,
    is_reverse=False,
    gate_activation='sigmoid',
    candidate_activation='tanh',
    h_0=None,
    origin_mode=False,
):
    r"""

    **Note: The input type of this must be LoDTensor. If the input type to be
    processed is Tensor, use** :ref:`api_fluid_layers_StaticRNN` .

    This operator is used to perform the calculations for a single layer of
    Gated Recurrent Unit (GRU) on full sequences step by step. The calculations
    in one time step support these two modes:

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

    :math:`x_t` is the input of current time step, but it is not from ``input`` .
    This operator does not include the calculations :math:`W_{ux}x_{t}, W_{rx}x_{t}, W_{cx}x_{t}` ,
    **Note** thus a fully-connect layer whose size is 3 times of ``size`` should
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
        input(Variable): A LoDTensor whose lod level is 1, representing the input
            after linear projection. Its shape should be :math:`[T, D \\times 3]` ,
            where :math:`T` stands for the total sequence lengths in this mini-batch,
            :math:`D` for the hidden size. The data type should be float32 or float64.
        size(int): Indicate the hidden size.
        param_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        is_reverse(bool, optional): Whether to compute in the reversed order of
            input sequences. Default False.
        gate_activation(str, optional): The activation function corresponding to
            :math:`act_g` in the formula. "sigmoid", "tanh", "relu" and "identity"
            are supported. Default "sigmoid".
        candidate_activation(str, optional): The activation function corresponding to
            :math:`act_c` in the formula. "sigmoid", "tanh", "relu" and "identity"
            are supported. Default "tanh".
        h_0 (Variable, optional): A Tensor representing the initial hidden state.
            It not provided, the default initial hidden state is 0. The shape is
            :math:`[N, D]` , where :math:`N` is the number of sequences in the
            mini-batch, :math:`D` for the hidden size. The data type should be
            same as ``input`` . Default None.

    Returns:
        Variable: A LoDTensor whose lod level is 1 and shape is :math:`[T, D]` , \
            where :math:`T` stands for the total sequence lengths in this mini-batch \
            :math:`D` for the hidden size. It represents GRU transformed sequence output, \
            and has the same lod and data type with ``input`` .

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            dict_dim, emb_dim = 128, 64
            data = fluid.data(name='sequence',
                      shape=[None],
                      dtype='int64',
                      lod_level=1)
            emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim = 512
            x = fluid.layers.fc(input=emb, size=hidden_dim * 3)
            hidden = fluid.layers.dynamic_gru(input=x, size=hidden_dim)
    """

    assert (
        _non_static_mode() is not True
    ), "please use gru instead of dynamic_gru in dygraph mode!"

    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'dynamic_gru'
    )

    check_type(h_0, 'h_0', (Variable, type(None)), 'dynamic_gru')
    if isinstance(h_0, Variable):
        check_variable_and_dtype(
            h_0, 'h_0', ['float32', 'float64'], 'dynamic_gru'
        )

    helper = LayerHelper('gru', **locals())
    dtype = helper.input_dtype()

    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 3 * size], dtype=dtype
    )
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=[1, 3 * size], dtype=dtype, is_bias=True
    )
    batch_size = input.shape[0]
    inputs = {'Input': input, 'Weight': weight, 'Bias': bias}
    if h_0:
        assert h_0.shape == (batch_size, size), (
            'The shape of h0 should be(batch_size, %d)' % size
        )
        inputs['H0'] = h_0

    hidden = helper.create_variable_for_type_inference(dtype)
    batch_gate = helper.create_variable_for_type_inference(dtype)
    batch_reset_hidden_prev = helper.create_variable_for_type_inference(dtype)
    batch_hidden = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type='gru',
        inputs=inputs,
        outputs={
            'Hidden': hidden,
            'BatchGate': batch_gate,
            'BatchResetHiddenPrev': batch_reset_hidden_prev,
            'BatchHidden': batch_hidden,
        },
        attrs={
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'activation': candidate_activation,
            'origin_mode': origin_mode,
        },
    )
    return hidden


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
