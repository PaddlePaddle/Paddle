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

from .layers import Layer
from paddle.fluid import layers

__all__ = [
    'CudnnLSTM', 'CudnnGRU'
]

class CudnnLSTM(Layer):
    """
    ****
    CUDNN LSTM implementation for dynamic graph.

    A four-gate Long Short-Term Memory network with no peephole connections.
    In the forward pass the output ht and cell output ct for a given iteration can be computed from the recurrent input ht-1, 
    the cell input ct-1 and the previous layer input xt given matrices W, R and biases bW, bR from the following equations:
        .. math::
            i_t = sigmoid(W_{ix}x_{t} + W_{ih}h_{t-1} + bx_i + bh_i)
            f_t = sigmoid(W_{fx}x_{t} + W_{fh}h_{t-1} + bx_f + bh_f)
            o_t = sigmoid(W_{ox}x_{t} + W_{oh}h_{t-1} + bx_o + bh_o)
            \\tilde{c_t} = tanh(W_{cx}x_t + W_{ch}h_{t-1} + bx_c + bh_c)
            c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c_t}
            h_t = o_t \\odot tanh(c_t)
    - W terms denote weight matrices (e.g. $W_{ix}$ is the matrix
    of weights from the input gate to the input)
    - The b terms denote bias vectors ($bx_i$ and $bh_i$ are the input gate bias vector).
    - sigmoid is the logistic sigmoid function.
    - $i, f, o$ and $c$ are the input gate, forget gate, output gate,
    and cell activation vectors, respectively, all of which have the same size as
    the cell output activation vector $h$.
    - The $\odot$ is the element-wise product of the vectors.
    - `tanh` is the activation functions.
    - $\tilde{c_t}$ is also called candidate hidden state,
    which is computed based on the current input and the previous hidden state.


    Args:
        hidden_size (integer): The hidden size used in the Unit.
        input_size (integer): The input size used in the Unit.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, lstm_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of LSTM unit.
            If it is set to None or one attribute of ParamAttr, lstm_unit will 
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized as zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cells (actNode).
                             Default: 'fluid.layers.tanh'
        forget_bias(float|1.0): forget bias used when computing forget gate
        dtype(string): data type used in this unit
    
    Returns:
        None

    Examples:
        .. code-block:: python
            from paddle import fluid
            from paddle.fluid.dygraph.rnn import CudnnLSTM
            batch_size = 64
            input_size = 128
            hidden_size = 256
            cudnn_lstm = CudnnLSTM(hidden_size, input_size)

            step_input_np = np.random.uniform(-0.1, 0.1, (
                batch_size, input_size)).astype('float64')
            pre_hidden_np = np.random.uniform(-0.1, 0.1, (
                batch_size, hidden_size)).astype('float64')
            pre_cell_np = np.random.uniform(-0.1, 0.1, (
                batch_size, hidden_size)).astype('float64')

            step_input_var = fluid.dygraph.to_variable(step_input_np)
            pre_hidden_var = fluid.dygraph.to_variable(pre_hidden_np)
            pre_cell_var = fluid.dygraph.to_variable(pre_cell_np)
            new_hidden, new_cell = cudnn_lstm(step_input_var, pre_hidden_var, pre_cell_var)
    """

    def __init__(self,
                 hidden_size,
                 input_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 dtype='float64'):
        super(CudnnLSTM, self).__init__(dtype)

        self._hidden_size = hidden_size
        self._input_size = input_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._dtype = dtype
        self._gate_activation = gate_activation or layers.sigmoid
        self._activation = activation or layers.tanh
        
        self._forget_bias = layers.fill_constant(
            [1], dtype=dtype, value=forget_bias)
        self._forget_bias.stop_gradient = False

        self._weight = self.create_parameter(
            attr=self._param_attr,
            shape=[self._input_size + self._hidden_size, 4 * self._hidden_size],
            dtype=dtype)

        self._bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[4 * self._hidden_size],
            dtype=dtype,
            is_bias=True)

    def forward(self, input, pre_hidden, pre_cell):
        concat_input_hidden = layers.concat([input, pre_hidden], 1)
        gate_input = layers.matmul(x=concat_input_hidden, y=self._weight)

        gate_input = layers.elementwise_add(gate_input, self._bias)
        i, j, f, o = layers.split(gate_input, num_or_sections=4, dim=-1)
        new_cell = layers.elementwise_add(
            layers.elementwise_mul(
                pre_cell,
                layers.sigmoid(layers.elementwise_add(f, self._forget_bias))),
            layers.elementwise_mul(layers.sigmoid(i), layers.tanh(j)))
        new_hidden = layers.tanh(new_cell) * layers.sigmoid(o)

        return new_hidden, new_cell

class CudnnGRU(Layer):
    """
    ****
    CUDNN LSTM implementation for dynamic graph. References:
    Learning Phrase Representations using RNN Encoder Decoder for Statistical
    Machine Translation:
      [Cho et al., 2014]
      (https://aclanthology.coli.uni-saarland.de/papers/D14-1179/d14-1179)
      ([pdf](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf))

    The algorithm can be described as the equations below.
        .. math::
            u_t & = sigmoid(W_{ux} x_{t} + W_{uh} h_{t-1} + b_u)
            r_t & = sigmoid(W_{rx} x_{t} + W_{rh} h_{t-1} + b_r)
            \\tilde{h_{t}} & = tanh(W_{cx} x_{t} + W_{ch} (r_t \\odot h_{t-1}) + b_c)
            h_t & = u_t h_{t-1} + (1-u_t) \\tilde{h_{t}}
    Args:
        hidden_size (integer): The hidden size used in the Unit.
        input_size (integer): The input size used in the Unit.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, gru_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of GRU unit.
            If it is set to None or one attribute of ParamAttr, gru_unit will 
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cell (actNode).
                             Default: 'fluid.layers.tanh'
        dtype(string): data type used in this unit
    
    Returns:
        None

    Examples:
        .. code-block:: python
            from paddle import fluid
            from paddle.fluid.dygraph.rnn import CudnnGRU
            batch_size = 64
            input_size = 128
            hidden_size = 256
            cudnn_gru = CudnnGRU(hidden_size, input_size)

            step_input_np = np.random.uniform(-0.1, 0.1, (
                batch_size, input_size)).astype('float64')
            pre_hidden_np = np.random.uniform(-0.1, 0.1, (
                batch_size, hidden_size)).astype('float64')
            step_input_var = fluid.dygraph.to_variable(step_input_np)
            pre_hidden_var = fluid.dygraph.to_variable(pre_hidden_np)

            new_hidden = cudnn_gru(step_input_var, pre_hidden_var)
    """

    def __init__(self,
                 hidden_size,
                 input_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 dtype='float64'):
        super(CudnnGRU, self).__init__()

        self._hidden_size = hidden_size
        self._input_size = input_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._dtype = dtype
        self._gate_activation = gate_activation or layers.sigmoid
        self._activation = activation or layers.tanh
        
        if self._param_attr is not None and self._param_attr.name is not None:
            gate_param_attr = copy.deepcopy(self._param_attr)
            candidate_param_attr = copy.deepcopy(self._param_attr)
            gate_param_attr.name += "_gate"
            candidate_param_attr.name += "_candidate"
        else:
            gate_param_attr = self._param_attr
            candidate_param_attr = self._param_attr

        self._gate_weight = self.create_parameter(
            attr=gate_param_attr,
            shape=[self._input_size + self._hidden_size, 2 * self._hidden_size],
            dtype=dtype)

        self._candidate_weight = self.create_parameter(
            attr=candidate_param_attr,
            shape=[self._input_size + self._hidden_size, self._hidden_size],
            dtype=dtype)

        if self._bias_attr is not None and self._bias_attr.name is not None:
            gate_bias_attr = copy.deepcopy(self._bias_attr)
            candidate_bias_attr = copy.deepcopy(self._bias_attr)
            gate_bias_attr.name += "_gate"
            candidate_bias_attr.name += "_candidate"
        else:
            gate_bias_attr = self._bias_attr
            candidate_bias_attr = self._bias_attr

        self._gate_bias = self.create_parameter(
            attr=gate_bias_attr,
            shape=[2 * self._hidden_size],
            dtype=dtype,
            is_bias=True)
        self._candidate_bias = self.create_parameter(
            attr=candidate_bias_attr,
            shape=[self._hidden_size],
            dtype=dtype,
            is_bias=True)

    def forward(self, input, pre_hidden):
        
        concat_input_hidden = layers.concat([input, pre_hidden], 1)

        gate_input = layers.matmul(x=concat_input_hidden, y=self._gate_weight)

        gate_input = layers.elementwise_add(gate_input, self._gate_bias)
        gate_input = self._gate_activation(gate_input)
        r, u = layers.split(gate_input, num_or_sections=2, dim=1)

        r_hidden = r * pre_hidden

        candidate = layers.matmul(
            layers.concat([input, r_hidden], 1), self._candidate_weight)
        candidate = layers.elementwise_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_hidden = u * pre_hidden + (1 - u) * c

        return new_hidden
