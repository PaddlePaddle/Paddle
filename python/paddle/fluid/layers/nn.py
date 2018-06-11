#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""
All layers just related to the neural network.
"""

from ..layer_helper import LayerHelper
from ..initializer import Normal, Constant
from ..framework import Variable
from ..param_attr import ParamAttr
from layer_function_generator import autodoc
from tensor import concat
import utils

__all__ = [
    'fc',
    'embedding',
    'dynamic_lstm',
    'dynamic_lstmp',
    'dynamic_gru',
    'gru_unit',
    'linear_chain_crf',
    'crf_decoding',
    'cos_sim',
    'cross_entropy',
    'square_error_cost',
    'chunk_eval',
    'sequence_conv',
    'conv2d',
    'sequence_pool',
    'sequence_softmax',
    'softmax',
    'pool2d',
    'batch_norm',
    'beam_search_decode',
    'conv2d_transpose',
    'sequence_expand',
    'lstm_unit',
    'reduce_sum',
    'reduce_mean',
    'reduce_max',
    'reduce_min',
    'reduce_prod',
    'sequence_first_step',
    'sequence_last_step',
    'dropout',
    'split',
    'ctc_greedy_decoder',
    'edit_distance',
    'l2_normalize',
    'matmul',
    'topk',
    'warpctc',
    'sequence_reshape',
    'transpose',
    'im2sequence',
    'nce',
    'beam_search',
    'row_conv',
    'multiplex',
    'layer_norm',
    'softmax_with_cross_entropy',
    'smooth_l1',
    'one_hot',
    'autoincreased_step_counter',
    'reshape',
    'lod_reset',
    'lrn',
    'pad',
    'label_smooth',
    'roi_pool',
    'dice_loss',
    'upsampling_bilinear2d',
    'random_crop',
]


def fc(input,
       size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       use_cudnn=False,
       use_mkldnn=False,
       act=None,
       is_test=False,
       name=None):
    """
    **Fully Connected Layer**

    The fully connected layer can take multiple tensors as its inputs. It
    creates a variable called weights for each input tensor, which represents
    a fully connected weight matrix from each input unit to each output unit.
    The fully connected layer multiplies each input tensor with its coresponding
    weight to produce an output Tensor. If multiple input tensors are given,
    the results of multiple multiplications will be sumed up. If bias_attr is
    not None, a bias variable will be created and added to the output. Finally,
    if activation is not None, it will be applied to the output as well.

    This process can be formulated as follows:

    .. math::

        Out = Act({\sum_{i=0}^{N-1}X_iW_i + b})

    In the above equation:

    * :math:`N`: Number of the input.
    * :math:`X_i`: The input tensor.
    * :math:`W`: The weights created by this layer.
    * :math:`b`: The bias parameter created by this layer (if needed).
    * :math:`Act`: The activation function.
    * :math:`Out`: The output tensor.

    Args:
        input (Variable|list of Variable): The input tensor(s) of this layer, and the dimension of
            the input tensor(s) is at least 2.
        size(int): The number of output units in this layer.
        num_flatten_dims (int, default 1): The fc layer can accept an input tensor with more than
            two dimensions. If this happens, the multidimensional tensor will first be flattened
            into a 2-dimensional matrix. The parameter `num_flatten_dims` determines how the input
            tensor is flattened: the first `num_flatten_dims` (inclusive, index starts from 1)
            dimensions will be flatten to form the first dimension of the final matrix (height of
            the matrix), and the rest `rank(X) - num_flatten_dims` dimensions are flattened to
            form the second dimension of the final matrix (width of the matrix). For example, suppose
            `X` is a 6-dimensional tensor with a shape [2, 3, 4, 5, 6], and `num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30].
        param_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for learnable
            parameters/weights of this layer.
        bias_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for the bias
            of this layer. If it is set to None, no bias will be added to the output units.
        act (str, default None): Activation to be applied to the output of this layer.
        is_test(bool): A flag indicating whether execution is in test phase.
        use_mkldnn(bool): Use mkldnn kernel or not, it is valid only when the mkldnn
            library is installed. Default: False
        name (str, default None): The name of this layer.

    Returns:
        A tensor variable storing the transformation result.

    Raises:
        ValueError: If rank of the input tensor is less than 2.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(
              name="data", shape=[32, 32], dtype="float32")
          fc = fluid.layers.fc(input=data, size=1000, act="tanh")
    """

    helper = LayerHelper("fc", **locals())

    dtype = helper.input_dtype()

    mul_results = []
    for input_var, param_attr in helper.iter_inputs_and_params():
        input_shape = input_var.shape
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[num_flatten_dims:], 1)
        ] + [size]

        w = helper.create_parameter(
            attr=param_attr, shape=param_shape, dtype=dtype, is_bias=False)
        tmp = helper.create_tmp_variable(dtype)
        helper.append_op(
            type="mul",
            inputs={"X": input_var,
                    "Y": w},
            outputs={"Out": tmp},
            attrs={"x_num_col_dims": num_flatten_dims,
                   "y_num_col_dims": 1})
        mul_results.append(tmp)

    if len(mul_results) == 1:
        pre_bias = mul_results[0]
    else:
        pre_bias = helper.create_tmp_variable(dtype)
        helper.append_op(
            type="sum", inputs={"X": mul_results}, outputs={"Out": pre_bias})
    # add bias
    pre_activation = helper.append_bias_op(pre_bias, dim_start=num_flatten_dims)
    # add activation
    return helper.append_activation(pre_activation)


def embedding(input,
              size,
              is_sparse=False,
              is_distributed=False,
              padding_idx=None,
              param_attr=None,
              dtype='float32'):
    """
    **Embedding Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    a lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    All the input variables are passed in as local variables to the LayerHelper
    constructor.

    Args:
        input(Variable): The tensor variable containing the IDs.
        size(tuple|list): The shape of the look up table parameter. It should
            have two elements which indicate the size of the dictionary of
            embeddings and the size of each embedding vector respectively.
        is_sparse(bool): The flag indicating whether to use sparse update.
        padding_idx(int|long|None): If :attr:`None`, it makes no effect to lookup.
            Otherwise the given :attr:`padding_idx` indicates padding the output
            with zeros whenever lookup encounters it in :attr:`input`. If
            :math:`padding_idx < 0`, the padding_idx to use in lookup is
            :math:`size[0] + dim`.
        param_attr(ParamAttr): Parameters for this layer
        dtype(np.dtype|core.VarDesc.VarType|str): The type of data : float32, float_16, int etc

    Returns:
        Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          dict_size = len(dataset.ids)
          data = fluid.layers.data(name='ids', shape=[32, 32], dtype='float32')
          fc = fluid.layers.embedding(input=data, size=[dict_size, 16])
    """

    helper = LayerHelper('embedding', **locals())
    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=dtype, is_bias=False)
    tmp = helper.create_tmp_variable(dtype)
    padding_idx = -1 if padding_idx is None else padding_idx if padding_idx >= 0 else (
        size[0] + padding_idx)
    helper.append_op(
        type='lookup_table',
        inputs={'Ids': input,
                'W': w},
        outputs={'Out': tmp},
        attrs={
            'is_sparse': is_sparse,
            'is_distributed': is_distributed,
            'padding_idx': padding_idx
        })
    return tmp


# TODO(qijun): expose H0 and C0
def dynamic_lstm(input,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 use_peepholes=True,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 cell_activation='tanh',
                 candidate_activation='tanh',
                 dtype='float32',
                 name=None):
    """
    **Dynamic LSTM Layer**

    The defalut implementation is diagonal/peephole connection
    (https://arxiv.org/pdf/1402.1128.pdf), the formula is as follows:

    .. math::

        i_t & = \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + W_{ic}c_{t-1} + b_i)

        f_t & = \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + W_{fc}c_{t-1} + b_f)

        \\tilde{c_t} & = act_g(W_{cx}x_t + W_{ch}h_{t-1} + b_c)

        o_t & = \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + W_{oc}c_t + b_o)

        c_t & = f_t \odot c_{t-1} + i_t \odot \\tilde{c_t}

        h_t & = o_t \odot act_h(c_t)

    where the :math:`W` terms denote weight matrices (e.g. :math:`W_{xi}` is
    the matrix of weights from the input gate to the input), :math:`W_{ic}, \
    W_{fc}, W_{oc}` are diagonal weight matrices for peephole connections. In
    our implementation, we use vectors to reprenset these diagonal weight
    matrices. The :math:`b` terms denote bias vectors (:math:`b_i` is the input
    gate bias vector), :math:`\sigma` is the non-linear activations, such as
    logistic sigmoid function, and :math:`i, f, o` and :math:`c` are the input
    gate, forget gate, output gate, and cell activation vectors, respectively,
    all of which have the same size as the cell output activation vector :math:`h`.

    The :math:`\odot` is the element-wise product of the vectors. :math:`act_g`
    and :math:`act_h` are the cell input and cell output activation functions
    and `tanh` is usually used for them. :math:`\\tilde{c_t}` is also called
    candidate hidden state, which is computed based on the current input and
    the previous hidden state.

    Set `use_peepholes` to `False` to disable peephole connection. The formula
    is omitted here, please refer to the paper
    http://www.bioinf.jku.at/publications/older/2604.pdf for details.

    Note that these :math:`W_{xi}x_{t}, W_{xf}x_{t}, W_{xc}x_{t}, W_{xo}x_{t}`
    operations on the input :math:`x_{t}` are NOT included in this operator.
    Users can choose to use fully-connect layer before LSTM layer.

    Args:
        input(Variable): The input of dynamic_lstm layer, which supports
                         variable-time length input sequence. The underlying
                         tensor in this Variable is a matrix with shape
                         (T X 4D), where T is the total time steps in this
                         mini-batch, D is the hidden size.
        size(int): 4 * hidden size.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
                               hidden-hidden weights.

                               - Weights = {:math:`W_{ch}, W_{ih}, \
                                                W_{fh}, W_{oh}`}
                               - The shape is (D x 4D), where D is the hidden
                                 size.
        bias_attr(ParamAttr|None): The bias attribute for the learnable bias
                              weights, which contains two parts, input-hidden
                              bias weights and peephole connections weights if
                              setting `use_peepholes` to `True`.

                              1. `use_peepholes = False`
                                - Biases = {:math:`b_c, b_i, b_f, b_o`}.
                                - The shape is (1 x 4D).
                              2. `use_peepholes = True`
                                - Biases = { :math:`b_c, b_i, b_f, b_o, W_{ic}, \
                                                 W_{fc}, W_{oc}`}.
                                - The shape is (1 x 7D).
        use_peepholes(bool): Whether to enable diagonal/peephole connections,
                             default `True`.
        is_reverse(bool): Whether to compute reversed LSTM, default `False`.
        gate_activation(str): The activation for input gate, forget gate and
                              output gate. Choices = ["sigmoid", "tanh", "relu",
                              "identity"], default "sigmoid".
        cell_activation(str): The activation for cell output. Choices = ["sigmoid",
                              "tanh", "relu", "identity"], default "tanh".
        candidate_activation(str): The activation for candidate hidden state.
                              Choices = ["sigmoid", "tanh",
                                  "relu", "identity"],
                              default "tanh".
        dtype(str): Data type. Choices = ["float32", "float64"], default "float32".
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        tuple: The hidden state, and cell state of LSTM. The shape of both \
        is (T x D), and lod is the same with the `input`.

    Examples:
        .. code-block:: python

            hidden_dim = 512
            forward_proj = fluid.layers.fc(input=input_seq, size=hidden_dim * 4,
                                           act=None, bias_attr=None)
            forward, _ = fluid.layers.dynamic_lstm(
                input=forward_proj, size=hidden_dim * 4, use_peepholes=False)
    """

    helper = LayerHelper('lstm', **locals())
    size = size / 4
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 4 * size], dtype=dtype)
    bias_size = [1, 7 * size]
    if not use_peepholes:
        bias_size[1] = 4 * size
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)

    hidden = helper.create_tmp_variable(dtype)
    cell = helper.create_tmp_variable(dtype)
    batch_gate = helper.create_tmp_variable(dtype)
    batch_cell_pre_act = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='lstm',
        inputs={'Input': input,
                'Weight': weight,
                'Bias': bias},
        outputs={
            'Hidden': hidden,
            'Cell': cell,
            'BatchGate': batch_gate,
            'BatchCellPreAct': batch_cell_pre_act
        },
        attrs={
            'use_peepholes': use_peepholes,
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'cell_activation': cell_activation,
            'candidate_activation': candidate_activation
        })
    return hidden, cell


def dynamic_lstmp(input,
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
                  name=None):
    """
    **Dynamic LSTMP Layer**

    LSTMP (LSTM with recurrent projection) layer has a separate projection
    layer after the LSTM layer, projecting the original hidden state to a
    lower-dimensional one, which is proposed to reduce the number of total
    parameters and furthermore computational complexity for the LSTM,
    espeacially for the case that the size of output units is relative
    large (https://research.google.com/pubs/archive/43905.pdf).

    The formula is as follows:

    .. math::

        i_t & = \sigma(W_{ix}x_{t} + W_{ir}r_{t-1} + W_{ic}c_{t-1} + b_i)

        f_t & = \sigma(W_{fx}x_{t} + W_{fr}r_{t-1} + W_{fc}c_{t-1} + b_f)

        \\tilde{c_t} & = act_g(W_{cx}x_t + W_{cr}r_{t-1} + b_c)

        o_t & = \sigma(W_{ox}x_{t} + W_{or}r_{t-1} + W_{oc}c_t + b_o)

        c_t & = f_t \odot c_{t-1} + i_t \odot \\tilde{c_t}

        h_t & = o_t \odot act_h(c_t)

        r_t & = \overline{act_h}(W_{rh}h_t)

    In the above formula:

    * :math:`W`: Denotes weight matrices (e.g. :math:`W_{xi}` is \
          the matrix of weights from the input gate to the input).
    * :math:`W_{ic}`, :math:`W_{fc}`, :math:`W_{oc}`: Diagonal weight \
          matrices for peephole connections. In our implementation, \
          we use vectors to reprenset these diagonal weight matrices.
    * :math:`b`: Denotes bias vectors (e.g. :math:`b_i` is the input gate \
          bias vector).
    * :math:`\sigma`: The activation, such as logistic sigmoid function.
    * :math:`i, f, o` and :math:`c`: The input gate, forget gate, output \
          gate, and cell activation vectors, respectively, all of which have \
          the same size as the cell output activation vector :math:`h`.
    * :math:`h`: The hidden state.
    * :math:`r`: The recurrent projection of the hidden state.
    * :math:`\\tilde{c_t}`: The candidate hidden state, whose \
          computation is based on the current input and previous hidden state.
    * :math:`\odot`: The element-wise product of the vectors.
    * :math:`act_g` and :math:`act_h`: The cell input and cell output \
          activation functions and `tanh` is usually used for them.
    * :math:`\overline{act_h}`: The activation function for the projection \
          output, usually using `identity` or same as :math:`act_h`.

    Set `use_peepholes` to `False` to disable peephole connection. The formula
    is omitted here, please refer to the paper
    http://www.bioinf.jku.at/publications/older/2604.pdf for details.

    Note that these :math:`W_{xi}x_{t}, W_{xf}x_{t}, W_{xc}x_{t}, W_{xo}x_{t}`
    operations on the input :math:`x_{t}` are NOT included in this operator.
    Users can choose to use fully-connected layer before LSTMP layer.

    Args:
        input(Variable): The input of dynamic_lstmp layer, which supports
                         variable-time length input sequence. The underlying
                         tensor in this Variable is a matrix with shape
                         (T X 4D), where T is the total time steps in this
                         mini-batch, D is the hidden size.
        size(int): 4 * hidden size.
        proj_size(int): The size of projection output.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
                               hidden-hidden weight and projection weight.

                               - Hidden-hidden weight = {:math:`W_{ch}, W_{ih}, \
                                                W_{fh}, W_{oh}`}.
                               - The shape of hidden-hidden weight is (P x 4D),
                                 where P is the projection size and D the hidden
                                 size.
                               - Projection weight = {:math:`W_{rh}`}.
                               - The shape of projection weight is (D x P).
        bias_attr(ParamAttr|None): The bias attribute for the learnable bias
                              weights, which contains two parts, input-hidden
                              bias weights and peephole connections weights if
                              setting `use_peepholes` to `True`.

                              1. `use_peepholes = False`
                                - Biases = {:math:`b_c, b_i, b_f, b_o`}.
                                - The shape is (1 x 4D).
                              2. `use_peepholes = True`
                                - Biases = { :math:`b_c, b_i, b_f, b_o, W_{ic}, \
                                                 W_{fc}, W_{oc}`}.
                                - The shape is (1 x 7D).
        use_peepholes(bool): Whether to enable diagonal/peephole connections,
                             default `True`.
        is_reverse(bool): Whether to compute reversed LSTM, default `False`.
        gate_activation(str): The activation for input gate, forget gate and
                              output gate. Choices = ["sigmoid", "tanh", "relu",
                              "identity"], default "sigmoid".
        cell_activation(str): The activation for cell output. Choices = ["sigmoid",
                              "tanh", "relu", "identity"], default "tanh".
        candidate_activation(str): The activation for candidate hidden state.
                              Choices = ["sigmoid", "tanh",
                                  "relu", "identity"],
                              default "tanh".
        proj_activation(str): The activation for projection output.
                              Choices = ["sigmoid", "tanh",
                                  "relu", "identity"],
                              default "tanh".
        dtype(str): Data type. Choices = ["float32", "float64"], default "float32".
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        tuple: The projection of hidden state, and cell state of LSTMP. The \
               shape of projection is (T x P), for the cell state which is \
               (T x D), and both LoD is the same with the `input`.

    Examples:
        .. code-block:: python

            hidden_dim, proj_dim = 512, 256
            fc_out = fluid.layers.fc(input=input_seq, size=hidden_dim * 4,
                                     act=None, bias_attr=None)
            proj_out, _ = fluid.layers.dynamic_lstmp(input=fc_out,
                                                     size=hidden_dim * 4,
                                                     proj_size=proj_dim,
                                                     use_peepholes=False,
                                                     is_reverse=True,
                                                     cell_activation="tanh",
                                                     proj_activation="tanh")
    """

    helper = LayerHelper('lstmp', **locals())
    size = size / 4
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[proj_size, 4 * size], dtype=dtype)
    proj_weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, proj_size], dtype=dtype)
    bias_size = [1, 7 * size]
    if not use_peepholes:
        bias_size[1] = 4 * size
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)

    projection = helper.create_tmp_variable(dtype)
    cell = helper.create_tmp_variable(dtype)
    ordered_proj0 = helper.create_tmp_variable(dtype)
    batch_hidden = helper.create_tmp_variable(dtype)
    batch_gate = helper.create_tmp_variable(dtype)
    batch_cell_pre_act = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='lstmp',
        inputs={
            'Input': input,
            'Weight': weight,
            'ProjWeight': proj_weight,
            'Bias': bias
        },
        outputs={
            'Projection': projection,
            'Cell': cell,
            'OrderedP0': ordered_proj0,
            'BatchHidden': batch_hidden,
            'BatchGate': batch_gate,
            'BatchCellPreAct': batch_cell_pre_act
        },
        attrs={
            'use_peepholes': use_peepholes,
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'cell_activation': cell_activation,
            'candidate_activation': candidate_activation,
            'proj_activation': proj_activation
        })
    return projection, cell


def dynamic_gru(input,
                size,
                param_attr=None,
                bias_attr=None,
                is_reverse=False,
                gate_activation='sigmoid',
                candidate_activation='tanh',
                h_0=None):
    """
    **Dynamic GRU Layer**

    Refer to `Empirical Evaluation of Gated Recurrent Neural Networks on
    Sequence Modeling <https://arxiv.org/abs/1412.3555>`_

    The formula is as follows:

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = (1-u_t) \odot h_{t-1} + u_t \odot \\tilde{h_t}

    The :math:`\odot` is the element-wise product of the vectors. :math:`act_g`
    is the update gate and reset gate activation function and :math:`sigmoid`
    is usually used for it. :math:`act_c` is the activation function for
    candidate hidden state and :math:`tanh` is usually used for it.

    Note that these :math:`W_{ux}x_{t}, W_{rx}x_{t}, W_{cx}x_{t}` operations on
    the input :math:`x_{t}` are NOT included in this operator. Users can choose
    to use fully-connect layer before GRU layer.

    Args:
        input(Variable): The input of dynamic_gru layer, which supports
            variable-time length input sequence. The underlying tensor in this
            Variable is a matrix with shape :math:`(T \\times 3D)`, where
            :math:`T` is the total time steps in this mini-batch, :math:`D`
            is the hidden size.
        size(int): The dimension of the gru cell.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            hidden-hidden weight matrix. Note:

            - The shape of the weight matrix is :math:`(T \\times 3D)`, where
              :math:`D` is the hidden size.
            - All elements in the weight matrix can be divided into two parts.
              The first part are weights of the update gate and reset gate with
              shape :math:`(D \\times 2D)`, and the second part are weights for
              candidate hidden state with shape :math:`(D \\times D)`.
        bias_attr(ParamAttr): The parameter attribute for learnable the
            hidden-hidden bias.
        is_reverse(bool): Whether to compute reversed GRU, default
            :attr:`False`.
        gate_activation(str): The activation for update gate and reset gate.
            Choices = ["sigmoid", "tanh", "relu", "identity"], default "sigmoid".
        activation(str): The activation for candidate hidden state.
            Choices = ["sigmoid", "tanh", "relu", "identity"], default "tanh".

    Returns:
        Variable: The hidden state of GRU. The shape is :math:`(T \\times D)`, \
            and lod is the same with the input.

    Examples:
        .. code-block:: python

            hidden_dim = 512
            x = fluid.layers.fc(input=data, size=hidden_dim * 3)
            hidden = fluid.layers.dynamic_gru(input=x, dim=hidden_dim)
    """

    helper = LayerHelper('gru', **locals())
    dtype = helper.input_dtype()

    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 3 * size], dtype=dtype)
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=[1, 3 * size], dtype=dtype, is_bias=True)
    inputs = {'Input': input, 'Weight': weight, 'Bias': bias}
    if h_0 != None:
        assert h_0.shape == (
            size, size), 'The shape of h0 should be(%d, %d)' % (size, size)
        inputs['h0'] = h_0

    hidden = helper.create_tmp_variable(dtype)
    batch_gate = helper.create_tmp_variable(dtype)
    batch_reset_hidden_prev = helper.create_tmp_variable(dtype)
    batch_hidden = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='gru',
        inputs=inputs,
        outputs={
            'Hidden': hidden,
            'BatchGate': batch_gate,
            'BatchResetHiddenPrev': batch_reset_hidden_prev,
            'BatchHidden': batch_hidden
        },
        attrs={
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'activation': candidate_activation
        })
    return hidden


def gru_unit(input,
             hidden,
             size,
             param_attr=None,
             bias_attr=None,
             activation='tanh',
             gate_activation='sigmoid'):
    """
    GRU unit layer. The equation of a gru step is:

        .. math::
            u_t & = actGate(xu_{t} + W_u h_{t-1} + b_u)

            r_t & = actGate(xr_{t} + W_r h_{t-1} + b_r)

            m_t & = actNode(xm_t + W_c dot(r_t, h_{t-1}) + b_m)

            h_t & = dot((1-u_t), m_t) + dot(u_t, h_{t-1})

    The inputs of gru unit includes :math:`z_t`, :math:`h_{t-1}`. In terms
    of the equation above, the :math:`z_t` is split into 3 parts -
    :math:`xu_t`, :math:`xr_t` and :math:`xm_t`. This means that in order to
    implement a full GRU unit operator for an input, a fully
    connected layer has to be applied, such that :math:`z_t = W_{fc}x_t`.

    The terms :math:`u_t` and :math:`r_t` represent the update and reset gates
    of the GRU cell. Unlike LSTM, GRU has one lesser gate. However, there is
    an intermediate candidate hidden output, which is denoted by :math:`m_t`.
    This layer has three outputs :math:`h_t`, :math:`dot(r_t, h_{t-1})`
    and concatenation of :math:`u_t`, :math:`r_t` and :math:`m_t`.

    Args:
        input (Variable): The fc transformed input value of current step.
        hidden (Variable): The hidden value of lstm unit from previous step.
        size (integer): The input dimension value.
        param_attr (ParamAttr): The weight parameters for gru unit. Default: None
        bias_attr (ParamAttr): The bias parameters for gru unit. Default: None
        activation (string): The activation type for cell (actNode).
                             Default: 'tanh'
        gate_activation (string): The activation type for gates (actGate).
                                  Default: 'sigmoid'

    Returns:
        tuple: The hidden value, reset-hidden value and gate values.

    Examples:

        .. code-block:: python

             # assuming we have x_t_data and prev_hidden of size=10
             x_t = fluid.layers.fc(input=x_t_data, size=30)
             hidden_val, r_h_val, gate_val = fluid.layers.gru_unit(input=x_t,
                                                    hidden = prev_hidden)

    """
    activation_dict = dict(
        identity=0,
        sigmoid=1,
        tanh=2,
        relu=3, )
    activation = activation_dict[activation]
    gate_activation = activation_dict[gate_activation]

    helper = LayerHelper('gru_unit', **locals())
    dtype = helper.input_dtype()
    size = size / 3

    # create weight
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 3 * size], dtype=dtype)

    gate = helper.create_tmp_variable(dtype)
    reset_hidden_pre = helper.create_tmp_variable(dtype)
    updated_hidden = helper.create_tmp_variable(dtype)
    inputs = {'Input': input, 'HiddenPrev': hidden, 'Weight': weight}
    # create bias
    if helper.bias_attr:
        bias_size = [1, 3 * size]
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)
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
        })

    return updated_hidden, reset_hidden_pre, gate


def linear_chain_crf(input, label, param_attr=None):
    helper = LayerHelper('linear_chain_crf', **locals())
    size = input.shape[1]
    transition = helper.create_parameter(
        attr=helper.param_attr,
        shape=[size + 2, size],
        dtype=helper.input_dtype())
    alpha = helper.create_tmp_variable(dtype=helper.input_dtype())
    emission_exps = helper.create_tmp_variable(dtype=helper.input_dtype())
    transition_exps = helper.create_tmp_variable(dtype=helper.input_dtype())
    log_likelihood = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='linear_chain_crf',
        inputs={"Emission": [input],
                "Transition": transition,
                "Label": label},
        outputs={
            "Alpha": [alpha],
            "EmissionExps": [emission_exps],
            "TransitionExps": transition_exps,
            "LogLikelihood": log_likelihood
        })

    return log_likelihood


def crf_decoding(input, param_attr, label=None):
    helper = LayerHelper('crf_decoding', **locals())
    transition = helper.get_parameter(param_attr.name)
    viterbi_path = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='crf_decoding',
        inputs={"Emission": [input],
                "Transition": transition,
                "Label": label},
        outputs={"ViterbiPath": [viterbi_path]})

    return viterbi_path


def cos_sim(X, Y):
    """
    This function performs the cosine similarity between two tensors
    X and Y and returns that as the output.
    """
    helper = LayerHelper('cos_sim', **locals())
    out = helper.create_tmp_variable(dtype=X.dtype)
    xnorm = helper.create_tmp_variable(dtype=X.dtype)
    ynorm = helper.create_tmp_variable(dtype=X.dtype)
    helper.append_op(
        type='cos_sim',
        inputs={'X': [X],
                'Y': [Y]},
        outputs={'Out': [out],
                 'XNorm': [xnorm],
                 'YNorm': [ynorm]})
    return out


def dropout(x, dropout_prob, is_test=False, seed=None, name=None):
    """
    Computes dropout.

    Drop or keep each element of `x` independently. Dropout is a regularization
    technique for reducing overfitting by preventing neuron co-adaption during
    training. The dropout operator randomly set (according to the given dropout
    probability) the outputs of some units to zero, while others are remain
    unchanged.

    Args:
       x(variable): The input tensor.
       dropout_prob(float): Probability of setting units to zero.
       is_test(bool): A flag indicating whether it is in test phrase or not.
       seed(int): A Python integer used to create random seeds. If this
                  parameter is set to None, a random seed is used.
                  NOTE: If an integer seed is given, always the same output
                  units will be dropped. DO NOT use a fixed seed in training.
       name(str|None): A name for this layer(optional). If set None, the layer
                    will be named automatically.

    Returns:
        Variable: A tensor variable.

    Examples:
        .. code-block:: python

          x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
          droped = fluid.layers.dropout(input=x, dropout_rate=0.5)
    """

    helper = LayerHelper('dropout', **locals())
    out = helper.create_tmp_variable(dtype=x.dtype)
    mask = helper.create_tmp_variable(dtype=x.dtype, stop_gradient=True)
    helper.append_op(
        type='dropout',
        inputs={'X': [x]},
        outputs={'Out': [out],
                 'Mask': [mask]},
        attrs={
            'dropout_prob': dropout_prob,
            'is_test': is_test,
            'fix_seed': seed is not None,
            'seed': seed if seed is not None else 0
        })
    return out


def cross_entropy(input, label, soft_label=False):
    """
    **Cross Entropy Layer**

    This layer computes the cross entropy between `input` and `label`. It
    supports both standard cross-entropy and soft-label cross-entropy loss
    computation.

    1) One-hot cross-entropy:
        `soft_label = False`, `Label[i, 0]` indicates the class index for sample i:

        .. math::

            Y[i] = -\log(X[i, Label[i]])

    2) Soft-label cross-entropy:
        `soft_label = True`, `Label[i, j]` indicates the soft label of class j
        for sample i:

        .. math::

            Y[i] = \sum_j{-Label[i, j] * log(X[i, j])}

       Please make sure that in this case the summation of each row of `label`
       equals one.

    3) One-hot cross-entropy with vecterized `label`:
         As a special case of 2), when each row of 'label' has only one
         non-zero element which is equal to 1, soft-label cross-entropy degenerates
         to a one-hot cross-entropy with one-hot label representation.

    Args:
        input (Variable|list):  a 2-D tensor with shape [N x D], where N is the
                                batch size and D is the number of classes. This
                                input is a probability computed by the previous
                                operator, which is almost always the result of
                                a softmax operator.
        label (Variable|list): the ground truth which is a 2-D tensor. When
                               `soft_label` is set to `False`, `label` is a
                               tensor<int64> with shape [N x 1]. When
                               `soft_label` is set to `True`, `label` is a
                               tensor<float/double> with shape [N x D].
        soft_label (bool): a flag indicating whether to
                                           interpretate the given labels as soft
                                           labels, default `False`.

    Returns:
         A 2-D tensor with shape [N x 1], the cross entropy loss.

    Raises:
        `ValueError`: 1) the 1st dimension of `input` and `label` are not equal.
                      2) when `soft_label == True`, and the 2nd dimension of
                         `input` and `label` are not equal.
                      3) when `soft_label == False`, and the 2nd dimension of
                         `label` is not 1.

    Examples:
        .. code-block:: python

          predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
          cost = fluid.layers.cross_entropy(input=predict, label=label)
    """
    helper = LayerHelper('cross_entropy', **locals())
    out = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='cross_entropy',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out]},
        attrs={"soft_label": soft_label})
    return out


def square_error_cost(input, label):
    """
    **Square error cost layer**

    This layer accepts input predictions and target label and returns the
    squared error cost.

    For predictions, :math:`X`, and target labels, :math:`Y`, the equation is:

    .. math::

        Out = (X - Y)^2

    In the above equation:

        * :math:`X`: Input predictions, a tensor.
        * :math:`Y`: Input labels, a tensor.
        * :math:`Out`: Output value, same shape with :math:`X`.

    Args:
       input(Variable): Input tensor, has predictions.
       label(Variable): Label tensor, has target labels.

    Returns:
        Variable: The tensor variable storing the element-wise squared error \
                  difference of input and label.

    Examples:
        .. code-block:: python

          y = layers.data(name='y', shape=[1], dtype='float32')
          y_predict = layers.data(name='y_predict', shape=[1], dtype='float32')
          cost = layers.square_error_cost(input=y_predict, label=y)

    """
    helper = LayerHelper('square_error_cost', **locals())
    minus_out = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='elementwise_sub',
        inputs={'X': [input],
                'Y': [label]},
        outputs={'Out': [minus_out]})

    square_out = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='square', inputs={'X': [minus_out]},
        outputs={'Out': [square_out]})
    return square_out


def chunk_eval(input,
               label,
               chunk_scheme,
               num_chunk_types,
               excluded_chunk_types=None):
    """
    This function computes and outputs the precision, recall and
    F1-score of chunk detection.
    """
    helper = LayerHelper("chunk_eval", **locals())

    # prepare output
    precision = helper.create_tmp_variable(dtype="float32")
    recall = helper.create_tmp_variable(dtype="float32")
    f1_score = helper.create_tmp_variable(dtype="float32")
    num_infer_chunks = helper.create_tmp_variable(dtype="int64")
    num_label_chunks = helper.create_tmp_variable(dtype="int64")
    num_correct_chunks = helper.create_tmp_variable(dtype="int64")

    helper.append_op(
        type="chunk_eval",
        inputs={"Inference": [input],
                "Label": [label]},
        outputs={
            "Precision": [precision],
            "Recall": [recall],
            "F1-Score": [f1_score],
            "NumInferChunks": [num_infer_chunks],
            "NumLabelChunks": [num_label_chunks],
            "NumCorrectChunks": [num_correct_chunks]
        },
        attrs={
            "num_chunk_types": num_chunk_types,
            "chunk_scheme": chunk_scheme,
            "excluded_chunk_types": excluded_chunk_types or []
        })
    return (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
            num_correct_chunks)


def sequence_conv(input,
                  num_filters,
                  filter_size=3,
                  filter_stride=1,
                  padding=None,
                  bias_attr=None,
                  param_attr=None,
                  act=None):
    """
    This function creates the op for sequence_conv, using the inputs and
    other convolutional configurations for the filters and stride as given
    in the input parameters to the function.
    """

    # FIXME(dzh) : want to unify the argument of python layer
    # function. So we ignore some unecessary attributes.
    # such as, padding_trainable, context_start.

    helper = LayerHelper('sequence_conv', **locals())
    dtype = helper.input_dtype()
    filter_shape = [filter_size * input.shape[1], num_filters]
    filter_param = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype)
    pre_bias = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='sequence_conv',
        inputs={
            'X': [input],
            'Filter': [filter_param],
        },
        outputs={"Out": pre_bias},
        attrs={
            'contextStride': filter_stride,
            'contextStart': -int(filter_size / 2),
            'contextLength': filter_size
        })
    pre_act = helper.append_bias_op(pre_bias)
    return helper.append_activation(pre_act)


def sequence_softmax(input, param_attr=None, bias_attr=None, use_cudnn=True):
    helper = LayerHelper('sequence_softmax', **locals())
    dtype = helper.input_dtype()
    softmax_out = helper.create_tmp_variable(dtype)
    helper.append_op(
        type="sequence_softmax",
        inputs={"X": input},
        outputs={"Out": softmax_out},
        attrs={"use_cudnn": use_cudnn})
    return softmax_out


def softmax(input, param_attr=None, bias_attr=None, use_cudnn=True, name=None):
    helper = LayerHelper('softmax', **locals())
    dtype = helper.input_dtype()
    softmax_out = helper.create_tmp_variable(dtype)
    helper.append_op(
        type="softmax",
        inputs={"X": input},
        outputs={"Out": softmax_out},
        attrs={"use_cudnn": use_cudnn})
    return softmax_out


def conv2d(input,
           num_filters,
           filter_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=None,
           param_attr=None,
           bias_attr=None,
           use_cudnn=True,
           use_mkldnn=False,
           act=None,
           name=None):
    """
    **Convlution2D Layer**

    The convolution2D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are in NCHW format. Where N is batch size, C is the number of
    channels, H is the height of the feature, and W is the width of the feature.
    The details of convolution layer, please refer UFLDL's `convolution,
    <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_ .
    If bias attribution and activation type are provided, bias is added to the
    output of the convolution, and the corresponding activation function is
    applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NCHW format.
    * :math:`W`: Filter value, a tensor with MCHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be
                   different.

    Example:

        - Input:

          Input shape: $(N, C_{in}, H_{in}, W_{in})$

          Filter shape: $(C_{out}, C_{in}, H_f, W_f)$

        - Output:
          Output shape: $(N, C_{out}, H_{out}, W_{out})$

        Where

        .. math::

        H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
        W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
       input(Variable): The input image with [N, C, H, W] format.
       num_filters(int): The number of filter. It is as same as the output
           image channel.
       filter_size(int|tuple|None): The filter size. If filter_size is a tuple,
           it must contain two integers, (filter_size_H, filter_size_W).
           Otherwise, the filter will be a square.
       stride(int|tuple): The stride size. If stride is a tuple, it must
           contain two integers, (stride_H, stride_W). Otherwise, the
           stride_H = stride_W = stride. Default: stride = 1.
       padding(int|tuple): The padding size. If padding is a tuple, it must
           contain two integers, (padding_H, padding_W). Otherwise, the
           padding_H = padding_W = padding. Default: padding = 0.
       dilation(int|tuple): The dilation size. If dilation is a tuple, it must
           contain two integers, (dilation_H, dilation_W). Otherwise, the
           dilation_H = dilation_W = dilation. Default: dilation = 1.
       groups(int): The groups number of the Conv2d Layer. According to grouped
           convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
           the first half of the filters is only connected to the first half
           of the input channels, while the second half of the filters is only
           connected to the second half of the input channels. Default: groups=1
       param_attr(ParamAttr): The parameters to the Conv2d Layer. Default: None
       bias_attr(ParamAttr): Bias parameter for the Conv2d layer. Default: None
       use_cudnn(bool): Use cudnn kernel or not, it is valid only when the cudnn
           library is installed. Default: True
       act(str): Activation type. Default: None
       name(str|None): A name for this layer(optional). If set None, the layer
           will be named automatically.

    Returns:
        Variable: The tensor variable storing the convolution and \
                  non-linearity activation result.

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(
              name='data', shape=[3, 32, 32], dtype='float32')
          conv2d = fluid.layers.conv2d(
              input=data, num_filters=2, filter_size=3, act="relu")
    """
    if stride is None:
        stride = [1, 1]

    num_channels = input.shape[1]

    l_type = 'conv2d'
    if (num_channels == groups and num_filters % num_channels == 0 and
            not use_cudnn):
        l_type = 'depthwise_conv2d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()

    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        num_filter_channels = num_channels / groups

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    input_shape = input.shape
    filter_shape = [num_filters, num_filter_channels] + filter_size

    def _get_default_param_initializer():
        std = (2.0 / (filter_size[0]**2 * num_channels))**0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    pre_bias = helper.create_tmp_variable(dtype)

    helper.append_op(
        type=l_type,
        inputs={
            'Input': input,
            'Filter': filter_param,
        },
        outputs={"Output": pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': use_mkldnn
        })

    pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)

    return helper.append_activation(pre_act)


def sequence_pool(input, pool_type):
    """
    This function add the operator for sequence pooling.
    It pools features of all time-steps of each instance, and is applied
    on top of the input using pool_type mentioned in the parameters.

    It supports four pool_type:

    - average: :math:`Out[i] = \\frac{\sum_i X_i}{N}`
    - sum:     :math:`Out[i] = \sum_jX_{ij}`
    - sqrt:    :math:`Out[i] = \\frac{\sum_jX_{ij}}{\sqrt{len(X_i)}}`
    - max:     :math:`Out[i] = max(X_i)`

    .. code-block:: text

       x is a 1-level LoDTensor:
         x.lod = [[0, 2, 5, 7]]
         x.data = [1, 3, 2, 4, 6, 5, 1]
         x.dims = [7, 1]

       then output is a Tensor:
         out.dim = [3, 1]
         with condition len(x.lod[-1]) - 1 == out.dims[0]

       for different pool_type:
         average: out.data = [2, 4, 3], where 2=(1+3)/2, 4=(2+4+6)/3, 3=(5+1)/2
         sum    : out.data = [4, 12, 6], where 4=1+3, 12=2+4+6, 6=5+1
         sqrt   : out.data = [2.82, 6.93, 4.24], where 2.82=(1+3)/sqrt(2),
                    6.93=(2+4+6)/sqrt(3), 4.24=(5+1)/sqrt(2)
         max    : out.data = [3, 6, 5], where 3=max(1,3), 6=max(2,4,6), 5=max(5,1)
         last   : out.data = [3, 6, 1], where 3=last(1,3), 6=last(2,4,6), 1=last(5,1)
         first  : out.data = [1, 2, 5], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)

    Args:
        input(variable): The input variable which is a LoDTensor.
        pool_type (string): The pooling type of sequence_pool.
            It supports average, sum, sqrt and max.

    Returns:
        The sequence pooling variable which is a Tensor.

    Examples:

        .. code-block:: python

             x = fluid.layers.data(name='x', shape=[7, 1],
                              dtype='float32', lod_level=1)
             avg_x = fluid.layers.sequence_pool(input=x, pool_type='average')
             sum_x = fluid.layers.sequence_pool(input=x, pool_type='sum')
             sqrt_x = fluid.layers.sequence_pool(input=x, pool_type='sqrt')
             max_x = fluid.layers.sequence_pool(input=x, pool_type='max')
             last_x = fluid.layers.sequence_pool(input=x, pool_type='last')
             first_x = fluid.layers.sequence_pool(input=x, pool_type='first')
    """
    helper = LayerHelper('sequence_pool', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_tmp_variable(dtype)
    max_index = helper.create_tmp_variable(dtype)

    helper.append_op(
        type="sequence_pool",
        inputs={"X": input},
        outputs={"Out": pool_out,
                 "MaxIndex": max_index},
        attrs={"pooltype": pool_type.upper()})

    # when pool_type is max, variable max_index is initialized,
    # so we stop the gradient explicitly here
    if pool_type == 'max':
        max_index.stop_gradient = True

    return pool_out


def sequence_first_step(input):
    """
    This funciton get the first step of sequence.

    .. code-block:: text

       x is a 1-level LoDTensor:
         x.lod = [[0, 2, 5, 7]]
         x.data = [1, 3, 2, 4, 6, 5, 1]
         x.dims = [7, 1]

       then output is a Tensor:
         out.dim = [3, 1]
         with condition len(x.lod[-1]) - 1 == out.dims[0]
         out.data = [1, 2, 5], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)

    Args:
        input(variable): The input variable which is a LoDTensor.

    Returns:
        The sequence's first step variable which is a Tensor.

    Examples:

        .. code-block:: python

             x = fluid.layers.data(name='x', shape=[7, 1],
                              dtype='float32', lod_level=1)
             x_first_step = fluid.layers.sequence_first_step(input=x)
    """
    return sequence_pool(input=input, pool_type="first")


def sequence_last_step(input):
    """
    This funciton get the last step of sequence.

    .. code-block:: text

       x is a 1-level LoDTensor:
         x.lod = [[0, 2, 5, 7]]
         x.data = [1, 3, 2, 4, 6, 5, 1]
         x.dims = [7, 1]

       then output is a Tensor:
         out.dim = [3, 1]
         with condition len(x.lod[-1]) - 1 == out.dims[0]
         out.data = [3, 6, 1], where 3=last(1,3), 6=last(2,4,6), 1=last(5,1)

    Args:
        input(variable): The input variable which is a LoDTensor.

    Returns:
        The sequence's last step variable which is a Tensor.

    Examples:

        .. code-block:: python

             x = fluid.layers.data(name='x', shape=[7, 1],
                              dtype='float32', lod_level=1)
             x_last_step = fluid.layers.sequence_last_step(input=x)
    """
    return sequence_pool(input=input, pool_type="last")


def pool2d(input,
           pool_size=-1,
           pool_type="max",
           pool_stride=1,
           pool_padding=0,
           global_pooling=False,
           use_cudnn=True,
           ceil_mode=False,
           use_mkldnn=False,
           name=None):
    """
    This function adds the operator for pooling in 2 dimensions, using the
    pooling configurations mentioned in input parameters.
    """
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))

    if global_pooling is False and pool_size == -1:
        raise ValueError(
            "When the global_pooling is False, pool_size must be passed "
            "and be a valid value. Received pool_size: " + str(pool_size))

    pool_size = utils.convert_to_list(pool_size, 2, 'pool_size')
    pool_padding = utils.convert_to_list(pool_padding, 2, 'pool_padding')
    pool_stride = utils.convert_to_list(pool_stride, 2, 'pool_stride')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    helper = LayerHelper('pool2d', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_tmp_variable(dtype)

    helper.append_op(
        type="pool2d",
        inputs={"X": input},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "global_pooling": global_pooling,
            "strides": pool_stride,
            "paddings": pool_padding,
            "use_cudnn": use_cudnn,
            "ceil_mode": ceil_mode,
            "use_mkldnn": use_mkldnn
        })

    return pool_out


def batch_norm(input,
               act=None,
               is_test=False,
               momentum=0.9,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               data_layout='NCHW',
               in_place=False,
               use_mkldnn=False,
               name=None,
               moving_mean_name=None,
               moving_variance_name=None,
               do_model_average_for_mean_and_var=False):
    """
    This function helps create an operator to implement
    the BatchNorm layer using the configurations from the input parameters.
    """
    helper = LayerHelper('batch_norm', **locals())
    dtype = helper.input_dtype()

    input_shape = input.shape
    if data_layout == 'NCHW':
        channel_num = input_shape[1]
    else:
        if data_layout == 'NHWC':
            channel_num = input_shape[-1]
        else:
            raise ValueError("unsupported data layout:" + data_layout)

    param_shape = [channel_num]

    # create parameter
    scale = helper.create_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype=dtype,
        default_initializer=Constant(1.0))

    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True)

    mean = helper.create_parameter(
        attr=ParamAttr(
            name=moving_mean_name,
            initializer=Constant(0.0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var),
        shape=param_shape,
        dtype=input.dtype)
    mean.stop_gradient = True

    variance = helper.create_parameter(
        attr=ParamAttr(
            name=moving_variance_name,
            initializer=Constant(1.0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var),
        shape=param_shape,
        dtype=input.dtype)
    variance.stop_gradient = True

    # create output
    # mean and mean_out share the same memory
    mean_out = mean
    # variance and variance out share the same memory
    variance_out = variance
    saved_mean = helper.create_tmp_variable(dtype=dtype, stop_gradient=True)
    saved_variance = helper.create_tmp_variable(dtype=dtype, stop_gradient=True)

    batch_norm_out = input if in_place else helper.create_tmp_variable(dtype)

    helper.append_op(
        type="batch_norm",
        inputs={
            "X": input,
            "Scale": scale,
            "Bias": bias,
            "Mean": mean,
            "Variance": variance
        },
        outputs={
            "Y": batch_norm_out,
            "MeanOut": mean_out,
            "VarianceOut": variance_out,
            "SavedMean": saved_mean,
            "SavedVariance": saved_variance
        },
        attrs={
            "momentum": momentum,
            "epsilon": epsilon,
            "is_test": is_test,
            "use_mkldnn": use_mkldnn
        })

    return helper.append_activation(batch_norm_out)


def layer_norm(input,
               scale=True,
               shift=True,
               begin_norm_axis=1,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               act=None,
               name=None):
    """
    **Layer Normalization**

    Assume feature vectors exist on dimensions
    :attr:`begin_norm_axis ... rank(input)` and calculate the moment statistics
    along these dimensions for each feature vector :math:`a` with size
    :math:`H`, then normalize each feature vector using the corresponding
    statistics. After that, apply learnable gain and bias on the normalized
    tensor to scale and shift if :attr:`scale` and :attr:`shift` are set.

    Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_

    The formula is as follows:

    .. math::

        \\mu & = \\frac{1}{H}\\sum_{i=1}^{H} a_i

        \\sigma & = \\sqrt{\\frac{1}{H}\sum_{i=1}^{H}(a_i - \\mu)^2}

        h & = f(\\frac{g}{\\sigma}(a - \\mu) + b)

    Args:
        input(Variable): The input tensor variable.
        scale(bool): Whether to learn the adaptive gain :math:`g` after
            normalization.
        shift(bool): Whether to learn the adaptive bias :math:`b` after
            normalization.
        begin_norm_axis(bool): The normalization will be performed along
            dimensions from :attr:`begin_norm_axis` to :attr:`rank(input)`.
        epsilon(float): The small value added to the variance to prevent
            division by zero.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            gain :math:`g`.
        bias_attr(ParamAttr|None): The parameter attribute for the learnable
            bias :math:`b`.
        act(str): Activation to be applied to the output of layer normalizaiton.

    Returns:
        Variable: A tensor variable with the same shape as the input.

    Examples:
        .. code-block:: python

            data = fluid.layers.data(
              name='data', shape=[3, 32, 32], dtype='float32')
            x = fluid.layers.layer_norm(input=data, begin_norm_axis=1)
    """
    helper = LayerHelper('layer_norm', **locals())
    dtype = helper.input_dtype()

    # create intput and parameters
    inputs = {'X': input}
    input_shape = input.shape
    param_shape = [reduce(lambda x, y: x * y, input_shape[begin_norm_axis:])]
    if scale:
        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=param_shape,
            dtype=dtype,
            default_initializer=Constant(1.0))
        inputs['Scale'] = scale
    if shift:
        assert bias_attr is not False
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True)
        inputs['Bias'] = bias

    # create output
    mean_out = helper.create_tmp_variable(dtype=dtype, stop_gradient=True)
    variance_out = helper.create_tmp_variable(dtype=dtype, stop_gradient=True)
    layer_norm_out = helper.create_tmp_variable(dtype)

    helper.append_op(
        type="layer_norm",
        inputs=inputs,
        outputs={
            "Y": layer_norm_out,
            "Mean": mean_out,
            "Variance": variance_out,
        },
        attrs={"epsilon": epsilon,
               "begin_norm_axis": begin_norm_axis})

    return helper.append_activation(layer_norm_out)


def beam_search_decode(ids, scores, name=None):
    helper = LayerHelper('beam_search_decode', **locals())
    sentence_ids = helper.create_tmp_variable(dtype=ids.dtype)
    sentence_scores = helper.create_tmp_variable(dtype=ids.dtype)

    helper.append_op(
        type="beam_search_decode",
        inputs={"Ids": ids,
                "Scores": scores},
        outputs={
            "SentenceIds": sentence_ids,
            "SentenceScores": sentence_scores
        })

    return sentence_ids, sentence_scores


def conv2d_transpose(input,
                     num_filters,
                     output_size=None,
                     filter_size=None,
                     padding=0,
                     stride=1,
                     dilation=1,
                     groups=None,
                     param_attr=None,
                     bias_attr=None,
                     use_cudnn=True,
                     act=None,
                     name=None):
    """
    **Convlution2D transpose layer**

    The convolution2D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCHW format. Where N is batch size, C is the number of channels,
    H is the height of the feature, and W is the width of the feature.
    Parameters(dilations, strides, paddings) are two elements. These two elements
    represent height and width, respectively. The details of convolution transpose
    layer, please refer to the following explanation and references
    `therein <http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf>`_.

    For each input :math:`X`, the equation is:

    .. math::

        Out = W \\ast X

    In the above equation:

    * :math:`X`: Input value, a tensor with NCHW format.
    * :math:`W`: Filter value, a tensor with MCHW format.
    * :math:`\\ast` : Convolution transpose operation.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be
                   different.

    Example:

        - Input:

          Input shape: $(N, C_{in}, H_{in}, W_{in})$

          Filter shape: $(C_{in}, C_{out}, H_f, W_f)$

        - Output:

          Output shape: $(N, C_{out}, H_{out}, W_{out})$

        Where

        .. math::

           H_{out} &= (H_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (H_f - 1) + 1 \\\\
           W_{out} &= (W_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (W_f - 1) + 1

    Args:
       input(Variable): The input image with [N, C, H, W] format.
       num_filters(int): The number of the filter. It is as same as the output
           image channel.
       output_size(int|tuple|None): The output image size. If output size is a
           tuple, it must contain two integers, (image_H, image_W). This
           parameter only works when filter_size is None.
       filter_size(int|tuple|None): The filter size. If filter_size is a tuple,
           it must contain two integers, (filter_size_H, filter_size_W).
           Otherwise, the filter will be a square. None if use output size to
           calculate filter_size.
       padding(int|tuple): The padding size. If padding is a tuple, it must
           contain two integers, (padding_H, padding_W). Otherwise, the
           padding_H = padding_W = padding. Default: padding = 0.
       stride(int|tuple): The stride size. If stride is a tuple, it must
           contain two integers, (stride_H, stride_W). Otherwise, the
           stride_H = stride_W = stride. Default: stride = 1.
       dilation(int|tuple): The dilation size. If dilation is a tuple, it must
           contain two integers, (dilation_H, dilation_W). Otherwise, the
           dilation_H = dilation_W = dilation. Default: dilation = 1.
       groups(int): The groups number of the Conv2d transpose layer. Inspired by
           grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
           when group=2, the first half of the filters is only connected to the
           first half of the input channels, while the second half of the
           filters is only connected to the second half of the input channels.
           Default: groups=1
       param_attr(ParamAttr): The parameters to the Conv2d_transpose Layer.
                              Default: None
       bias_attr(ParamAttr): Bias parameter for the Conv2d layer. Default: None
       use_cudnn(bool): Use cudnn kernel or not, it is valid only when the cudnn
           library is installed. Default: True
       act(str): Activation type. Default: None
       name(str|None): A name for this layer(optional). If set None, the layer
           will be named automatically.

    Returns:
       Variable: The tensor variable storing the convolution transpose result.

    Raises:
       ValueError: If the shapes of input, filter_size, stride, padding and
                   groups mismatch.

    Examples:
       .. code-block:: python

          data = fluid.layers.data(
              name='data', shape=[3, 32, 32], dtype='float32')
          conv2d_transpose = fluid.layers.conv2d_transpose(
              input=data, num_filters=2, filter_size=3)
    """
    helper = LayerHelper("conv2d_transpose", **locals())
    if not isinstance(input, Variable):
        raise TypeError("Input of conv2d_transpose must be Variable")
    input_channel = input.shape[1]

    padding = utils.convert_to_list(padding, 2, 'padding')
    stride = utils.convert_to_list(stride, 2, 'stride')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    if filter_size is None:
        if output_size is None:
            raise ValueError("output_size must be set when filter_size is None")
        if isinstance(output_size, int):
            output_size = [output_size, output_size]

        h_in = input.shape[2]
        w_in = input.shape[3]

        filter_size_h = (output_size[0] - (h_in - 1) * stride[0] + 2 *
                         padding[0] - 1) / dilation[0] + 1
        filter_size_w = (output_size[1] - (w_in - 1) * stride[1] + 2 *
                         padding[1] - 1) / dilation[1] + 1
        filter_size = [filter_size_h, filter_size_w]
    else:
        filter_size = utils.convert_to_list(filter_size, 2,
                                            'conv2d_transpose.filter_size')

    groups = 1 if groups is None else groups
    filter_shape = [input_channel, num_filters / groups] + filter_size
    img_filter = helper.create_parameter(
        dtype=input.dtype, shape=filter_shape, attr=helper.param_attr)

    pre_bias = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='conv2d_transpose',
        inputs={'Input': [input],
                'Filter': [img_filter]},
        outputs={'Output': pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn
        })

    pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    out = helper.append_activation(pre_act)
    return out


def sequence_expand(x, y, ref_level=-1, name=None):
    """Sequence Expand Layer. This layer will expand the input variable **x**
    according to specified level lod of **y**. Please note that lod level of
    **x** is at most 1 and rank of **x** is at least 2. When rank of **x**
    is greater than 2, then it would be viewed as a 2-D tensor.
    Following examples will explain how sequence_expand works:

    .. code-block:: text

        * Case 1
            x is a LoDTensor:
                x.lod  = [[0,   2,        4]]
                x.data = [[a], [b], [c], [d]]
                x.dims = [4, 1]

            y is a LoDTensor:
                y.lod = [[0,    2,    4],
                         [0, 3, 6, 7, 8]]

            ref_level: 0

            then output is a 1-level LoDTensor:
                out.lod =  [[0,   2,        4,        6,        8]]
                out.data = [[a], [b], [a], [b], [c], [d], [c], [d]]
                out.dims = [8, 1]

        * Case 2
            x is a Tensor:
                x.data = [[a], [b], [c]]
                x.dims = [3, 1]

            y is a LoDTensor:
                y.lod = [[0, 2, 2, 5]]

            ref_level: -1

            then output is a Tensor:
                out.data = [[a], [a], [c], [c], [c]]
                out.dims = [5, 1]
    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor.
        y (Variable): The input variable which is a LoDTensor.
        ref_level (int): Lod level of `y` to be referred by `x`. If set to -1,
                         refer the last level of lod.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The expanded variable which is a LoDTensor.

    Examples:
        .. code-block:: python

            x = fluid.layers.data(name='x', shape=[10], dtype='float32')
            y = fluid.layers.data(name='y', shape=[10, 20],
                             dtype='float32', lod_level=1)
            out = layers.sequence_expand(x=x, y=y, ref_level=0)
    """
    helper = LayerHelper('sequence_expand', input=x, **locals())
    dtype = helper.input_dtype()
    tmp = helper.create_tmp_variable(dtype)
    helper.append_op(
        type='sequence_expand',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': tmp},
        attrs={'ref_level': ref_level})
    return tmp


def beam_search(pre_ids, ids, scores, beam_size, end_id, level=0):
    '''
    This function implements the beam search algorithm.
    '''
    helper = LayerHelper('beam_search', **locals())
    score_type = scores.dtype
    id_type = ids.dtype

    selected_scores = helper.create_tmp_variable(dtype=score_type)
    selected_ids = helper.create_tmp_variable(dtype=id_type)

    helper.append_op(
        type='beam_search',
        inputs={
            'pre_ids': pre_ids,
            'ids': ids,
            'scores': scores,
        },
        outputs={
            'selected_ids': selected_ids,
            'selected_scores': selected_scores,
        },
        attrs={
            # TODO(ChunweiYan) to assure other value support
            'level': level,
            'beam_size': beam_size,
            'end_id': end_id,
        })

    return selected_ids, selected_scores


def lstm_unit(x_t,
              hidden_t_prev,
              cell_t_prev,
              forget_bias=0.0,
              param_attr=None,
              bias_attr=None,
              name=None):
    """Lstm unit layer. The equation of a lstm step is:

        .. math::

            i_t & = \sigma(W_{x_i}x_{t} + W_{h_i}h_{t-1} + b_i)

            f_t & = \sigma(W_{x_f}x_{t} + W_{h_f}h_{t-1} + b_f)

            c_t & = f_tc_{t-1} + i_t tanh (W_{x_c}x_t + W_{h_c}h_{t-1} + b_c)

            o_t & = \sigma(W_{x_o}x_{t} + W_{h_o}h_{t-1} + b_o)

            h_t & = o_t tanh(c_t)

    The inputs of lstm unit include :math:`x_t`, :math:`h_{t-1}` and
    :math:`c_{t-1}`. The 2nd dimensions of :math:`h_{t-1}` and :math:`c_{t-1}`
    should be same. The implementation separates the linear transformation and
    non-linear transformation apart. Here, we take :math:`i_t` as an example.
    The linear transformation is applied by calling a `fc` layer and the
    equation is:

        .. math::

            L_{i_t} = W_{x_i}x_{t} + W_{h_i}h_{t-1} + b_i

    The non-linear transformation is applied by calling `lstm_unit_op` and the
    equation is:

        .. math::

            i_t = \sigma(L_{i_t})

    This layer has two outputs including :math:`h_t` and :math:`o_t`.

    Args:
        x_t (Variable): The input value of current step, a 2-D tensor with shape
            M x N, M for batch size and N for input size.
        hidden_t_prev (Variable): The hidden value of lstm unit, a 2-D tensor
            with shape M x S, M for batch size and S for size of lstm unit.
        cell_t_prev (Variable): The cell value of lstm unit, a 2-D tensor with
            shape M x S, M for batch size and S for size of lstm unit.
        forget_bias (float): The forget bias of lstm unit.
        param_attr (ParamAttr): The attributes of parameter weights, used to set
            initializer, name etc.
        bias_attr (ParamAttr): The attributes of bias weights, if not False,
            bias weights will be created and be set to default value.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        tuple: The hidden value and cell value of lstm unit.

    Raises:
        ValueError: The ranks of **x_t**, **hidden_t_prev** and **cell_t_prev**
                    not be 2 or the 1st dimensions of **x_t**, **hidden_t_prev**
                    and **cell_t_prev** not be the same or the 2nd dimensions of
                    **hidden_t_prev** and **cell_t_prev** not be the same.

    Examples:

        .. code-block:: python

             x_t = fluid.layers.fc(input=x_t_data, size=10)
             prev_hidden = fluid.layers.fc(input=prev_hidden_data, size=30)
             prev_cell = fluid.layers.fc(input=prev_cell_data, size=30)
             hidden_value, cell_value = fluid.layers.lstm_unit(x_t=x_t,
                                                    hidden_t_prev=prev_hidden,
                                                    cell_t_prev=prev_cell)
    """
    helper = LayerHelper('lstm_unit', **locals())

    if len(x_t.shape) != 2:
        raise ValueError("Rank of x_t must be 2.")

    if len(hidden_t_prev.shape) != 2:
        raise ValueError("Rank of hidden_t_prev must be 2.")

    if len(cell_t_prev.shape) != 2:
        raise ValueError("Rank of cell_t_prev must be 2.")

    if x_t.shape[0] != hidden_t_prev.shape[0] or x_t.shape[
            0] != cell_t_prev.shape[0]:
        raise ValueError("The 1st dimensions of x_t, hidden_t_prev and "
                         "cell_t_prev must be the same.")

    if hidden_t_prev.shape[1] != cell_t_prev.shape[1]:
        raise ValueError("The 2nd dimensions of hidden_t_prev and "
                         "cell_t_prev must be the same.")

    if bias_attr is None:
        bias_attr = ParamAttr()

    size = cell_t_prev.shape[1]
    concat_out = concat(input=[x_t, hidden_t_prev], axis=1)
    fc_out = fc(input=concat_out,
                size=4 * size,
                param_attr=param_attr,
                bias_attr=bias_attr)
    dtype = x_t.dtype
    c = helper.create_tmp_variable(dtype)
    h = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='lstm_unit',
        inputs={"X": fc_out,
                "C_prev": cell_t_prev},
        outputs={"C": c,
                 "H": h},
        attrs={"forget_bias": forget_bias})

    return h, c


def reduce_sum(input, dim=None, keep_dim=False, name=None):
    """
    Computes the sum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (list|int|None): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`input` and return a
            Tensor variable with a single element, otherwise must be in the
            range :math:`[-rank(input), rank(input))`. If :math:`dim[i] < 0`,
            the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool|False): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        Variable: The reduced Tensor variable.

    Examples:
        .. code-block:: python

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_sum(x)  # [3.5]
            fluid.layers.reduce_sum(x, dim=0)  # [0.3, 0.5, 1.1, 1.6]
            fluid.layers.reduce_sum(x, dim=-1)  # [1.9, 1.6]
            fluid.layers.reduce_sum(x, dim=1, keep_dim=True)  # [[1.9], [1.6]]

            # x is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1, 2], [3, 4]],
            #      [[5, 6], [7, 8]]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_sum(x, dim=[1, 2]) # [10, 26]
            fluid.layers.reduce_sum(x, dim=[0, 1]) # [16, 20]

    """
    helper = LayerHelper('reduce_sum', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_sum',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out


def reduce_mean(input, dim=None, keep_dim=False, name=None):
    """
    Computes the mean of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (list|int|None): The dimensions along which the mean is computed. If
            :attr:`None`, compute the mean over all elements of :attr:`input`
            and return a Tensor variable with a single element, otherwise
            must be in the range :math:`[-rank(input), rank(input))`. If
            :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        Variable: The reduced Tensor variable.

    Examples:
        .. code-block:: python

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_mean(x)  # [0.4375]
            fluid.layers.reduce_mean(x, dim=0)  # [0.15, 0.25, 0.55, 0.8]
            fluid.layers.reduce_mean(x, dim=-1)  # [0.475, 0.4]
            fluid.layers.reduce_mean(
                x, dim=1, keep_dim=True)  # [[0.475], [0.4]]

            # x is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_mean(x, dim=[1, 2]) # [2.5, 6.5]
            fluid.layers.reduce_mean(x, dim=[0, 1]) # [4.0, 5.0]
    """
    helper = LayerHelper('reduce_mean', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_mean',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out


def reduce_max(input, dim=None, keep_dim=False, name=None):
    """
    Computes the maximum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (list|int|None): The dimension along which the maximum is computed.
            If :attr:`None`, compute the maximum over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        Variable: The reduced Tensor variable.

    Examples:
        .. code-block:: python

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_max(x)  # [0.9]
            fluid.layers.reduce_max(x, dim=0)  # [0.2, 0.3, 0.6, 0.9]
            fluid.layers.reduce_max(x, dim=-1)  # [0.9, 0.7]
            fluid.layers.reduce_max(x, dim=1, keep_dim=True)  # [[0.9], [0.7]]

            # x is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_max(x, dim=[1, 2]) # [4.0, 8.0]
            fluid.layers.reduce_max(x, dim=[0, 1]) # [7.0, 8.0]
    """
    helper = LayerHelper('reduce_max', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_max',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out


def reduce_min(input, dim=None, keep_dim=False, name=None):
    """
    Computes the minimum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (list|int|None): The dimensions along which the minimum is computed.
            If :attr:`None`, compute the minimum over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        Variable: The reduced Tensor variable.

    Examples:
        .. code-block:: python

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_min(x)  # [0.1]
            fluid.layers.reduce_min(x, dim=0)  # [0.1, 0.2, 0.5, 0.7]
            fluid.layers.reduce_min(x, dim=-1)  # [0.2, 0.1]
            fluid.layers.reduce_min(x, dim=1, keep_dim=True)  # [[0.2], [0.1]]

            # x is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_min(x, dim=[1, 2]) # [1.0, 5.0]
            fluid.layers.reduce_min(x, dim=[0, 1]) # [1.0, 2.0]
    """
    helper = LayerHelper('reduce_min', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_min',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out


def reduce_prod(input, dim=None, keep_dim=False, name=None):
    """
    Computes the product of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (list|int|None): The dimensions along which the product is performed. If
            :attr:`None`, multipy all elements of :attr:`input` and return a
            Tensor variable with a single element, otherwise must be in the
            range :math:`[-rank(input), rank(input))`. If :math:`dim[i] < 0`,
            the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool|False): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true.
        name(str|None): A name for this layer(optional). If set None, the
            layer will be named automatically.

    Returns:
        Variable: The reduced Tensor variable.

    Examples:
        .. code-block:: python

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_prod(x)  # [0.0002268]
            fluid.layers.reduce_prod(x, dim=0)  # [0.02, 0.06, 0.3, 0.63]
            fluid.layers.reduce_prod(x, dim=-1)  # [0.027, 0.0084]
            fluid.layers.reduce_prod(x, dim=1,
                                     keep_dim=True)  # [[0.027], [0.0084]]

            # x is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_prod(x, dim=[1, 2]) # [24.0, 1680.0]
            fluid.layers.reduce_prod(x, dim=[0, 1]) # [105.0, 384.0]
    """
    helper = LayerHelper('reduce_prod', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_prod',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out


def split(input, num_or_sections, dim=-1, name=None):
    """
    Split the input tensor into multiple sub-tensors.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor.
        num_or_sections (int|list): If :attr:`num_or_sections` is an integer,
            then the integer indicates the number of equal sized sub-tensors
            that the tensor will be divided into. If :attr:`num_or_sections`
            is a list of integers, the length of list indicates the number of
            sub-tensors and the integers indicate the sizes of sub-tensors'
            :attr:`dim` dimension orderly.
        dim (int): The dimension along which to split. If :math:`dim < 0`, the
            dimension to split along is :math:`rank(input) + dim`.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        List: The list of segmented tensor variables.

    Examples:
        .. code-block:: python

            # x is a Tensor variable with shape [3, 9, 5]:
            x0, x1, x2 = fluid.layers.split(x, num_or_sections=3, dim=1)
            x0.shape  # [3, 3, 5]
            x1.shape  # [3, 3, 5]
            x2.shape  # [3, 3, 5]
            x0, x1, x2 = fluid.layers.split(
                x, num_or_sections=[2, 3, 4], dim=1)
            x0.shape  # [3, 2, 5]
            x1.shape  # [3, 3, 5]
            x2.shape  # [3, 4, 5]
    """
    helper = LayerHelper('split', **locals())
    input_shape = input.shape
    dim = (len(input_shape) + dim) if dim < 0 else dim
    if isinstance(num_or_sections, int):
        assert num_or_sections > 1, 'num_or_sections must be more than 1.'
        num = num_or_sections
    else:
        assert len(num_or_sections) < input_shape[
            dim], 'len(num_or_sections) must not be more than input.shape[dim].'
        num = len(num_or_sections)
    outs = [
        helper.create_tmp_variable(dtype=helper.input_dtype())
        for i in range(num)
    ]
    helper.append_op(
        type='split',
        inputs={'X': input},
        outputs={'Out': outs},
        attrs={
            'num': num_or_sections if isinstance(num_or_sections, int) else 0,
            'sections': num_or_sections
            if isinstance(num_or_sections, list) else [],
            'axis': dim
        })
    return outs


def l2_normalize(x, axis, epsilon=1e-12, name=None):
    """
    **L2 normalize Layer**

    The l2 normalize layer normalizes `x` along dimension `axis` using an L2
    norm. For a 1-D tensor (`dim` is fixed to 0), this layer computes

    output = x / sqrt(max(sum(x**2), epsilon))

    For `x` with more dimensions, this layer independently normalizes each 1-D
    slice along dimension `axis`.

    Args:
       x(Variable|list): The input tensor to l2_normalize layer.
       axis(int): Dimension along which to normalize the input.
       epsilon(float): A lower bound value for `x`'s l2 norm. sqrt(epsilon) will
                       be used as the divisor if the l2 norm of `x` is less than
                       sqrt(epsilon).
       name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.


    Returns:
        Variable: The output tensor variable.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(name="data",
                                   shape=(3, 17, 13),
                                   dtype="float32")
          normed = fluid.layers.l2_normalize(x=data, axis=1)
    """

    if len(x.shape) == 1:
        axis = 0
    helper = LayerHelper("l2_normalize", **locals())

    square = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(type="square", inputs={"X": x}, outputs={"Out": square})

    reduced_sum = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type="reduce_sum",
        inputs={"X": square},
        outputs={"Out": reduced_sum},
        attrs={
            "dim": [1] if axis is None else [axis],
            "keep_dim": True,
            "reduce_all": False
        })

    # TODO(caoying) A lower bound value epsilon for the norm is needed to
    # imporve the numeric stability of reciprocal. This requires a maximum_op.
    rsquare = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type="reciprocal", inputs={"X": reduced_sum}, outputs={"Out": rsquare})

    # TODO(caoying) the current elementwise_mul operator does not support a
    # general broadcast rule which broadcasts input(Y) to have the same
    # dimension with Input(X) starting from a specified dimension. So this
    # exanpsion is requred. Once a general broadcast rule is spported, this
    # expanding canbe removed.
    rsquare_expanded = helper.create_tmp_variable(dtype=x.dtype)
    expand_times = [1] * len(x.shape)
    expand_times[axis] = int(x.shape[axis])
    helper.append_op(
        type="expand",
        inputs={"X": rsquare},
        outputs={"Out": rsquare_expanded},
        attrs={"expand_times": expand_times})

    out = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type="elementwise_mul",
        inputs={"X": x,
                "Y": rsquare_expanded},
        outputs={"Out": out})
    return out


def matmul(x, y, transpose_x=False, transpose_y=False, name=None):
    """
    Applies matrix multiplication to two tensors.

    Currently, the input tensors' rank can be any, but when the rank of any
    inputs is bigger than 3, this two inputs' rank should be equal.

    The actual behavior depends on the shapes of :math:`x`, :math:`y` and the
    flag values of :attr:`transpose_x`, :attr:`transpose_y`. Specifically:

    - If a transpose flag is specified, the last two dimensions of the tensor
      are transposed. If the tensor is rank-1 of shape :math:`[D]`, then for
      :math:`x` it is treated as :math:`[1, D]` in nontransposed form and as
      :math:`[D, 1]` in transposed form, whereas for :math:`y` it is the
      opposite: It is treated as :math:`[D, 1]` in nontransposed form and as
      :math:`[1, D]` in transposed form.

    - After transpose, the two tensors are 2-D or n-D and matrix multiplication
      performs in the following way.

      - If both are 2-D, they are multiplied like conventional matrices.
      - If either is n-D, it is treated as a stack of matrices residing in the
        last two dimensions and a batched matrix multiply supporting broadcast
        applies on the two tensors.

    Also note that if the raw tensor :math:`x` or :math:`y` is rank-1 and
    nontransposed, the prepended or appended dimension :math:`1` will be
    removed after matrix multiplication.

    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor.
        y (Variable): The input variable which is a Tensor or LoDTensor.
        transpose_x (bool): Whether to transpose :math:`x` before multiplication.
        transpose_y (bool): Whether to transpose :math:`y` before multiplication.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The product Tensor variable.

    Examples:
        .. code-block:: python

            # Examples to clarify shapes of the inputs and output
            # x: [B, ..., M, K], y: [B, ..., K, N]
            fluid.layers.matmul(x, y)  # out: [B, ..., M, N]

            # x: [B, M, K], y: [B, K, N]
            fluid.layers.matmul(x, y)  # out: [B, M, N]

            # x: [B, M, K], y: [K, N]
            fluid.layers.matmul(x, y)  # out: [B, M, N]

            # x: [M, K], y: [K, N]
            fluid.layers.matmul(x, y)  # out: [M, N]

            # x: [B, M, K], y: [K]
            fluid.layers.matmul(x, y)  # out: [B, M]

            # x: [K], y: [K]
            fluid.layers.matmul(x, y)  # out: [1]

            # x: [M], y: [N]
            fluid.layers.matmul(x, y, True, True)  # out: [M, N]
    """

    def __check_input(x, y):
        if len(y.shape) > len(x.shape):
            raise ValueError(
                "Invalid inputs for matmul. "
                "x's rank should be always greater than or equal to y'rank.")

        x_shape = list(x.shape)
        y_shape = list(y.shape)
        if len(x_shape) == 1:
            x_shape = [1] + x_shape
        if len(y_shape) == 1:
            y_shape = y_shape + [1]

        # check the inner 2 dimensions
        if transpose_x:
            x_shape[-2], x_shape[-1] = x_shape[-1], x_shape[-2]
        if transpose_y:
            y_shape[-2], y_shape[-1] = y_shape[-1], y_shape[-2]
        if x_shape[-1] != y_shape[-2]:
            raise ValueError("Invalid inputs for matmul.")

        if len(y_shape) > 2:
            for i, dim_x in enumerate(x_shape[:-2]):
                if dim_x != y_shape[i]:
                    raise ValueError("Invalid inputs for matmul.")

    __check_input(x, y)

    helper = LayerHelper('matmul', **locals())
    out = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type='matmul',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={'transpose_X': transpose_x,
               'transpose_Y': transpose_y})
    return out


def topk(input, k, name=None):
    """
    This operator is used to find values and indices of the k largest entries
    for the last dimension.

    If the input is a vector (rank=1), finds the k largest entries in the vector
    and outputs their values and indices as vectors. Thus values[j] is the j-th
    largest entry in input, and its index is indices[j].

    If the input is a Tensor with higher rank, this operator computes the top k
    entries along the last dimension.

    Args:
        input(Variable): The input variable which can be a vector or Tensor with
            higher rank.
        k(int): An integer value to specify the top k largest elements.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        values(Variable): The k largest elements along each last dimensional
            slice.
        indices(Variable): The indices of values within the last dimension of
            input.

    Examples:
        .. code-block:: python

            top5_values, top5_indices = layers.topk(input, k=5)
    """
    shape = input.shape
    if k < 1 and k >= shape[-1]:
        raise ValueError("k must be greater than 0 and less than %d." %
                         (shape[-1]))

    helper = LayerHelper("top_k", **locals())
    values = helper.create_tmp_variable(dtype=input.dtype)
    indices = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="top_k",
        inputs={"X": [input]},
        outputs={"Out": [values],
                 "Indices": [indices]},
        attrs={"k": k})
    values.stop_gradient = True
    indices.stop_gradient = True
    return values, indices


def edit_distance(input, label, normalized=True, ignored_tokens=None,
                  name=None):
    """
    EditDistance operator computes the edit distances between a batch of
    hypothesis strings and their references. Edit distance, also called
    Levenshtein distance, measures how dissimilar two strings are by counting
    the minimum number of operations to transform one string into anthor.
    Here the operations include insertion, deletion, and substitution.

    For example, given hypothesis string A = "kitten" and reference
    B = "sitting", the edit distance is 3 for A will be transformed into B
    at least after two substitutions and one insertion:

    "kitten" -> "sitten" -> "sittin" -> "sitting"

    Input(Hyps) is a LoDTensor consisting of all the hypothesis strings with
    the total number denoted by `batch_size`, and the separation is specified
    by the LoD information. And the `batch_size` reference strings are arranged
    in order in the same way in the LoDTensor Input(Refs).

    Output(Out) contains the `batch_size` results and each stands for the edit
    distance for a pair of strings respectively. If Attr(normalized) is true,
    the edit distance will be divided by the length of reference string.

    Args:

        input(Variable): The indices for hypothesis strings.

        label(Variable): The indices for reference strings.

        normalized(bool): Indicated whether to normalize the edit distance by
                          the length of reference string.

        ignored_tokens(list of int): Tokens that should be removed before
                                     calculating edit distance.

    Returns:
        Variable: sequence-to-sequence edit distance in shape [batch_size, 1].

    Examples:
        .. code-block:: python

            x = fluid.layers.data(name='x', shape=[8], dtype='float32')
            y = fluid.layers.data(name='y', shape=[7], dtype='float32')

            cost = fluid.layers.edit_distance(input=x,label=y)
    """
    helper = LayerHelper("edit_distance", **locals())

    # remove some tokens from input and labels
    if ignored_tokens is not None and len(ignored_tokens) > 0:
        erased_input = helper.create_tmp_variable(dtype="int64")
        erased_label = helper.create_tmp_variable(dtype="int64")

        helper.append_op(
            type="sequence_erase",
            inputs={"X": [input]},
            outputs={"Out": [erased_input]},
            attrs={"tokens": ignored_tokens})
        input = erased_input

        helper.append_op(
            type="sequence_erase",
            inputs={"X": [label]},
            outputs={"Out": [erased_label]},
            attrs={"tokens": ignored_tokens})
        label = erased_label

    # edit distance op
    edit_distance_out = helper.create_tmp_variable(dtype="int64")
    sequence_num = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="edit_distance",
        inputs={"Hyps": [input],
                "Refs": [label]},
        outputs={"Out": [edit_distance_out],
                 "SequenceNum": [sequence_num]},
        attrs={"normalized": normalized})

    return edit_distance_out, sequence_num


def ctc_greedy_decoder(input, blank, name=None):
    """
    This op is used to decode sequences by greedy policy by below steps:
    1. Get the indexes of max value for each row in input. a.k.a.
       numpy.argmax(input, axis=0).
    2. For each sequence in result of step1, merge repeated tokens between two
       blanks and delete all blanks.

    A simple example as below:

    .. code-block:: text

        Given:

        input.data = [[0.6, 0.1, 0.3, 0.1],
                      [0.3, 0.2, 0.4, 0.1],
                      [0.1, 0.5, 0.1, 0.3],
                      [0.5, 0.1, 0.3, 0.1],

                      [0.5, 0.1, 0.3, 0.1],
                      [0.2, 0.2, 0.2, 0.4],
                      [0.2, 0.2, 0.1, 0.5],
                      [0.5, 0.1, 0.3, 0.1]]

        input.lod = [[0, 4, 8]]

        Then:

        output.data = [[2],
                       [1],
                       [3]]

        output.lod = [[0, 2, 3]]

    Args:

        input(Variable): (LoDTensor<float>), the probabilities of
                         variable-length sequences, which is a 2-D Tensor with
                         LoD information. It's shape is [Lp, num_classes + 1],
                         where Lp is the sum of all input sequences' length and
                         num_classes is the true number of classes. (not
                         including the blank label).

        blank(int): the blank label index of Connectionist Temporal
                    Classification (CTC) loss, which is in thehalf-opened
                    interval [0, num_classes + 1).

    Returns:
        Variable: CTC greedy decode result. If all the sequences in result were
        empty, the result LoDTensor will be [-1] with LoD [[0]] and dims [1, 1].

    Examples:
        .. code-block:: python

            x = fluid.layers.data(name='x', shape=[8], dtype='float32')

            cost = fluid.layers.ctc_greedy_decoder(input=x, blank=0)
    """
    helper = LayerHelper("ctc_greedy_decoder", **locals())
    _, topk_indices = topk(input, k=1)

    # ctc align op
    ctc_out = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="ctc_align",
        inputs={"Input": [topk_indices]},
        outputs={"Output": [ctc_out]},
        attrs={"merge_repeated": True,
               "blank": blank})
    return ctc_out


def warpctc(input, label, blank=0, norm_by_times=False):
    """
    An operator integrating the open source Warp-CTC library
    (https://github.com/baidu-research/warp-ctc)
    to compute Connectionist Temporal Classification (CTC) loss.
    It can be aliased as softmax with CTC, since a native softmax activation is
    interated to the Warp-CTC library, to to normlize values for each row of the
    input tensor.

    Args:
       input(Variable): (LodTensor, default: LoDTensor<float>),
         the unscaled probabilities of variable-length sequences,
         which is a 2-D Tensor with LoD information.
         It's shape is [Lp, num_classes + 1], where Lp is the sum of all input
         sequences' length and num_classes is the true number of classes.
         (not including the blank label).
       label(Variable): (LodTensor, default: LoDTensor<int>), the ground truth
         of variable-length sequence, which is a 2-D Tensor with LoD
         information. It is of the shape [Lg, 1], where Lg is th sum of
         all labels' length.
       blank: (int, default: 0), the blank label index of Connectionist
         Temporal Classification (CTC) loss, which is in the
         half-opened interval [0, num_classes + 1).
       norm_by_times: (bool, default: false), whether to normalize
       the gradients by the number of time-step, which is also the
       sequence's length. There is no need to normalize the gradients
       if warpctc layer was follewed by a mean_op.

    Returns:
        Variable: The Connectionist Temporal Classification (CTC) loss,
        which is a 2-D Tensor of the shape [batch_size, 1].

    Examples:
        .. code-block:: python
            y = layers.data(
                name='y', shape=[11, 8], dtype='float32', lod_level=1)
            y_predict = layers.data(
                name='y_predict', shape=[11, 1], dtype='float32')
            cost = layers.warpctc(input=y_predict, label=y)

    """
    helper = LayerHelper('warpctc', **locals())
    loss_out = helper.create_tmp_variable(dtype=input.dtype)
    grad_out = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='warpctc',
        inputs={'Logits': [input],
                'Label': [label]},
        outputs={'WarpCTCGrad': [grad_out],
                 'Loss': [loss_out]},
        attrs={'blank': blank,
               'norm_by_times': norm_by_times})
    return loss_out


def sequence_reshape(input, new_dim):
    """
    **Sequence Reshape Layer**

    This layer will rearrange the input sequences. The new dimension is set by
    user. Length of each sequence is computed according to original length,
    original dimension and new dimension. The following example will help to
    illustrate the function of this layer:

    .. code-block:: text

        x is a LoDTensor:
            x.lod  = [[0, 2, 6]]
            x.data = [[1, 2], [3, 4],
                      [5, 6], [7, 8], [9, 10], [11, 12]]
            x.dims = [6, 2]

        set new_dim = 4

        then out is a LoDTensor:
            out.lod  = [[0, 1, 3]]
            out.data = [[1, 2, 3, 4],
                        [5, 6, 7, 8], [9, 10, 11, 12]]
            out.dims = [3, 4]

    Currently, only 1-level LoDTensor is supported and please make sure
    (original length * original dimension) can be divided by new dimension with
    no remainder for each sequence.

    Args:
       input (Variable): (LodTensor, default: LoDTensor<float>), a 2-D LoDTensor
                with shape being [N, M] where M for dimension.
       new_dim (int): New dimension which the input LoDTensor is reshaped to.

    Returns:
        Variable: Reshaped LoDTensor according to new dimension.

    Examples:
        .. code-block:: python

            x = fluid.layers.data(name='x', shape=[5, 20],
                              dtype='float32', lod_level=1)
            x_reshaped = layers.sequence_reshape(input=x, new_dim=10)
    """
    helper = LayerHelper('sequence_reshape', **locals())
    out = helper.create_tmp_variable(helper.input_dtype())
    helper.append_op(
        type='sequence_reshape',
        inputs={'X': [input]},
        outputs={'Out': [out]},
        attrs={'new_dim': new_dim})
    return out


@autodoc()
def nce(input,
        label,
        num_total_classes,
        sample_weight=None,
        param_attr=None,
        bias_attr=None,
        num_neg_samples=None):
    helper = LayerHelper('nce', **locals())
    assert isinstance(input, Variable)
    dim = input.shape[1]
    assert isinstance(label, Variable)
    num_true_class = label.shape[1]
    w = helper.create_parameter(
        attr=helper.param_attr,
        shape=[num_total_classes, dim],
        is_bias=False,
        dtype=input.dtype)
    b = helper.create_parameter(
        attr=helper.bias_attr,
        shape=[num_total_classes, 1],
        is_bias=True,
        dtype=input.dtype)
    cost = helper.create_tmp_variable(dtype=input.dtype)
    sample_logits = helper.create_tmp_variable(dtype=input.dtype)
    sample_labels = helper.create_tmp_variable(dtype=label.dtype)

    if num_neg_samples is None:
        num_neg_samples = 10
    else:
        num_neg_samples = int(num_neg_samples)

    attrs = {
        'num_total_classes': int(num_total_classes),
        'num_neg_samples': num_neg_samples
    }

    helper.append_op(
        type='nce',
        inputs={
            'Input': input,
            'Label': label,
            'Weight': w,
            'Bias': b,
            'SampleWeight': sample_weight if sample_weight is not None else []
        },
        outputs={
            'Cost': cost,
            'SampleLogits': sample_logits,
            'SampleLabels': sample_labels
        },
        attrs=attrs)
    return cost / (num_neg_samples + 1)


def transpose(x, perm, name=None):
    """
    **transpose Layer**

    Permute the dimensions of `input` according to `perm`.

    The `i`-th dimension  of the returned tensor will correspond to the
    perm[i]-th dimension of `input`.

    Args:
       input (Variable): (Tensor), A Tensor.
       perm (list): A permutation of the dimensions of `input`.

    Returns:
        Variable: A transposed Tensor.

    Examples:
        .. code-block:: python

            x = fluid.layers.data(name='x', shape=[5, 10, 15], dtype='float32')
            x_transposed = layers.transpose(x, perm=[1, 0, 2])
    """

    if len(perm) != len(x.shape):
        raise ValueError(
            "Input(perm) is the permutation of dimensions of Input(input). "
            "It's length shoud be equal to Input(input)'s rank.")
    for idx, dim in enumerate(perm):
        if dim >= len(x.shape):
            raise ValueError(
                "Each element in perm should be less than x's rank. "
                "%d-th element in perm is %d which accesses x's rank %d." %
                (idx, perm[idx], len(x.shape)))

    helper = LayerHelper('transpose', **locals())
    out = helper.create_tmp_variable(x.dtype)
    helper.append_op(
        type='transpose',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'axis': perm})
    return out


def im2sequence(input, filter_size=1, stride=1, padding=0, name=None):
    """
    Extracts image patches from the input tensor to form a tensor of shape
    {input.batch_size * output_height * output_width, filter_size_H *
    filter_size_W * input.channels} which is similar with im2col.
    This op use filter / kernel to scan images and convert these images to
    sequences. After expanding, the number of time step are
    output_height * output_width for an image, in which output_height and
    output_width are calculated by below equation:

    .. math::

        output\_size = 1 + \
            (2 * padding + img\_size - block\_size + stride - 1) / stride

    And the dimension of each time step is block_y * block_x * input.channels.

    Args:
        input (Variable): The input should be a tensor in NCHW format.

        filter_size(int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.

        stride(int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.

        padding(int|tuple): The padding size. If padding is a tuple, it can
            contain two integers like (padding_H, padding_W) which means
            padding_up = padding_down = padding_H and
            padding_left = padding_right = padding_W. Or it can use
            (padding_up, padding_left, padding_down, padding_right) to indicate
            paddings of four direction. Otherwise, a scalar padding means
            padding_up = padding_down = padding_left = padding_right = padding
            Default: padding = 0.

        name (int): The name of this layer. It is optional.

    Returns:
        output: The output is a LoDTensor with shape
        {input.batch_size * output_height * output_width,
        filter_size_H * filter_size_W * input.channels}.
        If we regard output as a matrix, each row of this matrix is
        a step of a sequence.

    Examples:

    As an example:

        .. code-block:: text

            Given:

            x = [[[[ 6.  2.  1.]
                   [ 8.  3.  5.]
                   [ 0.  2.  6.]]

                  [[ 2.  4.  4.]
                   [ 6.  3.  0.]
                   [ 6.  4.  7.]]]

                 [[[ 6.  7.  1.]
                   [ 5.  7.  9.]
                   [ 2.  4.  8.]]

                  [[ 1.  2.  1.]
                   [ 1.  3.  5.]
                   [ 9.  0.  8.]]]]

            x.dims = {2, 2, 3, 3}

            And:

            filter = [2, 2]
            stride = [1, 1]
            padding = [0, 0]

            Then:

            output.data = [[ 6.  2.  8.  3.  2.  4.  6.  3.]
                           [ 2.  1.  3.  5.  4.  4.  3.  0.]
                           [ 8.  3.  0.  2.  6.  3.  6.  4.]
                           [ 3.  5.  2.  6.  3.  0.  4.  7.]
                           [ 6.  7.  5.  7.  1.  2.  1.  3.]
                           [ 7.  1.  7.  9.  2.  1.  3.  5.]
                           [ 5.  7.  2.  4.  1.  3.  9.  0.]
                           [ 7.  9.  4.  8.  3.  5.  0.  8.]]

            output.dims = {8, 9}

            output.lod = [[0, 4, 8]]

        The simple usage is:

        .. code-block:: python

            output = fluid.layers.im2sequence(
                input=layer, stride=[1, 1], filter_size=[2, 2])

    """

    if isinstance(filter_size, int):
        filter_size = [filter_size, filter_size]
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(padding, int):
        padding = [padding, padding]
    if len(padding) == 2:
        padding.append(padding[0])
        padding.append(padding[1])

    helper = LayerHelper('im2sequence', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='im2sequence',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'kernels': filter_size,
            'strides': stride,
            'paddings': padding,
        })
    return out


def row_conv(input, future_context_size, param_attr=None, act=None):
    """Row Conv Operator. This layer will apply lookahead convolution to
    **input**. The input variable should be a 2D LoDTensor with shape [T, D].
    Parameters with shape [future_context_size + 1, D] will be created. The math
    equation of row convolution is as follows:

    .. math::
        Out_{i} = \sum_{j = i} ^ {i + \\tau} X_{j} \odot W_{i - j}

    In the above equation:

    * :math:`Out_{i}`: The i-th row of output variable with shape [1, D].
    * :math:`\\tau`: Future context size.
    * :math:`X_{j}`: The j-th row of input variable with shape [1, D].
    * :math:`W_{i-j}`: The (i-j)-th row of parameters with shape [1, D].

    More details about row_conv please refer to the paper \
    (http://www.cs.cmu.edu/~dyogatam/papers/wang+etal.iclrworkshop2016.pdf) and
    the design document \
    (https://github.com/PaddlePaddle/Paddle/issues/2228#issuecomment-303903645).

    Args:
        input (Variable): Input variable, a 2D LoDTensor with shape [T, D].
        future_context_size (int): Future context size. Please note, the shape
            of convolution kernel is [future_context_size + 1, D].
        param_attr (ParamAttr): Attributes of parameters, including
            name, initializer etc.
        act (str): Non-linear activation to be applied to output variable.

    Returns:
        Variable: The output tensor with same shape as input tensor.

    Examples:
        .. code-block:: python

            x = fluid.layers.data(name='x', shape=[16],
                            dtype='float32', lod_level=1)
            out = fluid.layers.row_conv(input=x, future_context_size=2)
    """
    helper = LayerHelper('row_conv', **locals())
    dtype = helper.input_dtype()
    filter_shape = [future_context_size + 1, input.shape[1]]
    filter_param = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype)
    out = helper.create_tmp_variable(dtype)
    helper.append_op(
        type='row_conv',
        inputs={'X': [input],
                'Filter': [filter_param]},
        outputs={'Out': [out]})
    return helper.append_activation(out)


def multiplex(inputs, index):
    """
    **Multiplex Layer**

    Referring to the given index variable, this layer selects rows from the
    input variables to construct a multiplex variable. Assuming that there are
    :math:`m` input variables and :math:`I_i` represents the i-th input
    variable and :math:`i` is in [0, :math:`m`). All input variables are
    tensors with same shape [:math:`d_0`, :math:`d_1`, ..., :math:`d_R`].
    Please note that rank of the input tensor should be at least 2. Each input
    variable will be treated as a 2-D matrix with shape [:math:`M`, :math:`N`]
    where :math:`M` for :math:`d_0` and :math:`N` for :math:`d_1` * :math:`d_2`
    * ... * :math:`d_R`. Let :math:`I_i[j]` be the j-th row of the i-th input
    variable. The given index variable should be a 2-D tensor with shape
    [:math:`M`, 1]. Let `ID[i]` be the i-th index value of the index variable.
    Then the output variable will be a tensor with shape [:math:`d_0`,
    :math:`d_1`, ..., :math:`d_R`]. If we treat the output tensor as a 2-D
    matrix with shape [:math:`M`, :math:`N`] and let :math:`O[i]` be the i-th
    row of the matrix, then `O[i]` is equal to :math:`I_{ID[i]}[i]`.

    Args:
       inputs (list): A list of variables to gather from. All variables have the
                same shape and the rank is at least 2.
       index (Variable): Tensor<int32>, index variable which is a 2-D tensor
                with shape [M, 1] where M is the batch size.

    Returns:
        Variable: Multiplex variable gathered from input variables.

    Examples:
        .. code-block:: python

            x1 = fluid.layers.data(name='x1', shape=[4], dtype='float32')
            x2 = fluid.layers.data(name='x2', shape=[4], dtype='float32')
            index = fluid.layers.data(name='index', shape=[1], dtype='int32')
            out = fluid.layers.multiplex(inputs=[x1, x2], index=index)
    """
    helper = LayerHelper('multiplex', **locals())

    if not isinstance(inputs, list) and len(inputs) < 2:
        raise ValueError("inputs should be a list object and contains at least "
                         "2 elements.")

    out = helper.create_tmp_variable(inputs[0].dtype)
    helper.append_op(
        type='multiplex',
        inputs={'X': inputs,
                'Ids': index},
        outputs={'Out': [out]})
    return out


def softmax_with_cross_entropy(logits, label, soft_label=False):
    """
    **Softmax With Cross Entropy Operator.**

    Cross entropy loss with softmax is used as the output layer extensively. This
    operator computes the softmax normalized values for each row of the input
    tensor, after which cross-entropy loss is computed. This provides a more
    numerically stable gradient.

    Because this operator performs a softmax on logits internally, it expects
    unscaled logits. This operator should not be used with the output of
    softmax operator since that would produce incorrect results.

    When the attribute soft_label is set false, this operators expects mutually
    exclusive hard labels, each sample in a batch is in exactly one class with a
    probability of 1.0. Each sample in the batch will have a single label.

    The equation is as follows:

    1) Hard label (one-hot label, so every sample has exactly one class)

    .. math::

        loss_j =  -\\text{logit}_{label_j} +
        \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{logit}_i)\\right), j = 1,..., K

    2) Soft label (each sample can have a distribution over all classes)

    .. math::

        loss_j =  -\\sum_{i=0}^{K}\\text{label}_i
        \\left(\\text{logit}_i - \\log\\left(\\sum_{i=0}^{K}
        \\exp(\\text{logit}_i)\\right)\\right), j = 1,...,K

    Args:
        logits (Variable): The unscaled log probabilities, which is a 2-D tensor
            with shape [N x K]. N is the batch_size, and K is the class number.
        label (Variable): The ground truth which is a 2-D tensor. If soft_label
            is set to false, Label is a Tensor<int64> with shape [N x 1]. If
            soft_label is set to true, Label is a Tensor<float/double> with
        soft_label (bool): A flag to indicate whether to interpretate the given
            labels as soft labels. By default, `soft_label` is set to False.
    Returns:
        Variable: The cross entropy loss is a 2-D tensor with shape [N x 1].

    Examples:
        .. code-block:: python

            data = fluid.layers.data(name='data', shape=[128], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            fc = fluid.layers.fc(input=data, size=100)
            out = fluid.layers.softmax_with_cross_entropy(
                logits=fc, label=label)
    """
    helper = LayerHelper('softmax_with_cross_entropy', **locals())
    softmax = helper.create_tmp_variable(dtype=logits.dtype)
    loss = helper.create_tmp_variable(dtype=logits.dtype)
    helper.append_op(
        type='softmax_with_cross_entropy',
        inputs={'Logits': logits,
                'Label': label},
        outputs={'Softmax': softmax,
                 'Loss': loss},
        attrs={'soft_label': soft_label})
    return loss


def smooth_l1(x, y, inside_weight=None, outside_weight=None, sigma=None):
    """
    **Smooth L1 Loss Operator. **

    This operator computes the smooth L1 loss for X and Y.
    The operator takes the first dimension of X and Y as batch size.
    For each instance, it computes the smooth L1 loss element by element first
    and then sums all the losses. So the shape of Out is [batch_size, 1].

    Args:
        x (Variable): A tensor with rank at least 2. The input value of smooth
            L1 loss op with shape [batch_size, dim1, ..., dimN].
        y (Variable): A tensor with rank at least 2. The target value of smooth
            L1 loss op with same shape as x.
        inside_weight (Variable|None):  A tensor with rank at least 2. This
            input is optional and should have same shape with x. If provided,
            the result of (x - y) will be multiplied by this tensor element by
            element.
        outside_weight (Variable|None): A tensor with rank at least 2. This
            input is optional and should have same shape with x. If provided,
            the out smooth L1 loss will be multiplied by this tensor element
            by element.
        sigma (float|None): Hyper parameter of smooth L1 loss op. A float scalar
            with default value 1.0.
    Returns:
        Variable: A tensor with rank be 2. The output smooth L1 loss with
            shape [batch_size, 1].

    Examples:
        .. code-block:: python

            data = fluid.layers.data(name='data', shape=[128], dtype='float32')
            label = fluid.layers.data(
                name='label', shape=[100], dtype='float32')
            fc = fluid.layers.fc(input=data, size=100)
            out = fluid.layers.smooth_l1(x=fc, y=label)
    """
    helper = LayerHelper('smooth_l1_loss', **locals())
    diff = helper.create_tmp_variable(dtype=x.dtype)
    loss = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type='smooth_l1_loss',
        inputs={
            'X': x,
            'Y': y,
            'InsideWeight': inside_weight,
            'OutsideWeight': outside_weight
        },
        outputs={'Diff': diff,
                 'Out': loss},
        attrs={'sigma': sigma})
    return loss


def one_hot(input, depth):
    """
    One Hot Operator. This operator creates the one-hot representations for input
    index values. The following example will help to explain the function of this
    operator.

    Args:
        input(variable):  A Tensor/LodTensor of indices, last dimension must be 1.
        depth(scalar): an interger defining the depth of the one hot dimension.

    Returns:
         The one-hot tensor or LodTensor, same as input.

    Examples:
        .. code-block:: python

        X is a LoDTensor:
          X.lod = [[0, 1, 4]]
          X.shape = [4, 1]
          X.data = [[1], [1], [3], [0]]
        set depth = 4
        Out is a LoDTensor:
          Out.lod = [[0, 1, 4]]
          Out.shape = [4, 4]
          Out.data = [[0., 1., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 0., 1.],
                      [1., 0., 0., 0.]]
    """
    helper = LayerHelper("one_hot", **locals())
    one_hot_out = helper.create_tmp_variable(dtype='float32')
    helper.append_op(
        type="one_hot",
        inputs={'X': input},
        attrs={'depth': depth},
        outputs={'Out': one_hot_out})
    return one_hot_out


def autoincreased_step_counter(counter_name=None, begin=1, step=1):
    """
    NOTE: The counter will be automatically increased by 1 every mini-batch
    Return the run counter of the main program, which is started with 1.

    Args:
        counter_name(str): The counter name, default is '@STEP_COUNTER@'.
        begin(int): The first value of this counter.
        step(int): The increment step between each execution.

    Returns(Variable): The global run counter.
    """
    helper = LayerHelper('global_step_counter')
    if counter_name is None:
        counter_name = '@STEP_COUNTER@'
    counter, is_new_var = helper.create_or_get_global_variable(
        name=counter_name, dtype='int64', shape=[1], persistable=True)
    if is_new_var:
        helper.set_variable_initializer(
            counter, initializer=Constant(
                value=begin - 1, force_cpu=True))
        helper.main_program.global_block().prepend_op(
            type='increment',
            inputs={'X': [counter]},
            outputs={'Out': [counter]},
            attrs={'step': float(step)})
        counter.stop_gradient = True

    return counter


def reshape(x, shape, actual_shape=None, act=None, inplace=True, name=None):
    """
    Gives a new shape to the input Tensor without changing its data.

    The target shape can be given by :attr:`shape` or :attr:`actual_shape`.
    :attr:`shape` is a list of integer while :attr:`actual_shape` is a tensor
    variable. :attr:`actual_shape` has a higher priority than :attr:`shape`
    if it is provided, while :attr:`shape` still should be set correctly to
    gurantee shape inference in compile-time.

    Some tricks exist when specifying the target shape.

    1. -1 means the value of this dimension is inferred from the total element
    number of x and remaining dimensions. Thus one and only one dimension can
    be set -1.

    2. 0 means the actual dimension value is going to be copied from the
    corresponding dimension of x. The indice of 0s in shape can not exceed
    Rank(X).

    Here are some examples to explain it.

    1. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    is [6, 8], the reshape operator will transform x into a 2-D tensor with
    shape [6, 8] and leaving x's data unchanged.

    2. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    specified is [2, 3, -1, 2], the reshape operator will transform x into a
    4-D tensor with shape [2, 3, 4, 2] and leaving x's data unchanged. In this
    case, one dimension of the target shape is set to -1, the value of this
    dimension is inferred from the total element number of x and remaining
    dimensions.

    3. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    is [-1, 0, 3, 2], the reshape operator will transform x into a 4-D tensor
    with shape [2, 4, 3, 2] and leaving x's data unchanged. In this case,
    besides -1, 0 means the actual dimension value is going to be copied from
    the corresponding dimension of x.

    Args:
        input(variable): The input tensor.
        shape(list): The new shape. At most one dimension of the new shape can
                     be -1.
        actual_shape(variable): An optional input. If provided, reshape
                                according to this given shape rather than
                                :attr:`shape` specifying shape. That is to
                                say :attr:`actual_shape` has a higher priority
                                than :attr:`shape`.
        act (str): The non-linear activation to be applied to output variable.
        inplace(bool): If this flag is set true, a new output tensor is created
                       whose data is copied from input x, otherwise the output
                       shares data with input without copying.

    Returns(variable): The output tensor.

    Examples:
        .. code-block:: python

            data = fluid.layers.data(
                name='data', shape=[2, 4, 6], dtype='float32')
            reshaped = fluid.layers.reshape(
                x=data, shape=[-1, 0, 3, 2], act='tanh', inplace=True)
    """

    if not (isinstance(shape, list) or isinstance(shape, tuple)):
        raise ValueError("Input shape must be a python lsit or tuple.")

    # Validate the shape
    unk_dim_idx = -1
    for dim_idx, dim_size in enumerate(shape):
        if dim_size == -1:
            assert unk_dim_idx == -1, (
                "Only one dimension in shape can be unknown.")
            unk_dim_idx = dim_idx
        elif dim_size == 0:
            assert dim_idx < len(x.shape), (
                "The indice of 0s in shape can not exceed Rank(X).")
        else:
            assert dim_size > 0, (
                "Each dimension size given in shape must not be negtive "
                "except one unknown dimension.")

    helper = LayerHelper("reshape", **locals())
    reshaped = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type="reshape",
        inputs={"X": x,
                "Shape": actual_shape}
        if isinstance(actual_shape, Variable) else {"X": x},
        attrs={"shape": shape,
               "inplace": inplace},
        outputs={"Out": reshaped})

    return helper.append_activation(reshaped)


def lod_reset(x, y=None, target_lod=None):
    """
    LoD Reset Operator. Set LoD of **x** to a new one specified by **y** or
    **target_lod**. When **y** provided, **y.lod** would be considered as target
    LoD first, otherwise **y.data** would be considered as target LoD. If **y**
    is not provided, target LoD should be specified by **target_lod**.
    If target LoD is specified by **Y.data** or **target_lod**, only one level
    LoD is supported.

    .. code-block:: text

        * Example 1:

            Given a 1-level LoDTensor x:
                x.lod =  [[ 0,     2,                   5      6 ]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            target_lod: [0, 4, 6]

            then we get a 1-level LoDTensor:
                out.lod =  [[ 0,                   4,            6 ]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

        * Example 2:

            Given a 1-level LoDTensor x:
                x.lod =  [[ 0,     2,                   5      6 ]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            y is a Tensor:
                y.data = [[0, 2, 6]]
                y.dims = [1, 3]

            then we get a 1-level LoDTensor:
                out.lod =  [[ 0,     2,                          6 ]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

        * Example 3:

            Given a 1-level LoDTensor x:
                x.lod =  [[ 0,      2,                   5     6 ]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            y is a 2-level LoDTensor:
                y.lod =  [[0, 2, 4], [0, 2, 5, 6]]
                y.data = [[1.1], [2.1], [3.1], [4.1], [5.1], [6.1]]
                y.dims = [6, 1]

            then we get a 2-level LoDTensor:
                out.lod =  [[0, 2, 4], [0, 2, 5, 6]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

    Args:
        x (Variable): Input variable which could be a Tensor or LodTensor.
        y (Variable|None): If provided, output's LoD would be derived from y.
        target_lod (list|tuple|None): One level LoD which should be considered
                                      as target LoD when y not provided.

    Returns:
        Variable: Output variable with LoD specified by this operator.

    Raises:
        ValueError: If y and target_lod are both None.

    Examples:
        .. code-block:: python

            x = layers.data(name='x', shape=[10])
            y = layers.data(name='y', shape=[10, 20], lod_level=2)
            out = layers.lod_reset(x=x, y=y)
    """
    helper = LayerHelper("lod_reset", **locals())
    out = helper.create_tmp_variable(dtype=x.dtype)
    if y is not None:
        helper.append_op(
            type="lod_reset", inputs={'X': x,
                                      'Y': y}, outputs={'Out': out})
    elif target_lod is not None:
        helper.append_op(
            type="lod_reset",
            inputs={'X': x},
            attrs={'target_lod': target_lod},
            outputs={'Out': out})
    else:
        raise ValueError("y and target_lod should not be both None.")

    return out


def lrn(input, n=5, k=1.0, alpha=1e-4, beta=0.75, name=None):
    """
    Local Response Normalization Layer. This layer performs a type of
    "lateral inhibition" by normalizing over local input regions.

    The formula is as follows:

    .. math::

        Output(i, x, y) = Input(i, x, y) / \left(
        k + \alpha \sum\limits^{\min(C, c + n/2)}_{j = \max(0, c - n/2)}
        (Input(j, x, y))^2 \right)^{\beta}

    In the above equation:

    * :math:`n`: The number of channels to sum over.
    * :math:`k`: The offset (avoid being divided by 0).
    * :math:`alpha`: The scaling parameter.
    * :math:`beta`: The exponent parameter.

    Refer to `ImageNet Classification with Deep Convolutional Neural Networks
    <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

    Args:
        input (Variable): The input tensor of this layer, and the dimension of input tensor must be 4.
        n (int, default 5): The number of channels to sum over.
        k (float, default 1.0): An offset (usually positive to avoid dividing by 0).
        alpha (float, default 1e-4): The scaling parameter.
        beta (float, default 0.75): The exponent.
        name (str, default None): A name for this operation.

    Raises:
        ValueError: If rank of the input tensor is not 4.

    Returns:
        A tensor variable storing the transformation result.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(
              name="data", shape=[3, 112, 112], dtype="float32")
          lrn = fluid.layers.lrn(input=data)
    """
    helper = LayerHelper('lrn', **locals())
    dtype = helper.input_dtype()
    input_shape = input.shape
    dims = len(input_shape)

    if dims != 4:
        raise ValueError(
            "dims of input must be 4(not %d), and it's order must be NCHW" %
            (dims))

    mid_out = helper.create_tmp_variable(dtype=dtype, stop_gradient=True)
    lrn_out = helper.create_tmp_variable(dtype)
    helper.append_op(
        type="lrn",
        inputs={"X": input},
        outputs={
            "Out": lrn_out,
            "MidOut": mid_out,
        },
        attrs={"n": n,
               "k": k,
               "alpha": alpha,
               "beta": beta})

    return lrn_out


def pad(x, paddings, pad_value=0., name=None):
    """
    Pads a tensor with a constant value given by :attr:`pad_value`, and the
    padded width is specified by :attr:`paddings`.

    Specifically, the number of values padded before the contents of :attr:`x`
    in dimension :attr:`i` is indicated by :attr:`paddings[i]`, and the number
    of values padded after the contents of :attr:`x` in dimension :attr:`i` is
    indicated by :attr:`paddings[i+1]`.

    See below for an example.

    .. code-block:: text

        Given:
            x = [[1, 2], [3, 4]]

            paddings = [0, 1, 1, 2]

            pad_value = 0

        Return:

            out = [[0, 1, 2, 0, 0]
                   [0, 3, 4, 0, 0]
                   [0, 0, 0, 0, 0]]

    Args:
        x (Variable): The input tensor variable.
        paddings (list): A list of integers. Its elements specify the padded
                         width before and after for each dimension in turn.
                         The length of :attr:paddings must be
                         :math:`rank(x) \\times 2`.
        pad_value (float): The constant value used to pad.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The padded tensor variable.

    Examples:
        .. code-block:: python

            # x is a rank 2 tensor variable.
            out = fluid.layers.pad(
                x=x, paddings=[0, 1, 1, 2], pad_value=0.)
    """
    helper = LayerHelper('pad', input=x, **locals())
    dtype = helper.input_dtype()
    out = helper.create_tmp_variable(dtype)
    helper.append_op(
        type='pad',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'paddings': paddings,
               'pad_value': float(pad_value)})
    return out


def label_smooth(label,
                 prior_dist=None,
                 epsilon=0.1,
                 dtype="float32",
                 name=None):
    """
    Label smoothing is a mechanism to regularize the classifier layer and is
    called label-smoothing regularization (LSR).

    Label smoothing is proposed to encourage the model to be less confident,
    since optimizing the log-likelihood of the correct label directly may
    cause overfitting and reduce the ability of the model to adapt. Label
    smoothing replaces the ground-truth label :math:`y` with the weighted sum
    of itself and some fixed distribution :math:`\mu`. For class :math:`k`,
    i.e.

    .. math::

        \\tilde{y_k} = (1 - \epsilon) * y_k + \epsilon * \mu_k,

    where :math:`1 - \epsilon` and :math:`\epsilon` are the weights
    respectively, and :math:`\\tilde{y}_k` is the smoothed label. Usually
    uniform distribution is used for :math:`\mu`.

    See more details about label smoothing in https://arxiv.org/abs/1512.00567.

    Args:
        label(Variable): The input variable containing the label data. The
                          label data should use one-hot representation.
        prior_dist(Variable): The prior distribution to be used to smooth
                              labels. If not provided, an uniform distribution
                              is used. The shape of :attr:`prior_dist` should
                              be :math:`(1, class\_num)`.
        epsilon(float): The weight used to mix up the original ground-truth
                        distribution and the fixed distribution.
        dtype(np.dtype|core.VarDesc.VarType|str): The type of data : float32,
                                                  float_64, int etc.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The tensor variable containing the smoothed labels.

    Examples:
        .. code-block:: python

            label = layers.data(name="label", shape=[1], dtype="float32")
            one_hot_label = layers.one_hot(input=label, depth=10)
            smooth_label = layers.label_smooth(
                label=one_hot_label, epsilon=0.1, dtype="float32")
    """
    if epsilon > 1. or epsilon < 0.:
        raise ValueError("The value of epsilon must be between 0 and 1.")
    helper = LayerHelper("label_smooth", **locals())
    label.stop_gradient = True
    smooth_label = helper.create_tmp_variable(dtype)
    helper.append_op(
        type="label_smooth",
        inputs={"X": label,
                "PriorDist": prior_dist} if prior_dist else {"X": label},
        outputs={"Out": smooth_label},
        attrs={"epsilon": float(epsilon)})
    return smooth_label


def roi_pool(input, rois, pooled_height=1, pooled_width=1, spatial_scale=1.0):
    """
    Region of interest pooling (also known as RoI pooling) is to perform
        is to perform max pooling on inputs of nonuniform sizes to obtain
        fixed-size feature maps (e.g. 7*7).
    The operator has three steps:
        1. Dividing each region proposal into equal-sized sections with
           the pooled_width and pooled_height
        2. Finding the largest value in each section
        3. Copying these max values to the output buffer

    Args:
        input (Variable): The input for ROI pooling.
        rois (Variable): ROIs (Regions of Interest) to pool over. It should
                         be a 2-D one level LoTensor of shape [num_rois, 4].
                         The layout is [x1, y1, x2, y2], where (x1, y1)
                         is the top left coordinates, and (x2, y2) is the
                         bottom right coordinates. The num_rois is the
                         total number of ROIs in this batch data.
        pooled_height (integer): The pooled output height. Default: 1
        pooled_width (integer): The pooled output width. Default: 1
        spatial_scale (float): Multiplicative spatial scale factor. To
                               translate ROI coords from their input scale
                               to the scale used when pooling. Default: 1.0

    Returns:
        pool_out (Variable): The output is a 4-D tensor of the shape
                             (num_rois, channels, pooled_h, pooled_w).

    Examples:
        .. code-block:: python

            pool_out = fluid.layers.roi_pool(input=x, rois=rois, 7, 7, 1.0)
    """
    helper = LayerHelper('roi_pool', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_tmp_variable(dtype)
    argmaxes = helper.create_tmp_variable(dtype='int32')
    helper.append_op(
        type="roi_pool",
        inputs={"X": input,
                "ROIs": rois},
        outputs={"Out": pool_out,
                 "Argmax": argmaxes},
        attrs={
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "spatial_scale": spatial_scale
        })
    return pool_out


def dice_loss(input, label, epsilon=0.00001):
    """
    **Dice loss Layer**
    Dice loss for comparing the similarity of two batch of data,
    usually is used for binary image segmentation i.e. labels are binary.
    The dice loss can be defined as below equation:

    .. math::

        dice\_loss &= 1 - \\frac{2 * intersection\_area}{total\_area} \\\\
                  &= \\frac{(total\_area - intersection\_area) - intersection\_area}{total\_area} \\\\
                  &= \\frac{(union\_area - intersection\_area)}{total\_area}


    Args:
        input (Variable): The predictions with rank>=2. The first dimension is batch size,
                          and the last dimension is class number.
        label (Variable): The groud truth with the same rank with input. The first dimension
                          is batch size, and the last dimension is 1.
        epsilon (float): The epsilon will be added to the numerator and denominator.
                         If both input and label are empty, it makes sure dice is 1.
                         Default: 0.00001

    Returns:
        dice_loss (Variable): The dice loss with shape [1].

    Examples:
        .. code-block:: python

            predictions = fluid.layers.softmax(x)
            loss = fluid.layers.dice_loss(input=predictions, label=label, 2)
    """
    label = one_hot(label, depth=input.shape[-1])
    reduce_dim = range(1, len(input.shape))
    inse = reduce_sum(input * label, dim=reduce_dim)
    dice_denominator = reduce_sum(
        input, dim=reduce_dim) + reduce_sum(
            label, dim=reduce_dim)
    dice_score = 1 - inse * 2 / (dice_denominator + epsilon)
    return reduce_mean(dice_score)


def upsampling_bilinear2d(input, out_shape=None, scale=None, name=None):
    """
    The mathematical meaning of upsampling_bilinear2d is also called
    Bilinear interpolation.
    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this layer) on a rectilinear 2D grid.

    For details, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation

    Args:
        input (Variable): The input tensor of bilinear interpolation,
                          This is a 4-D tensor of the shape
                          (num_batches, channels, in_h, in_w).
        out_shape(list|tuple|None): Output shape of bilinear interpolation
                                    layer, the shape is (out_h, out_w).
                                    Default: None
        scale(int|None): The multiplier for the input height or width.
                         At least one of out_shape or scale must be set.
                         And out_shape has a higher priority than scale.
                         Default: None
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        out (Variable): The output is a 4-D tensor of the shape
                        (num_batches, channls, out_h, out_w).

    Examples:
        .. code-block:: python

            out = fluid.layers.bilinear_interp(input, out_shape=[12, 12])
    """
    if out_shape is None and scale is None:
        raise ValueError("One of out_shape and scale must not be None")
    helper = LayerHelper('bilinear_interp', **locals())
    dtype = helper.input_dtype()

    def _is_list_or_turple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    if out_shape is not None:
        if not (_is_list_or_turple_(out_shape) and len(out_shape) == 2):
            raise ValueError('out_shape should be a list or tuple ',
                             'with length 2, (out_h, out_w).')
        out_shape = list(map(int, out_shape))
        out_h = out_shape[0]
        out_w = out_shape[1]
    else:
        out_h = int(input.shape[2] * scale)
        out_w = int(input.shape[3] * scale)

    out = helper.create_tmp_variable(dtype)
    helper.append_op(
        type="bilinear_interp",
        inputs={"X": input},
        outputs={"Out": out},
        attrs={"out_h": out_h,
               "out_w": out_w})
    return out


def random_crop(input, shape, seed=1):
    helper = LayerHelper("random_crop", **locals())
    dtype = helper.input_dtype()
    out = helper.create_tmp_variable(dtype)
    if isinstance(seed, int):
        seed_value = seed
        seed = helper.create_tmp_variable(dtype="int64")
        helper.append_op(
            type="fill_constant",
            inputs={},
            outputs={"Out": seed},
            attrs={
                "dtype": seed.dtype,
                "shape": [1],
                "value": float(seed_value),
                "force_cpu": True
            })
    elif not isinstance(seed, Variable):
        raise ValueError("'seed' must be a Variable or an int.")
    seed_out = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="random_crop",
        inputs={"X": input,
                "Seed": seed},
        outputs={"Out": out,
                 "SeedOut": seed_out},
        attrs={"shape": shape})
    return out
