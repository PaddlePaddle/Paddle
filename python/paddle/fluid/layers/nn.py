# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import warnings
import six
import os
import inspect
from ..layer_helper import LayerHelper
from ..initializer import Normal, Constant, NumpyArrayInitializer
from ..framework import Variable, OpProtoHolder, in_dygraph_mode
from ..dygraph import base
from ..param_attr import ParamAttr
from .layer_function_generator import autodoc, templatedoc, _generate_doc_string_
from .tensor import concat, assign, fill_constant, zeros
from . import utils
from .. import unique_name
from functools import reduce
from .. import core
from ..dygraph import layers
from ..data_feeder import convert_dtype

__all__ = [
    'fc',
    'center_loss',
    'embedding',
    'dynamic_lstm',
    'dynamic_lstmp',
    'dynamic_gru',
    'gru_unit',
    'linear_chain_crf',
    'crf_decoding',
    'cos_sim',
    'cross_entropy',
    'bpr_loss',
    'square_error_cost',
    'chunk_eval',
    'sequence_conv',
    'conv2d',
    'conv3d',
    'sequence_pool',
    'sequence_softmax',
    'softmax',
    'pool2d',
    'pool3d',
    'adaptive_pool2d',
    'adaptive_pool3d',
    'batch_norm',
    'instance_norm',
    'data_norm',
    'beam_search_decode',
    'conv2d_transpose',
    'conv3d_transpose',
    'sequence_expand',
    'sequence_expand_as',
    'sequence_pad',
    'sequence_unpad',
    'lstm_unit',
    'reduce_sum',
    'reduce_mean',
    'reduce_max',
    'reduce_min',
    'reduce_prod',
    'reduce_all',
    'reduce_any',
    'sequence_first_step',
    'sequence_last_step',
    'sequence_slice',
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
    'sampled_softmax_with_cross_entropy',
    'hsigmoid',
    'beam_search',
    'row_conv',
    'multiplex',
    'layer_norm',
    'group_norm',
    'spectral_norm',
    'softmax_with_cross_entropy',
    'smooth_l1',
    'one_hot',
    'autoincreased_step_counter',
    'reshape',
    'squeeze',
    'unsqueeze',
    'lod_reset',
    'lod_append',
    'lrn',
    'pad',
    'pad_constant_like',
    'label_smooth',
    'roi_pool',
    'roi_align',
    'dice_loss',
    'image_resize',
    'image_resize_short',
    'resize_bilinear',
    'resize_trilinear',
    'resize_nearest',
    'gather',
    'gather_nd',
    'scatter',
    'scatter_nd_add',
    'scatter_nd',
    'sequence_scatter',
    'random_crop',
    'mean_iou',
    'relu',
    'selu',
    'log',
    'crop',
    'crop_tensor',
    'rank_loss',
    'margin_rank_loss',
    'elu',
    'relu6',
    'pow',
    'stanh',
    'hard_sigmoid',
    'swish',
    'prelu',
    'brelu',
    'leaky_relu',
    'soft_relu',
    'flatten',
    'sequence_mask',
    'stack',
    'pad2d',
    'unstack',
    'sequence_enumerate',
    'unique',
    'unique_with_counts',
    'expand',
    'sequence_concat',
    'scale',
    'elementwise_add',
    'elementwise_div',
    'elementwise_sub',
    'elementwise_mul',
    'elementwise_max',
    'elementwise_min',
    'elementwise_pow',
    'elementwise_mod',
    'elementwise_floordiv',
    'uniform_random_batch_size_like',
    'gaussian_random',
    'sampling_id',
    'gaussian_random_batch_size_like',
    'sum',
    'slice',
    'strided_slice',
    'shape',
    'rank',
    'size',
    'logical_and',
    'logical_or',
    'logical_xor',
    'logical_not',
    'clip',
    'clip_by_norm',
    'mean',
    'mul',
    'sigmoid_cross_entropy_with_logits',
    'maxout',
    'space_to_depth',
    'affine_grid',
    'sequence_reverse',
    'affine_channel',
    'similarity_focus',
    'hash',
    'grid_sampler',
    'log_loss',
    'add_position_encoding',
    'bilinear_tensor_product',
    'merge_selected_rows',
    'get_tensor_from_selected_rows',
    'lstm',
    'shuffle_channel',
    'temporal_shift',
    'py_func',
    'psroi_pool',
    'prroi_pool',
    'teacher_student_sigmoid_loss',
    'huber_loss',
    'kldiv_loss',
    'npair_loss',
    'pixel_shuffle',
    'fsp_matrix',
    'continuous_value_model',
    'where',
    'sign',
    'deformable_conv',
    'unfold',
    'deformable_roi_pooling',
    'filter_by_instag',
    'shard_index',
    'hard_swish',
    'mse_loss',
]

kIgnoreIndex = -100


def fc(input,
       size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       act=None,
       name=None):
    """
    **Fully Connected Layer**

    This function creates a fully connected layer in the network. It can take
    one or multiple tensors as its inputs(input can be a list of Variable, see
    Args in detail). It creates a variable called weights for each input tensor,
    which represents a fully connected weight matrix from each input unit to
    each output unit. The fully connected layer multiplies each input tensor
    with its corresponding weight to produce an output Tensor with shape [M, `size`],
    where M is batch size. If multiple input tensors are given, the results of
    multiple output tensors with shape [M, `size`] will be summed up. If bias_attr
    is not None, a bias variable will be created and added to the output.
    Finally, if activation is not None, it will be applied to the output as well.

    When the input is single tensor:

    .. math::

        Out = Act({XW + b})

    When the input are multiple tensors:

    .. math::

        Out = Act({\sum_{i=0}^{N-1}X_iW_i + b})

    In the above equation:

    * :math:`N`: Number of the input. N equals to len(input) if input is list of Variable.
    * :math:`X_i`: The i-th input tensor.
    * :math:`W_i`: The i-th weights matrix corresponding i-th input tensor.
    * :math:`b`: The bias parameter created by this layer (if needed).
    * :math:`Act`: The activation function.
    * :math:`Out`: The output tensor.

    See below for an example.

    .. code-block:: text

        Given:
            data_1.data = [[[0.1, 0.2],
                           [0.3, 0.4]]]
            data_1.shape = (1, 2, 2) # 1 is batch_size

            data_2 = [[[0.1, 0.2, 0.3]]]
            data_2.shape = (1, 1, 3)

            out = fluid.layers.fc(input=[data_1, data_2], size=2)

        Then:
            out.data = [[0.18669507, 0.1893476]]
            out.shape = (1, 2)

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
            `X` is a 5-dimensional tensor with a shape [2, 3, 4, 5, 6], and `num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30].
        param_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for learnable
            parameters/weights of this layer.
        bias_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for the bias
            of this layer. If it is set to False, no bias will be added to the output units.
            If it is set to None, the bias is initialized zero. Default: None.
        act (str, default None): Activation to be applied to the output of this layer.
        name (str, default None): The name of this layer.

    Returns:
        Variable: The transformation result.

    Raises:
        ValueError: If rank of the input tensor is less than 2.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          # when input is single tensor
          data = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
          fc = fluid.layers.fc(input=data, size=1000, act="tanh")

          # when input are multiple tensors
          data_1 = fluid.layers.data(name="data_1", shape=[32, 32], dtype="float32")
          data_2 = fluid.layers.data(name="data_2", shape=[24, 36], dtype="float32")
          fc = fluid.layers.fc(input=[data_1, data_2], size=1000, act="tanh")
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
        tmp = helper.create_variable_for_type_inference(dtype)
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
        pre_bias = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="sum",
            inputs={"X": mul_results},
            outputs={"Out": pre_bias},
            attrs={"use_mkldnn": False})
    # add bias
    pre_activation = helper.append_bias_op(pre_bias, dim_start=num_flatten_dims)
    # add activation
    return helper.append_activation(pre_activation)


def center_loss(input,
                label,
                num_classes,
                alpha,
                param_attr,
                update_center=True):
    """
    **Center loss Cost layer**
    
    This layer accepts input (deep features,the output of the last hidden layer)
    and target label and return the center loss cost
    
    For deep features, :math:`X`, and target labels, :math:`Y`, the equation is:
    
    .. math::

        Out = \\frac{1}{2}(X - Y)^2

    Args:
        input (Variable): a 2-D tensor with shape[N x M].
        label (Variable): the groud truth which is a 2-D tensor
                         with shape[N x 1],where N is the batch size.
        num_classes (int): the number of classification categories.
        alpha (float|Variable): learning rate of centers.
        param_attr (ParamAttr): Attribute initializer of centers. 
        update_center (bool): whether to update value of center.

    Returns:
        Variable: 2-D tensor with shape [N * 1] 

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid 

          input = fluid.layers.data(name='x',shape=[20,30],dtype='float32')
          label = fluid.layers.data(name='y',shape=[20,1],dtype='int64')
          num_classes = 1000
          alpha = 0.01
          param_attr = fluid.initializer.Xavier(uniform=False)
          center_loss=fluid.layers.center_loss(input=input,
                 label=label,
                 num_classes=1000,
                 alpha=alpha,
                 param_attr=fluid.initializer.Xavier(uniform=False),
                 update_center=True)
    """
    helper = LayerHelper('center_loss', **locals())
    dtype = helper.input_dtype()
    centers_shape = [num_classes, input.shape[1]]
    centers_param = helper.create_parameter(
        attr=param_attr, shape=centers_shape, dtype=dtype)
    centers_param.stop_gradient = True
    if isinstance(alpha, Variable):
        alpha_param = alpha
    else:
        assert isinstance(alpha, float)
        alpha_param = helper.create_variable(
            name="centerloss_alpha",
            shape=[1],
            dtype="float32",
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=True,
            stop_gradient=True,
            initializer=Constant(alpha))

    centersdiff = helper.create_variable_for_type_inference(dtype=input.dtype)
    loss = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='center_loss',
        inputs={
            'X': [input],
            'Label': [label],
            'Centers': [centers_param],
            'CenterUpdateRate': [alpha_param]
        },
        outputs={
            'SampleCenterDiff': [centersdiff],
            'Loss': [loss],
            'CentersOut': [centers_param]
        },
        attrs={'cluster_num': num_classes,
               'need_update': update_center})
    return loss


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
        input(Variable): Input is a Tensor<int64> Variable, which contains the IDs information.
            The value of the input IDs should satisfy :math:`0<= id < size[0]`.
        size(tuple|list): The shape of the look up table parameter. It should
            have two elements which indicate the size of the dictionary of
            embeddings and the size of each embedding vector respectively.
        is_sparse(bool): The flag indicating whether to use sparse update.
        is_distributed(bool): Whether to run lookup table from remote parameter server.
        padding_idx(int|long|None): It will output all-zero padding data whenever
            lookup encounters :math:`padding\_idx` in Ids. If set :attr:`None`, it makes
            no effect to output. If :math:`padding\_idx < 0`, the :math:`padding\_idx`
            will automatically be converted to :math:`size[0] + padding\_idx` to use.
            Default: None.
        param_attr(ParamAttr): Parameters for this layer.
        dtype(np.dtype|core.VarDesc.VarType|str): The dtype refers to the data type of output
            tensor. It can be float32, float_16, int etc.

    Returns:
        Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='sequence', shape=[1], dtype='int64', lod_level=1)
          emb = fluid.layers.embedding(input=data, size=[128, 64])    
    """

    helper = LayerHelper('embedding', **locals())
    remote_prefetch = is_sparse and (not is_distributed)
    if remote_prefetch:
        assert is_sparse is True and is_distributed is False
    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=dtype, is_bias=False)
    tmp = helper.create_variable_for_type_inference(dtype)
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
            'remote_prefetch': remote_prefetch,
            'padding_idx': padding_idx
        })
    return tmp


def _pull_box_sparse(input, size, dtype='float32'):
    """
    **Pull Box Sparse Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    BoxPS lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    Args:
        input(Variable|list of Variable): Input is a Tensor<int64> Variable, which 
            contains the IDs information.
        size(int): The embedding size parameter, which indicates the size of 
            each embedding vector respectively.
        dtype(str): The dtype refers to the data type of output tensor. Only supports 
	    float32 now.

    Returns:
        Variable|list of Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='sequence', shape=[1], dtype='int64', lod_level=1)
          emb = fluid.layers.pull_box_sparse(input=data, size=[11])    
    """
    helper = LayerHelper('pull_box_sparse', **locals())
    if dtype != 'float32':
        raise ValueError(
            "BoxPS only support float type embedding now, and your type is: " +
            dtype)
    helper.input_dtype()
    inputs = helper.multiple_input()
    outs = [
        helper.create_variable_for_type_inference(dtype)
        for i in range(len(inputs))
    ]
    helper.append_op(
        type='pull_box_sparse',
        inputs={'Ids': inputs},
        outputs={'Out': outs},
        attrs={'size': size})
    if len(outs) == 1:
        return outs[0]
    return outs


@templatedoc(op_type="lstm")
def dynamic_lstm(input,
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
                 name=None):
    """
    ${comment}

    Args:
        input (Variable): ${input_comment}
        size (int): 4 * hidden size.
        h_0(Variable): The initial hidden state is an optional input, default is zero.
                       This is a tensor with shape (N x D), where N is the
                       batch size and D is the hidden size.
        c_0(Variable): The initial cell state is an optional input, default is zero.
                       This is a tensor with shape (N x D), where N is the
                       batch size. `h_0` and `c_0` can be NULL but only at the same time.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
                               hidden-hidden weights.

                               - Weights = {:math:`W_{ch}, W_{ih}, \
                                                W_{fh}, W_{oh}`}
                               - The shape is (D x 4D), where D is the hidden
                                 size.

                               If it is set to None or one attribute of ParamAttr,
                               dynamic_lstm will create ParamAttr as param_attr.
                               If the Initializer of the param_attr is not set, the
                               parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The bias attribute for the learnable bias
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

                              If it is set to None or one attribute of ParamAttr,
                              dynamic_lstm will create ParamAttr as bias_attr.
                              If the Initializer of the bias_attr is not set,
                              the bias is initialized zero. Default: None.
        use_peepholes (bool): ${use_peepholes_comment}
        is_reverse (bool): ${is_reverse_comment}
        gate_activation (str): ${gate_activation_comment}
        cell_activation (str): ${cell_activation_comment}
        candidate_activation (str): ${candidate_activation_comment}
        dtype (str): Data type. Choices = ["float32", "float64"], default "float32".
        name (str|None): A name for this layer(optional). If set None, the layer
                         will be named automatically.

    Returns:
        tuple: The hidden state, and cell state of LSTM. The shape of both \
        is (T x D), and lod is the same with the `input`.

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid
            emb_dim = 256
            vocab_size = 10000
            hidden_dim = 512
            
            data = fluid.layers.data(name='x', shape=[1],
                         dtype='int32', lod_level=1)
            emb = fluid.layers.embedding(input=data, size=[vocab_size, emb_dim], is_sparse=True)

            forward_proj = fluid.layers.fc(input=emb, size=hidden_dim * 4,
                                           bias_attr=False)

            forward, _ = fluid.layers.dynamic_lstm(
                input=forward_proj, size=hidden_dim * 4, use_peepholes=False)
    """
    assert in_dygraph_mode(
    ) is not True, "please use lstm instead of dynamic_lstm in dygraph mode!"
    assert bias_attr is not False, "bias_attr should not be False in dynamic_lstmp."
    helper = LayerHelper('lstm', **locals())
    size = size // 4
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 4 * size], dtype=dtype)
    bias_size = [1, 7 * size]
    if not use_peepholes:
        bias_size[1] = 4 * size
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)

    hidden = helper.create_variable_for_type_inference(dtype)
    cell = helper.create_variable_for_type_inference(dtype)
    batch_gate = helper.create_variable_for_type_inference(dtype)
    batch_cell_pre_act = helper.create_variable_for_type_inference(dtype)
    inputs = {'Input': input, 'Weight': weight, 'Bias': bias}
    batch_size = input.shape[0]
    if h_0:
        assert h_0.shape == (batch_size, size), \
            'The shape of h0 should be (batch_size, %d)' % size
        inputs['H0'] = h_0
    if c_0:
        assert c_0.shape == (batch_size, size), \
            'The shape of c0 should be (batch_size, %d)' % size
        inputs['C0'] = c_0

    helper.append_op(
        type='lstm',
        inputs=inputs,
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


def lstm(input,
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
         seed=-1):
    """
    If Device is GPU, This op will use cudnn LSTM implementation

    A four-gate Long Short-Term Memory network with no peephole connections.
    In the forward pass the output ht and cell output ct for a given iteration can be computed from the recurrent input ht-1,
    the cell input ct-1 and the previous layer input xt given matrices W, R and biases bW, bR from the following equations:

    .. math::

       i_t &= \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + bx_i + bh_i)

       f_t &= \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + bx_f + bh_f)

       o_t &= \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + bx_o + bh_o)

       \\tilde{c_t} &= tanh(W_{cx}x_t + W_{ch}h_{t-1} + bx_c + bh_c)

       c_t &= f_t \odot c_{t-1} + i_t \odot \\tilde{c_t}

       h_t &= o_t \odot tanh(c_t)

    - $W$ terms denote weight matrices (e.g. $W_{ix}$ is the matrix
      of weights from the input gate to the input)
    - The b terms denote bias vectors ($bx_i$ and $bh_i$ are the input gate bias vector).
    - sigmoid is the logistic sigmoid function.
    - $i, f, o$ and $c$ are the input gate, forget gate, output gate,
      and cell activation vectors, respectively, all of which have the same size as
      the cell output activation vector $h$.
    - The :math:`\odot` is the element-wise product of the vectors.
    - :math:`tanh` is the activation functions.
    - :math:`\\tilde{c_t}` is also called candidate hidden state,
      which is computed based on the current input and the previous hidden state.

    Where sigmoid is the sigmoid operator: :math:`sigmoid(x) = 1 / (1 + e^{-x})` , * represents a point-wise multiplication,
    X represensts a matrix multiplication


    Args:
        input (Variable): LSTM input tensor, shape MUST be ( seq_len x batch_size x input_size )
        init_h(Variable): The initial hidden state of the LSTM
                       This is a tensor with shape ( num_layers x batch_size x hidden_size)
                       if is_bidirec = True, shape should be ( num_layers*2 x batch_size x hidden_size)
        init_c(Variable): The initial cell state of the LSTM.
                       This is a tensor with shape ( num_layers x batch_size x hidden_size )
                       if is_bidirec = True, shape should be ( num_layers*2 x batch_size x hidden_size)
        max_len (int): max length of LSTM. the first dim of input tensor CAN NOT greater than max_len
        hidden_size (int): hidden size of the LSTM
        num_layers (int): total layers number of the LSTM
        dropout_prob(float|0.0): dropout prob, dropout ONLY work between rnn layers, NOT between time steps
                             There is NO dropout work on rnn output of the last RNN layers
        is_bidirec (bool): If it is bidirectional
        is_test (bool): If it is in test phrase
        name (str|None): A name for this layer(optional). If set None, the layer
                         will be named automatically.
        default_initializer(Initialize|None): Where use initializer to initialize the Weight
                         If set None, defaule initializer will be used
        seed(int): Seed for dropout in LSTM, If it's -1, dropout will use random seed


    Returns:
        rnn_out(Tensor),last_h(Tensor),last_c(Tensor):

                        Three tensors, rnn_out, last_h, last_c:

                        - rnn_out is result of LSTM hidden, shape is (seq_len x batch_size x hidden_size) \
                          if is_bidirec set to True, shape will be ( seq_len x batch_sze x hidden_size*2)
                        - last_h is the hidden state of the last step of LSTM \
                          shape is ( num_layers x batch_size x hidden_size ) \
                          if is_bidirec set to True, shape will be ( num_layers*2 x batch_size x hidden_size)
                        - last_c(Tensor): the cell state of the last step of LSTM \
                          shape is ( num_layers x batch_size x hidden_size ) \
                          if is_bidirec set to True, shape will be ( num_layers*2 x batch_size x hidden_size)


    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            emb_dim = 256
            vocab_size = 10000
            data = fluid.layers.data(name='x', shape=[-1, 100, 1],
                         dtype='int32')
            emb = fluid.layers.embedding(input=data, size=[vocab_size, emb_dim], is_sparse=True)
            batch_size = 20
            max_len = 100
            dropout_prob = 0.2
            input_size = 100
            hidden_size = 150
            num_layers = 1
            init_h = layers.fill_constant( [num_layers, batch_size, hidden_size], 'float32', 0.0 )
            init_c = layers.fill_constant( [num_layers, batch_size, hidden_size], 'float32', 0.0 )
            rnn_out, last_h, last_c = layers.lstm( emb, init_h, init_c, \
                    max_len, hidden_size, num_layers, \
                    dropout_prob=dropout_prob)
    """

    helper = LayerHelper('cudnn_lstm', **locals())

    dtype = input.dtype
    input_shape = list(input.shape)
    input_size = input_shape[-1]
    weight_size = 0
    for i in range(num_layers):
        if i == 0:
            input_weight_size = (input_size * hidden_size) * 4
        else:
            if is_bidirec:
                input_weight_size = (hidden_size * 2 * hidden_size) * 4
            else:
                input_weight_size = (hidden_size * hidden_size) * 4

        hidden_weight_size = (hidden_size * hidden_size) * 4

        if is_bidirec:
            weight_size += (input_weight_size + hidden_weight_size) * 2
            weight_size += hidden_size * 8 * 2
        else:
            weight_size += input_weight_size + hidden_weight_size
            weight_size += hidden_size * 8

    weight = helper.create_parameter(
        attr=helper.param_attr,
        shape=[weight_size],
        dtype=dtype,
        default_initializer=default_initializer)

    out = helper.create_variable_for_type_inference(dtype)
    last_h = helper.create_variable_for_type_inference(dtype)
    last_c = helper.create_variable_for_type_inference(dtype)

    cache = helper.create_variable(
        persistable=True, type=core.VarDesc.VarType.RAW, stop_gradient=True)

    helper.append_op(
        type='cudnn_lstm',
        inputs={
            'Input': input,
            'InitH': init_h,
            'InitC': init_c,
            'W': weight,
            'Cache': cache,
        },
        outputs={
            'Out': out,
            'last_h': last_h,
            'last_c': last_c,
        },
        attrs={
            'max_len': max_len,
            'is_bidirec': is_bidirec,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'is_test': is_test,
            'dropout_prob': dropout_prob,
            'seed': seed,
        })
    return out, last_h, last_c


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
                  name=None,
                  h_0=None,
                  c_0=None,
                  cell_clip=None,
                  proj_clip=None):
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
          we use vectors to represent these diagonal weight matrices.
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

                               If it is set to None or one attribute of ParamAttr,
                               dynamic_lstm will create ParamAttr as param_attr.
                               If the Initializer of the param_attr is not set, the
                               parameter is initialized with Xavier. Default: None.
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

                              If it is set to None or one attribute of ParamAttr,
                              dynamic_lstm will create ParamAttr as bias_attr.
                              If the Initializer of the bias_attr is not set,
                              the bias is initialized zero. Default: None.
        use_peepholes(bool): Whether to enable diagonal/peephole connections,
                             default `True`.
        is_reverse(bool): Whether to compute reversed LSTM, default `False`.
        gate_activation(str): The activation for input gate, forget gate and
                              output gate. Choices = ["sigmoid", "tanh", "relu",
                              "identity"], default "sigmoid".
        cell_activation(str): The activation for cell output. Choices = ["sigmoid",
                              "tanh", "relu", "identity"], default "tanh".
        candidate_activation(str): The activation for candidate hidden state.
                              Choices = ["sigmoid", "tanh", "relu", "identity"],
                              default "tanh".
        proj_activation(str): The activation for projection output.
                              Choices = ["sigmoid", "tanh", "relu", "identity"],
                              default "tanh".
        dtype(str): Data type. Choices = ["float32", "float64"], default "float32".
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.
        h_0(Variable): The initial hidden state is an optional input, default is zero.
                       This is a tensor with shape (N x D), where N is the
                       batch size and D is the projection size.
        c_0(Variable): The initial cell state is an optional input, default is zero.
                       This is a tensor with shape (N x D), where N is the
                       batch size. `h_0` and `c_0` can be NULL but only at the same time.
        cell_clip(float): If provided the cell state is clipped
                             by this value prior to the cell output activation.
        proj_clip(float): If `num_proj > 0` and `proj_clip` is
                            provided, then the projected values are clipped elementwise to within
                            `[-proj_clip, proj_clip]`.

    Returns:
        tuple: A tuple of two output variable: the projection of hidden state, \
               and cell state of LSTMP. The shape of projection is (T x P), \
               for the cell state which is (T x D), and both LoD is the same \
               with the `input`.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            dict_dim, emb_dim = 128, 64
            data = fluid.layers.data(name='sequence', shape=[1],
                                     dtype='int32', lod_level=1)
            emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim, proj_dim = 512, 256
            fc_out = fluid.layers.fc(input=emb, size=hidden_dim * 4,
                                     act=None, bias_attr=None)
            proj_out, _ = fluid.layers.dynamic_lstmp(input=fc_out,
                                                     size=hidden_dim * 4,
                                                     proj_size=proj_dim,
                                                     use_peepholes=False,
                                                     is_reverse=True,
                                                     cell_activation="tanh",
                                                     proj_activation="tanh")
    """

    assert in_dygraph_mode(
    ) is not True, "please use lstm instead of dynamic_lstmp in dygraph mode!"

    assert bias_attr is not False, "bias_attr should not be False in dynamic_lstmp."
    helper = LayerHelper('lstmp', **locals())
    size = size // 4
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[proj_size, 4 * size], dtype=dtype)
    proj_weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, proj_size], dtype=dtype)
    bias_size = [1, 7 * size]
    if not use_peepholes:
        bias_size[1] = 4 * size
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)

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
        'Bias': bias
    }
    batch_size = input.shape[0]
    if h_0:
        assert h_0.shape == (batch_size, proj_size), \
            'The shape of h0 should be (batch_size, %d)' % proj_size
        inputs['H0'] = h_0
    if c_0:
        assert c_0.shape == (batch_size, size), \
            'The shape of c0 should be (batch_size, %d)' % size
        inputs['C0'] = c_0

    if cell_clip:
        assert cell_clip >= 0, "cell_clip should not be negtive."
    if proj_clip:
        assert proj_clip >= 0, "proj_clip should not be negtive."

    helper.append_op(
        type='lstmp',
        inputs=inputs,
        outputs={
            'Projection': projection,
            'Cell': cell,
            'BatchHidden': batch_hidden,
            'BatchGate': batch_gate,
            'BatchCellPreAct': batch_cell_pre_act
        },
        attrs={
            'use_peepholes': use_peepholes,
            'cell_clip': cell_clip,
            'proj_clip': proj_clip,
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
                h_0=None,
                origin_mode=False):
    """
    **Gated Recurrent Unit (GRU) Layer**

    if origin_mode is False, then the equation of a gru step is from paper
    `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
    Modeling <https://arxiv.org/pdf/1412.3555.pdf>`_ .

    The formula is as follows:

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = (1-u_t) \odot h_{t-1} + u_t \odot \\tilde{h_t}


    if origin_mode is True then the equation is from paper
    Learning Phrase Representations using RNN Encoder-Decoder for Statistical
    Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \\tilde{h_t}

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

            If it is set to None or one attribute of ParamAttr, dynamic_gru will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias
            of GRU.Note that the bias with :math:`(1 \\times 3D)` concatenates
            the bias in the update gate, reset gate and candidate calculations.
            If it is set to False, no bias will be applied to the update gate,
            reset gate and candidate calculations. If it is set to None or one
            attribute of ParamAttr, dynamic_gru will create ParamAttr as
            bias_attr. If the Initializer of the bias_attr is not set, the bias
            is initialized zero. Default: None.
        is_reverse(bool): Whether to compute reversed GRU, default
            :attr:`False`.
        gate_activation(str): The activation for update gate and reset gate.
            Choices = ["sigmoid", "tanh", "relu", "identity"], default "sigmoid".
        candidate_activation(str): The activation for candidate hidden state.
            Choices = ["sigmoid", "tanh", "relu", "identity"], default "tanh".
        h_0 (Variable): This is initial hidden state. If not set, default is
            zero. This is a tensor with shape (N x D), where N is the number of
            total time steps of input mini-batch feature and D is the hidden
            size.

    Returns:
        Variable: The hidden state of GRU. The shape is :math:`(T \\times D)`, \
            and sequence length is the same with the input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            dict_dim, emb_dim = 128, 64
            data = fluid.layers.data(name='sequence', shape=[1],
                                     dtype='int32', lod_level=1)
            emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim = 512
            x = fluid.layers.fc(input=emb, size=hidden_dim * 3)
            hidden = fluid.layers.dynamic_gru(input=x, size=hidden_dim)
    """

    assert in_dygraph_mode(
    ) is not True, "please use gru instead of dynamic_gru in dygraph mode!"

    helper = LayerHelper('gru', **locals())
    dtype = helper.input_dtype()

    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 3 * size], dtype=dtype)
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=[1, 3 * size], dtype=dtype, is_bias=True)
    batch_size = input.shape[0]
    inputs = {'Input': input, 'Weight': weight, 'Bias': bias}
    if h_0:
        assert h_0.shape == (
            batch_size, size
        ), 'The shape of h0 should be(batch_size, %d)' % size
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
            'BatchHidden': batch_hidden
        },
        attrs={
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'activation': candidate_activation,
            'origin_mode': origin_mode
        })
    return hidden


def gru_unit(input,
             hidden,
             size,
             param_attr=None,
             bias_attr=None,
             activation='tanh',
             gate_activation='sigmoid',
             origin_mode=False):
    """
    **GRU unit layer**

    if origin_mode is True, then the equation of a gru step is from paper
    `Learning Phrase Representations using RNN Encoder-Decoder for Statistical
    Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_

        .. math::
            u_t & = actGate(xu_{t} + W_u h_{t-1} + b_u)

            r_t & = actGate(xr_{t} + W_r h_{t-1} + b_r)

            m_t & = actNode(xm_t + W_c dot(r_t, h_{t-1}) + b_m)

            h_t & = dot(u_t, h_{t-1}) + dot((1-u_t), m_t)

    if origin_mode is False, then the equation of a gru step is from paper
    `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
    Modeling <https://arxiv.org/pdf/1412.3555.pdf>`_

        .. math::
            u_t & = actGate(xu_{t} + W_u h_{t-1} + b_u)

            r_t & = actGate(xr_{t} + W_r h_{t-1} + b_r)

            m_t & = actNode(xm_t + W_c dot(r_t, h_{t-1}) + b_m)

            h_t & = dot((1-u_t), h_{t-1}) + dot(u_t, m_t)


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
        hidden (Variable): The hidden value of gru unit from previous step.
        size (integer): The input dimension value.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            hidden-hidden weight matrix. Note:

            - The shape of the weight matrix is :math:`(T \\times 3D)`, where
              :math:`D` is the hidden size.
            - All elements in the weight matrix can be divided into two parts.
              The first part are weights of the update gate and reset gate with
              shape :math:`(D \\times 2D)`, and the second part are weights for
              candidate hidden state with shape :math:`(D \\times D)`.

            If it is set to None or one attribute of ParamAttr, gru_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias
            of GRU.Note that the bias with :math:`(1 \\times 3D)` concatenates
            the bias in the update gate, reset gate and candidate calculations.
            If it is set to False, no bias will be applied to the update gate,
            reset gate and candidate calculations. If it is set to None or one
            attribute of ParamAttr, gru_unit will create ParamAttr as
            bias_attr. If the Initializer of the bias_attr is not set, the bias
            is initialized zero. Default: None.
        activation (string): The activation type for cell (actNode).
                             Default: 'tanh'
        gate_activation (string): The activation type for gates (actGate).
                                  Default: 'sigmoid'

    Returns:
        tuple: The hidden value, reset-hidden value and gate values.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            dict_dim, emb_dim = 128, 64
            data = fluid.layers.data(name='step_data', shape=[1], dtype='int32')
            emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim = 512
            x = fluid.layers.fc(input=emb, size=hidden_dim * 3)
            pre_hidden = fluid.layers.data(
                name='pre_hidden', shape=[hidden_dim], dtype='float32')
            hidden = fluid.layers.gru_unit(
                input=x, hidden=pre_hidden, size=hidden_dim * 3)

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
    size = size // 3

    # create weight
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 3 * size], dtype=dtype)

    gate = helper.create_variable_for_type_inference(dtype)
    reset_hidden_pre = helper.create_variable_for_type_inference(dtype)
    updated_hidden = helper.create_variable_for_type_inference(dtype)
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


@templatedoc()
def linear_chain_crf(input, label, param_attr=None, length=None):
    """
    Linear Chain CRF.

    ${comment}

    Args:
        input(${emission_type}): ${emission_comment}
        label(${label_type}): ${label_comment}
        Length(${length_type}): ${length_comment}
        param_attr(ParamAttr): The attribute of the learnable parameter for transition parameter.

    Returns:
        output(${emission_exps_type}): ${emission_exps_comment} \n
        output(${transition_exps_type}): ${transition_exps_comment} \n
        output(${log_likelihood_type}): ${log_likelihood_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            #define net structure, using LodTensor
            train_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(train_program, startup_program):
                input_data = fluid.layers.data(name='input_data', shape=[10], dtype='float32', lod_level=1)
                label = fluid.layers.data(name='label', shape=[1], dtype='int', lod_level=1)
                emission= fluid.layers.fc(input=input_data, size=10, act="tanh")
                crf_cost = fluid.layers.linear_chain_crf(
                    input=emission,
                    label=label,
                    param_attr=fluid.ParamAttr(
                    name='crfw',
                    learning_rate=0.01)) 
            use_cuda = False
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)    
            #define data, using LoDTensor
            a = fluid.create_lod_tensor(np.random.rand(12,10).astype('float32'), [[3,3,4,2]], place)
            b = fluid.create_lod_tensor(np.array([[1],[1],[2],[3],[1],[1],[1],[3],[1],[1],[1],[1]]),[[3,3,4,2]] , place)
            feed1 = {'input_data':a,'label':b}
            loss= exe.run(train_program,feed=feed1, fetch_list=[crf_cost])
            print(loss) 

            #define net structure, using padding
            train_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(train_program, startup_program):
                input_data2 = fluid.layers.data(name='input_data2', shape=[10,10], dtype='float32')
                label2 = fluid.layers.data(name='label2', shape=[10,1], dtype='int')
                label_length = fluid.layers.data(name='length', shape=[1], dtype='int')
                emission2= fluid.layers.fc(input=input_data2, size=10, act="tanh", num_flatten_dims=2)
                crf_cost2 = fluid.layers.linear_chain_crf(
                    input=emission2,
                    label=label2,
                    length=label_length,
                    param_attr=fluid.ParamAttr(
                     name='crfw',
                     learning_rate=0.01))

            use_cuda = False
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)

            #define data, using padding
            cc=np.random.rand(4,10,10).astype('float32')
            dd=np.random.rand(4,10,1).astype('int64')
            ll=np.array([[3,3,4,2]])
            feed2 = {'input_data2':cc,'label2':dd,'length':ll}

            loss2= exe.run(train_program,feed=feed2, fetch_list=[crf_cost2])
            print(loss2) 
            
            #you can use find_var to get transition parameter.
            transition=np.array(fluid.global_scope().find_var('crfw').get_tensor())
            print(transition)
    """
    helper = LayerHelper('linear_chain_crf', **locals())
    size = input.shape[1]
    transition = helper.create_parameter(
        attr=helper.param_attr,
        shape=[size + 2, size],
        dtype=helper.input_dtype())
    alpha = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    emission_exps = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    transition_exps = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    log_likelihood = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    this_inputs = {
        "Emission": [input],
        "Transition": transition,
        "Label": [label]
    }
    if length:
        this_inputs['length'] = [length]
    helper.append_op(
        type='linear_chain_crf',
        inputs=this_inputs,
        outputs={
            "Alpha": [alpha],
            "EmissionExps": [emission_exps],
            "TransitionExps": transition_exps,
            "LogLikelihood": log_likelihood
        })

    return log_likelihood


@templatedoc()
def crf_decoding(input, param_attr, label=None):
    """
    ${comment}

    Args:
        input(${emission_type}): ${emission_comment}

        param_attr(ParamAttr): The parameter attribute for training.

        label(${label_type}): ${label_comment}

    Returns:
        Variable: ${viterbi_path_comment}

    Examples:
        .. code-block:: python

           import paddle.fluid as fluid
           images = fluid.layers.data(name='pixel', shape=[784], dtype='float32')
           label = fluid.layers.data(name='label', shape=[1], dtype='int32')
           hidden = fluid.layers.fc(input=images, size=2)
           crf = fluid.layers.linear_chain_crf(input=hidden, label=label, 
                     param_attr=fluid.ParamAttr(name="crfw"))
           crf_decode = fluid.layers.crf_decoding(input=hidden, 
                     param_attr=fluid.ParamAttr(name="crfw"))
    """
    helper = LayerHelper('crf_decoding', **locals())
    transition = helper.get_parameter(param_attr.name)
    viterbi_path = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    helper.append_op(
        type='crf_decoding',
        inputs={"Emission": [input],
                "Transition": transition,
                "Label": label},
        outputs={"ViterbiPath": [viterbi_path]})

    return viterbi_path


@templatedoc()
def cos_sim(X, Y):
    """
    ${comment}

    Args:
        X (Variable): ${x_comment}.
        Y (Variable): ${y_comment}.

    Returns:
        Variable: the output of cosine(X, Y).

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[3, 7], dtype='float32', append_batch_size=False)
            y = fluid.layers.data(name='y', shape=[1, 7], dtype='float32', append_batch_size=False)
            out = fluid.layers.cos_sim(x, y)
    """
    helper = LayerHelper('cos_sim', **locals())
    out = helper.create_variable_for_type_inference(dtype=X.dtype)
    xnorm = helper.create_variable_for_type_inference(dtype=X.dtype)
    ynorm = helper.create_variable_for_type_inference(dtype=X.dtype)
    helper.append_op(
        type='cos_sim',
        inputs={'X': [X],
                'Y': [Y]},
        outputs={'Out': [out],
                 'XNorm': [xnorm],
                 'YNorm': [ynorm]})
    return out


def dropout(x,
            dropout_prob,
            is_test=False,
            seed=None,
            name=None,
            dropout_implementation="downgrade_in_infer"):
    """
    Computes dropout.

    Drop or keep each element of `x` independently. Dropout is a regularization
    technique for reducing overfitting by preventing neuron co-adaption during
    training. The dropout operator randomly sets (according to the given dropout
    probability) the outputs of some units to zero, while others are remain
    unchanged.

    dropout op can be removed from the program to make the program more efficient.

    Args:
        x (Variable): The input tensor variable.
        dropout_prob (float): Probability of setting units to zero.
        is_test (bool): A flag indicating whether it is in test phrase or not.
        seed (int): A Python integer used to create random seeds. If this
                    parameter is set to None, a random seed is used.
                    NOTE: If an integer seed is given, always the same output
                    units will be dropped. DO NOT use a fixed seed in training.
        name (str|None): A name for this layer(optional). If set None, the layer
                         will be named automatically.
        dropout_implementation(string): ['downgrade_in_infer'(default)|'upscale_in_train']

                                        1. downgrade_in_infer(default), downgrade the outcome at inference

                                           - train: out = input * mask
                                           - inference: out = input * (1.0 - dropout_prob)

                                           (mask is a tensor same shape with input, value is 0 or 1
                                           ratio of 0 is dropout_prob)
                                        2. upscale_in_train, upscale the outcome at training time

                                           - train: out = input * mask / ( 1.0 - dropout_prob )
                                           - inference: out = input

                                           (mask is a tensor same shape with input, value is 0 or 1
                                           ratio of 0 is dropout_prob)


    Returns:
        Variable: A tensor variable is the shape with `x`.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            droped = fluid.layers.dropout(x, dropout_prob=0.5)
    """

    helper = LayerHelper('dropout', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    mask = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.UINT8, stop_gradient=True)

    if (seed is None or seed == 0) and helper.main_program.random_seed != 0:
        seed = helper.main_program.random_seed

    helper.append_op(
        type='dropout',
        inputs={'X': [x]},
        outputs={'Out': [out],
                 'Mask': [mask]},
        attrs={
            'dropout_prob': dropout_prob,
            'is_test': is_test,
            'fix_seed': seed is not None,
            'seed': seed if seed is not None else 0,
            'dropout_implementation': dropout_implementation,
        })
    return out


def cross_entropy(input, label, soft_label=False, ignore_index=kIgnoreIndex):
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
                                           labels. Default: `False`.
        ignore_index (int): Specifies a target value that is ignored and does
                            not contribute to the input gradient. Only valid
                            if soft_label is set to False. Default: kIgnoreIndex

    Returns:
         A 2-D tensor with shape [N x 1], the cross entropy loss.

    Raises:
         ValueError:

                      1. the 1st dimension of ``input`` and ``label`` are not equal.

                      2. when ``soft_label == True``, and the 2nd dimension of
                         ``input`` and ``label`` are not equal.

                      3. when ``soft_label == False``, and the 2nd dimension of
                         ``label`` is not 1.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          classdim = 7
          x = fluid.layers.data(name='x', shape=[3, 7], dtype='float32', append_batch_size=False)
          label = fluid.layers.data(name='label', shape=[3, 1], dtype='float32', append_batch_size=False)
          predict = fluid.layers.fc(input=x, size=classdim, act='softmax')
          cost = fluid.layers.cross_entropy(input=predict, label=label)
    """
    if not soft_label:
        return cross_entropy2(input, label, ignore_index)
    helper = LayerHelper('cross_entropy', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='cross_entropy',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out]},
        attrs={"soft_label": soft_label,
               "ignore_index": ignore_index})
    return out


def cross_entropy2(input, label, ignore_index=kIgnoreIndex):
    helper = LayerHelper('cross_entropy2', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    xshape = helper.create_variable_for_type_inference(dtype=input.dtype)
    match_x = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='cross_entropy2',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out],
                 'MatchX': [match_x],
                 'XShape': [xshape]},
        attrs={'ignore_index': ignore_index})
    return out


def bpr_loss(input, label, name=None):
    """
    **Bayesian Personalized Ranking Loss Operator**

    This operator belongs to pairwise ranking loss. Label is the desired item.
    The loss at a given point in one session is defined as:

    .. math::
        Y[i] = 1/(N[i] - 1) * \sum_j{\log(\sigma(X[i, Label[i]]-X[i, j]))}

    Learn more details by reading paper <session-based recommendations with recurrent
    neural networks>.

    Args:
        input (Variable|list):  a 2-D tensor with shape [N x D], where N is the
                                batch size and D is the number of classes.
                                This input is not probability but logits.
        label (Variable|list):  the ground truth which is a 2-D tensor.  `label`
                                is a tensor<int64> with shape [N x 1].
        name (str|None):        A name for this layer(optional). If set None, the
                                layer will be named automatically. Default: None.
    Returns:
        A 2-D tensor with shape [N x 1], the bpr loss.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid

          neg_size = 10
          label = fluid.layers.data(
                    name="label", shape=[1], dtype="int64")
          predict = fluid.layers.data(
                    name="predict", shape=[neg_size + 1], dtype="float32")
          cost = fluid.layers.bpr_loss(input=predict, label=label)
    """
    helper = LayerHelper('bpr_loss', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='bpr_loss',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out]})
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
        input (Variable): Input tensor, has predictions.
        label (Variable): Label tensor, has target labels.

    Returns:
        Variable: The tensor variable storing the element-wise squared error \
                  difference of input and label.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          y = fluid.layers.data(name='y', shape=[1], dtype='float32')
          y_predict = fluid.layers.data(name='y_predict', shape=[1], dtype='float32')
          cost = fluid.layers.square_error_cost(input=y_predict, label=y)

    """
    helper = LayerHelper('square_error_cost', **locals())
    minus_out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='elementwise_sub',
        inputs={'X': [input],
                'Y': [label]},
        outputs={'Out': [minus_out]})

    square_out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='square', inputs={'X': [minus_out]},
        outputs={'Out': [square_out]})
    return square_out


@templatedoc()
def chunk_eval(input,
               label,
               chunk_scheme,
               num_chunk_types,
               excluded_chunk_types=None,
               seq_length=None):
    """
    **Chunk Evaluator**

    This function computes and outputs the precision, recall and
    F1-score of chunk detection.

    For some basics of chunking, please refer to
    `Chunking with Support Vector Machines <https://aclanthology.info/pdf/N/N01/N01-1025.pdf>`_ .

    ChunkEvalOp computes the precision, recall, and F1-score of chunk detection,
    and supports IOB, IOE, IOBES and IO (also known as plain) tagging schemes.
    Here is a NER example of labeling for these tagging schemes:

    .. code-block:: python

       ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========
              Li     Ming    works  at  Agricultural   Bank   of    China  in  Beijing.
       ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========
       IO     I-PER  I-PER   O      O   I-ORG          I-ORG  I-ORG I-ORG  O   I-LOC
       IOB    B-PER  I-PER   O      O   B-ORG          I-ORG  I-ORG I-ORG  O   B-LOC
       IOE    I-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   E-LOC
       IOBES  B-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   S-LOC
       ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========

    There are three chunk types(named entity types) including PER(person), ORG(organization)
    and LOC(LOCATION), and we can see that the labels have the form <tag type>-<chunk type>.

    Since the calculations actually use label ids rather than labels, extra attention
    should be paid when mapping labels to ids to make CheckEvalOp work. The key point
    is that the listed equations are satisfied by ids.

    .. code-block:: python

       tag_type = label % num_tag_type
       chunk_type = label / num_tag_type

    where `num_tag_type` is the num of tag types in the tagging scheme, `num_chunk_type`
    is the num of chunk types, and `tag_type` get its value from the following table.

    .. code-block:: python

       Scheme Begin Inside End   Single
        plain   0     -      -     -
        IOB     0     1      -     -
        IOE     -     0      1     -
        IOBES   0     1      2     3

    Still use NER as example, assuming the tagging scheme is IOB while chunk types are ORG,
    PER and LOC. To satisfy the above equations, the label map can be like this:

    .. code-block:: python

       B-ORG  0
       I-ORG  1
       B-PER  2
       I-PER  3
       B-LOC  4
       I-LOC  5
       O      6

    It's not hard to verify the equations noting that the num of chunk types
    is 3 and the num of tag types in IOB scheme is 2. For example, the label
    id of I-LOC is 5, the tag type id of I-LOC is 1, and the chunk type id of
    I-LOC is 2, which consistent with the results from the equations.

    Args:
        input (Variable): prediction output of the network.
        label (Variable): label of the test data set.
        chunk_scheme (str): ${chunk_scheme_comment}
        num_chunk_types (int): ${num_chunk_types_comment}
        excluded_chunk_types (list): ${excluded_chunk_types_comment}
        seq_length(Variable): 1-D Tensor specifying sequence length when input and label are Tensor type.

    Returns:
        tuple: tuple containing: precision, recall, f1_score,
        num_infer_chunks, num_label_chunks,
        num_correct_chunks

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            dict_size = 10000
            label_dict_len = 7
            sequence = fluid.layers.data(
                name='id', shape=[1], lod_level=1, dtype='int64')
            embedding = fluid.layers.embedding(
                input=sequence, size=[dict_size, 512])
            hidden = fluid.layers.fc(input=embedding, size=512)
            label = fluid.layers.data(
                name='label', shape=[1], lod_level=1, dtype='int32')
            crf = fluid.layers.linear_chain_crf(
                input=hidden, label=label, param_attr=fluid.ParamAttr(name="crfw"))
            crf_decode = fluid.layers.crf_decoding(
                input=hidden, param_attr=fluid.ParamAttr(name="crfw"))
            fluid.layers.chunk_eval(
                input=crf_decode,
                label=label,
                chunk_scheme="IOB",
                num_chunk_types=(label_dict_len - 1) / 2)
    """
    helper = LayerHelper("chunk_eval", **locals())

    # prepare output
    precision = helper.create_variable_for_type_inference(dtype="float32")
    recall = helper.create_variable_for_type_inference(dtype="float32")
    f1_score = helper.create_variable_for_type_inference(dtype="float32")
    num_infer_chunks = helper.create_variable_for_type_inference(dtype="int64")
    num_label_chunks = helper.create_variable_for_type_inference(dtype="int64")
    num_correct_chunks = helper.create_variable_for_type_inference(
        dtype="int64")

    this_input = {"Inference": [input], "Label": [label]}

    if seq_length:
        this_input["SeqLength"] = [seq_length]

    helper.append_op(
        type="chunk_eval",
        inputs=this_input,
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


@templatedoc()
def sequence_conv(input,
                  num_filters,
                  filter_size=3,
                  filter_stride=1,
                  padding=True,
                  padding_start=None,
                  bias_attr=None,
                  param_attr=None,
                  act=None,
                  name=None):
    """
    The sequence_conv receives input sequences with variable length and other convolutional
    configuration parameters for the filter and stride to apply the convolution operation.
    It fills all-zero padding data on both sides of the sequence by default to ensure that
    the output is the same length as the input. You can customize the padding behavior by
    configuring the parameter :attr:`padding\_start`.
    
    **Warning:** the parameter :attr:`padding` take no effect and will be deprecated in the future.

    .. code-block:: text

            Here we'll illustrate the details of the padding operation:
            For a mini-batch of 2 variable lengths sentences, containing 3, and 1 time-steps:
            Assumed input (X) is a [4, M, N] float LoDTensor, and X->lod()[0] = [0, 3, 4].
            Besides, for the sake of simplicity, we assume M=1 and N=2.
            X = [[a1, a2;
                  b1, b2;
                  c1, c2]
                 [d1, d2]]

            This is to say that input (X) has 4 words and the dimension of each word
            representation is 2.

            * Case1:

                If padding_start is -1 and filter_size is 3.
                The length of padding data is calculated as follows:
                up_pad_len = max(0, -padding_start) = 1
                down_pad_len = max(0, filter_size + padding_start - 1) = 1

                The output of the input sequence after padding is:
                data_aftet_padding = [[0,  0,  a1, a2, b1, b2;
                                       a1, a2, b1, b2, c1, c2;
                                       b1, b2, c1, c2, 0,  0 ]
                                      [0,  0,  d1, d2, 0,  0 ]]

                It will be multiplied by the filter weight to get the final output.

    Args:
        input (Variable): ${x_comment}
        num_filters (int): the number of filters.
        filter_size (int): the height of filter, the width is hidden size by default.
        filter_stride (int): stride of the filter. Currently only supports :attr:`stride` = 1.
        padding (bool): the parameter :attr:`padding` take no effect and will be discarded in the
            future. Currently, it will always pad input to make sure the length of the output is
            the same as input whether :attr:`padding` is set true or false. Because the length of
            input sequence may be shorter than :attr:`filter\_size`, which will cause the convolution
            result to not be computed correctly. These padding data will not be trainable or updated
            while trainnig. 
        padding_start (int|None): It is used to indicate the start index for padding the input
            sequence, which can be negative. The negative number means to pad
            :attr:`|padding_start|` time-steps of all-zero data at the beginning of each instance.
            The positive number means to skip :attr:`padding_start` time-steps of each instance,
            and it will pad :math:`filter\_size + padding\_start - 1` time-steps of all-zero data
            at the end of the sequence to ensure that the output is the same length as the input.
            If set None, the same length :math:`\\frac{filter\_size}{2}` of data will be filled
            on both sides of the sequence. If set 0, the length of :math:`filter\_size - 1` data
            is padded at the end of each input sequence.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of sequence_conv.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, sequence_conv
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of sequence_conv. If it is set to None or one attribute of ParamAttr, sequence_conv
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None.

    Returns:
        Variable: output of sequence_conv

    Examples:

        .. code-block:: python

             import paddle.fluid as fluid

             x = fluid.layers.data(name='x', shape=[10,10], append_batch_size=False, dtype='float32')
             x_conved = fluid.layers.sequence_conv(input=x, num_filters=2, filter_size=3, padding_start=-1)
    """

    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_conv', **locals())
    dtype = helper.input_dtype()
    filter_shape = [filter_size * input.shape[1], num_filters]
    filter_param = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype)
    pre_bias = helper.create_variable_for_type_inference(dtype)
    if padding_start is None:
        padding_start = -int(filter_size // 2)

    helper.append_op(
        type='sequence_conv',
        inputs={
            'X': [input],
            'Filter': [filter_param],
        },
        outputs={"Out": pre_bias},
        attrs={
            'contextStride': filter_stride,
            'contextStart': padding_start,
            'contextLength': filter_size,
        })
    pre_act = helper.append_bias_op(pre_bias)
    return helper.append_activation(pre_act)


def sequence_softmax(input, use_cudnn=False, name=None):
    """
    This function computes the softmax activation among all time-steps for each
    sequence. The dimension of each time-step should be 1. Thus, the shape of
    input Tensor can be either :math:`[N, 1]` or :math:`[N]`, where :math:`N`
    is the sum of the length of all sequences.

    For i-th sequence in a mini-batch:

    .. math::

        Out(X[lod[i]:lod[i+1]], :) = \\frac{\exp(X[lod[i]:lod[i+1], :])}{\sum(\exp(X[lod[i]:lod[i+1], :]))}

    For example, for a mini-batch of 3 sequences with variable-length,
    each containing 2, 3, 2 time-steps, the lod of which is [0, 2, 5, 7],
    then softmax will be computed among :math:`X[0:2, :]`, :math:`X[2:5, :]`,
    :math:`X[5:7, :]`, and :math:`N` turns out to be 7.

    Args:
        input (Variable): The input variable which is a LoDTensor.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn \
            library is installed. Default: False.
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None.

    Returns:
        Variable: output of sequence_softmax

    Examples:

        .. code-block:: python

             import paddle.fluid as fluid
             x = fluid.layers.data(name='x', shape=[7, 1],
                              dtype='float32', lod_level=1)
             x_sequence_softmax = fluid.layers.sequence_softmax(input=x)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_softmax', **locals())
    dtype = helper.input_dtype()
    softmax_out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="sequence_softmax",
        inputs={"X": input},
        outputs={"Out": softmax_out},
        attrs={"use_cudnn": use_cudnn})
    return softmax_out


def softmax(input, use_cudnn=False, name=None, axis=-1):
    """
    The input of the softmax operator is a tensor of any rank. The output tensor
    has the same shape as the input.

    The dimension :attr:`axis` of the input tensor will be permuted to the last.
    Then the input tensor will be logically flattened to a 2-D matrix. The matrix's
    second dimension(row length) is the same as the dimension :attr:`axis` of the input
    tensor, and the first dimension(column length) is the product of all other
    dimensions of the input tensor. For each row of the matrix, the softmax operator
    squashes the K-dimensional(K is the width of the matrix, which is also the size
    of the input tensor's dimension :attr:`axis`) vector of arbitrary real values to a
    K-dimensional vector of real values in the range [0, 1] that add up to 1.

    It computes the exponential of the given dimension and the sum of exponential
    values of all the other dimensions in the K-dimensional vector input.
    Then the ratio of the exponential of the given dimension and the sum of
    exponential values of all the other dimensions is the output of the softmax
    operator.

    For each row :math:`i` and each column :math:`j` in the matrix, we have:

    .. math::

        Out[i, j] = \\frac{\exp(X[i, j])}{\sum_j(exp(X[i, j])}

    Args:
        input (Variable): The input variable.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn \
            library is installed. To improve numerical stablity, set use_cudnn to \
            False by default. Default: False
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None.
        axis (int): The index of dimension to perform softmax calculations, it should
            be in range :math:`[-1, rank - 1]`, while :math:`rank` is the rank of
            input variable. Default: -1.

    Returns:
        Variable: output of softmax

    Examples:

        .. code-block:: python

             import paddle.fluid as fluid
             x = fluid.layers.data(name='x', shape=[2], dtype='float32')
             fc = fluid.layers.fc(input=x, size=10)
             # perform softmax in the second dimension
             softmax = fluid.layers.softmax(input=fc, axis=1)
             # perform softmax in the last dimension
             softmax = fluid.layers.softmax(input=fc, axis=-1)

    """
    helper = LayerHelper('softmax', **locals())
    if not isinstance(input, Variable):
        raise TypeError(
            "The type of 'input' in softmax must be Variable, but received %s" %
            (type(input)))
    if convert_dtype(input.dtype) not in ['float32', 'float64']:
        raise TypeError(
            "The data type of 'input' in softmax must be float32 or float64, but received %s."
            % (convert_dtype(input.dtype)))

    dtype = helper.input_dtype()
    softmax_out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="softmax",
        inputs={"X": input},
        outputs={"Out": softmax_out},
        attrs={"axis": axis,
               "use_cudnn": use_cudnn})
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
           act=None,
           name=None):
    """
    The convolution2D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCHW format, where N is batch size, C is the number of
    channels, H is the height of the feature, and W is the width of the feature.
    Filter is in MCHW format, where M is the number of output image channels,
    C is the number of input image channels, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input image channels divided by the groups.
    Please refer to UFLDL's `convolution
    <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_
    for more details.
    If bias attribution and activation type are provided, bias is added to the
    output of the convolution, and the corresponding activation function is
    applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    Where:

    * :math:`X`: Input value, a tensor with NCHW format.
    * :math:`W`: Filter value, a tensor with MCHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Note:
        padding mode is 'SAME' and 'VALID' can reference this link<https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/PaddleGAN/network/base_network.py#L181>`_

    Args:
        input (Variable): The input image with [N, C, H, W] format.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple): The filter size. If filter_size 
            is a tuple, it must contain two integers, (filter_size_height, 
            filter_size_width). Otherwise, filter_size_height = filter_\
            size_width = filter_size.
        stride (int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_height, stride_width). Otherwise,
            stride_height = stride_width = stride. Default: stride = 1.
        padding (int|tuple): The padding size. If padding is a tuple, it must
            contain two integers, (padding_height, padding_width). Otherwise,
            padding_height = padding_width =  padding. Default: padding = 0.
        dilation (int|tuple): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_height, dilation_width). Otherwise,
            dilation_height = dilation_width = dilation. Default: dilation = 1.
        groups (int): The groups number of the Conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None

    Returns:
        Variable: The tensor variable storing the convolution and \
                  non-linearity activation result.

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
          conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu")
    """

    num_channels = input.shape[1]
    assert param_attr is not False, "param_attr should not be False here."
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
        num_filter_channels = num_channels // groups

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    input_shape = input.shape
    filter_shape = [num_filters, int(num_filter_channels)] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * num_channels
        std = (2.0 / filter_elem_num)**0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    pre_bias = helper.create_variable_for_type_inference(dtype)

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
            'use_mkldnn': False,
            'fuse_relu_before_depthwise_conv': False
        })

    pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)

    return helper.append_activation(pre_act)


def conv3d(input,
           num_filters,
           filter_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=None,
           param_attr=None,
           bias_attr=None,
           use_cudnn=True,
           act=None,
           name=None):
    """
    **Convlution3D Layer**

    The convolution3D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are in NCDHW format. Where N is batch size C is the number of
    channels, D is the depth of the feature, H is the height of the feature,
    and W is the width of the feature. Convlution3D is similar with Convlution2D
    but adds one dimension(depth). If bias attribution and activation type are
    provided, bias is added to the output of the convolution, and the
    corresponding activation function is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NCDHW format.
    * :math:`W`: Filter value, a tensor with MCDHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, D_f, H_f, W_f)`

        - Output:
          Output shape: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        .. math::

            D_{out}&= \\frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (D_f - 1) + 1))}{strides[0]} + 1 \\\\
            H_{out}&= \\frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (H_f - 1) + 1))}{strides[1]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (W_f - 1) + 1))}{strides[2]} + 1

    Args:
        input (Variable): The input image with [N, C, D, H, W] format.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple): The filter size. If filter_size is a tuple,
            it must contain three integers, (filter_size_depth, filter_size_height, 
            filter_size_width). Otherwise, filter_size_depth = filter_size_height = \
            filter_size_width = filter_size.
        stride (int|tuple): The stride size. If stride is a tuple, it must
            contain three integers, (stride_depth, stride_height, stride_width). Otherwise,
            stride_depth = stride_height = stride_width = stride. Default: stride = 1.
        padding (int|tuple): The padding size. If padding is a tuple, it must
            contain three integers, (padding_depth, padding_height, padding_width). Otherwise,
            padding_depth = padding_height = padding_width = padding. Default: padding = 0.
        dilation (int|tuple): The dilation size. If dilation is a tuple, it must
            contain three integers, (dilation_depth, dilation_height, dilation_width). Otherwise,
            dilation_depth = dilation_height = dilation_width = dilation. Default: dilation = 1.
        groups (int): The groups number of the Conv3d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv3d. If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv3d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None.

    Returns:
        Variable: The tensor variable storing the convolution and \
                  non-linearity activation result.

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='data', shape=[3, 12, 32, 32], dtype='float32')
          conv3d = fluid.layers.conv3d(input=data, num_filters=2, filter_size=3, act="relu")
    """

    l_type = 'conv3d'
    assert param_attr is not False, "param_attr should not be False here."
    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()

    num_channels = input.shape[1]

    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        num_filter_channels = num_channels // groups

    filter_size = utils.convert_to_list(filter_size, 3, 'filter_size')
    stride = utils.convert_to_list(stride, 3, 'stride')
    padding = utils.convert_to_list(padding, 3, 'padding')
    dilation = utils.convert_to_list(dilation, 3, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    input_shape = input.shape
    filter_shape = [num_filters, num_filter_channels] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * filter_size[
            2] * num_channels
        std = (2.0 / filter_elem_num)**0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    pre_bias = helper.create_variable_for_type_inference(dtype)

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
            'use_mkldnn': False
        })

    pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)

    return helper.append_activation(pre_act)


def sequence_pool(input, pool_type, is_test=False, pad_value=0.0):
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

       x is a 1-level LoDTensor and **pad_value** = 0.0:
         x.lod = [[2, 3, 2, 0]]
         x.data = [1, 3, 2, 4, 6, 5, 1]
         x.dims = [7, 1]

       then output is a Tensor:
         out.dim = [4, 1]
         with condition len(x.lod[-1]) == out.dims[0]

       for different pool_type:
         average: out.data = [2, 4, 3, 0.0], where 2=(1+3)/2, 4=(2+4+6)/3, 3=(5+1)/2
         sum    : out.data = [4, 12, 6, 0.0], where 4=1+3, 12=2+4+6, 6=5+1
         sqrt   : out.data = [2.82, 6.93, 4.24, 0.0], where 2.82=(1+3)/sqrt(2),
                    6.93=(2+4+6)/sqrt(3), 4.24=(5+1)/sqrt(2)
         max    : out.data = [3, 6, 5, 0.0], where 3=max(1,3), 6=max(2,4,6), 5=max(5,1)
         last   : out.data = [3, 6, 1, 0.0], where 3=last(1,3), 6=last(2,4,6), 1=last(5,1)
         first  : out.data = [1, 2, 5, 0.0], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)

         and all above 0.0 = **pad_value**.

    Args:
        input (variable): The input variable which is a LoDTensor.
        pool_type (string): The pooling type of sequence_pool.
            It supports average, sum, sqrt and max.
        is_test (bool): Used to distinguish training from scoring mode. Default False.
        pad_value (float): Used to pad the pooling result for empty input sequence.

    Returns:
        The sequence pooling variable which is a Tensor.

    Examples:

        .. code-block:: python

             import paddle.fluid as fluid

             x = fluid.layers.data(name='x', shape=[7, 1],
                              dtype='float32', lod_level=1)
             avg_x = fluid.layers.sequence_pool(input=x, pool_type='average')
             sum_x = fluid.layers.sequence_pool(input=x, pool_type='sum')
             sqrt_x = fluid.layers.sequence_pool(input=x, pool_type='sqrt')
             max_x = fluid.layers.sequence_pool(input=x, pool_type='max')
             last_x = fluid.layers.sequence_pool(input=x, pool_type='last')
             first_x = fluid.layers.sequence_pool(input=x, pool_type='first')
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_pool', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    max_index = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type="sequence_pool",
        inputs={"X": input},
        outputs={"Out": pool_out,
                 "MaxIndex": max_index},
        attrs={
            "pooltype": pool_type.upper(),
            "is_test": is_test,
            "pad_value": pad_value
        })

    # when pool_type is max, variable max_index is initialized,
    # so we stop the gradient explicitly here
    if pool_type == 'max':
        max_index.stop_gradient = True

    return pool_out


@templatedoc()
def sequence_concat(input, name=None):
    """
    ${comment}

    Args:
        input(list): List of Variables to be concatenated.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        Variable: Output variable of the concatenation.

    Examples:
        .. code-block:: python

           import paddle.fluid as fluid
           x = fluid.layers.data(name='x', shape=[10], dtype='float32')
           y = fluid.layers.data(name='y', shape=[10], dtype='float32')
           out = fluid.layers.sequence_concat(input=[x, y])
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_concat', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='sequence_concat', inputs={'X': input}, outputs={'Out': [out]})
    return out


def sequence_first_step(input):
    """
    This function gets the first step of sequence.

    .. code-block:: text

       x is a 1-level LoDTensor:
         x.lod = [[2, 3, 2]]
         x.data = [1, 3, 2, 4, 6, 5, 1]
         x.dims = [7, 1]

       then output is a Tensor:
         out.dim = [3, 1]
         with condition len(x.lod[-1]) == out.dims[0]
         out.data = [1, 2, 5], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)

    Args:
        input(variable): The input variable which is a LoDTensor.

    Returns:
        The sequence's first step variable which is a Tensor.

    Examples:

        .. code-block:: python

             import paddle.fluid as fluid
             x = fluid.layers.data(name='x', shape=[7, 1],
                              dtype='float32', lod_level=1)
             x_first_step = fluid.layers.sequence_first_step(input=x)
    """
    return sequence_pool(input=input, pool_type="first")


def sequence_last_step(input):
    """
    This function gets the last step of sequence.

    .. code-block:: text

       x is a 1-level LoDTensor:
         x.lod = [[2, 3, 2]]
         x.data = [1, 3, 2, 4, 6, 5, 1]
         x.dims = [7, 1]

       then output is a Tensor:
         out.dim = [3, 1]
         with condition len(x.lod[-1]) == out.dims[0]
         out.data = [3, 6, 1], where 3=last(1,3), 6=last(2,4,6), 1=last(5,1)

    Args:
        input(variable): The input variable which is a LoDTensor.

    Returns:
        The sequence's last step variable which is a Tensor.

    Examples:

        .. code-block:: python

             import paddle.fluid as fluid
             x = fluid.layers.data(name='x', shape=[7, 1],
                              dtype='float32', lod_level=1)
             x_last_step = fluid.layers.sequence_last_step(input=x)
    """
    return sequence_pool(input=input, pool_type="last")


def sequence_slice(input, offset, length, name=None):
    """
    **Sequence Slice Layer**

    The layer crops a subsequence from given sequence with given start
    offset and subsequence length.

    It only supports sequence data (LoDTensor with lod_level equal to 1).

    .. code-block:: text

              - Case:

            Given the input Variable **input**:

                input.data = [[a1, a2], [b1, b2], [c1, c2], [d1, d2], [e1, e2]],
                input.lod = [[3, 2]],
                input.dims = (5, 2),

            with offset.data = [[0], [1]] and length.data = [[2], [1]],

            the output Variable will be

                out.data = [[a1, a2], [b1, b2], [e1, e2]],
                out.lod = [[2, 1]],
                out.dims = (3, 2).

    Note:
          The first dimension size of **input**, **offset** and **length**
          should be equal. The **offset** should start from 0.

    Args:
        input(Variable): The input Variable which consists of the complete
                         sequences.
        offset(Variable): The offset to slice each sequence.
        length(Variable): The length of each subsequence.
        name(str|None): A name for this layer(optional). If set None, the
                        layer will be named automatically.

    Returns:
        Variable: The output subsequences.

    Examples:

        .. code-block:: python

             import paddle.fluid as fluid
             import numpy as np
             seqs = fluid.layers.data(name='x', shape=[10, 5],
                              dtype='float32', lod_level=1)
             offset = fluid.layers.assign(input=np.array([[0, 1]]).astype("int32"))
             length = fluid.layers.assign(input=np.array([[2, 1]]).astype("int32"))
             subseqs = fluid.layers.sequence_slice(input=seqs, offset=offset,
                                                   length=length)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper("sequence_slice", **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)

    offset.stop_gradient = True
    length.stop_gradient = True

    helper.append_op(
        type="sequence_slice",
        inputs={"X": input,
                "Offset": offset,
                "Length": length},
        outputs={"Out": out})

    return out


@templatedoc()
def pool2d(input,
           pool_size=-1,
           pool_type="max",
           pool_stride=1,
           pool_padding=0,
           global_pooling=False,
           use_cudnn=True,
           ceil_mode=False,
           name=None,
           exclusive=True):
    """
    ${comment}

    Args:
        input (Variable): The input tensor of pooling operator. The format of
                          input tensor is NCHW, where N is batch size, C is
                          the number of channels, H is the height of the
                          feature, and W is the width of the feature.
        pool_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be a square of an int.
        pool_type: ${pooling_type_comment}
        pool_stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain two integers, (pool_stride_Height, pool_stride_Width).
            Otherwise, the pool stride size will be a square of an int.
        pool_padding (int|list|tuple): The pool padding size. If pool padding size is a tuple,
            it must contain two integers, (pool_padding_on_Height, pool_padding_on_Width).
            Otherwise, the pool padding size will be a square of an int.
        global_pooling (bool): ${global_pooling_comment}
        use_cudnn (bool): ${use_cudnn_comment}
        ceil_mode (bool): ${ceil_mode_comment}
        name (str|None): A name for this layer(optional). If set None, the
                        layer will be named automatically.
        exclusive (bool): Whether to exclude padding points in average pooling
                          mode, default is true

    Returns:
        Variable: The pooling result.

    Raises:
        ValueError: If 'pool_type' is not "max" nor "avg"
        ValueError: If 'global_pooling' is False and 'pool_size' is -1
        ValueError: If 'use_cudnn' is not a bool value.

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(
              name='data', shape=[3, 32, 32], dtype='float32')
          pool2d = fluid.layers.pool2d(
                            input=data,
                            pool_size=2,
                            pool_type='max',
                            pool_stride=1,
                            global_pooling=False)
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

    l_type = 'pool2d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=l_type,
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
            "use_mkldnn": False,
            "exclusive": exclusive,
        })

    return pool_out


@templatedoc()
def pool3d(input,
           pool_size=-1,
           pool_type="max",
           pool_stride=1,
           pool_padding=0,
           global_pooling=False,
           use_cudnn=True,
           ceil_mode=False,
           name=None,
           exclusive=True):
    """
    ${comment}

    Args:
        input (Variable): The input tensor of pooling operator. The format of
                          input tensor is NCDHW, where N is batch size, C is
                          the number of channels, D is the depth of the feature,
                          H is the height of the feature, and W is the width
                          of the feature.
        pool_size (int|list|tuple): The pool kernel size. If pool kernel size 
            is a tuple or list, it must contain three integers, 
            (pool_size_Depth, pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be the cube of an int.
        pool_type (string): ${pooling_type_comment}
        pool_stride (int): stride of the pooling layer.
        pool_padding (int): padding size.
        global_pooling (bool): ${global_pooling_comment}
        use_cudnn (bool): ${use_cudnn_comment}
        ceil_mode (bool): ${ceil_mode_comment}
        name (str): A name for this layer(optional). If set None, the layer
            will be named automatically.
        exclusive (bool): Whether to exclude padding points in average pooling
                          mode, default is true

    Returns:
        Variable: output of pool3d layer.

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(
              name='data', shape=[3, 32, 32, 32], dtype='float32')
          pool3d = fluid.layers.pool3d(
                            input=data,
                            pool_size=2,
                            pool_type='max',
                            pool_stride=1,
                            global_pooling=False)
    """
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))

    if global_pooling is False and pool_size == -1:
        raise ValueError(
            "When the global_pooling is False, pool_size must be passed "
            "and be a valid value. Received pool_size: " + str(pool_size))

    pool_size = utils.convert_to_list(pool_size, 3, 'pool_size')
    pool_padding = utils.convert_to_list(pool_padding, 3, 'pool_padding')
    pool_stride = utils.convert_to_list(pool_stride, 3, 'pool_stride')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    l_type = "pool3d"
    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=l_type,
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
            "use_mkldnn": False,
            "exclusive": exclusive,
        })

    return pool_out


@templatedoc(op_type="pool2d")
def adaptive_pool2d(input,
                    pool_size,
                    pool_type="max",
                    require_index=False,
                    name=None):
    """
    **Adaptive Pool2d Operator**
    The adaptive_pool2d operation calculates the output based on the input, pool_size,
    pool_type parameters. Input(X) and output(Out) are in NCHW format, where N is batch
    size, C is the number of channels, H is the height of the feature, and W is
    the width of the feature. Parameters(pool_size) should contain two elements which
    represent height and width, respectively. Also the H and W dimensions of output(Out)
    is same as Parameter(pool_size).

    For average adaptive pool2d:

    ..  math::

       hstart &= floor(i * H_{in} / H_{out})

       hend &= ceil((i + 1) * H_{in} / H_{out})

       wstart &= floor(j * W_{in} / W_{out})

       wend &= ceil((j + 1) * W_{in} / W_{out})

       Output(i ,j) &= \\frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}

    Args:
        input (Variable): The input tensor of pooling operator. The format of
                          input tensor is NCHW, where N is batch size, C is
                          the number of channels, H is the height of the
                          feature, and W is the width of the feature.
        pool_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
        pool_type: ${pooling_type_comment}
        require_index (bool): If true, the index of max pooling point will be returned along
            with outputs. It cannot be set in average pooling type.
        name (str|None): A name for this layer(optional). If set None, the
                        layer will be named automatically.

    Returns:
        Variable: The pooling result.

    Raises:
        ValueError: 'pool_type' is not 'max' nor 'avg'.
        ValueError: invalid setting 'require_index' true when 'pool_type' is 'avg'.
        ValueError: 'pool_size' should be a list or tuple with length as 2.

    Examples:
        .. code-block:: python

          # suppose input data in shape of [N, C, H, W], `pool_size` is [m, n],
          # output shape is [N, C, m, n], adaptive pool divide H and W dimentions
          # of input data into m * n grids averagely and performs poolings in each
          # grid to get output.
          # adaptive average pool performs calculations as follow:
          #
          #     for i in range(m):
          #         for j in range(n):
          #             hstart = floor(i * H / m)
          #             hend = ceil((i + 1) * H / m)
          #             wstart = floor(i * W / n)
          #             wend = ceil((i + 1) * W / n)
          #             output[:, :, i, j] = avg(input[:, :, hstart: hend, wstart: wend])
          #
          import paddle.fluid as fluid
          data = fluid.layers.data(
              name='data', shape=[3, 32, 32], dtype='float32')
          pool_out = fluid.layers.adaptive_pool2d(
                            input=data,
                            pool_size=[3, 3],
                            pool_type='avg')
    """
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))

    if pool_type == "avg" and require_index:
        raise ValueError(
            "invalid setting 'require_index' true when 'pool_type' is 'avg'.")

    pool_size = utils.convert_to_list(pool_size, 2, 'pool_size')

    if pool_type == "max":
        l_type = 'max_pool2d_with_index'
    else:
        l_type = "pool2d"

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    outputs = {"Out": pool_out}
    if pool_type == "max":
        mask = helper.create_variable_for_type_inference(dtype)
        outputs["Mask"] = mask

    helper.append_op(
        type=l_type,
        inputs={"X": input},
        outputs=outputs,
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "adaptive": True,
        })

    return (pool_out, mask) if require_index else pool_out


@templatedoc(op_type="pool3d")
def adaptive_pool3d(input,
                    pool_size,
                    pool_type="max",
                    require_index=False,
                    name=None):
    """
    **Adaptive Pool3d Operator**
    The adaptive_pool3d operation calculates the output based on the input, pool_size,
    pool_type parameters. Input(X) and output(Out) are in NCDHW format, where N is batch
    size, C is the number of channels, D is the depth of the feature, H is the height of
    the feature, and W is the width of the feature. Parameters(pool_size) should contain
    three elements which represent height and width, respectively. Also the D, H and W
    dimensions of output(Out) is same as Parameter(pool_size).

    For average adaptive pool3d:

    ..  math::

      dstart &= floor(i * D_{in} / D_{out})

      dend &= ceil((i + 1) * D_{in} / D_{out})

      hstart &= floor(j * H_{in} / H_{out})

      hend &= ceil((j + 1) * H_{in} / H_{out})

      wstart &= floor(k * W_{in} / W_{out})

      wend &= ceil((k + 1) * W_{in} / W_{out})

      Output(i ,j, k) &= \\frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{(dend - dstart) * (hend - hstart) * (wend - wstart)}

    Args:
        input (Variable): The input tensor of pooling operator. The format of
                          input tensor is NCDHW, where N is batch size, C is
                          the number of channels, D is the depth of the feature,
                          H is the height of the feature, and W is the width of the feature.
        pool_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain three integers, (Depth, Height, Width).
        pool_type: ${pooling_type_comment}
        require_index (bool): If true, the index of max pooling point will be returned along
            with outputs. It cannot be set in average pooling type.
        name (str|None): A name for this layer(optional). If set None, the
                        layer will be named automatically.

    Returns:
        Variable: The pooling result.

    Raises:
        ValueError: 'pool_type' is not 'max' nor 'avg'.
        ValueError: invalid setting 'require_index' true when 'pool_type' is 'avg'.
        ValueError: 'pool_size' should be a list or tuple with length as 2.

    Examples:
        .. code-block:: python

          # suppose input data in shape of [N, C, D, H, W], `pool_size` is [l, m, n],
          # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimentions
          # of input data into l * m * n grids averagely and performs poolings in each
          # grid to get output.
          # adaptive average pool performs calculations as follow:
          #
          #     for i in range(l):
          #         for j in range(m):
          #             for k in range(n):
          #                 dstart = floor(i * D / l)
          #                 dend = ceil((i + 1) * D / l)
          #                 hstart = floor(j * H / m)
          #                 hend = ceil((j + 1) * H / m)
          #                 wstart = floor(k * W / n)
          #                 wend = ceil((k + 1) * W / n)
          #                 output[:, :, i, j, k] =
          #                     avg(input[:, :, dstart:dend, hstart: hend, wstart: wend])
          #

          import paddle.fluid as fluid

          data = fluid.layers.data(
              name='data', shape=[3, 32, 32, 32], dtype='float32')
          pool_out = fluid.layers.adaptive_pool3d(
                            input=data,
                            pool_size=[3, 3, 3],
                            pool_type='avg')
    """
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))

    if pool_type == "avg" and require_index:
        raise ValueError(
            "invalid setting 'require_index' true when 'pool_type' is 'avg'.")

    pool_size = utils.convert_to_list(pool_size, 3, 'pool_size')

    if pool_type == "max":
        l_type = 'max_pool3d_with_index'
    else:
        l_type = "pool3d"

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    outputs = {"Out": pool_out}
    if pool_type == "max":
        mask = helper.create_variable_for_type_inference(dtype)
        outputs["Mask"] = mask

    helper.append_op(
        type=l_type,
        inputs={"X": input},
        outputs=outputs,
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "adaptive": True,
        })

    return (pool_out, mask) if require_index else pool_out


def batch_norm(input,
               act=None,
               is_test=False,
               momentum=0.9,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               data_layout='NCHW',
               in_place=False,
               name=None,
               moving_mean_name=None,
               moving_variance_name=None,
               do_model_average_for_mean_and_var=False,
               fuse_with_relu=False,
               use_global_stats=False):
    """
    **Batch Normalization Layer**

    Can be used as a normalizer function for conv2d and fully_connected operations.
    The required data format for this layer is one of the following:

    1. NHWC `[batch, in_height, in_width, in_channels]`

    2. NCHW `[batch, in_channels, in_height, in_width]`

    Refer to `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_
    for more details.

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

        moving\_mean = moving\_mean * momentum + mini-batch\_mean * (1. - momentum)
        moving\_var = moving\_var * momentum + mini-batch\_var * (1. - momentum)
        moving_mean and moving_var is global mean and global variance.


    When use_global_stats = True, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are not the statistics of one mini-batch.
    They are global (or running) statistics. (It usually got from the
    pre-trained model.)
    The training and testing (or inference) have the same behavior:

    ..  math::

        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}}  \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta

    Note:
        if build_strategy.sync_batch_norm=True, the batch_norm in network will use 
        sync_batch_norm automatically.

    Args:
        input(variable): The rank of input variable can be 2, 3, 4, 5.
        act(string, Default None): Activation type, linear|relu|prelu|...
        is_test (bool, Default False): A flag indicating whether it is in
            test phrase or not.
        momentum(float, Default 0.9): The value used for the moving_mean and
            moving_var computation. The updated formula is:
            :math:`moving\_mean = moving\_mean * momentum + new\_mean * (1. - momentum)`
            :math:`moving\_var = moving\_var * momentum + new\_var * (1. - momentum)`
            Default is 0.9.
        epsilon(float, Default 1e-05): A value added to the denominator for
            numerical stability. Default is 1e-5.
        param_attr(ParamAttr|None): The parameter attribute for Parameter `scale`
             of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
	     will create ParamAttr as param_attr, the name of scale can be set in ParamAttr.
	     If the Initializer of the param_attr is not set, the parameter is initialized 
	     with Xavier. Default: None.
        bias_attr(ParamAttr|None): The parameter attribute for the bias of batch_norm.
             If it is set to None or one attribute of ParamAttr, batch_norm
	     will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr. 
	     If the Initializer of the bias_attr is not set, the bias is initialized zero. 
	     Default: None.
        data_layout(string, default NCHW): NCHW|NHWC
        in_place(bool, Default False): Make the input and output of batch norm reuse memory.
        name(string, Default None): A name for this layer(optional). If set None, the layer
            will be named automatically.
        moving_mean_name(string, Default None): The name of moving_mean which store the global Mean. If it 
            is set to None, batch_norm will save global mean with a random name, otherwise, batch_norm 
            will save global mean with the string.
        moving_variance_name(string, Default None): The name of the moving_variance which store the global Variance.
            If it is set to None, batch_norm will save global variance with a random name, otherwise, batch_norm 
            will save global variance with the string.
        do_model_average_for_mean_and_var(bool, Default False): Do model average for mean and variance or not.
        fuse_with_relu (bool): if True, this OP performs relu after batch norm.
        use_global_stats(bool, Default False): Whether to use global mean and
            variance. In inference or test mode, set use_global_stats to true
            or is_test to true, and the behavior is equivalent.
            In train mode, when setting use_global_stats True, the global mean
            and variance are also used during train period.

    Returns:
        Variable: A tensor variable which is the result after applying batch normalization on the input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[3, 7, 3, 7], dtype='float32', append_batch_size=False)
            hidden1 = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
            hidden2 = fluid.layers.batch_norm(input=hidden1)
    """
    assert bias_attr is not False, "bias_attr should not be False in batch_norm."
    helper = LayerHelper('batch_norm', **locals())
    dtype = helper.input_dtype()

    # use fp32 for bn parameter
    if dtype == core.VarDesc.VarType.FP16:
        dtype = core.VarDesc.VarType.FP32

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
        dtype=dtype)
    mean.stop_gradient = True

    variance = helper.create_parameter(
        attr=ParamAttr(
            name=moving_variance_name,
            initializer=Constant(1.0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var),
        shape=param_shape,
        dtype=dtype)
    variance.stop_gradient = True

    # create output
    # mean and mean_out share the same memory
    mean_out = mean
    # variance and variance out share the same memory
    variance_out = variance
    saved_mean = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    saved_variance = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)

    batch_norm_out = input if in_place else helper.create_variable_for_type_inference(
        dtype)

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
            "data_layout": data_layout,
            "use_mkldnn": False,
            "fuse_with_relu": fuse_with_relu,
            "use_global_stats": use_global_stats
        })

    return helper.append_activation(batch_norm_out)


def instance_norm(input,
                  epsilon=1e-05,
                  param_attr=None,
                  bias_attr=None,
                  name=None):
    """
    **Instance Normalization Layer**

    Can be used as a normalizer function for conv2d and fully_connected operations.
    The required data format for this layer is one of the following:

    DataLayout: NCHW `[batch, in_channels, in_height, in_width]`

    Refer to `Instance Normalization: The Missing Ingredient for 
    Fast Stylization <https://arxiv.org/pdf/1607.08022.pdf>`_
    for more details.

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{HW} \\sum_{i=1}^{HW} x_i \\qquad &//\\
        \\ mean of one  feature map in mini-batch \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{HW} \\sum_{i=1}^{HW}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ variance of one feature map in mini-batch \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}}  \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta

    Args:
        input(variable): The rank of input variable can be 2, 3, 4, 5.
        epsilon(float, Default 1e-05): A value added to the denominator for
            numerical stability. Default is 1e-5.
        param_attr(ParamAttr|None): The parameter attribute for Parameter `scale`
             of instance_norm. If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as param_attr, the name of scale can be set in ParamAttr.
	     If the Initializer of the param_attr is not set, the parameter is initialized 
	     with Xavier. Default: None.
        bias_attr(ParamAttr|None): The parameter attribute for the bias of instance_norm.
             If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr. 
	     If the Initializer of the bias_attr is not set, the bias is initialized zero. 
	     Default: None.
        name(string, Default None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: A tensor variable which is the result after applying instance normalization on the input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[3, 7, 3, 7], dtype='float32', append_batch_size=False)
            hidden1 = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
            hidden2 = fluid.layers.instance_norm(input=hidden1)
    """
    assert bias_attr is not False, "bias_attr should not be False in instance_norm."
    helper = LayerHelper('instance_norm', **locals())
    dtype = helper.input_dtype()

    # use fp32 for in parameter
    if dtype == core.VarDesc.VarType.FP16:
        dtype = core.VarDesc.VarType.FP32

    input_shape = input.shape
    channel_num = input_shape[1]

    param_shape = [channel_num]

    # create parameter
    scale = helper.create_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype=dtype,
        default_initializer=Constant(1.0))
    bias = helper.create_parameter(
        attr=helper.bias_attr,
        shape=param_shape,
        dtype=dtype,
        is_bias=True,
        default_initializer=Constant(0.0))

    # create output
    saved_mean = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    saved_variance = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)

    instance_norm_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type="instance_norm",
        inputs={
            "X": input,
            "Scale": scale,
            "Bias": bias,
        },
        outputs={
            "Y": instance_norm_out,
            "SavedMean": saved_mean,
            "SavedVariance": saved_variance
        },
        attrs={"epsilon": epsilon, })

    return instance_norm_out


def data_norm(input,
              act=None,
              epsilon=1e-05,
              param_attr=None,
              data_layout='NCHW',
              in_place=False,
              name=None,
              moving_mean_name=None,
              moving_variance_name=None,
              do_model_average_for_mean_and_var=False):
    """
    **Data Normalization Layer**

    Can be used as a normalizer function for conv2d and fully_connected operations.
    The required data format for this layer is one of the following:

    1. NHWC `[batch, in_height, in_width, in_channels]`

    2. NCHW `[batch, in_channels, in_height, in_width]`

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

    Args:
        input(variable): The input variable which is a LoDTensor.
        act(string, Default None): Activation type, linear|relu|prelu|...
        epsilon(float, Default 1e-05):
        param_attr(ParamAttr): The parameter attribute for Parameter `scale`.
        data_layout(string, default NCHW): NCHW|NHWC
        in_place(bool, Default False): Make the input and output of batch norm reuse memory.
        name(string, Default None): A name for this layer(optional). If set None, the layer
            will be named automatically.
        moving_mean_name(string, Default None): The name of moving_mean which store the global Mean.
        moving_variance_name(string, Default None): The name of the moving_variance which store the global Variance.
        do_model_average_for_mean_and_var(bool, Default False): Do model average for mean and variance or not.

    Returns:
        Variable: A tensor variable which is the result after applying data normalization on the input.

    Examples:

        .. code-block:: python
            
            import paddle.fluid as fluid

            hidden1 = fluid.layers.data(name="hidden1", shape=[200])
            hidden2 = fluid.layers.data_norm(name="hidden2", input=hidden1)
    """
    helper = LayerHelper('data_norm', **locals())
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

    batch_size_default = 1e4
    batch_sum_default = 0.0
    batch_square_sum_default = 1e4

    if param_attr and isinstance(param_attr, dict):
        batch_size_default = param_attr.get("batch_size", 1e4)
        batch_sum_default = param_attr.get("batch_sum", 0.0)
        batch_square_sum_default = param_attr.get("batch_square", 1e4)

    # create parameter
    batch_size = helper.create_parameter(
        attr=ParamAttr(
            name=name + '.batch_size',
            initializer=Constant(value=float(batch_size_default)),
            trainable=True),
        shape=param_shape,
        dtype=input.dtype)

    batch_sum = helper.create_parameter(
        attr=ParamAttr(
            name=name + '.batch_sum',
            initializer=Constant(value=float(batch_sum_default)),
            trainable=True),
        shape=param_shape,
        dtype=input.dtype)

    batch_square_sum = helper.create_parameter(
        attr=ParamAttr(
            name=name + '.batch_square_sum',
            initializer=Constant(value=float(batch_square_sum_default)),
            trainable=True),
        shape=param_shape,
        dtype=input.dtype)

    means = helper.create_variable(dtype=dtype, stop_gradient=True)
    scales = helper.create_variable(dtype=dtype, stop_gradient=True)

    data_norm_out = input if in_place else helper.create_variable(dtype=dtype)

    helper.append_op(
        type="data_norm",
        inputs={
            "X": input,
            "BatchSize": batch_size,
            "BatchSum": batch_sum,
            "BatchSquareSum": batch_square_sum
        },
        outputs={"Y": data_norm_out,
                 "Means": means,
                 "Scales": scales},
        attrs={"epsilon": epsilon})

    return helper.append_activation(data_norm_out)


@templatedoc()
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
    ${comment}

    The formula is as follows:

    ..  math::

        \\mu & = \\frac{1}{H}\\sum_{i=1}^{H} a_i

        \\sigma & = \\sqrt{\\frac{1}{H}\sum_{i=1}^{H}(a_i - \\mu)^2}

        h & = f(\\frac{g}{\\sigma}(a - \\mu) + b)

    * :math:`a`: the vector representation of the summed inputs to the neurons
    in that layer.

    * :math:`H`: the number of hidden units in a layers

    * :math:`g`: the trainable scale parameter.

    * :math:`b`: the trainable bias parameter.

    Args:
        input(Variable): The input tensor variable.
        scale(bool): Whether to learn the adaptive gain :math:`g` after
            normalization. Default True.
        shift(bool): Whether to learn the adaptive bias :math:`b` after
            normalization. Default True.
        begin_norm_axis(int): The normalization will be performed along
            dimensions from :attr:`begin_norm_axis` to :attr:`rank(input)`.
            Default 1.
        epsilon(float): The small value added to the variance to prevent
            division by zero. Default 1e-05.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            gain :math:`g`. If :attr:`scale` is False, :attr:`param_attr` is
            omitted. If :attr:`scale` is True and :attr:`param_attr` is None,
            a default :code:`ParamAttr` would be added as scale. The
            :attr:`param_attr` is initialized as 1 if it is added. Default None.
        bias_attr(ParamAttr|None): The parameter attribute for the learnable
            bias :math:`b`. If :attr:`shift` is False, :attr:`bias_attr` is
            omitted. If :attr:`shift` is True and :attr:`param_attr` is None,
            a default :code:`ParamAttr` would be added as bias. The
            :attr:`bias_attr` is initialized as 0 if it is added. Default None.
        act(str): Activation to be applied to the output of layer normalizaiton.
                  Default None.
        name(str): The name of this layer. It is optional. Default None, and a
                   unique name would be generated automatically.

    Returns:
        ${y_comment}

    Examples:

        >>> import paddle.fluid as fluid
        >>> data = fluid.layers.data(name='data', shape=[3, 32, 32],
        >>>                          dtype='float32')
        >>> x = fluid.layers.layer_norm(input=data, begin_norm_axis=1)
    """
    assert in_dygraph_mode(
    ) is not True, "please use FC instead of fc in dygraph mode!"
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
    mean_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    variance_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    layer_norm_out = helper.create_variable_for_type_inference(dtype)

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


@templatedoc()
def group_norm(input,
               groups,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               act=None,
               data_layout='NCHW',
               name=None):
    """
    **Group Normalization Layer**

    Refer to `Group Normalization <https://arxiv.org/abs/1803.08494>`_ .

    Args:
        input(Variable): The input tensor variable.
        groups(int): The number of groups that divided from channels.
        epsilon(float): The small value added to the variance to prevent
            division by zero.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            scale :math:`g`. If it is set to False, no scale will be added to the output units.
            If it is set to None, the bias is initialized one. Default: None.
        bias_attr(ParamAttr|None): The parameter attribute for the learnable
            bias :math:`b`. If it is set to False, no bias will be added to the output units.
            If it is set to None, the bias is initialized zero. Default: None.
        act(str): Activation to be applied to the output of group normalizaiton.
        data_layout(string, default NCHW): NCHW(num_batch, channels, h, w) or NHWC(num_batch, h, w, channels).
        name (str): The name of this layer. It is optional.

    Returns:
        Variable: A tensor variable which is the result after applying group normalization on the input.

    Examples:

        >>> import paddle.fluid as fluid
        >>> data = fluid.layers.data(name='data', shape=[8, 32, 32],
        >>>                          dtype='float32')
        >>> x = fluid.layers.group_norm(input=data, groups=4)
    """
    helper = LayerHelper('group_norm', **locals())
    dtype = helper.input_dtype()

    # create intput and parameters
    inputs = {'X': input}
    input_shape = input.shape
    if data_layout != 'NCHW' and data_layout != 'NHWC':
        raise ValueError(
            "Param(data_layout) of Op(fluid.layers.group_norm) got wrong value: received "
            + data_layout + " but only NCHW or NHWC supported.")
    channel_num = input_shape[1] if data_layout == 'NCHW' else input_shape[-1]
    param_shape = [channel_num]
    if param_attr:
        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=param_shape,
            dtype=dtype,
            default_initializer=Constant(1.0))
        inputs['Scale'] = scale
    if bias_attr:
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True)
        inputs['Bias'] = bias

    # create output
    mean_out = helper.create_variable(dtype=dtype, stop_gradient=True)
    variance_out = helper.create_variable(dtype=dtype, stop_gradient=True)
    group_norm_out = helper.create_variable(dtype=dtype)

    helper.append_op(
        type="group_norm",
        inputs=inputs,
        outputs={
            "Y": group_norm_out,
            "Mean": mean_out,
            "Variance": variance_out,
        },
        attrs={
            "epsilon": epsilon,
            "groups": groups,
            "data_layout": data_layout
        })

    return helper.append_activation(group_norm_out)


@templatedoc()
def spectral_norm(weight, dim=0, power_iters=1, eps=1e-12, name=None):
    """
    **Spectral Normalization Layer**

    This layer calculates the spectral normalization value of weight parameters of
    fc, conv1d, conv2d, conv3d layers which should be 2-D, 3-D, 4-D, 5-D
    Parameters. Calculations are showed as follows.

    Step 1:
    Generate vector U in shape of [H], and V in shape of [W].
    While H is the :attr:`dim` th dimension of the input weights,
    and W is the product result of remaining dimensions.

    Step 2:
    :attr:`power_iters` shoule be a positive interger, do following
    calculations with U and V for :attr:`power_iters` rounds.

    .. math:: 

        \mathbf{v} := \\frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}

        \mathbf{u} := \\frac{\mathbf{W}^{T} \mathbf{v}}{\|\mathbf{W}^{T} \mathbf{v}\|_2}

    Step 3:
    Calculate :math:`\sigma(\mathbf{W})` and normalize weight values.

    .. math::

        \sigma(\mathbf{W}) = \mathbf{u}^{T} \mathbf{W} \mathbf{v}

        \mathbf{W} = \\frac{\mathbf{W}}{\sigma(\mathbf{W})}
                

    Refer to `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_ .

    Args:
        weight(${weight_type}): ${weight_comment}
        dim(int): ${dim_comment}
        power_iters(int): ${power_iters_comment}
        eps(float): ${eps_comment}
        name (str): The name of this layer. It is optional.

    Returns:
        Variable: A tensor variable of weight parameters after spectral normalization.

    Examples:
       .. code-block:: python

            import paddle.fluid as fluid

            weight = fluid.layers.data(name='weight', shape=[2, 8, 32, 32], 
                                       append_batch_size=False, dtype='float32')
            x = fluid.layers.spectral_norm(weight=weight, dim=1, power_iters=2)
    """
    helper = LayerHelper('spectral_norm', **locals())
    dtype = weight.dtype

    # create intput and parameters
    inputs = {'Weight': weight}
    input_shape = weight.shape
    h = input_shape[dim]
    w = np.prod(input_shape) // h

    u = helper.create_parameter(
        attr=ParamAttr(),
        shape=[h],
        dtype=dtype,
        default_initializer=Normal(0., 1.))
    u.stop_gradient = True
    inputs['U'] = u
    v = helper.create_parameter(
        attr=ParamAttr(),
        shape=[w],
        dtype=dtype,
        default_initializer=Normal(0., 1.))
    inputs['V'] = v
    v.stop_gradient = True

    # create output
    out = helper.create_variable(dtype=dtype)

    helper.append_op(
        type="spectral_norm",
        inputs=inputs,
        outputs={"Out": out, },
        attrs={
            "dim": dim,
            "power_iters": power_iters,
            "eps": eps,
        })

    return out


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
    `therein <https://arxiv.org/pdf/1603.07285.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    Where:

    * :math:`X`: Input value, a tensor with NCHW format.
    * :math:`W`: Filter value, a tensor with MCHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

           H^\prime_{out} &= (H_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (H_f - 1) + 1 \\\\
           W^\prime_{out} &= (W_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (W_f - 1) + 1 \\\\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[0] ] \\\\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[1] ] 

    Note:
          if output_size is None, :math:`H_{out} = H^\prime_{out}, W_{out} = W^\prime_{out}`; 
          else, the :math:`H_{out}` of the output size must between :math:`H^\prime_{out}` 
          and :math:`H^\prime_{out} + strides[0]`, and the :math:`W_{out}` of the output size must 
          between :math:`W^\prime_{out}` and :math:`W^\prime_{out} + strides[1]`, 
          conv2d_transpose can compute the kernel size automatically.

    Args:
        input(Variable): The input image with [N, C, H, W] format.
        num_filters(int): The number of the filter. It is as same as the output
            image channel.
        output_size(int|tuple|None): The output image size. If output size is a
            tuple, it must contain two integers, (image_height, image_width). None if use
            filter_size, padding, and stride to calculate output_size.
            if output_size and filter_size are specified at the same time, They
            should follow the formula above.
        filter_size(int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_height, filter_size_width).
            Otherwise, filter_size_height = filter_size_width = filter_size. None if 
            use output size to calculate filter_size.
        padding(int|tuple): The padding size. If padding is a tuple, it must
            contain two integers, (padding_height, padding_width). Otherwise, 
            padding_height = padding_width = padding. Default: padding = 0.
        stride(int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_height, stride_width). Otherwise,
            stride_height = stride_width = stride. Default: stride = 1.
        dilation(int|tuple): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_height, dilation_width). Otherwise, 
            dilation_height = dilation_width = dilation. Default: dilation = 1.
        groups(int): The groups number of the Conv2d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups = 1.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv2d_transpose. If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv2d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn(bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: True.

    Returns:
        Variable: The tensor variable storing the convolution transpose result.

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
       .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
          conv2d_transpose = fluid.layers.conv2d_transpose(input=data, num_filters=2, filter_size=3)
    """
    assert param_attr is not False, "param_attr should not be False in conv2d_transpose."
    input_channel = input.shape[1]

    op_type = 'conv2d_transpose'
    if (input_channel == groups and num_filters == input_channel and
            not use_cudnn):
        op_type = 'depthwise_conv2d_transpose'

    helper = LayerHelper(op_type, **locals())
    if not isinstance(input, Variable):
        raise TypeError("Input of conv2d_transpose must be Variable")

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
                         padding[0] - 1) // dilation[0] + 1
        filter_size_w = (output_size[1] - (w_in - 1) * stride[1] + 2 *
                         padding[1] - 1) // dilation[1] + 1
        filter_size = [filter_size_h, filter_size_w]
    else:
        filter_size = utils.convert_to_list(filter_size, 2,
                                            'conv2d_transpose.filter_size')

    if output_size is None:
        output_size = []
    elif isinstance(output_size, list) or isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 2, 'output_size')
    else:
        raise ValueError("output_size should be list or int")
    padding = utils.convert_to_list(padding, 2, 'padding')
    groups = 1 if groups is None else groups
    filter_shape = [input_channel, num_filters // groups] + filter_size

    img_filter = helper.create_parameter(
        dtype=input.dtype, shape=filter_shape, attr=helper.param_attr)

    pre_bias = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=op_type,
        inputs={'Input': [input],
                'Filter': [img_filter]},
        outputs={'Output': pre_bias},
        attrs={
            'output_size': output_size,
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn
        })

    pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    out = helper.append_activation(pre_act)
    return out


def conv3d_transpose(input,
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
    **Convlution3D transpose layer**

    The convolution3D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCDHW format. Where N is batch size, C is the number of channels,
    D is the depth of the feature, H is the height of the feature, and W
    is the width of the feature. Parameters(dilations, strides, paddings) are
    two elements. These two elements represent height and width, respectively.
    The details of convolution transpose layer, please refer to the following
    explanation and references `therein <https://arxiv.org/pdf/1603.07285.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NCDHW format.
    * :math:`W`: Filter value, a tensor with MCDHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, D_f, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        .. math::

           D_{out} &= (D_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (D_f - 1) + 1 \\\\
           H_{out} &= (H_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (H_f - 1) + 1 \\\\
           W_{out} &= (W_{in} - 1) * strides[2] - 2 * paddings[2] + dilations[2] * (W_f - 1) + 1

    Args:
        input(Variable): The input image with [N, C, D, H, W] format.
        num_filters(int): The number of the filter. It is as same as the output
            image channel.
        output_size(int|tuple|None): The output image size. If output size is a
            tuple, it must contain three integers, (image_D, image_H, image_W). This
            parameter only works when filter_size is None.
        filter_size(int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain three integers, (filter_size_depth, filter_size_height, \
            filter_size_width). Otherwise, filter_size_depth = filter_size_height = \
            filter_size_width = filter_size. None if use output size to
            calculate filter_size.
        padding(int|tuple): The padding size. If padding is a tuple, it must
            contain three integers, (padding_depth, padding_height, padding_width). Otherwise,
            padding_depth = padding_height = padding_width = padding. Default: padding = 0.
        stride(int|tuple): The stride size. If stride is a tuple, it must
            contain three integers, (stride_depth, stride_height, stride_width). Otherwise,
            stride_depth = stride_height = stride_width = stride. Default: stride = 1.
        dilation(int|tuple): The dilation size. If dilation is a tuple, it must
            contain three integers, (dilation_depth, dilation_height, dilation_width). Otherwise,
            dilation_depth = dilation_height = dilation_width = dilation. Default: dilation = 1.
        groups(int): The groups number of the Conv3d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups=1
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv3d_transpose. If it is set to None or one attribute of ParamAttr, conv3d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv3d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn(bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The tensor variable storing the convolution transpose result.

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
       .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='data', shape=[3, 12, 32, 32], dtype='float32')
          conv3d_transpose = fluid.layers.conv3d_transpose(input=data, num_filters=2, filter_size=3)
    """
    assert param_attr is not False, "param_attr should not be False in conv3d_transpose."
    l_type = "conv3d_transpose"
    helper = LayerHelper(l_type, **locals())
    if not isinstance(input, Variable):
        raise TypeError("Input of conv3d_transpose must be Variable")
    input_channel = input.shape[1]

    padding = utils.convert_to_list(padding, 3, 'padding')
    stride = utils.convert_to_list(stride, 3, 'stride')
    dilation = utils.convert_to_list(dilation, 3, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    if filter_size is None:
        if output_size is None:
            raise ValueError("output_size must be set when filter_size is None")
        if isinstance(output_size, int):
            output_size = [output_size, output_size]

        d_in = input.shape[2]
        h_in = input.shape[3]
        w_in = input.shape[4]

        filter_size_d = (output_size[0] - (d_in - 1) * stride[0] + 2 *
                         padding[0] - 1) // dilation[0] + 1
        filter_size_h = (output_size[1] - (h_in - 1) * stride[1] + 2 *
                         padding[1] - 1) // dilation[1] + 1
        filter_size_w = (output_size[2] - (w_in - 1) * stride[2] + 2 *
                         padding[2] - 1) // dilation[2] + 1
        filter_size = [filter_size_d, filter_size_h, filter_size_w]
    else:
        filter_size = utils.convert_to_list(filter_size, 3,
                                            'conv3d_transpose.filter_size')

    groups = 1 if groups is None else groups
    filter_shape = [input_channel, num_filters // groups] + filter_size
    img_filter = helper.create_parameter(
        dtype=input.dtype, shape=filter_shape, attr=helper.param_attr)

    pre_bias = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=l_type,
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
                x.lod  = [[2,        2]]
                x.data = [[a], [b], [c], [d]]
                x.dims = [4, 1]

            y is a LoDTensor:
                y.lod = [[2,    2],
                         [3, 3, 1, 1]]

            ref_level: 0

            then output is a 1-level LoDTensor:
                out.lod =  [[2,        2,        2,        2]]
                out.data = [[a], [b], [a], [b], [c], [d], [c], [d]]
                out.dims = [8, 1]

        * Case 2
            x is a Tensor:
                x.data = [[a], [b], [c]]
                x.dims = [3, 1]

            y is a LoDTensor:
                y.lod = [[2, 0, 3]]

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
	
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            x = fluid.layers.data(name='x', shape=[10], dtype='float32')
            y = fluid.layers.data(name='y', shape=[10, 20],
                             dtype='float32', lod_level=1)
            out = layers.sequence_expand(x=x, y=y, ref_level=0)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_expand', input=x, **locals())
    dtype = helper.input_dtype()
    tmp = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='sequence_expand',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': tmp},
        attrs={'ref_level': ref_level})
    return tmp


def sequence_expand_as(x, y, name=None):
    """Sequence Expand As Layer. This layer will expand the input variable **x**
    according to the zeroth level lod of **y**. Current implementation requires
    the level number of Input(Y)'s lod must be 1, and the first dimension of
    Input(X) should be equal to the size of Input(Y)'s zeroth level lod, and
    lod of Input(X) is not considered.

    Following examples will explain how sequence_expand_as works:

    .. code-block:: text

        * Case 1:

            Given a 1-level LoDTensor input(X)
                X.data = [[a], [b], [c], [d]]
                X.dims = [4, 1]
            and input(Y)
                Y.lod = [[0, 3, 6, 7, 8]]
            ref_level: 0
            then we get 1-level LoDTensor
                Out.lod =  [[0,            3,              6,  7,  8]]
                Out.data = [[a], [a], [a], [b], [b], [b], [c], [d]]
                Out.dims = [8, 1]

        * Case 2:

            Given a common Tensor input(X)
                X.data = [[a, b], [c, d], [e, f]]
                X.dims = [3, 2]
            and input(Y)
                Y.lod = [[0, 2, 3, 6]]
            ref_level: 0
            then we get a common LoDTensor
                Out.lod =  [[0,             2,     3,                    6]]
                Out.data = [[a, b], [a, b] [c, d], [e, f], [e, f], [e, f]]
                Out.dims = [6, 2]

    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor.
        y (Variable): The input variable which is a LoDTensor.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The expanded variable which is a LoDTensor.

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            x = fluid.layers.data(name='x', shape=[10], dtype='float32')
            y = fluid.layers.data(name='y', shape=[10, 20],
                             dtype='float32', lod_level=1)
            out = layers.sequence_expand_as(x=x, y=y)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_expand_as', input=x, **locals())
    dtype = helper.input_dtype()
    tmp = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='sequence_expand_as',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': tmp})
    return tmp


@templatedoc()
def sequence_pad(x, pad_value, maxlen=None, name=None):
    """
    ${comment}

    Args:
        x(Variable): Input variable which should contain lod information.
        pad_value(Variable): The Variable that holds values that will be fill
            into padded steps. It can be a scalar or a tensor whose shape
            equals to time steps in sequences. If it's a scalar, it will be
            automatically broadcasted to the shape of time step.
        maxlen(int, default None): The length of padded sequences. It can be
            None or any positive int. When it is None, all sequences will be
            padded up to the length of the longest one among them; when it a
            certain positive value, it must be greater than the length of the
            longest original sequence.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The padded sequence batch and the original lengths before
                  padding. All sequences has the same length.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy

            x = fluid.layers.data(name='x', shape=[10, 5],
                             dtype='float32', lod_level=1)
            pad_value = fluid.layers.assign(
                input=numpy.array([0.0], dtype=numpy.float32))
            out = fluid.layers.sequence_pad(x=x, pad_value=pad_value)
    """

    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_pad', input=x, **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    length = helper.create_variable_for_type_inference(dtype)

    pad_value.stop_gradient = True
    length.stop_gradient = True

    if maxlen is None:
        maxlen = -1
    helper.append_op(
        type='sequence_pad',
        inputs={'X': x,
                'PadValue': pad_value},
        outputs={'Out': out,
                 'Length': length},
        attrs={'padded_length': maxlen})
    return out, length


def sequence_unpad(x, length, name=None):
    """
    **Sequence Unpad Layer**

    This layer removes the padding data in the input sequences and convert
    them into sequences with actual length as output, identitied by lod
    information.

    .. code-block:: text

	Example:

	Given input Variable **x**:
	    x.data = [[ 1.0,  2.0,  3.0,  4.0,  5.0],
		      [ 6.0,  7.0,  8.0,  9.0, 10.0],
		      [11.0, 12.0, 13.0, 14.0, 15.0]],

	in which there are 3 sequences padded to length 5, and the acutal length
	specified by input Variable **length**:

	    length.data = [2, 3, 4],

	after unpadding, the output Variable will be:

	    out.data = [[1.0, 2.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0]]
	    out.lod = [[2, 3, 4]]

    Args:
        x(Variable): Input Variable which contains the padded sequences with
            equal length.
        length(Variable): The Variable that specifies the actual ength of
            sequences after unpadding.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The Variable contains the unpadded sequences.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy

            # pad data
            x = fluid.layers.data(name='x', shape=[10, 5], dtype='float32', lod_level=1)
            pad_value = fluid.layers.assign(input=numpy.array([0.0], dtype=numpy.float32))
            pad_data, len = fluid.layers.sequence_pad(x=x, pad_value=pad_value)
            
            # upad data
            unpad_data = fluid.layers.sequence_unpad(x=pad_data, length=len)
    """

    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_unpad', input=x, **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)

    length.stop_gradient = True

    helper.append_op(
        type='sequence_unpad',
        inputs={'X': x,
                'Length': length},
        outputs={'Out': out})
    return out


def beam_search(pre_ids,
                pre_scores,
                ids,
                scores,
                beam_size,
                end_id,
                level=0,
                is_accumulated=True,
                name=None,
                return_parent_idx=False):
    """
    Beam search is a classical algorithm for selecting candidate words in a
    machine translation task.

    Refer to `Beam search <https://en.wikipedia.org/wiki/Beam_search>`_
    for more details.

    This layer does the search in beams for one time step. Specifically, it
    selects the top-K candidate word ids of current step from :attr:`ids`
    according to their :attr:`scores` for all source sentences, where K is
    :attr:`beam_size` and :attr:`ids, scores` are predicted results from the
    computation cell. If :attr:`ids` is not set, it will be calculated out
    according to :attr:`scores`. Additionally, :attr:`pre_ids` and
    :attr:`pre_scores` are the output of beam_search at previous step, they
    are needed for special use to handle ended candidate translations.

    Note that if :attr:`is_accumulated` is :attr:`True`, the :attr:`scores`
    passed in should be accumulated scores. Else, the :attr:`scores` are
    considered as the straightforward scores and will be transformed to the
    log field and accumulated the :attr:`pre_scores` in this operator.
    Length penalty should be done with extra operators before calculating the
    accumulated scores if needed.

    Please see the following demo for a fully beam search usage example:

        fluid/tests/book/test_machine_translation.py

    Args:
        pre_ids(Variable): The LodTensor variable which is the output of
            beam_search at previous step. It should be a LodTensor with shape
            :math:`(batch_size, 1)` and lod
            :math:`[[0, 1, ... , batch_size], [0, 1, ..., batch_size]]` at the
            first step.
        pre_scores(Variable): The LodTensor variable which is the output of
            beam_search at previous step.
        ids(Variable): The LodTensor variable containing the candidates ids.
            Its shape should be :math:`(batch_size \\times beam_size, K)`,
            where :math:`K` supposed to be :attr:`beam_size`.
        scores(Variable): The LodTensor variable containing the accumulated
            scores corresponding to :attr:`ids` and its shape is the same as
            the shape of :attr:`ids`.
        beam_size(int): The beam width used in beam search.
        end_id(int): The id of end token.
        level(int, default 0): It can be ignored and mustn't change currently.
            It means the source level of lod, which is explained as following.
            The lod level of :attr:`ids` should be 2. The first level is source
            level which describes how many prefixes (branchs) for each source
            sentece (beam), and the second level is sentence level which
            describes how these candidates belong to the prefix. The paths
            linking prefixes and selected candidates are organized and reserved
            in lod.
        is_accumulated(bool, default True): Whether the input :attr:`score` is
             accumulated scores.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.
        return_parent_idx(bool): Whether to return an extra Tensor variable 
                        preserving the selected_ids' parent indice in pre_ids
                        in output, which can be used to gather cell states at
                        the next time step.

    Returns:
        Variable: The LodTensor tuple containing the selected ids and the \
            corresponding scores. If :attr:`return_parent_idx` is :attr:`True`, \
            an extra Tensor variable preserving the selected_ids' parent indice \
            is included.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            # Suppose `probs` contains predicted results from the computation
            # cell and `pre_ids` and `pre_scores` is the output of beam_search
            # at previous step.
            beam_size = 4
            end_id = 1
            pre_ids = fluid.layers.data(
                name='pre_id', shape=[1], lod_level=2, dtype='int64')
            pre_scores = fluid.layers.data(
                name='pre_scores', shape=[1], lod_level=2, dtype='float32')
            probs = fluid.layers.data(
                name='probs', shape=[10000], dtype='float32')
            topk_scores, topk_indices = fluid.layers.topk(probs, k=beam_size)
            accu_scores = fluid.layers.elementwise_add(
                x=fluid.layers.log(x=topk_scores),
                y=fluid.layers.reshape(pre_scores, shape=[-1]),
                axis=0)
            selected_ids, selected_scores = fluid.layers.beam_search(
                pre_ids=pre_ids,
                pre_scores=pre_scores,
                ids=topk_indices,
                scores=accu_scores,
                beam_size=beam_size,
                end_id=end_id)
    """
    helper = LayerHelper('beam_search', **locals())
    score_type = pre_scores.dtype
    id_type = pre_ids.dtype

    inputs = {"pre_ids": pre_ids, "pre_scores": pre_scores, "scores": scores}
    if ids is not None:
        inputs["ids"] = ids

    selected_scores = helper.create_variable_for_type_inference(
        dtype=score_type)
    selected_ids = helper.create_variable_for_type_inference(dtype=id_type)
    # parent_idx is a tensor used to gather cell states at the next time
    # step. Though lod in selected_ids can also be used to gather by
    # sequence_expand, it is not efficient.
    # gather_op's index input only supports int32 dtype currently
    parent_idx = helper.create_variable_for_type_inference(dtype="int32")

    helper.append_op(
        type='beam_search',
        inputs=inputs,
        outputs={
            'selected_ids': selected_ids,
            'selected_scores': selected_scores,
            'parent_idx': parent_idx
        },
        attrs={
            # TODO(ChunweiYan) to assure other value support
            'level': level,
            'beam_size': beam_size,
            'end_id': end_id,
            'is_accumulated': is_accumulated,
        })
    if return_parent_idx:
        return selected_ids, selected_scores, parent_idx
    else:
        return selected_ids, selected_scores


def beam_search_decode(ids, scores, beam_size, end_id, name=None):
    """
    Beam Search Decode Layer. This layer constructs the full hypotheses for
    each source sentence by walking back along the LoDTensorArray :attr:`ids`
    whose lods can be used to restore the path in the beam search tree.
    Please see the following demo for a fully beam search usage example:
        fluid/tests/book/test_machine_translation.py

    Args:
        ids(Variable): The LodTensorArray variable containing the selected ids
            of all steps.
        scores(Variable): The LodTensorArray variable containing the selected
            scores of all steps.
        beam_size(int): The beam width used in beam search.
        end_id(int): The id of end token.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The LodTensor pair containing the generated id sequences \
            and the corresponding scores. The shapes and lods of the two \
            LodTensor are same. The lod level is 2 and the two levels \
            separately indicate how many hypotheses each source sentence has \
            and how many ids each hypothesis has.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            # Suppose `ids` and `scores` are LodTensorArray variables reserving
            # the selected ids and scores of all steps
            ids = fluid.layers.create_array(dtype='int64')
            scores = fluid.layers.create_array(dtype='float32')
            finished_ids, finished_scores = fluid.layers.beam_search_decode(
                ids, scores, beam_size=5, end_id=0)
    """
    helper = LayerHelper('beam_search_decode', **locals())
    sentence_ids = helper.create_variable_for_type_inference(dtype=ids.dtype)
    sentence_scores = helper.create_variable_for_type_inference(dtype=ids.dtype)

    helper.append_op(
        type="beam_search_decode",
        inputs={"Ids": ids,
                "Scores": scores},
        outputs={
            "SentenceIds": sentence_ids,
            "SentenceScores": sentence_scores
        },
        attrs={"beam_size": beam_size,
               "end_id": end_id})

    return sentence_ids, sentence_scores


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

    This layer has two outputs including :math:`h_t` and :math:`c_t`.

    Args:
        x_t (Variable): The input value of current step, a 2-D tensor with shape
            M x N, M for batch size and N for input size.
        hidden_t_prev (Variable): The hidden value of lstm unit, a 2-D tensor
            with shape M x S, M for batch size and S for size of lstm unit.
        cell_t_prev (Variable): The cell value of lstm unit, a 2-D tensor with
            shape M x S, M for batch size and S for size of lstm unit.
        forget_bias (float): The forget bias of lstm unit.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
                               hidden-hidden weights.
                               If it is set to None or one attribute of ParamAttr,
                               lstm_unit will create ParamAttr as param_attr.
                               If the Initializer of the param_attr is not set, the
                               parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The bias attribute for the learnable bias
                              weights. If it is set to False, no bias will be added
                              to the output units. If it is set to None or one attribute of ParamAttr,
                              lstm_unit will create ParamAttr as bias_attr.
                              If the Initializer of the bias_attr is not set,
                              the bias is initialized zero. Default: None.
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

            import paddle.fluid as fluid

            dict_dim, emb_dim, hidden_dim = 128, 64, 512
            data = fluid.layers.data(name='step_data', shape=[1], dtype='int32')
            x = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
            pre_hidden = fluid.layers.data(
                name='pre_hidden', shape=[hidden_dim], dtype='float32')
            pre_cell = fluid.layers.data(
                name='pre_cell', shape=[hidden_dim], dtype='float32')
            hidden = fluid.layers.lstm_unit(
                x_t=x,
                hidden_t_prev=pre_hidden,
                cell_t_prev=pre_cell)
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
    c = helper.create_variable_for_type_inference(dtype)
    h = helper.create_variable_for_type_inference(dtype)

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

            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.layers.data(name='x', shape=[4, 2], dtype='float32')
            fluid.layers.reduce_sum(x)  # [3.5]
            fluid.layers.reduce_sum(x, dim=0)  # [0.3, 0.5, 1.1, 1.6]
            fluid.layers.reduce_sum(x, dim=-1)  # [1.9, 1.6]
            fluid.layers.reduce_sum(x, dim=1, keep_dim=True)  # [[1.9], [1.6]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1, 2], [3, 4]],
            #      [[5, 6], [7, 8]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.layers.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_sum(y, dim=[1, 2]) # [10, 26]
            fluid.layers.reduce_sum(y, dim=[0, 1]) # [16, 20]

    """
    helper = LayerHelper('reduce_sum', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
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
    Computes the mean of the input tensor's elements along the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (list|int|None): The dimension along which the mean is computed. If
            `None`, compute the mean over all elements of :attr:`input`
            and return a variable with a single element, otherwise it
            must be in the range :math:`[-rank(input), rank(input))`. If
            :math:`dim[i] < 0`, the dimension to reduce is
            :math:`rank(input) + dim[i]`.
        keep_dim (bool): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true.
        name(str|None): A name for this layer(optional). If set `None`, the layer
                       will be named automatically.

    Returns:
        Variable: The reduced mean Variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            x = fluid.layers.data(name='x', shape=[4, 2], dtype='float32')
            fluid.layers.reduce_mean(x)  # [0.4375]
            fluid.layers.reduce_mean(x, dim=0)  # [0.15, 0.25, 0.55, 0.8]
            fluid.layers.reduce_mean(x, dim=-1)  # [0.475, 0.4]
            fluid.layers.reduce_mean(x, dim=1, keep_dim=True)  # [[0.475], [0.4]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            y = fluid.layers.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_mean(y, dim=[1, 2]) # [2.5, 6.5]
            fluid.layers.reduce_mean(y, dim=[0, 1]) # [4.0, 5.0]
    """
    helper = LayerHelper('reduce_mean', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
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

            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            x = fluid.layers.data(name='x', shape=[4, 2], dtype='float32')
            fluid.layers.reduce_max(x)  # [0.9]
            fluid.layers.reduce_max(x, dim=0)  # [0.2, 0.3, 0.6, 0.9]
            fluid.layers.reduce_max(x, dim=-1)  # [0.9, 0.7]
            fluid.layers.reduce_max(x, dim=1, keep_dim=True)  # [[0.9], [0.7]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            y = fluid.layers.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_max(y, dim=[1, 2]) # [4.0, 8.0]
            fluid.layers.reduce_max(y, dim=[0, 1]) # [7.0, 8.0]
    """
    helper = LayerHelper('reduce_max', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
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

            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            x = fluid.layers.data(name='x', shape=[4, 2], dtype='float32')
            fluid.layers.reduce_min(x)  # [0.1]
            fluid.layers.reduce_min(x, dim=0)  # [0.1, 0.2, 0.5, 0.7]
            fluid.layers.reduce_min(x, dim=-1)  # [0.2, 0.1]
            fluid.layers.reduce_min(x, dim=1, keep_dim=True)  # [[0.2], [0.1]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            y = fluid.layers.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_min(y, dim=[1, 2]) # [1.0, 5.0]
            fluid.layers.reduce_min(y, dim=[0, 1]) # [1.0, 2.0]
    """
    helper = LayerHelper('reduce_min', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
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

            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            x = fluid.layers.data(name='x', shape=[4, 2], dtype='float32')
            fluid.layers.reduce_prod(x)  # [0.0002268]
            fluid.layers.reduce_prod(x, dim=0)  # [0.02, 0.06, 0.3, 0.63]
            fluid.layers.reduce_prod(x, dim=-1)  # [0.027, 0.0084]
            fluid.layers.reduce_prod(x, dim=1,
                                     keep_dim=True)  # [[0.027], [0.0084]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            y = fluid.layers.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_prod(y, dim=[1, 2]) # [24.0, 1680.0]
            fluid.layers.reduce_prod(y, dim=[0, 1]) # [105.0, 384.0]
    """
    helper = LayerHelper('reduce_prod', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
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


def reduce_all(input, dim=None, keep_dim=False, name=None):
    """
    Computes the ``logical and`` of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (list|int|None): The dimension along which the logical and is computed.
            If :attr:`None`, compute the logical and over all elements of
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
        
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            import numpy as np

            # x is a bool Tensor variable with following elements:
            #    [[True, False]
            #     [True, True]]
            x = layers.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
            x = layers.cast(x, 'bool')

            out = layers.reduce_all(x)  # False 
            out = layers.reduce_all(x, dim=0)  # [True, False]
            out = layers.reduce_all(x, dim=-1)  # [False, True]
            out = layers.reduce_all(x, dim=1, keep_dim=True)  # [[False], [True]]

    """
    helper = LayerHelper('reduce_all', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_all',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out


def reduce_any(input, dim=None, keep_dim=False, name=None):
    """
    Computes the ``logical or`` of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (list|int|None): The dimension along which the logical or is computed.
            If :attr:`None`, compute the logical or over all elements of
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

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            import numpy as np

            # x is a bool Tensor variable with following elements:
            #    [[True, False]
            #     [False, False]]
            x = layers.assign(np.array([[1, 0], [0, 0]], dtype='int32'))
            x = layers.cast(x, 'bool')

            out = layers.reduce_any(x)  # True
            out = layers.reduce_any(x, dim=0)  # [True, False]
            out = layers.reduce_any(x, dim=-1)  # [True, False]
            out = layers.reduce_any(x, dim=1,
                                     keep_dim=True)  # [[True], [False]]

    """
    helper = LayerHelper('reduce_any', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_any',
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
        list(Variable): The list of segmented tensor variables.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            # input is a variable which shape is [-1, 3, 9, 5]
            input = fluid.layers.data(
                 name="input", shape=[3, 9, 5], dtype="float32")

            x0, x1, x2 = fluid.layers.split(input, num_or_sections=3, dim=2)
            # x0.shape [-1, 3, 3, 5]
            # x1.shape [-1, 3, 3, 5]
            # x2.shape [-1, 3, 3, 5]

            x0, x1, x2 = fluid.layers.split(input, num_or_sections=3, dim=2)
            # x0.shape [-1, 3, 2, 5]
            # x1.shape [-1, 3, 3, 5]
            # x2.shape [-1, 3, 4, 5]
    """
    helper = LayerHelper('split', **locals())
    input_shape = input.shape
    dim = (len(input_shape) + dim) if dim < 0 else dim
    if isinstance(num_or_sections, int):
        assert num_or_sections > 1, 'num_or_sections must be more than 1.'
        num = num_or_sections
    else:
        assert len(num_or_sections) <= input_shape[
            dim], 'len(num_or_sections) must not be more than input.shape[dim].'
        num = len(num_or_sections)
    outs = [
        helper.create_variable_for_type_inference(dtype=helper.input_dtype())
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

    .. math::

        y = \\frac{x}{ \sqrt{\sum {x^2} + epsion }}

    For `x` with more dimensions, this layer independently normalizes each 1-D
    slice along dimension `axis`.

    Args:
        x(Variable|list): The input tensor to l2_normalize layer.
        axis(int): The axis on which to apply normalization. If `axis < 0`, \
            the dimension to normalization is rank(X) + axis. -1 is the
            last dimension.
        epsilon(float): The epsilon value is used to avoid division by zero, \
            the default value is 1e-12.
        name(str|None): A name for this layer(optional). If set None, the layer \
            will be named automatically.

    Returns:
        Variable: The output tensor variable is the same shape with `x`.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.layers.data(name="data",
                                     shape=(3, 17, 13),
                                     dtype="float32")
            normed = fluid.layers.l2_normalize(x=data, axis=1)
    """

    if len(x.shape) == 1:
        axis = 0
    helper = LayerHelper("l2_normalize", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    norm = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="norm",
        inputs={"X": x},
        outputs={"Out": out,
                 "Norm": norm},
        attrs={
            "axis": 1 if axis is None else axis,
            "epsilon": epsilon,
        })
    return out


def matmul(x, y, transpose_x=False, transpose_y=False, alpha=1.0, name=None):
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
        alpha (float): The scale of output. Default 1.0.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The product Tensor (or LoDTensor) variable.

    Examples:
        .. code-block:: python

            # Examples to clarify shapes of the inputs and output
            # x: [B, ..., M, K], y: [B, ..., K, N]
            # fluid.layers.matmul(x, y)  # out: [B, ..., M, N]

            # x: [B, M, K], y: [B, K, N]
            # fluid.layers.matmul(x, y)  # out: [B, M, N]

            # x: [B, M, K], y: [K, N]
            # fluid.layers.matmul(x, y)  # out: [B, M, N]

            # x: [M, K], y: [K, N]
            # fluid.layers.matmul(x, y)  # out: [M, N]

            # x: [B, M, K], y: [K]
            # fluid.layers.matmul(x, y)  # out: [B, M]

            # x: [K], y: [K]
            # fluid.layers.matmul(x, y)  # out: [1]

            # x: [M], y: [N]
            # fluid.layers.matmul(x, y, True, True)  # out: [M, N]

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[2, 3], dtype='float32')
            y = fluid.layers.data(name='y', shape=[3, 2], dtype='float32')
            out = fluid.layers.matmul(x, y, True, True)
    """

    def __check_input(x, y):
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
            raise ValueError("Invalid inputs for matmul. x: %s, y: %s\n" %
                             (x_shape, y_shape))

        if len(y_shape) > 2 and len(x_shape) > 2:
            for i, dim_x in enumerate(x_shape[:-2]):
                # don't check neg shape
                if dim_x < 0 or y_shape[i] < 0:
                    continue
                if dim_x != y_shape[i]:
                    raise ValueError("Invalid inputs for matmul. x(%s), y(%s)" %
                                     (x.shape, y.shape))

    __check_input(x, y)

    helper = LayerHelper('matmul', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='matmul',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={
            'transpose_X': transpose_x,
            'transpose_Y': transpose_y,
            'alpha': float(alpha),
        })
    return out


def topk(input, k, name=None):
    """
    This operator is used to find values and indices of the k largest entries
    for the last dimension.

    If the input is a vector (1-D Tensor), finds the k largest entries in the vector
    and outputs their values and indices as vectors. Thus values[j] is the j-th
    largest entry in input, and its index is indices[j].

    If the input is a Tensor with higher rank, this operator computes the top k
    entries along the last dimension.

    For example:

    .. code-block:: text

        If:
            input = [[5, 4, 2, 3],
                     [9, 7, 10, 25],
                     [6, 2, 10, 1]]
            k = 2

        Then:
            The first output:
            values = [[5, 4],
                      [10, 25],
                      [6, 10]]

            The second output:
            indices = [[0, 1],
                       [2, 3],
                       [0, 2]]

    Args:
        input(Variable): The input variable which can be a vector or Tensor with
            higher rank.
        k(int | Variable):  The number of top elements to look for along the last dimension
                 of input.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.
                       Default: None

    Returns:
        Tuple[Variable]: A tuple with two elements. Each element is a Variable.
        The first one is k largest elements along each last
        dimensional slice. The second one is indices of values
        within the last dimension of input.

    Raises:
        ValueError: If k < 1 or k is not less than the last dimension of input

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            input = layers.data(name="input", shape=[13, 11], dtype='float32')
            top5_values, top5_indices = layers.topk(input, k=5)
    """
    helper = LayerHelper("top_k", **locals())
    values = helper.create_variable_for_type_inference(dtype=input.dtype)
    indices = helper.create_variable_for_type_inference(dtype="int64")
    inputs = {"X": [input]}
    attrs = None
    if isinstance(k, Variable):
        inputs['K'] = k
    else:
        attrs = {'k': k}
    helper.append_op(
        type="top_k",
        inputs=inputs,
        outputs={"Out": [values],
                 "Indices": [indices]},
        attrs=attrs)
    values.stop_gradient = True
    indices.stop_gradient = True
    return values, indices


def edit_distance(input,
                  label,
                  normalized=True,
                  ignored_tokens=None,
                  input_length=None,
                  label_length=None):
    """
    Edit distance operator computes the edit distances between a batch of
    hypothesis strings and their references. Edit distance, also called
    Levenshtein distance, measures how dissimilar two strings are by counting
    the minimum number of operations to transform one string into anthor.
    Here the operations include insertion, deletion, and substitution.

    For example, given hypothesis string A = "kitten" and reference
    B = "sitting", the edit distance is 3 for A will be transformed into B
    at least after two substitutions and one insertion:

    "kitten" -> "sitten" -> "sittin" -> "sitting"

    The input is a LoDTensor/Tensor consisting of all the hypothesis strings with
    the total number denoted by `batch_size`, and the separation is specified
    by the LoD information or input_length. And the `batch_size` reference strings are arranged
    in order in the same way as `input`.

    The output contains the `batch_size` results and each stands for the edit
    distance for a pair of strings respectively. If Attr(normalized) is true,
    the edit distance will be divided by the length of reference string.

    Args:
        input(Variable): The indices for hypothesis strings, it should have rank 2 and dtype int64.
        label(Variable): The indices for reference strings, it should have rank 2 and dtype int64.
        normalized(bool, default True): Indicated whether to normalize the edit distance by
                          the length of reference string.
        ignored_tokens(list<int>, default None): Tokens that should be removed before
                                     calculating edit distance.
        input_length(Variable): The length for each sequence in `input` if it's of Tensor type, it should have shape `[batch_size]` and dtype int64.
        label_length(Variable): The length for each sequence in `label` if it's of Tensor type, it should have shape `[batch_size]` and dtype int64.

    Returns:
        edit_distance_out(Variable): edit distance result in shape [batch_size, 1]. \n
        sequence_num(Variable): sequence number in shape [].
        

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid

            # using LoDTensor
            x_lod = fluid.layers.data(name='x_lod', shape=[1], dtype='int64', lod_level=1)
            y_lod = fluid.layers.data(name='y_lod', shape=[1], dtype='int64', lod_level=1)
            distance_lod, seq_num_lod = fluid.layers.edit_distance(input=x_lod, label=y_lod)

            # using Tensor
            x_seq_len = 5
            y_seq_len = 6
            x_pad = fluid.layers.data(name='x_pad', shape=[x_seq_len], dtype='int64')
            y_pad = fluid.layers.data(name='y_pad', shape=[y_seq_len], dtype='int64')
            x_len = fluid.layers.data(name='x_len', shape=[], dtype='int64')
            y_len = fluid.layers.data(name='y_len', shape=[], dtype='int64')
            distance_pad, seq_num_pad = fluid.layers.edit_distance(input=x_pad, label=y_pad, input_length=x_len, label_length=y_len)

    """
    helper = LayerHelper("edit_distance", **locals())

    # remove some tokens from input and labels
    if ignored_tokens is not None and len(ignored_tokens) > 0:
        erased_input = helper.create_variable_for_type_inference(dtype="int64")
        erased_label = helper.create_variable_for_type_inference(dtype="int64")

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

    this_inputs = {"Hyps": [input], "Refs": [label]}
    if input_length and label_length:
        this_inputs['HypsLength'] = [input_length]
        this_inputs['RefsLength'] = [label_length]

    # edit distance op
    edit_distance_out = helper.create_variable_for_type_inference(dtype="int64")
    sequence_num = helper.create_variable_for_type_inference(dtype="int64")
    helper.append_op(
        type="edit_distance",
        inputs=this_inputs,
        outputs={"Out": [edit_distance_out],
                 "SequenceNum": [sequence_num]},
        attrs={"normalized": normalized})

    return edit_distance_out, sequence_num


def ctc_greedy_decoder(input,
                       blank,
                       input_length=None,
                       padding_value=0,
                       name=None):
    """
    This op is used to decode sequences by greedy policy by below steps:

    1. Get the indexes of max value for each row in input. a.k.a.
       numpy.argmax(input, axis=0).
    2. For each sequence in result of step1, merge repeated tokens between two
       blanks and delete all blanks.

    A simple example as below:

    .. code-block:: text

        Given:
        for lod mode:

        input.data = [[0.6, 0.1, 0.3, 0.1],
                      [0.3, 0.2, 0.4, 0.1],
                      [0.1, 0.5, 0.1, 0.3],
                      [0.5, 0.1, 0.3, 0.1],

                      [0.5, 0.1, 0.3, 0.1],
                      [0.2, 0.2, 0.2, 0.4],
                      [0.2, 0.2, 0.1, 0.5],
                      [0.5, 0.1, 0.3, 0.1]]

        input.lod = [[4, 4]]

        Computation:

        step1: Apply argmax to first input sequence which is input.data[0:4]. Then we get:
               [[0], [2], [1], [0]]
        step2: merge repeated tokens and remove blank which is 0. Then we get first output sequence:
               [[2], [1]]

        Finally:

        output.data = [[2],
                       [1],
                       [3]]

        output.lod = [[2, 1]]

        for padding mode:

         input.data = [[[0.6, 0.1, 0.3, 0.1],
                        [0.3, 0.2, 0.4, 0.1],
                        [0.1, 0.5, 0.1, 0.3],
                        [0.5, 0.1, 0.3, 0.1]],

                       [[0.5, 0.1, 0.3, 0.1],
                        [0.2, 0.2, 0.2, 0.4],
                        [0.2, 0.2, 0.1, 0.5],
                        [0.5, 0.1, 0.3, 0.1]]]

        input_length.data = [[4], [4]]
        input.shape = [2, 4, 4]

        step1: Apply argmax to first input sequence which is input.data[0:4]. Then we get:
               [[0], [2], [1], [0]], for input.data[4:8] is [[0], [3], [3], [0]], shape is [2,4,1]
        step2: Change the argmax result to use padding mode, then argmax result is 
                [[0, 2, 1, 0], [0, 3, 3, 0]], shape is [2, 4], lod is [], input_length is [[4], [4]]
        step3: Apply ctc_align to padding argmax result, padding_value is 0

        Finally:
        output.data = [[2, 1, 0, 0],
                       [3, 0, 0, 0]]
        output_length.data = [[2], [1]]




    Args:

        input(Variable): (LoDTensor<float>), the probabilities of
                         variable-length sequences. When in lod mode, it is a 2-D Tensor with
                         LoD information. It's shape is [Lp, num_classes + 1] 
                         where Lp is the sum of all input sequences' length and
                         num_classes is the true number of classes. When in padding mode,
                         it is a 3-D Tensor with padding, It's shape is [batch_size, N, num_classes + 1].
                         (not including the blank label).
        blank(int): the blank label index of Connectionist Temporal
                    Classification (CTC) loss, which is in thehalf-opened
                    interval [0, num_classes + 1).
        input_length(Variable, optional): (LoDTensor<int>), shape is [batch_size, 1], when in lod mode, input_length
                                 is None.
        padding_value(int): padding value.
        name (str, optional): The name of this layer. It is optional.

    Returns:
        output(Variable): For lod mode, CTC greedy decode result which is a 2-D tensor with shape [Lp, 1]. \
                  'Lp' is the sum if all output sequences' length. If all the sequences \
                  in result were empty, the result LoDTensor will be [-1] with  \
                  LoD [[]] and dims [1, 1]. For padding mode, CTC greedy decode result is a 2-D tensor \
                  with shape [batch_size, N], output length's shape is [batch_size, 1] which is length \
                  of every sequence in output.
        output_length(Variable, optional): length of each sequence of output for padding mode.

    Examples:
        .. code-block:: python

            # for lod mode
            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[8], dtype='float32')
            cost = fluid.layers.ctc_greedy_decoder(input=x, blank=0)

            # for padding mode
            x_pad = fluid.layers.data(name='x_pad', shape=[4,8], dtype='float32')
            x_pad_len = fluid.layers.data(name='x_pad_len', shape=[1], dtype='int64')
            out, out_len = fluid.layers.ctc_greedy_decoder(input=x_pad, blank=0,
                            input_length=x_pad_len)

    """
    helper = LayerHelper("ctc_greedy_decoder", **locals())
    _, topk_indices = topk(input, k=1)

    # ctc align op
    ctc_out = helper.create_variable_for_type_inference(dtype="int64")

    if input_length is None:
        helper.append_op(
            type="ctc_align",
            inputs={"Input": [topk_indices]},
            outputs={"Output": [ctc_out]},
            attrs={"merge_repeated": True,
                   "blank": blank})
        return ctc_out
    else:
        ctc_out_len = helper.create_variable_for_type_inference(dtype="int64")
        ctc_input = squeeze(topk_indices, [2])

        helper.append_op(
            type="ctc_align",
            inputs={"Input": [ctc_input],
                    "InputLength": [input_length]},
            outputs={"Output": [ctc_out],
                     "OutputLength": [ctc_out_len]},
            attrs={
                "merge_repeated": True,
                "blank": blank,
                "padding_value": padding_value
            })
        return ctc_out, ctc_out_len


def warpctc(input,
            label,
            blank=0,
            norm_by_times=False,
            input_length=None,
            label_length=None):
    """
    An operator integrating the open source Warp-CTC library
    (https://github.com/baidu-research/warp-ctc)
    to compute Connectionist Temporal Classification (CTC) loss.
    It can be aliased as softmax with CTC, since a native softmax activation is
    interated to the Warp-CTC library, to to normlize values for each row of the
    input tensor.

    Args:
       input (Variable): The unscaled probabilities of variable-length sequences,
         which is a 2-D Tensor with LoD information, or a 3-D Tensor without Lod
         information. When it is a 2-D LodTensor, it's shape is 
         [Lp, num_classes + 1], where Lp is the sum of all input
         sequences' length and num_classes is the true number of classes.
         (not including the blank label). When it is a 3-D Tensor, it's shape 
         is [max_logit_length, batch_size, num_classes + 1],
         where max_logit_length is the length of the longest
         input logit sequence.
       label (Variable): The ground truth of variable-length sequence,
         which is a 2-D Tensor with LoD information or a 2-D Tensor without
         LoD information. When it is a 2-D LoDTensor or 2-D Tensor, 
         it is of the shape [Lg, 1], where Lg is th sum of all labels' length.
       blank (int, default 0): The blank label index of Connectionist
         Temporal Classification (CTC) loss, which is in the
         half-opened interval [0, num_classes + 1).
       norm_by_times(bool, default false): Whether to normalize the gradients
         by the number of time-step, which is also the sequence's length.
         There is no need to normalize the gradients if warpctc layer was
         follewed by a mean_op.
       input_length(Variable): The length for each input sequence if it is 
         of Tensor type, it should have shape `[batch_size]` and dtype int64.
       label_length(Variable): The length for each label sequence if it is
         of Tensor type, it should have shape `[batch_size]` and dtype int64.

    Returns:
        Variable: The Connectionist Temporal Classification (CTC) loss,
        which is a 2-D Tensor of the shape [batch_size, 1].

    Examples:
        .. code-block:: python

            # using LoDTensor
            import paddle.fluid as fluid
            import numpy as np
            
            label = fluid.layers.data(name='label', shape=[12, 1],
                                      dtype='float32', lod_level=1)
            predict = fluid.layers.data(name='predict', 
                                        shape=[11, 8],
                                        dtype='float32',lod_level=1)
            cost = fluid.layers.warpctc(input=predict, label=label)

            # using Tensor
            input_length = fluid.layers.data(name='logits_length', shape=[11],
                                         dtype='int64')
            label_length = fluid.layers.data(name='labels_length', shape=[12],
                                         dtype='int64')
            target = fluid.layers.data(name='target', shape=[12, 1],
                                       dtype='int32')
            # length of the longest logit sequence
            max_seq_length = 4
            # number of logit sequences
            batch_size = 4
            output = fluid.layers.data(name='output', 
                                       shape=[max_seq_length, batch_size, 8],
                                       dtype='float32')
            loss = fluid.layers.warpctc(input=output,label=target,
                                        input_length=input_length,
                                        label_length=label_length)

    """
    helper = LayerHelper('warpctc', **locals())
    this_inputs = {'Logits': [input], 'Label': [label]}
    if input_length and label_length:
        this_inputs['LogitsLength'] = [input_length]
        this_inputs['LabelLength'] = [label_length]

    loss_out = helper.create_variable_for_type_inference(dtype=input.dtype)
    grad_out = helper.create_variable_for_type_inference(dtype=input.dtype)

    helper.append_op(
        type='warpctc',
        inputs=this_inputs,
        outputs={'WarpCTCGrad': [grad_out],
                 'Loss': [loss_out]},
        attrs={
            'blank': blank,
            'norm_by_times': norm_by_times,
        })
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
            x.data = [[1,  2], [3,  4],
                      [5,  6], [7,  8],
                      [9, 10], [11, 12]]
            x.dims = [6, 2]

        set new_dim = 4

        then out is a LoDTensor:

            out.lod  = [[0, 1, 3]]

            out.data = [[1,  2,  3,  4],
                        [5,  6,  7,  8],
                        [9, 10, 11, 12]]
            out.dims = [3, 4]

    Currently, only 1-level LoDTensor is supported and please make sure
    (original length * original dimension) can be divided by new dimension with
    no remainder for each sequence.

    Args:

       input (Variable): A 2-D LoDTensor with shape being [N, M] where M for dimension.
       new_dim (int): New dimension that the input LoDTensor is reshaped to.

    Returns:

        Variable: Reshaped LoDTensor according to new dimension.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[2, 6], append_batch_size=False, dtype='float32', lod_level=1)
            x_reshaped = fluid.layers.sequence_reshape(input=x, new_dim=4)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_reshape', **locals())
    out = helper.create_variable_for_type_inference(helper.input_dtype())
    helper.append_op(
        type='sequence_reshape',
        inputs={'X': [input]},
        outputs={'Out': [out]},
        attrs={'new_dim': new_dim})
    return out


# FIXME(wuyi): let docstring_checker.py understand @autodoc.
# For now, the comments in c++ use types like Tensor, but in python side
# the type is often "Variable", and arguments may vary.
@templatedoc(op_type="nce")
def nce(input,
        label,
        num_total_classes,
        sample_weight=None,
        param_attr=None,
        bias_attr=None,
        num_neg_samples=None,
        name=None,
        sampler="uniform",
        custom_dist=None,
        seed=0,
        is_sparse=False):
    """
    ${comment}

    Args:
        input (Variable): input variable.
        label (Variable): label.
        num_total_classes (int):${num_total_classes_comment}
        sample_weight (Variable|None): A Variable of shape [batch_size, 1]
            storing a weight for each sample. The default weight for each
            sample is 1.0.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
             of nce. If it is set to None or one attribute of ParamAttr, nce
             will create ParamAttr as param_attr. If the Initializer of the param_attr
             is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of nce.
             If it is set to False, no bias will be added to the output units.
             If it is set to None or one attribute of ParamAttr, nce
             will create ParamAttr as bias_attr. If the Initializer of the bias_attr
             is not set, the bias is initialized zero. Default: None.
        num_neg_samples (int): ${num_neg_samples_comment}
        name (str|None): A name for this layer(optional). If set None, the layer
             will be named automatically. Default: None.
        sampler (str): The sampler used to sample class from negtive classes.
                       It can be 'uniform', 'log_uniform' or 'custom_dist'.
                       default: 'uniform'.
        custom_dist (float[]): A float[] with size=num_total_classes.
                       It is used when sampler is set to 'custom_dist'.
                       custom_dist[i] is the probsbility of i-th class to be sampled.
                       default: None.
        seed (int): The seed used in sampler. default: 0.
        is_sparse(bool): The flag indicating whether to use sparse update, the weight@GRAD and bias@GRAD will be changed to SelectedRows.

    Returns:
        Variable: The output nce loss.

    Examples:
        .. code-block:: python


            import paddle.fluid as fluid
            import numpy as np

            window_size = 5
            words = []
            for i in xrange(window_size):
                words.append(fluid.layers.data(
                    name='word_{0}'.format(i), shape=[1], dtype='int64'))

            dict_size = 10000
            label_word = int(window_size / 2) + 1

            embs = []
            for i in xrange(window_size):
                if i == label_word:
                    continue

                emb = fluid.layers.embedding(input=words[i], size=[dict_size, 32],
                                   param_attr='embed', is_sparse=True)
                embs.append(emb)

            embs = fluid.layers.concat(input=embs, axis=1)
            loss = fluid.layers.nce(input=embs, label=words[label_word],
                      num_total_classes=dict_size, param_attr='nce.w_0',
                      bias_attr='nce.b_0')

             #or use custom distribution
             dist = np.array([0.05,0.5,0.1,0.3,0.05])
             loss = fluid.layers.nce(input=embs, label=words[label_word],
                       num_total_classes=5, param_attr='nce.w_1',
                       bias_attr='nce.b_1',
                       num_neg_samples=3,
                       sampler="custom_dist",
                       custom_dist=dist)
    """
    helper = LayerHelper('nce', **locals())
    assert isinstance(input, Variable)
    assert isinstance(label, Variable)

    dim = input.shape[1]
    num_true_class = label.shape[1]
    w = helper.create_parameter(
        attr=helper.param_attr,
        shape=[num_total_classes, dim],
        is_bias=False,
        dtype=input.dtype)
    inputs = {}
    if helper.bias_attr:
        b = helper.create_parameter(
            attr=helper.bias_attr,
            shape=[num_total_classes, 1],
            is_bias=True,
            dtype=input.dtype)
        inputs['Bias'] = b
    cost = helper.create_variable_for_type_inference(dtype=input.dtype)
    sample_logits = helper.create_variable_for_type_inference(dtype=input.dtype)
    sample_labels = helper.create_variable_for_type_inference(dtype=label.dtype)

    inputs['Input'] = input
    inputs['Label'] = label
    inputs['Weight'] = w
    inputs['SampleWeight'] = sample_weight if sample_weight is not None else []

    if sampler == "uniform":
        sampler = 0
    elif sampler == "log_uniform":
        sampler = 1
    elif sampler == "custom_dist":
        assert custom_dist is not None
        # assert isinstance(custom_dist, Variable)

        custom_dist_len = num_total_classes
        alias_probs_ = [0] * custom_dist_len
        alias_ = [0] * custom_dist_len
        bigs = []
        littles = []
        for i in range(custom_dist_len):
            normal_prob = custom_dist[i] * custom_dist_len
            if normal_prob - 1.0 > 0:
                bigs.append((i, normal_prob))
            elif 1.0 - normal_prob > 0:
                littles.append((i, normal_prob))
            else:
                alias_probs_[i] = normal_prob
                alias_[i] = -1

        while len(bigs) and len(littles):
            big = bigs.pop(0)
            little = littles.pop(0)

            big_idx = big[0]
            big_prob = big[1]

            alias_probs_[little[0]] = little[1]
            alias_[little[0]] = big_idx
            big_left = big[1] + little[1] - 1
            if big_left - 1.0 > 0:
                bigs.append((big_idx, big_left))
            elif 1.0 - big_left > 0:
                littles.append((big_idx, big_left))
            else:
                alias_probs_[big_idx] = big_left
                alias_[big_idx] = -1

        if len(bigs):
            big = bigs.pop(0)
            alias_probs_[big[0]] = 1.0
            alias_[big[0]] = -1
        if len(littles):
            little = littles.pop(0)
            alias_probs_[little[0]] = 1.0
            alias_[little[0]] = -1

        def _init_by_numpy_array(numpy_array):
            ret = helper.create_parameter(
                attr=ParamAttr(),
                shape=numpy_array.shape,
                dtype=numpy_array.dtype,
                default_initializer=NumpyArrayInitializer(numpy_array))
            ret.stop_gradient = True
            return ret

        inputs['CustomDistProbs'] = _init_by_numpy_array(
            np.array(custom_dist).astype('float32'))
        inputs['CustomDistAlias'] = _init_by_numpy_array(
            np.array(alias_).astype('int32'))
        inputs['CustomDistAliasProbs'] = _init_by_numpy_array(
            np.array(alias_probs_).astype('float32'))
        sampler = 2
    else:
        raise Exception("Unsupported sampler type.")

    if num_neg_samples is None:
        num_neg_samples = 10
    else:
        num_neg_samples = int(num_neg_samples)

    remote_prefetch = is_sparse
    print(
        "With sparse mode, if your models has only small parameter prefetch may cause speed down"
    )

    attrs = {
        'num_total_classes': int(num_total_classes),
        'num_neg_samples': num_neg_samples,
        'seed': seed,
        'sampler': sampler,
        'is_sparse': is_sparse,
        'remote_prefetch': remote_prefetch
    }

    helper.append_op(
        type='nce',
        inputs=inputs,
        outputs={
            'Cost': cost,
            'SampleLogits': sample_logits,
            'SampleLabels': sample_labels
        },
        attrs=attrs)
    return cost / (num_neg_samples + 1)


def hsigmoid(input,
             label,
             num_classes,
             param_attr=None,
             bias_attr=None,
             name=None,
             path_table=None,
             path_code=None,
             is_custom=False,
             is_sparse=False):
    """
    The hierarchical sigmoid operator is used to accelerate the training
    process of language model. This operator organizes the classes into a
    complete binary tree, or you can use is_custom to pass your own tree to
    implement hierarchical. Each leaf node represents a class(a word) and each
    internal node acts as a binary classifier. For each word there's a unique
    path from root to it's leaf node, hsigmoid calculate the cost for each
    internal node on the path, and sum them to get a total cost. hsigmoid can
    achive a acceleration from :math:`O(N)` to :math:`O(logN)`, where :math:`N`
    represents the size of word dict.

    Using default tree you can Refer to `Hierarchical Probabilistic Neural Network Language Model
    <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`_

    And if you want to use the costumed tree by set 'is_custom' as true you may need to do following things first:

    1. using your word dict to build a binary tree, each leaf node should be an word of your word dict
    2. build a dict to store word_id -> word's leaf to root path, we call it path_table.
    3. build a dict to store word_id -> code of word's leaf to root path, we call it path_code. Code
       means label of each binary classification, using 1 indicate true, 0 indicate false.
    4. now, each word should has its path and code along the path, you can pass a batch of path and code
       related to the same batch of inputs.

    Args:
        input (Variable): The input tensor variable with shape
            :math:`[N \\times D]`, where :math:`N` is the size of mini-batch,
            and :math:`D` is the feature size.
        label (Variable): The tensor variable contains labels of training data.
            It's a tensor with shape is :math:`[N \\times 1]`.
        num_classes: (int), The number of classes, must not be less than 2. with default tree this has to be set,
            it should never be None under is_custom=False, but while is_custom is true, it should be non leaf num
            which indicates the num of classes using by binary classify.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
             of hsigmoid. If it is set to None or one attribute of ParamAttr, hsigmoid
             will create ParamAttr as param_attr. If the Initializer of the param_attr
             is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of hsigmoid.
             If it is set to False, no bias will be added to the output units.
             If it is set to None or one attribute of ParamAttr, hsigmoid
             will create ParamAttr as bias_attr. If the Initializer of the bias_attr
             is not set, the bias is initialized zero. Default: None.
        name (str|None): A name for this layer(optional). If set None, the layer
             will be named automatically. Default: None.
        path_table: (Variable|None) this variable can store each batch of samples' path to root,
            it should be in leaf -> root order
            path_table should have the same shape with path_code, and for each sample i path_table[i] indicates a np.array like
            structure and each element in this array is indexes in parent nodes' Weight Matrix.
        path_code:  (Variable|None) this variable can store each batch of samples' code,
            each code consist with every code of parent nodes. it should be in leaf -> root order
        is_custom: (bool|False)using user defined binary tree instead of default complete binary tree, if costum is
             set you need to set path_table/path_code/num_classes, otherwise num_classes should be set
        is_sparse: (bool|False)using sparse update instead of dense update, if set, the gradient
             of W and input will be sparse.

    Returns:
        Out: (LodTensor) The cost of hierarchical sigmoid operator. the shape is [N, 1]

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[2], dtype='float32')
            y = fluid.layers.data(name='y', shape=[1], dtype='int64')
            out = fluid.layers.hsigmoid(input=x, label=y, num_classes=6)
    """

    helper = LayerHelper('hierarchical_sigmoid', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    pre_out = helper.create_variable_for_type_inference(dtype)
    dim = input.shape[1]
    if ((num_classes is None) or (num_classes < 2)) and (not is_custom):
        raise ValueError(
            "num_classes must not be less than 2 with default tree")

    if (not is_custom) and (is_sparse):
        print("Sparse mode should not be used without custom tree")
        is_sparse = False

    if (not is_custom) and ((path_table is not None) or
                            (path_code is not None)):
        raise ValueError(
            "only num_classes should be passed without custom tree")

    if (is_custom) and (path_code is None):
        raise ValueError("path_code should not be None with custom tree")
    elif (is_custom) and (path_table is None):
        raise ValueError("path_table should not be None with custom tree")
    elif (is_custom) and (num_classes is None):
        raise ValueError("num_classes should not be None with custom tree")
    else:
        pass

    weights = None
    remote_prefetch = is_sparse
    print(
        "With sparse mode, if your models has only small parameter prefetch may cause speed down"
    )
    if not is_custom:
        weights = helper.create_parameter(
            attr=helper.param_attr,
            shape=[num_classes - 1, dim],
            is_bias=False,
            dtype=input.dtype)
    else:
        weights = helper.create_parameter(
            attr=helper.param_attr,
            shape=[num_classes, dim],
            is_bias=False,
            dtype=input.dtype)
    inputs = {
        "X": input,
        "W": weights,
        "PathTable": path_table,
        "PathCode": path_code,
        "Label": label
    }
    if helper.bias_attr:
        if not is_custom:
            bias = helper.create_parameter(
                attr=helper.bias_attr,
                shape=[num_classes - 1, 1],
                is_bias=True,
                dtype=input.dtype)
            inputs['Bias'] = bias
        else:
            bias = helper.create_parameter(
                attr=helper.bias_attr,
                shape=[num_classes, 1],
                is_bias=True,
                dtype=input.dtype)
            inputs['Bias'] = bias
    helper.append_op(
        type="hierarchical_sigmoid",
        inputs=inputs,
        outputs={"Out": out,
                 "PreOut": pre_out,
                 "W_Out": weights},
        attrs={
            "num_classes": num_classes,
            "is_sparse": is_sparse,
            "remote_prefetch": remote_prefetch
        })
    return out


def transpose(x, perm, name=None):
    """
    Permute the dimensions of `input` according to `perm`.

    The `i`-th dimension  of the returned tensor will correspond to the
    perm[i]-th dimension of `input`.

    Args:
        x (Variable): The input Tensor.
        perm (list): A permutation of the dimensions of `input`.
        name (str): The name of this layer. It is optional.

    Returns:
        Variable: A transposed Tensor.

    Examples:
        .. code-block:: python

            # use append_batch_size=False to avoid prepending extra
            # batch size in shape
            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[5, 10, 15],
                            dtype='float32', append_batch_size=False)
            x_transposed = fluid.layers.transpose(x, perm=[1, 0, 2])
    """

    if len(perm) != len(x.shape):
        raise ValueError(
            "Input(perm) is the permutation of dimensions of Input(input). "
            "Its length should be equal to Input(input)'s rank.")
    for idx, dim in enumerate(perm):
        if dim >= len(x.shape):
            raise ValueError(
                "Each element in perm should be less than x's rank. "
                "%d-th element in perm is %d which accesses x's rank %d." %
                (idx, perm[idx], len(x.shape)))

    helper = LayerHelper('transpose', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    x_shape = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='transpose2',
        inputs={'X': [x]},
        outputs={'Out': [out],
                 'XShape': [x_shape]},
        attrs={'axis': perm})
    return out


def im2sequence(input,
                filter_size=1,
                stride=1,
                padding=0,
                input_image_size=None,
                out_stride=1,
                name=None):
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

        input_image_size(Variable): the input contains image real size.It's dim
            is [batchsize, 2]. It is dispensable.It is just for batch inference.

        out_stride(int|tuple): The scaling of image through CNN. It is
            dispensable. It is valid only when input_image_size is not null.
            If out_stride is tuple,  it must contain two intergers,
            (out_stride_H, out_stride_W). Otherwise,
            the out_stride_H = out_stride_W = out_stride.

        name (int): The name of this layer. It is optional.

    Returns:
        output: The output is a LoDTensor with shape
        {input.batch_size * output_height * output_width,
        filter_size_H * filter_size_W * input.channels}.
        If we regard output as a matrix, each row of this matrix is
        a step of a sequence.

    Examples:

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

            output.dims = {8, 8}

            output.lod = [[4, 4]]

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.layers.data(name='data', shape=[3, 32, 32],
                                     dtype='float32')
            output = fluid.layers.im2sequence(
                input=data, stride=[1, 1], filter_size=[2, 2])


    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")

    if isinstance(filter_size, int):
        filter_size = [filter_size, filter_size]
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(padding, int):
        padding = [padding, padding]
    if len(padding) == 2:
        padding.append(padding[0])
        padding.append(padding[1])
    inputs = {"X": input}
    attrs = {"kernels": filter_size, "strides": stride, "paddings": padding}
    if input_image_size:
        if isinstance(out_stride, int):
            out_stride = [out_stride, out_stride]
        inputs["Y"] = input_image_size
        attrs["out_stride"] = out_stride
    helper = LayerHelper('im2sequence', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='im2sequence', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


@templatedoc()
def row_conv(input, future_context_size, param_attr=None, act=None):
    """
    ${comment}

    Args:
        input (${x_type}): ${x_comment}.
        future_context_size (int): Future context size. Please note, the shape
            of convolution kernel is [future_context_size + 1, D].
        param_attr (ParamAttr): Attributes of parameters, including
            name, initializer etc.
        act (str): Non-linear activation to be applied to output variable.

    Returns:
        ${out_comment}.

    Examples:
        >>> import paddle.fluid as fluid
        >>> x = fluid.layers.data(name='x', shape=[16],
        >>>                        dtype='float32', lod_level=1)
        >>> out = fluid.layers.row_conv(input=x, future_context_size=2)
    """
    helper = LayerHelper('row_conv', **locals())
    dtype = helper.input_dtype()
    filter_shape = [future_context_size + 1, input.shape[1]]
    filter_param = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype)
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='row_conv',
        inputs={'X': [input],
                'Filter': [filter_param]},
        outputs={'Out': [out]})
    return helper.append_activation(out)


@templatedoc()
def multiplex(inputs, index):
    """
    ${comment}

    For Example:

    .. code-block:: text

        case 1:

        Given:

        X = [[[0,0,3,4], [0,1,3,4], [0,2,4,4], [0,3,3,4]],
             [[1,0,3,4], [1,1,7,8], [1,2,4,2], [1,3,3,4]],
             [[2,0,3,4], [2,1,7,8], [2,2,4,2], [2,3,3,4]],
             [[3,0,3,4], [3,1,7,8], [3,2,4,2], [3,3,3,4]]]

        index = [3,0,1,2]

        out:[[3 0 3 4]    // X[3,0] (3 = index[i], 0 = i); i=0
             [0 1 3 4]    // X[0,1] (0 = index[i], 1 = i); i=1
             [1 2 4 2]    // X[1,2] (0 = index[i], 2 = i); i=2
             [2 3 3 4]]   // X[2,3] (0 = index[i], 3 = i); i=3

        case 2:

        Given:

        X = [[[0,0,3,4], [0,1,3,4], [0,2,4,4], [0,3,3,4]],
             [[1,0,3,4], [1,1,7,8], [1,2,4,2], [1,3,3,4]]]

        index = [1,0]

        out:[[1 0 3 4]    // X[1,0] (3 = index[0], 0 = i); i=1
             [0 1 3 4]    // X[0,1] (0 = index[1], 1 = i); i=2
             [0 2 4 4]    // X[0,2] (0 = 0, 2 = i); i=3
             [0 3 3 4]]   // X[0,3] (0 = 0, 3 = i); i=4

    Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        x1 = fluid.layers.data(name='x1', shape=[4], dtype='float32')
        x2 = fluid.layers.data(name='x2', shape=[4], dtype='float32')
        index = fluid.layers.data(name='index', shape=[1], dtype='int32')
        out = fluid.layers.multiplex(inputs=[x1, x2], index=index)

    Args:
       inputs (list): ${x_comment}.
       index (${ids_type}): ${ids_comment}.

    Returns:
        ${out_comment}.
    """
    helper = LayerHelper('multiplex', **locals())

    if not isinstance(inputs, list) and len(inputs) < 2:
        raise ValueError("inputs should be a list object and contains at least "
                         "2 elements.")

    out = helper.create_variable_for_type_inference(inputs[0].dtype)
    helper.append_op(
        type='multiplex',
        inputs={'X': inputs,
                'Ids': index},
        outputs={'Out': [out]})
    return out


def softmax_with_cross_entropy(logits,
                               label,
                               soft_label=False,
                               ignore_index=kIgnoreIndex,
                               numeric_stable_mode=True,
                               return_softmax=False,
                               axis=-1):
    """
    **Softmax With Cross Entropy Operator.**

    Cross entropy loss with softmax is used as the output layer extensively. This
    operator computes the softmax normalized values for dimension :attr:`axis` of 
    the input tensor, after which cross-entropy loss is computed. This provides 
    a more numerically stable gradient.

    Because this operator performs a softmax on logits internally, it expects
    unscaled logits. This operator should not be used with the output of
    softmax operator since that would produce incorrect results.

    When the attribute :attr:`soft_label` is set :attr:`False`, this operators 
    expects mutually exclusive hard labels, each sample in a batch is in exactly 
    one class with a probability of 1.0. Each sample in the batch will have a 
    single label.

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

    3) If :attr:`numeric_stable_mode` is :attr:`True`, softmax is calculated 
    first by:

    .. math::

        max_j &= \\max_{i=0}^{K}{\\text{logit}_i}

        log\\_max\\_sum_j &= \\log\\sum_{i=0}^{K}\\exp(logit_i - max_j)

        softmax_j &= \\exp(logit_j - max_j - {log\\_max\\_sum}_j)

    and then cross entropy loss is calculated by softmax and label.

    Args:
        logits (Variable): The input tensor of unscaled log probabilities.
        label (Variable): The ground truth  tensor. If :attr:`soft_label`
            is set to :attr:`True`, Label is a Tensor<float/double> in the 
            same shape with :attr:`logits`. If :attr:`soft_label` is set to 
            :attr:`True`, Label is a Tensor<int64> in the same shape with 
            :attr:`logits` expect shape in dimension :attr:`axis` as 1.
        soft_label (bool): A flag to indicate whether to interpretate the given
            labels as soft labels. Default False.
        ignore_index (int): Specifies a target value that is ignored and does
                            not contribute to the input gradient. Only valid
                            if :attr:`soft_label` is set to :attr:`False`. 
                            Default: kIgnoreIndex
        numeric_stable_mode (bool): A flag to indicate whether to use a more
                                    numerically stable algorithm. Only valid
                                    when :attr:`soft_label` is :attr:`False` 
                                    and GPU is used. When :attr:`soft_label` 
                                    is :attr:`True` or CPU is used, the 
                                    algorithm is always numerically stable.
                                    Note that the speed may be slower when use
                                    stable algorithm. Default: True
        return_softmax (bool): A flag indicating whether to return the softmax
                               along with the cross entropy loss. Default: False
        axis (int): The index of dimension to perform softmax calculations. It 
                    should be in range :math:`[-1, rank - 1]`, while :math:`rank`
                    is the rank of input :attr:`logits`. Default: -1.

    Returns:
        Variable or Tuple of two Variables: Return the cross entropy loss if \
                                            `return_softmax` is False, otherwise the tuple \
                                            (loss, softmax), softmax is in the same shape \
                                            with input logits and cross entropy loss is in \
                                            the same shape with input logits except shape \
                                            in dimension :attr:`axis` as 1.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            data = fluid.layers.data(name='data', shape=[128], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            fc = fluid.layers.fc(input=data, size=100)
            out = fluid.layers.softmax_with_cross_entropy(
                logits=fc, label=label)
    """
    helper = LayerHelper('softmax_with_cross_entropy', **locals())
    softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
    loss = helper.create_variable_for_type_inference(dtype=logits.dtype)
    helper.append_op(
        type='softmax_with_cross_entropy',
        inputs={'Logits': logits,
                'Label': label},
        outputs={'Softmax': softmax,
                 'Loss': loss},
        attrs={
            'soft_label': soft_label,
            'ignore_index': ignore_index,
            'numeric_stable_mode': numeric_stable_mode,
            'axis': axis
        })

    if return_softmax:
        return loss, softmax

    return loss


def sampled_softmax_with_cross_entropy(logits,
                                       label,
                                       num_samples,
                                       num_true=1,
                                       remove_accidental_hits=True,
                                       use_customized_samples=False,
                                       customized_samples=None,
                                       customized_probabilities=None,
                                       seed=0):
    """
    **Sampled Softmax With Cross Entropy Operator.**

    Cross entropy loss with sampled softmax is used as the output layer for 
    larger output classes extensively. This operator samples a number of samples
    for all examples, and computes the softmax normalized values for each 
    row of the sampled tensor, after which cross-entropy loss is computed. 

    Because this operator performs a softmax on logits internally, it expects
    unscaled logits. This operator should not be used with the output of
    softmax operator since that would produce incorrect results.
    
    For examples with T true labels (T >= 1), we assume that each true label has
    a probability of 1/T. For each sample, S samples are generated using a
    log uniform distribution. True labels are concatenated with these samples to
    form T + S samples for each example. So, assume the shape of logits is
    [N x K], the shape for samples is [N x (T+S)]. For each sampled label, a 
    probability is calculated, which corresponds to the Q(y|x) in 
    [Jean et al., 2014](http://arxiv.org/abs/1412.2007).
    
    Logits are sampled according to the sampled labels. Then if 
    remove_accidental_hits is True, if a sample[i, j] accidentally hits true 
    labels, then the corresponding sampled_logits[i, j] is minus by 1e20 to 
    make its softmax result close to zero. Then sampled logits are subtracted by
    logQ(y|x), these sampled logits and re-indexed labels are used to compute 
    a softmax with cross entropy.

    Args:
        logits (Variable): The unscaled log probabilities, which is a 2-D tensor
            with shape [N x K]. N is the batch_size, and K is the class number.
        label (Variable): The ground truth which is a 2-D tensor. Label is a 
            Tensor<int64> with shape [N x T], where T is the number of true 
            labels per example. 
        num_samples (int): The number for each example, num_samples should be 
            less than the number of class.
        num_true(int): The number of target classes per training example.
        remove_accidental_hits (bool): A flag indicating whether to remove 
            accidental hits when sampling. If True and if a sample[i, j] 
            accidentally hits true labels, then the corresponding 
            sampled_logits[i, j] is minus by 1e20 to make its softmax result 
            close to zero. Default is True.
        use_customized_samples (bool): Whether to use custom samples and probabities to sample
            logits.
        customized_samples (Variable): User defined samples, which is a 2-D tensor
            with shape [N, T + S]. S is the num_samples, and T is the number of true 
            labels per example. 
        customized_probabilities (Variable): User defined probabilities of samples, 
            a 2-D tensor which has the same shape with customized_samples.
        seed (int): The random seed for generating random number, which is used
            in the process of sampling. Default is 0.

    Returns:
        Variable: Return the cross entropy loss which is a 2-D tensor with shape
                  [N x 1].

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            input = fluid.layers.data(name='data', shape=[256], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            fc = fluid.layers.fc(input=input, size=100)
            out = fluid.layers.sampled_softmax_with_cross_entropy(
                      logits=fc, label=label, num_samples=25)
    """
    helper = LayerHelper('sample_logits', **locals())
    samples = helper.create_variable_for_type_inference(dtype='int64')
    probabilities = helper.create_variable_for_type_inference(
        dtype=logits.dtype)
    sampled_logits \
        = helper.create_variable_for_type_inference(dtype=logits.dtype)
    sampled_label = helper.create_variable_for_type_inference(dtype='int64')
    sampled_softlabel = helper.create_variable_for_type_inference(
        dtype=logits.dtype)
    logits_dim = helper.create_variable_for_type_inference(dtype=logits.dtype)
    labels_dim = helper.create_variable_for_type_inference(dtype=label.type)

    helper.append_op(
        type='sample_logits',
        inputs={
            'Logits': logits,
            'Labels': label,
            'CustomizedSamples': customized_samples,
            'CustomizedProbabilities': customized_probabilities
        },
        outputs={
            'Samples': samples,
            'Probabilities': probabilities,
            'SampledLabels': sampled_label,
            'SampledLogits': sampled_logits,
            'LogitsDim': logits_dim,
            'LabelsDim': labels_dim
        },
        attrs={
            'use_customized_samples': use_customized_samples,
            'uniq': True,
            'remove_accidental_hits': remove_accidental_hits,
            'num_samples': num_samples,
            'seed': seed
        })
    loss = helper.create_variable_for_type_inference(dtype=logits.dtype)
    softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
    helper.append_op(
        type='one_hot',
        inputs={'X': sampled_label},
        attrs={'depth': num_samples + 1},
        outputs={'Out': sampled_softlabel})

    helper.append_op(
        type='softmax_with_cross_entropy',
        inputs={'Logits': sampled_logits,
                'Label': sampled_softlabel},
        outputs={'Softmax': softmax,
                 'Loss': loss},
        attrs={
            'soft_label': True,
            'ignore_index': False,
            'numeric_stable_mode': False
        })
    return loss / num_true


def smooth_l1(x, y, inside_weight=None, outside_weight=None, sigma=None):
    """
    This layer computes the smooth L1 loss for Variable :attr:`x` and :attr:`y`.
    It takes the first dimension of :attr:`x` and :attr:`y` as batch size.
    For each instance, it computes the smooth L1 loss element by element first
    and then sums all the losses. So the shape of ouput Variable is
    [batch_size, 1].

    Args:
        x (Variable): A tensor with rank at least 2. The input value of smooth
            L1 loss op with shape [batch_size, dim1, ..., dimN].
        y (Variable): A tensor with rank at least 2. The target value of smooth
            L1 loss op with same shape as :attr:`x`.
        inside_weight (Variable|None):  A tensor with rank at least 2. This
            input is optional and should have same shape with :attr:`x`. If
            provided, the result of (:attr:`x` - :attr:`y`) will be multiplied
            by this tensor element by element.
        outside_weight (Variable|None): A tensor with rank at least 2. This
            input is optional and should have same shape with :attr:`x`. If
            provided, the out smooth L1 loss will be multiplied by this tensor
            element by element.
        sigma (float|None): Hyper parameter of smooth L1 loss layer. A float
           scalar with default value 1.0.

    Returns:
        Variable: The output smooth L1 loss with shape [batch_size, 1].

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.layers.data(name='data', shape=[128], dtype='float32')
            label = fluid.layers.data(
                name='label', shape=[100], dtype='float32')
            fc = fluid.layers.fc(input=data, size=100)
            out = fluid.layers.smooth_l1(x=fc, y=label)
    """

    helper = LayerHelper('smooth_l1_loss', **locals())
    diff = helper.create_variable_for_type_inference(dtype=x.dtype)
    loss = helper.create_variable_for_type_inference(dtype=x.dtype)
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
        attrs={'sigma': sigma if sigma is not None else 1.0})
    return loss


def one_hot(input, depth, allow_out_of_range=False):
    """
    This layer creates the one-hot representations for input indices.

    Args:
        input(Variable): Input indices, last dimension must be 1.
        depth(scalar): An interger defining the depth of the one-hot dimension.
        allow_out_of_range(bool): A bool value indicating whether the input
            indices could be out of range [0, depth). When input indices are
            out of range, exceptions is raised if allow_out_of_range is False,
            or zero-filling representations is created if it is set True

    Returns:
        Variable: The one-hot representations of input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            one_hot_label = fluid.layers.one_hot(input=label, depth=10)
    """
    helper = LayerHelper("one_hot", **locals())

    one_hot_out = helper.create_variable_for_type_inference(dtype='float32')

    if in_dygraph_mode():
        inputs = {'X': input}
        attrs = {'depth': depth}
    else:
        if not isinstance(depth, Variable):
            # user attribute 
            inputs = {'X': input}
            attrs = {'depth': depth}
        else:
            depth.stop_gradient = True
            inputs = {'X': input, 'depth_tensor': depth}
            attrs = {}
    helper.append_op(
        type="one_hot",
        inputs=inputs,
        attrs=attrs,
        outputs={'Out': one_hot_out})
    one_hot_out.stop_gradient = True
    return one_hot_out


def autoincreased_step_counter(counter_name=None, begin=1, step=1):
    """
    Create an auto-increase variable
    which will be automatically increased by 1 every mini-batch
    Return the run counter of the main program, default is started from 1.

    Args:
        counter_name(str): The counter name, default is '@STEP_COUNTER@'.
        begin(int): The first value of this counter.
        step(int): The increment step between each execution.

    Returns:
        Variable: The global run counter.

    Examples:
        .. code-block:: python

           import paddle.fluid as fluid
           global_step = fluid.layers.autoincreased_step_counter(
               counter_name='@LR_DECAY_COUNTER@', begin=0, step=1)
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
        helper.main_program.global_block()._prepend_op(
            type='increment',
            inputs={'X': [counter]},
            outputs={'Out': [counter]},
            attrs={'step': float(step)})
        counter.stop_gradient = True

    return counter


def reshape(x, shape, actual_shape=None, act=None, inplace=False, name=None):
    """
    Gives a new shape to the input Tensor without changing its data.

    The target shape can be given by :attr:`shape` or :attr:`actual_shape`.
    :attr:`shape` is a list of integer or tensor variable while :attr:`actual_shape` is a tensor
    variable. :attr:`actual_shape` has a higher priority than :attr:`shape`
    if it is provided and it only contains integer, while :attr:`shape` still should be set correctly to
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

    **Warning:** the parameter :attr:`actual_shape` will be deprecated in the future and only use :attr:`shape` instead.

    Args:
        x(variable): The input tensor.
        shape(list|tuple|Variable): The new shape. At most one dimension of the new shape can
                     be -1. If :attr:`shape` is a list or tuple, it can contain Variable or not and
                     the shape of Variable must be [1].

        actual_shape(variable): An optional input. If provided, reshape
                                according to this given shape rather than
                                :attr:`shape` specifying shape. That is to
                                say :attr:`actual_shape` has a higher priority
                                than :attr:`shape(list|tuple)` but not :attr:`shape(Variable)`. \
                                This argument :attr:`actual_shape` will be removed in a future version. \
                                Instructions for updating: :attr:`actual_shape` is deprecated,
                                only use :attr:`shape` instead.
        act (str): The non-linear activation to be applied to the reshaped tensor
                   variable.
        inplace(bool): If ``inplace`` is `True`, the input and output of ``layers.reshape``
                       are the same variable, otherwise, the input and output of
                       ``layers.reshape`` are different variables. Note that if :attr:`x`
                       is more than one layer's input, ``inplace`` must be :attr:`False`.
        name (str): The name of this layer. It is optional.

    Returns:
        Variable: The reshaped tensor variable if :attr:`act` is None. It is a \
                  new tensor variable if :attr:`inplace` is :attr:`False`, \
                  otherwise it is :attr:`x`. If :attr:`act` is not None, return \
                  the activated tensor variable.

    Raises:
        TypeError: if actual_shape is neither Variable nor None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            data_1 = fluid.layers.data(
                name='data_1', shape=[2, 4, 6], dtype='float32')
            reshaped_1 = fluid.layers.reshape(
                x=data_1, shape=[-1, 0, 3, 2], inplace=True)

            # example 2:
            # attr shape is a list which contains tensor Variable.
            data_2 = fluid.layers.fill_constant([2,25], "int32", 3)
            dim = fluid.layers.fill_constant([1], "int32", 5)
            reshaped_2 = fluid.layers.reshape(data_2, shape=[dim, 10])
    """

    if not isinstance(shape, (list, tuple, Variable)):
        raise TypeError(
            "Input shape must be an Variable or python list or tuple.")

    if not isinstance(actual_shape, Variable) and (actual_shape is not None):
        raise TypeError("actual_shape should either be Variable or None.")

    helper = LayerHelper("reshape2", **locals())
    inputs = {"X": x}
    attrs = {}

    def contain_var(one_list):
        for ele in one_list:
            if isinstance(ele, Variable):
                return True
        return False

    def get_new_shape_tensor(list_shape):
        new_shape_tensor = []
        for dim in list_shape:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_shape_tensor.append(dim)
            else:
                assert (isinstance(dim, int))
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', dim, force_cpu=True, out=temp_out)
                new_shape_tensor.append(temp_out)
        return new_shape_tensor

    def get_attr_shape(list_shape):
        unk_dim_idx = -1
        attrs_shape = []
        for dim_idx, dim_size in enumerate(list_shape):
            if isinstance(dim_size, Variable):
                attrs_shape.append(-1)
            else:
                attrs_shape.append(dim_size)
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
        return attrs_shape

    if in_dygraph_mode():
        inputs = {'X': x}
        attrs = {'shape': shape}
    else:
        if isinstance(shape, Variable):
            shape.stop_gradient = True
            inputs["Shape"] = shape
        elif isinstance(shape, (list, tuple)):
            assert len(shape) > 0, (
                "The size of argument(shape) can't be zero.")
            attrs["shape"] = get_attr_shape(shape)
            if contain_var(shape):
                inputs['ShapeTensor'] = get_new_shape_tensor(shape)
            elif isinstance(actual_shape, Variable):
                actual_shape.stop_gradient = True
                inputs["Shape"] = actual_shape

    out = x if inplace else helper.create_variable_for_type_inference(
        dtype=x.dtype)
    x_shape = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="reshape2",
        inputs=inputs,
        attrs=attrs,
        outputs={"Out": out,
                 "XShape": x_shape})

    return helper.append_activation(out)


def squeeze(input, axes, name=None):
    """
    Remove single-dimensional entries from the shape of a tensor. Takes a
    parameter axes with a list of axes to squeeze. If axes is not provided, all
    the single dimensions will be removed from the shape. If an axis is
    selected with shape entry not equal to one, an error is raised.

    For example:

    .. code-block:: text

        Case 1:

          Given
            X.shape = (1, 3, 1, 5)
          and
            axes = [0]
          we get:
            Out.shape = (3, 1, 5)

        Case 2:

          Given
            X.shape = (1, 3, 1, 5)
          and
            axes = []
          we get:
            Out.shape = (3, 5)

    Args:
        input (Variable): The input variable to be squeezed.
        axes (list): List of integers, indicating the dimensions to be squeezed.
        name (str|None): Name for this layer.

    Returns:
        Variable: Output squeezed variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            x = layers.data(name='x', shape=[5, 1, 10])
            y = layers.squeeze(input=x, axes=[1])
    """
    assert not in_dygraph_mode(), (
        "squeeze layer is not supported in dygraph mode yet.")
    helper = LayerHelper("squeeze", **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    x_shape = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type="squeeze2",
        inputs={"X": input},
        attrs={"axes": axes},
        outputs={"Out": out,
                 "XShape": x_shape})

    return out


def unsqueeze(input, axes, name=None):
    """
    Insert single-dimensional entries to the shape of a tensor. Takes one
    required argument axes, a list of dimensions that will be inserted.
    Dimension indices in axes are as seen in the output tensor.

    For example:

    .. code-block:: text

      Given a tensor such that tensor with shape [3, 4, 5],
      then Unsqueezed tensor with axes=[0, 4] has shape [1, 3, 4, 5, 1].

    Args:
        input (Variable): The input variable to be unsqueezed.
        axes (list): List of integers, indicating the dimensions to be inserted.
        name (str|None): Name for this layer.

    Returns:
        Variable: Output unsqueezed variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[5, 10])
            y = fluid.layers.unsqueeze(input=x, axes=[1])
    """
    helper = LayerHelper("unsqueeze", **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    x_shape = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type="unsqueeze2",
        inputs={"X": input},
        attrs={"axes": axes},
        outputs={"Out": out,
                 "XShape": x_shape})

    return out


def lod_reset(x, y=None, target_lod=None):
    """
    Set LoD of :attr:`x` to a new one specified by :attr:`y` or
    :attr:`target_lod`. When :attr:`y` provided, :attr:`y.lod` would be
    considered as target LoD first, otherwise :attr:`y.data` would be
    considered as target LoD. If :attr:`y` is not provided, target LoD should
    be specified by :attr:`target_lod`. If target LoD is specified by
    :attr:`y.data` or :attr:`target_lod`, only one level LoD is supported.

    .. code-block:: text

        * Example 1:

            Given a 1-level LoDTensor x:
                x.lod =  [[ 2,           3,                   1 ]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            target_lod: [4, 2]

            then we get a 1-level LoDTensor:
                out.lod =  [[4,                          2]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

        * Example 2:

            Given a 1-level LoDTensor x:
                x.lod =  [[2,            3,                   1]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            y is a Tensor:
                y.data = [[2, 4]]
                y.dims = [1, 3]

            then we get a 1-level LoDTensor:
                out.lod =  [[2,            4]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

        * Example 3:

            Given a 1-level LoDTensor x:
                x.lod =  [[2,            3,                   1]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            y is a 2-level LoDTensor:
                y.lod =  [[2, 2], [2, 2, 1, 1]]
                y.data = [[1.1], [2.1], [3.1], [4.1], [5.1], [6.1]]
                y.dims = [6, 1]

            then we get a 2-level LoDTensor:
                out.lod =  [[2, 2], [2, 2, 1, 1]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

    Args:
        x (Variable): Input variable which could be a Tensor or LoDTensor.
        y (Variable|None): If provided, output's LoD would be derived
                           from :attr:`y`.
        target_lod (list|tuple|None): One level LoD which should be considered
                                      as target LoD when :attr:`y` not provided.

    Returns:
        Variable: Output variable with LoD specified by this layer.

    Raises:
        ValueError: If :attr:`y` and :attr:`target_lod` are both None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[10])
            y = fluid.layers.data(name='y', shape=[10, 20], lod_level=2)
            out = fluid.layers.lod_reset(x=x, y=y)
    """
    helper = LayerHelper("lod_reset", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
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
        raise ValueError("y and target_lod should not be both none.")
    return out


def lod_append(x, level):
    """
    Append level to LoD of :attr:`x`.

    .. code-block:: text

        * Example 1:

            given a 1-level LoDTensor x:
                x.lod =  [[ 2,           3,                   1 ]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            level: [1, 1, 1, 1, 1, 1, 1]

            then we get a 2-level LoDTensor:
                x.lod =  [[ 2, 3, 1 ], [1, 1, 1, 1, 1, 1]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

    Args:
        x (Variable): Input variable which could be a tensor or LoDTensor.
        level (list|tuple|Variable): The LoD level to be appended into LoD of x.

    Returns:
        Variable: Output variable with new LoD level.

    Raises:
        ValueError: If :attr:`y` is None or and :attr:`level` is not Iterator.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[6, 10], lod_level=1)
            out = fluid.layers.lod_append(x, [1,1,1,1,1,1])
    """
    from collections import Iterable
    if x is None:
        raise ValueError("Input(x) can't be None.")
    if (not isinstance(level, Iterable)) and (not isinstance(level, Variable)):
        raise ValueError("Input(level) must be list, tuple or Variable.")

    helper = LayerHelper("lod_append", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {'X': x}
    attrs = {'append': True}

    if isinstance(level, Variable):
        inputs['Y'] = level
    else:
        attrs['target_lod'] = level
    helper.append_op(
        type="lod_reset", inputs=inputs, attrs=attrs, outputs={'Out': out})
    return out


def lrn(input, n=5, k=1.0, alpha=1e-4, beta=0.75, name=None):
    """
    Local Response Normalization Layer. This layer performs a type of
    "lateral inhibition" by normalizing over local input regions.

    The formula is as follows:

    .. math::

      Output(i, x, y) = Input(i, x, y) / \\left(k + \\alpha \\sum\\limits^{\\min(C-1, i + n/2)}_{j = \\max(0, i - n/2)}(Input(j, x, y))^2\\right)^{\\beta}

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

          import paddle.fluid as fluid
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

    mid_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    lrn_out = helper.create_variable_for_type_inference(dtype)
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
    in dimension :attr:`i` is indicated by :attr:`paddings[2i]`, and the number
    of values padded after the contents of :attr:`x` in dimension :attr:`i` is
    indicated by :attr:`paddings[2i+1]`.

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
            import paddle.fluid as fluid
            x = fluid.layers.data(name='data', shape=[224], dtype='float32')
            out = fluid.layers.pad(
                x=x, paddings=[0, 1, 1, 2], pad_value=0.)
    """
    helper = LayerHelper('pad', input=x, **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='pad',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'paddings': paddings,
               'pad_value': float(pad_value)})
    return out


def pad_constant_like(x, y, pad_value=0., name=None):
    """
    Pad input(Y) with :attr:`pad_value`, the number of values padded to
    the edges of each axis is specified by the difference of the shape
    of X and Y. ((0, shape_x_0 - shape_y_0), ... (0, shape_x_n - shape_y_n))
    unique pad widths for each axis. The input should be a k-D
    tensor(k > 0 and k < 7).

    See below for an example.

    .. code-block:: text

        Given:
            X = [[[[ 0,  1,  2],
                   [ 3,  4,  5]],
                  [[ 6,  7,  8],
                   [ 9, 10, 11]],
                  [[12, 13, 14],
                   [15, 16, 17]]],
                 [[[18, 19, 20],
                   [21, 22, 23]],
                  [[24, 25, 26],
                   [27, 28, 29]],
                  [[30, 31, 32],
                   [33, 34, 35]]]]
            X.shape = (2, 3, 2, 3)

            Y = [[[[35, 36, 37]],
                  [[38, 39, 40]],
                  [[41, 42, 43]]]]
            Y.shape = (1, 3, 1, 3)
		And
            pad_value = -1,

        Return:
            Out = [[[[35, 36, 37],
                     [-1, -1, -1]],
                    [[38, 39, 40],
                     [-1, -1, -1]],
                    [[41, 42, 43],
                     [-1, -1, -1]]],
                  [[[-1, -1, -1],
                    [-1, -1, -1]],
                   [[-1, -1, -1],
                    [-1, -1, -1]],
                   [[-1, -1, -1],
                    [-1, -1, -1]]]]
            Out.shape = (2, 3, 2, 3)

    Args:
        x (Variable): The input tensor variable.
        y (Variable): The input tensor variable.
        pad_value (float): The constant value used to pad.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The padded tensor variable.

    Examples:
        .. code-block:: python

            # x is a rank 4 tensor variable, x.shape = (2, 3, 2, 3)
            # y is a rank 4 tensor variable, y.shape = (1, 3, 1, 3)
            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[2,3,2,3], dtype='float32')
            y = fluid.layers.data(name='y', shape=[1,3,1,3], dtype='float32')
            out = fluid.layers.pad_constant_like(x=x, y=y, pad_value=0.)
            # out is a rank 4 tensor variable, and out.shape = [2, 3 ,2 , 3]
    """
    helper = LayerHelper('pad_constant_like', input=x, **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='pad_constant_like',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={'pad_value': float(pad_value)})
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
            
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            label = layers.data(name="label", shape=[1], dtype="float32")
            one_hot_label = layers.one_hot(input=label, depth=10)
            smooth_label = layers.label_smooth(
                label=one_hot_label, epsilon=0.1, dtype="float32")
    """
    if epsilon > 1. or epsilon < 0.:
        raise ValueError("The value of epsilon must be between 0 and 1.")
    helper = LayerHelper("label_smooth", **locals())
    label.stop_gradient = True
    smooth_label = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="label_smooth",
        inputs={"X": label,
                "PriorDist": prior_dist} if prior_dist else {"X": label},
        outputs={"Out": smooth_label},
        attrs={"epsilon": float(epsilon)})
    return smooth_label


@templatedoc()
def roi_pool(input, rois, pooled_height=1, pooled_width=1, spatial_scale=1.0):
    """
    ${comment}

    Args:
        input (Variable): ${x_comment}
        rois (Variable): ROIs (Regions of Interest) to pool over.It should be
                         a 2-D LoDTensor of shape (num_rois, 4), the lod level
                         is 1. Given as [[x1, y1, x2, y2], ...], (x1, y1) is
                         the top left coordinates, and (x2, y2) is the bottom
                         right coordinates.
        pooled_height (integer): ${pooled_height_comment} Default: 1
        pooled_width (integer): ${pooled_width_comment} Default: 1
        spatial_scale (float): ${spatial_scale_comment} Default: 1.0

    Returns:
        Variable: ${out_comment}.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.layers.data(
                name='x', shape=[8, 112, 112], dtype='float32')
            rois = fluid.layers.data(
                name='roi', shape=[4], lod_level=1, dtype='float32')
            pool_out = fluid.layers.roi_pool(
                input=x,
                rois=rois,
                pooled_height=7,
                pooled_width=7,
                spatial_scale=1.0)

    """
    helper = LayerHelper('roi_pool', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    argmaxes = helper.create_variable_for_type_inference(dtype='int32')
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


@templatedoc()
def roi_align(input,
              rois,
              pooled_height=1,
              pooled_width=1,
              spatial_scale=1.0,
              sampling_ratio=-1,
              name=None):
    """
    ${comment}

    Args:
        input (Variable): ${x_comment}
        rois (Variable): ROIs (Regions of Interest) to pool over.It should be
                         a 2-D LoDTensor of shape (num_rois, 4), the lod level
                         is 1. Given as [[x1, y1, x2, y2], ...], (x1, y1) is
                         the top left coordinates, and (x2, y2) is the bottom
                         right coordinates. 
        pooled_height (integer): ${pooled_height_comment} Default: 1
        pooled_width (integer): ${pooled_width_comment} Default: 1
        spatial_scale (float): ${spatial_scale_comment} Default: 1.0
        sampling_ratio(intger): ${sampling_ratio_comment} Default: -1

    Returns:
        Variable: ${out_comment}.
    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(
                name='data', shape=[256, 32, 32], dtype='float32')
            rois = fluid.layers.data(
                name='rois', shape=[4], dtype='float32')
            align_out = fluid.layers.roi_align(input=x,
                                               rois=rois,
                                               pooled_height=7,
                                               pooled_width=7,
                                               spatial_scale=0.5,
                                               sampling_ratio=-1)
    """
    helper = LayerHelper('roi_align', **locals())
    dtype = helper.input_dtype()
    align_out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="roi_align",
        inputs={"X": input,
                "ROIs": rois},
        outputs={"Out": align_out},
        attrs={
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "spatial_scale": spatial_scale,
            "sampling_ratio": sampling_ratio
        })
    return align_out


def dice_loss(input, label, epsilon=0.00001):
    """
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

            import paddle.fluid as fluid
            x = fluid.layers.data(name='data', shape = [3, 224, 224, 2], dtype='float32')
            label = fluid.layers.data(name='label', shape=[3, 224, 224, 1], dtype='float32')
            predictions = fluid.layers.softmax(x)
            loss = fluid.layers.dice_loss(input=predictions, label=label)
    """
    label = one_hot(label, depth=input.shape[-1])
    reduce_dim = list(range(1, len(input.shape)))
    inse = reduce_sum(input * label, dim=reduce_dim)
    dice_denominator = reduce_sum(
        input, dim=reduce_dim) + reduce_sum(
            label, dim=reduce_dim)
    dice_score = 1 - inse * 2 / (dice_denominator + epsilon)
    return reduce_mean(dice_score)


def image_resize(input,
                 out_shape=None,
                 scale=None,
                 name=None,
                 resample='BILINEAR',
                 actual_shape=None,
                 align_corners=True,
                 align_mode=1,
                 data_format='NCHW'):
    """
    **Resize a Batch of Images**

    The input must be a 4-D Tensor of the shape (num_batches, channels, in_h, in_w) 
    or (num_batches, in_h, in_w, channels), or a 5-D Tensor of the shape 
    (num_batches, channels, in_d, in_h, in_w) or (num_batches, in_d, in_h, in_w, channels), 
    and the resizing only applies on the three dimensions(depth, hight and width).

    **Warning:** the parameter :attr:`actual_shape` will be deprecated in the
    future and only use :attr:`out_shape` instead.

    Supporting resample methods:

        'BILINEAR' : Bilinear interpolation

        'TRILINEAR' : Trilinear interpolation

        'NEAREST' : Nearest neighbor interpolation

    Nearest neighbor interpolation is to perform nearest neighbor interpolation
    in both the 3rd dimention(in height direction) and the 4th dimention(in width 
    direction) on input tensor.
            
    Bilinear interpolation is an extension of linear interpolation for 
    interpolating functions of two variables (e.g. H-direction and 
    W-direction in this op) on a rectilinear 2D grid. The key idea is 
    to perform linear interpolation first in one direction, and then 
    again in the other direction.

    Trilinear interpolation is an extension of linear interpolation for 
    interpolating functions of three variables (e.g. D-direction, 
    H-direction and W-direction in this op) on a rectilinear 3D grid. 
    The linear interpolation is performed on three directions.

    Align_corners and align_mode are optinal parameters,the calculation method 
    of interpolation can be selected by them.

    Example:

    .. code-block:: text

        For scale:
          
            if align_corners = True && out_size > 1 :

              scale_factor = (in_size-1.0)/(out_size-1.0)
            
            else:
              
              scale_factor = float(in_size/out_size)
            
          
        Nearest neighbor interpolation:
          
          if:
              align_corners = False

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = floor (H_{in} * scale_{factor})
              W_out = floor (W_{in} * scale_{factor})

          else:
              align_corners = True

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = round(H_{in} * scale_{factor})
              W_out = round(W_{in} * scale_{factor})

        Bilinear interpolation:

          if:
              align_corners = False , align_mode = 0
              
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5

          else:
           
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        Trilinear interpolation:

          if:
              align_corners = False , align_mode = 0
              
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              
              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5


          else:
           
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:

              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}
          
    For details of nearest neighbor interpolation, please refer to Wikipedia: 
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.

    For details of bilinear interpolation, please refer to Wikipedia: 
    https://en.wikipedia.org/wiki/Bilinear_interpolation.

    For details of trilinear interpolation, please refer to Wikipedia: 
    https://en.wikipedia.org/wiki/Trilinear_interpolation.



    Args:
        input (Variable): 4-D or 5-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        out_shape(list|tuple|Variable|None): Output shape of image resize
             layer, the shape is (out_h, out_w) when input is a 4-D Tensor and is
             (out_d, out_h, out_w) when input is a 5-D Tensor. Default: None. If 
             a list, each element can be an integer or a Tensor Variable of shape: [1].
             If a Tensor Variable, its dimensions size should be a 1.
        scale(float|Variable|None): The multiplier for the input height or width. At
             least one of :attr:`out_shape` or :attr:`scale` must be set.
             And :attr:`out_shape` has a higher priority than :attr:`scale`.
             Default: None.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.
        resample(str): The resample method. It supports 'BILINEAR', 'TRILINEAR'
                       and 'NEAREST' currently. Default: 'BILINEAR'
        actual_shape(Variable): An optional input to specify output shape
                                dynamically. If provided, image resize
                                according to this given shape rather than
                                :attr:`out_shape` and :attr:`scale` specifying
                                shape. That is to say actual_shape has the
                                highest priority. It is recommended to use
                                :attr:`out_shape` if you want to specify output 
                                shape dynamically, because :attr:`actual_shape` 
                                will be deprecated. When using actual_shape to 
                                specify output shape, one of :attr:`out_shape` 
                                and :attr:`scale` should also be set, otherwise 
                                errors would be occured in graph constructing stage.
                                Default: None
        align_corners(bool) :  An optional bool, If True, the centers of the 4 corner pixels of the 
                               input and output tensors are aligned, preserving the values at the 
                               corner pixels.
                               Default: True
        align_mode(int)  :  An optional for bilinear interpolation. can be \'0\' 
                            for src_idx = scale*(dst_indx+0.5)-0.5 , can be \'1\' for 
                            src_idx = scale*dst_index.
        data_format(str, optional): NCHW(num_batches, channels, height, width) or 
                                    NHWC(num_batches, height, width, channels) for 4-D Tensor,
                                    NCDHW(num_batches, channels, depth, height, width) or 
                                    NDHWC(num_batches, depth, height, width, channels) for 5-D Tensor.
                                    Default: 'NCHW'.

    Returns:
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),
        or 5-D Tensor of the shape (num_batches, channels, out_d, out_h, out_w) or (num_batches, out_d, out_h, out_w, channels).

    Raises:
        TypeError: out_shape should be a list or tuple or Variable.
        TypeError: actual_shape should either be Variable or None.
        ValueError: The 'resample' of image_resize can only be 'BILINEAR',
                    'TRILINEAR' or 'NEAREST' currently.
        ValueError: 'BILINEAR' and 'NEAREST' only support 4-D tensor.
        ValueError: 'TRILINEAR' only support 5-D tensor.
        ValueError: One of out_shape and scale must not be None.
        ValueError: out_shape length should be 2 for input 4-D tensor.
        ValueError: out_shape length should be 3 for input 5-D tensor.
        ValueError: scale should be greater than zero.
        TypeError: align_corners shoule be a bool value
        ValueError: align_mode can only be '0' or '1'
        ValueError: data_format can only be 'NCHW', 'NHWC', 'NCDHW' or 'NDHWC'.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.layers.data(name="input", shape=[3, 6, 9], dtype="float32")
            # input.shape = [-1, 3, 6, 9], where -1 indicates batch size, and it will get the exact value in runtime.

            out0 = fluid.layers.image_resize(input, out_shape=[12, 12], resample="NEAREST")
            # out0.shape = [-1, 3, 12, 12], it means out0.shape[0] = input.shape[0] in runtime.

            # out_shape is a list in which each element is a integer or a tensor Variable
            dim1 = fluid.layers.data(name="dim1", shape=[1], dtype="int32", append_batch_size=False)
            out1 = fluid.layers.image_resize(input, out_shape=[12, dim1], resample="NEAREST")
            # out1.shape = [-1, 3, 12, -1]

            # out_shape is a 1-D tensor Variable
            shape_tensor = fluid.layers.data(name="shape_tensor", shape=[2], dtype="int32", append_batch_size=False)
            out2 = fluid.layers.image_resize(input, out_shape=shape_tensor, resample="NEAREST")
            # out2.shape = [-1, 3, -1, -1]

            # when use actual_shape
            actual_shape_tensor = fluid.layers.data(name="actual_shape_tensor", shape=[2], dtype="int32", append_batch_size=False)
            out3 = fluid.layers.image_resize(input, out_shape=[4, 4], resample="NEAREST", actual_shape=actual_shape_tensor)
            # out3.shape = [-1, 3, 4, 4]

            # scale is a Variable
            scale_tensor = fluid.layers.data(name="scale", shape=[1], dtype="float32", append_batch_size=False)
            out4 = fluid.layers.image_resize(input, scale=scale_tensor)
            # out4.shape = [-1, 3, -1, -1]

    """
    resample_methods = {
        'BILINEAR': 'bilinear',
        'TRILINEAR': 'trilinear',
        'NEAREST': 'nearest',
    }
    if resample not in resample_methods:
        raise ValueError(
            "The 'resample' of image_resize can only be 'BILINEAR', 'TRILINEAR' "
            "or 'NEAREST' currently.")
    resample_type = resample_methods[resample]

    if resample in ['BILINEAR', 'NEAREST'] and len(input.shape) != 4:
        raise ValueError("'BILINEAR' and 'NEAREST' only support 4-D tensor.")
    if resample == 'TRILINEAR' and len(input.shape) != 5:
        raise ValueError("'TRILINEAR'only support 5-D tensor.")

    if not isinstance(align_corners, bool):
        raise TypeError("Attr align_corners should be a bool value")
    if align_mode != 0 and align_mode != 1:
        raise ValueError("align_mode can only be 0 or 1")

    if out_shape is None and scale is None:
        raise ValueError("One of out_shape and scale must not be None.")
    helper = LayerHelper('{}_interp'.format(resample_type), **locals())
    dtype = helper.input_dtype()

    if len(input.shape) == 4 and data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: " + data_format +
            " received but only `NCHW` or `NHWC` supported for 4-D input.")
    elif len(input.shape) == 5 and data_format not in ['NCDHW', 'NDHWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: " + data_format +
            " received but only `NCDHW` or `NDHWC` supported for 5-D input.")

    def _is_list_or_turple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    if data_format == 'NCHW' or data_format == 'NCDHW':
        data_layout = 'NCHW'
    if data_format == 'NHWC' or data_format == 'NDHWC':
        data_layout = 'NHWC'

    inputs = {"X": input}
    attrs = {
        "out_d": -1,
        "out_h": -1,
        "out_w": -1,
        "interp_method": resample_type,
        "align_corners": align_corners,
        "align_mode": align_mode,
        "data_layout": data_layout
    }

    if out_shape is not None:
        if isinstance(out_shape, Variable):
            out_shape.stop_gradient = True
            inputs['OutSize'] = out_shape
        else:
            if not (_is_list_or_turple_(out_shape)):
                raise TypeError(
                    "out_shape should be a list or tuple or Variable.")
            # Validate the shape
            contain_var = False
            for dim_idx, dim_size in enumerate(out_shape):
                if isinstance(dim_size, Variable):
                    contain_var = True
                    continue
                assert dim_size > 0, (
                    "Each dimension size given in out_shape must be greater than 0."
                )

            if contain_var:
                new_size_tensor = []
                size_list = []
                for dim in out_shape:
                    if isinstance(dim, Variable):
                        dim.stop_gradient = True
                        new_size_tensor.append(dim)
                        size_list.append(-1)
                    else:
                        assert (isinstance(dim, int))
                        temp_out = helper.create_variable_for_type_inference(
                            'int32')
                        fill_constant(
                            [1], 'int32', dim, force_cpu=True, out=temp_out)
                        new_size_tensor.append(temp_out)
                        size_list.append(dim)
                inputs['SizeTensor'] = new_size_tensor

            if len(input.shape) == 4:
                if len(out_shape) != 2:
                    raise ValueError("out_shape length should be 2 for "
                                     "input 4-D tensor.")
                if contain_var:
                    attrs['out_h'] = size_list[0]
                    attrs['out_w'] = size_list[1]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_h'] = out_shape[0]
                    attrs['out_w'] = out_shape[1]
            if len(input.shape) == 5:
                if len(out_shape) != 3:
                    raise ValueError("out_shape length should be 3 for "
                                     "input 5-D tensor.")
                if contain_var:
                    attrs['out_d'] = size_list[0]
                    attrs['out_h'] = size_list[1]
                    attrs['out_w'] = size_list[2]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_d'] = out_shape[0]
                    attrs['out_h'] = out_shape[1]
                    attrs['out_w'] = out_shape[2]

    else:
        if isinstance(scale, Variable):
            scale.stop_gradient = True
            inputs["Scale"] = scale
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError("scale should be greater than zero.")
            attrs['scale'] = float(scale)

    if isinstance(actual_shape, Variable):
        warnings.warn(
            "actual_shape will be deprecated, it is recommended to use "
            "out_shape instead of actual_shape to specify output shape dynamically."
        )
        actual_shape.stop_gradient = True
        inputs["OutSize"] = actual_shape
    elif actual_shape is not None:
        raise TypeError("actual_shape should either be Variable or None.")

    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='{}_interp'.format(resample_type),
        inputs=inputs,
        outputs={"Out": out},
        attrs=attrs)
    return out


@templatedoc(op_type="bilinear_interp")
def resize_bilinear(input,
                    out_shape=None,
                    scale=None,
                    name=None,
                    actual_shape=None,
                    align_corners=True,
                    align_mode=1,
                    data_format='NCHW'):
    """
    Resize input by performing bilinear interpolation based on given
    output shape which specified by actual_shape, out_shape and scale
    in priority order.

    **Warning:** the parameter :attr:`actual_shape` will be deprecated in 
    the future and only use :attr:`out_shape` instead.

    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.

    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation

    Align_corners and align_mode are optinal parameters,the calculation 
    method of interpolation can be selected by them.

    Example:

    .. code-block:: text

        For scale:
          
            if align_corners = True && out_size > 1 :

              scale_factor = (in_size-1.0)/(out_size-1.0)
            
            else:
              
              scale_factor = float(in_size/out_size)

        Bilinear interpolation:

          if:
              align_corners = False , align_mode = 0
              
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5

          else:

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

    Args:
        input(${x_type}): 4-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        out_shape(list|tuple|Variable|None): Output shape of resize bilinear
            layer, the shape is (out_h, out_w).Default: None. If a list, each 
            element can be an integer or a Tensor Variable with shape: [1]. If a 
            Tensor Variable, its dimension size should be 1.
        scale(float|Variable|None): The multiplier for the input height or width. At
             least one of :attr:`out_shape` or :attr:`scale` must be set. 
             And :attr:`out_shape` has a higher priority than :attr:`scale`. 
             Default: None.
        name(str|None): The output variable name.
        actual_shape(Variable): An optional input to specify output shape
                                dynamically. If provided, image resize
                                according to this given shape rather than
                                :attr:`out_shape` and :attr:`scale` specifying
                                shape. That is to say actual_shape has the
                                highest priority. It is recommended to use
                                :attr:`out_shape` if you want to specify output 
                                shape dynamically, because :attr:`actual_shape` 
                                will be deprecated. When using actual_shape to 
                                specify output shape, one of :attr:`out_shape` 
                                and :attr:`scale` should also be set, otherwise 
                                errors would be occured in graph constructing stage.
                                Default: None
        align_corners(bool): ${align_corners_comment}
        align_mode(bool): ${align_mode_comment}
        data_format(str, optional): NCHW(num_batches, channels, height, width) or 
                                    NHWC(num_batches, height, width, channels). Default: 'NCHW'.

    Returns:
        A 4-D Tensor in shape of (num_batches, channels, out_h, out_w) or
        (num_batches, out_h, out_w, channels).

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.layers.data(name="input", shape=[3, 6, 9], dtype="float32")
            # input.shape = [-1, 3, 6, 9], where -1 indicates batch size, and it will get the exact value in runtime.

            out0 = fluid.layers.resize_bilinear(input, out_shape=[12, 12])
            # out0.shape = [-1, 3, 12, 12], it means out0.shape[0] = input.shape[0] in runtime.

            # out_shape is a list in which each element is a integer or a tensor Variable
            dim1 = fluid.layers.data(name="dim1", shape=[1], dtype="int32", append_batch_size=False)
            out1 = fluid.layers.resize_bilinear(input, out_shape=[12, dim1])
            # out1.shape = [-1, 3, 12, -1]

            # out_shape is a 1-D tensor Variable
            shape_tensor = fluid.layers.data(name="shape_tensor", shape=[2], dtype="int32", append_batch_size=False)
            out2 = fluid.layers.resize_bilinear(input, out_shape=shape_tensor)
            # out2.shape = [-1, 3, -1, -1]

            # when use actual_shape
            actual_shape_tensor = fluid.layers.data(name="actual_shape_tensor", shape=[2], dtype="int32", append_batch_size=False)
            out3 = fluid.layers.resize_bilinear(input, out_shape=[4, 4], actual_shape=actual_shape_tensor)
            # out3.shape = [-1, 3, 4, 4]

            # scale is a Variable
            scale_tensor = fluid.layers.data(name="scale", shape=[1], dtype="float32", append_batch_size=False)
            out4 = fluid.layers.resize_bilinear(input, scale=scale_tensor)
            # out4.shape = [-1, 3, -1, -1]
    """

    return image_resize(input, out_shape, scale, name, 'BILINEAR', actual_shape,
                        align_corners, align_mode, data_format)


@templatedoc(op_type="trilinear_interp")
def resize_trilinear(input,
                     out_shape=None,
                     scale=None,
                     name=None,
                     actual_shape=None,
                     align_corners=True,
                     align_mode=1,
                     data_format='NCDHW'):
    """
    Resize input by performing trilinear interpolation based on given
    output shape which specified by actual_shape, out_shape and scale
    in priority order.

    **Warning:** the parameter :attr:`actual_shape` will be deprecated 
    in the future and only use :attr:`out_shape` instead.

    Trilinear interpolation is an extension of linear interpolation for 
    interpolating functions of three variables (e.g. D-direction, 
    H-direction and W-direction in this op) on a rectilinear 3D grid. 
    The linear interpolation is performed on three directions.

    For details of trilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Trilinear_interpolation

    Align_corners and align_mode are optinal parameters,the calculation 
    method of interpolation can be selected by them.

    Example:

    .. code-block:: text

        For scale:
          
            if align_corners = True && out_size > 1 :

              scale_factor = (in_size-1.0)/(out_size-1.0)
            
            else:
              
              scale_factor = float(in_size/out_size)     

        Bilinear interpolation:

          if:

              align_corners = False , align_mode = 0
              
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              
              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5

          else:

              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:

              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

    Args:
        input(${x_type}): 5-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        out_shape(list|tuple|Variable|None): Output shape of resize bilinear
            layer, the shape is (out_d, out_h, out_w). Default: None. If a list, 
            each element can be  an integer or a Tensor Variable with shape: [1]. If 
            a Tensor Variable, its dimension size should be 1.
        scale(float|Variable|None): The multiplier for the input depth, height or width.
             At least one of :attr:`out_shape` or :attr:`scale` must be set. 
             And :attr:`out_shape` has a higher priority than :attr:`scale`. 
             Default: None.
        name(str|None): The output variable name.
        actual_shape(Variable): An optional input to specify output shape
                                dynamically. If provided, image resize
                                according to this given shape rather than
                                :attr:`out_shape` and :attr:`scale` specifying
                                shape. That is to say actual_shape has the
                                highest priority. It is recommended to use
                                :attr:`out_shape` if you want to specify output 
                                shape dynamically, because :attr:`actual_shape` 
                                will be deprecated. When using actual_shape to 
                                specify output shape, one of :attr:`out_shape` 
                                and :attr:`scale` should also be set, otherwise 
                                errors would be occured in graph constructing stage.
                                Default: None
        align_corners(bool): ${align_corners_comment}
        align_mode(bool): ${align_mode_comment}
        data_format(str, optional): NCDHW(num_batches, channels, depth, height, width) or 
                                    NDHWC(num_batches, depth, height, width, channels).
                                    Default: 'NCDHW'.

    Returns:
        A 5-D Tensor in shape of (num_batches, channels, out_d, out_h, out_w) or 
        (num_batches, out_d, out_h, out_w, channels).

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.layers.data(name="input", shape=[3, 6, 9, 11], dtype="float32")
            # input.shape = [-1, 3, 6, 9, 11], where -1 indicates batch size, and it will get the exact value in runtime.

            out0 = fluid.layers.resize_trilinear(input, out_shape=[12, 12, 12])
            # out0.shape = [-1, 3, 12, 12, 12], it means out0.shape[0] = input.shape[0] in runtime.

            # out_shape is a list in which each element is a integer or a tensor Variable
            dim1 = fluid.layers.data(name="dim1", shape=[1], dtype="int32", append_batch_size=False)
            out1 = fluid.layers.resize_trilinear(input, out_shape=[12, dim1, 4])
            # out1.shape = [-1, 3, 12, -1, 4]

            # out_shape is a 1-D tensor Variable
            shape_tensor = fluid.layers.data(name="shape_tensor", shape=[3], dtype="int32", append_batch_size=False)
            out2 = fluid.layers.resize_trilinear(input, out_shape=shape_tensor)
            # out2.shape = [-1, 3, -1, -1, -1]

            # when use actual_shape
            actual_shape_tensor = fluid.layers.data(name="actual_shape_tensor", shape=[3], dtype="int32", append_batch_size=False)
            out3 = fluid.layers.resize_trilinear(input, out_shape=[4, 4, 8], actual_shape=actual_shape_tensor)
            # out3.shape = [-1, 3, 4, 4, 8]

            # scale is a Variable
            scale_tensor = fluid.layers.data(name="scale", shape=[1], dtype="float32", append_batch_size=False)
            out4 = fluid.layers.resize_trilinear(input, scale=scale_tensor)
            # out4.shape = [-1, 3, -1, -1, -1]
    """

    return image_resize(input, out_shape, scale, name, 'TRILINEAR',
                        actual_shape, align_corners, align_mode, data_format)


@templatedoc(op_type="nearest_interp")
def resize_nearest(input,
                   out_shape=None,
                   scale=None,
                   name=None,
                   actual_shape=None,
                   align_corners=True,
                   data_format='NCHW'):
    """
    Resize input by performing nearest neighbor interpolation in both the
    height direction and the width direction based on given output shape 
    which is specified by actual_shape, out_shape and scale in priority order.

    **Warning:** the parameter :attr:`actual_shape` will be deprecated in the 
    future and only use :attr:`out_shape` instead.

    Example:

    .. code-block:: text

        For scale:
          
            if align_corners = True && out_size > 1 :
              scale_factor = (in_size-1.0)/(out_size-1.0)
            
            else:
              
              scale_factor = float(in_size/out_size)
          
        Nearest neighbor interpolation:
          
          if:
              align_corners = False

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = floor(H_{in} * scale_{factor})
              W_out = floor(W_{in} * scale_{factor})

          else:
              align_corners = True

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = round(H_{in} * scale_{factor})
              W_out = round(W_{in} * scale_{factor})


    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation

    Args:
        input(${x_type}): 4-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        out_shape(list|tuple|Variable|None): Output shape of resize nearest
            layer, the shape is (out_h, out_w). Default: None. If a list, each 
            element can be integer or a tensor Variable with shape: [1]. If a 
            tensor Variable, its dimension size should be 1.
        scale(float|Variable|None): The multiplier for the input height or width. At
             least one of :attr:`out_shape` or :attr:`scale` must be set. 
             And :attr:`out_shape` has a higher priority than :attr:`scale`. 
             Default: None.
        name(str|None): The output variable name.
        actual_shape(Variable): An optional input to specify output shape
                                dynamically. If provided, image resize
                                according to this given shape rather than
                                :attr:`out_shape` and :attr:`scale` specifying
                                shape. That is to say actual_shape has the
                                highest priority. It is recommended to use
                                :attr:`out_shape` if you want to specify output 
                                shape dynamically, because :attr:`actual_shape` 
                                will be deprecated. When using actual_shape to 
                                specify output shape, one of :attr:`out_shape` 
                                and :attr:`scale` should also be set, otherwise 
                                errors would be occured in graph constructing stage.
                                Default: None
        align_corners(bool): ${align_corners_comment}
        data_format(str, optional): NCHW(num_batches, channels, height, width) or 
                                    NHWC(num_batches, height, width, channels).
                                    Default: 'NCHW'.

    Returns:
        A 4-D Tensor in shape of (num_batches, channels, out_h, out_w) or 
        (num_batches, out_h, out_w, channels).

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.layers.data(name="input", shape=[3, 6, 9], dtype="float32")
            # input.shape = [-1, 3, 6, 9], where -1 indicates batch size, and it will get the exact value in runtime.

            out0 = fluid.layers.resize_nearest(input, out_shape=[12, 12])
            # out0.shape = [-1, 3, 12, 12], it means out0.shape[0] = input.shape[0] in runtime.

            # out_shape is a list in which each element is a integer or a tensor Variable
            dim1 = fluid.layers.data(name="dim1", shape=[1], dtype="int32", append_batch_size=False)
            out1 = fluid.layers.resize_nearest(input, out_shape=[12, dim1])
            # out1.shape = [-1, 3, 12, -1]

            # out_shape is a 1-D tensor Variable
            shape_tensor = fluid.layers.data(name="resize_shape", shape=[2], dtype="int32", append_batch_size=False)
            out2 = fluid.layers.resize_nearest(input, out_shape=shape_tensor)
            # out2.shape = [-1, 3, -1, -1]

            # when use actual_shape
            actual_shape_tensor = fluid.layers.data(name="actual_shape_tensor", shape=[2], dtype="int32", append_batch_size=False)
            out3 = fluid.layers.resize_nearest(input, out_shape=[4, 4], actual_shape=actual_shape_tensor)
            # out3.shape = [-1, 3, 4, 4]

            # scale is a Variable
            scale_tensor = fluid.layers.data(name="scale", shape=[1], dtype="float32", append_batch_size=False)
            out4 = fluid.layers.resize_nearest(input, scale=scale_tensor)
            # out4.shape = [-1, 3, -1, -1]
    """

    return image_resize(
        input,
        out_shape,
        scale,
        name,
        'NEAREST',
        actual_shape,
        align_corners,
        align_mode=1,
        data_format=data_format)


def image_resize_short(input, out_short_len, resample='BILINEAR'):
    """
    Resize a batch of images. The short edge of input images will be
    resized to the given 'out_short_len'. The long edge of input images
    will be resized proportionately to make images' length-width ratio
    constant.

    Args:
        input (Variable): The input tensor of image resize layer,
                          This is a 4-D tensor of the shape
                          (num_batches, channels, in_h, in_w).
        out_short_len(int): The length of output images' short edge.
        resample (str): resample method, default: BILINEAR.

    Returns:
        Variable: The output is a 4-D tensor of the shape
        (num_batches, channls, out_h, out_w).

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.layers.data(name="input", shape=[3,6,9], dtype="float32")
            out = fluid.layers.image_resize_short(input, out_short_len=3)
    """
    in_shape = input.shape
    if len(in_shape) != 4:
        raise ValueError(
            "The rank of input must be 4 (num_batches, channels, in_h, in_w).")
    hw = in_shape[2:4]
    short_idx = hw.index(min(hw))
    long_idx = 1 - short_idx
    out_shape = list(hw)
    out_shape[short_idx] = out_short_len
    out_shape[long_idx] = int(
        float(out_shape[long_idx]) * (float(out_short_len) / float(hw[
            short_idx])) + 0.5)
    return image_resize(input=input, out_shape=out_shape, resample=resample)


def gather(input, index, overwrite=True):
    """
    **Gather Layer**

    Output is obtained by gathering entries of the outer-most dimension
    of X indexed by `index` and concatenate them together.

    .. math::

        Out = X[Index]


    .. code-block:: text


                Given:

                X = [[1, 2],
                     [3, 4],
                     [5, 6]]

                Index = [1, 2]

                Then:

                Out = [[3, 4],
                       [5, 6]]

    Args:
        input (Variable): The source input with rank>=1.
        index (Variable): The index input with rank=1.
        overwrite (bool): The mode that updating the grad when has same index.
            If True, use the overwrite mode to update the grad of the same index,
	    if False, use the accumulate mode to update the grad of the same index. 
	    Default value is True.
	    


    Returns:
        output (Variable): The output is a tensor with the same rank as input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[-1, 5], dtype='float32')
            index = fluid.layers.data(name='index', shape=[-1, 1], dtype='int32')
            output = fluid.layers.gather(x, index)
    """
    helper = LayerHelper('gather', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="gather",
        inputs={"X": input,
                "Index": index},
        outputs={"Out": out},
        attrs={'overwrite': overwrite})
    return out


def gather_nd(input, index, name=None):
    """
    **Gather Nd Layer**

    This function is actually a high-dimensional extension of :code:`gather` 
    and supports for simultaneous indexing by multiple axes. :attr:`index` is a 
    K-dimensional integer tensor, which is regarded as a (K-1)-dimensional 
    tensor of :attr:`index` into :attr:`input`, where each element defines 
    a slice of params:

    .. math::

        output[(i_0, ..., i_{K-2})] = input[index[(i_0, ..., i_{K-2})]]

    Obviously, :code:`index.shape[-1] <= input.rank` . And, the output tensor has
    shape :code:`index.shape[:-1] + input.shape[index.shape[-1]:]` .

    .. code-block:: text

            Given:
                input = [[[ 0,  1,  2,  3],
                          [ 4,  5,  6,  7],
                          [ 8,  9, 10, 11]],
                         [[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]]]
                input.shape = (2, 3, 4)

            * Case 1:
                index = [[1]]
                
                gather_nd(input, index)  
                         = [input[1, :, :]] 
                         = [[12, 13, 14, 15],
                            [16, 17, 18, 19],
                            [20, 21, 22, 23]]

            * Case 2:
                index = [[0,2]]

                gather_nd(input, index)
                         = [input[0, 2, :]]
                         = [8, 9, 10, 11]

            * Case 3:
                index = [[1, 2, 3]]

                gather_nd(input, index)
                         = [input[1, 2, 3]]
                         = [23]

    Args:
        input (Variable): The source input
        index (Variable): The index input with rank > 1, index.shape[-1] <= input.rank
        name (str|None): A name for this layer(optional). If set None, the
                         layer will be named automatically

    Returns:
        output (Variable): A tensor with the shape index.shape[:-1] + input.shape[index.shape[-1]:]

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[3, 4, 5], dtype='float32')
            index = fluid.layers.data(name='index', shape=[2, 2], dtype='int32')
            output = fluid.layers.gather_nd(x, index)

    """
    helper = LayerHelper('gather_nd', **locals())
    dtype = helper.input_dtype()
    if name is None:
        output = helper.create_variable_for_type_inference(dtype)
    else:
        output = helper.create_variable(
            name=name, dtype=dtype, persistable=False)
    helper.append_op(
        type="gather_nd",
        inputs={"X": input,
                "Index": index},
        outputs={"Out": output})
    return output


def scatter(input, index, updates, name=None, overwrite=True):
    """
    **Scatter Layer**

    Output is obtained by updating the input on selected indices on the first
    axis.

    .. math::

        Out = X
        Out[Ids] = Updates

    Args:
        input (Variable): The source input with rank>=1.
        index (Variable): The index input with rank=1. Its dtype should be
                          int32 or int64 as it is used as indexes.
        updates (Variable): The updated value of scatter op.
        name (str|None): The output variable name. Default None.
        overwrite (bool): The mode that updating the output when has same index.
            If True, use the overwrite mode to update the output of the same index,
	    if False, use the accumulate mode to update the output of the same index. 
	    Default value is True.You can set overwrite=False to implement scatter_add.

    Returns:
        output (Variable): The output is a tensor with the same shape as input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            input = fluid.layers.data(name='data', shape=[3, 5, 9], dtype='float32', append_batch_size=False)
            index = fluid.layers.data(name='index', shape=[3], dtype='int64', append_batch_size=False)
            updates = fluid.layers.data(name='update', shape=[3, 5, 9], dtype='float32', append_batch_size=False)

            output = fluid.layers.scatter(input, index, updates)
    """
    helper = LayerHelper('scatter', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="scatter",
        inputs={"X": input,
                "Ids": index,
                "Updates": updates},
        attrs={'overwrite': overwrite},
        outputs={"Out": out})
    return out


def scatter_nd_add(ref, index, updates, name=None):
    """
    **Scatter_nd_add Layer**

    Output is obtained by applying sparse addition to a single value
    or slice in a Variable. :attr:`ref` is a Tensor with rank :math:`R` 
    and :attr:`index` is a Tensor with rank :math:`K` . Thus, :attr:`index` 
    has shape :math:`[i_0, i_1, ..., i_{K-2}, Q]` where :math:`Q \leq R` . :attr:`updates` 
    is a Tensor with rank :math:`K - 1 + R - Q` and its
    shape is :math:`index.shape[:-1] + ref.shape[index.shape[-1]:]` .
    According to the :math:`[i_0, i_1, ..., i_{K-2}]` of :attr:`index` ,
    add the corresponding :attr:`updates` slice to the :attr:`ref` slice
    which is obtained by the last one dimension of :attr:`index` .

    .. code-block:: text
        
        Given:

        * Case 1:
            ref = [0, 1, 2, 3, 4, 5]
            index = [[1], [2], [3], [1]]
            updates = [9, 10, 11, 12]

          we get:
             
            output = [0, 22, 12, 14, 4, 5]

        * Case 2:
            ref = [[65, 17], [-14, -25]]
            index = [[], []]
            updates = [[[-1, -2], [1, 2]],
                       [[3, 4], [-3, -4]]]
            ref.shape = (2, 2)
            index.shape = (2, 0)
            updates.shape = (2, 2, 2)

          we get:
             
            output = [[67, 19], [-16, -27]]

    Args:
        ref (Variable): The ref input.
        index (Variable): The index input with rank > 1 and index.shape[-1] <= ref.rank.
                          Its dtype should be int32 or int64 as it is used as indexes.
        updates (Variable): The updated value of scatter_nd_add op, and it must have the same type
                            as ref. It must have the shape index.shape[:-1] + ref.shape[index.shape[-1]:]
        name (str|None): The output variable name. Default None.

    Returns:
        output (Variable): The output is a tensor with the same shape and type as ref.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            ref = fluid.layers.data(name='ref', shape=[3, 5, 9, 10], dtype='float32', append_batch_size=False)
            index = fluid.layers.data(name='index', shape=[3, 2], dtype='int32', append_batch_size=False)
            updates = fluid.layers.data(name='update', shape=[3, 9, 10], dtype='float32', append_batch_size=False)

            output = fluid.layers.scatter_nd_add(ref, index, updates)
    """
    if ref.dtype != updates.dtype:
        raise ValueError("ref and updates must have same data type.")

    helper = LayerHelper('scatter_nd_add', **locals())
    dtype = helper.input_dtype()
    if name is None:
        output = helper.create_variable_for_type_inference(dtype)
    else:
        output = helper.create_variable(
            name=name, dtype=dtype, persistable=False)
    helper.append_op(
        type="scatter_nd_add",
        inputs={"X": ref,
                "Index": index,
                "Updates": updates},
        outputs={"Out": output})
    return output


def scatter_nd(index, updates, shape, name=None):
    """
    **Scatter_nd Layer**

    Output is obtained by scattering the :attr:`updates` in a new tensor according 
    to :attr:`index` . This op is similar to :code:`scatter_nd_add`, except the 
    tensor of :attr:`shape` is zero-initialized. Correspondingly, :code:`scatter_nd(index, updates, shape)` 
    is equal to :code:`scatter_nd_add(fluid.layers.zeros(shape, updates.dtype), index, updates)` . 
    If :attr:`index` has repeated elements, then the corresponding updates are accumulated. 
    Because of the numerical approximation issues, the different order of repeated elements 
    in :attr:`index` may cause different results. The specific calculation method can be 
    seen :code:`scatter_nd_add` . This op is the inverse of the :code:`gather_nd` op.

    Args:
        index (Variable): The index input with rank > 1 and index.shape[-1] <= len(shape).
                          Its dtype should be int32 or int64 as it is used as indexes.
        updates (Variable): The updated value of scatter_nd op. 
                            It must have the shape index.shape[:-1] + shape[index.shape[-1]:]
        shape(tuple|list): Shape of output tensor.
        name (str|None): The output variable name. Default None.

    Returns:
        output (Variable): The output is a tensor with the same type as :attr:`updates` .

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            index = fluid.layers.data(name='index', shape=[3, 2], dtype='int64', append_batch_size=False)
            updates = fluid.layers.data(name='update', shape=[3, 9, 10], dtype='float32', append_batch_size=False)
            shape = [3, 5, 9, 10]

            output = fluid.layers.scatter_nd(index, updates, shape)
    """
    return scatter_nd_add(zeros(shape, updates.dtype), index, updates, name)


def sequence_scatter(input, index, updates, name=None):
    """
    **Sequence Scatter Layer**

    This operator scatters the Updates tensor to the input X. It uses the LoD
    information of Ids to select the rows to update, and use the values in Ids as
    the columns to update in each row of X.

    Here is an example:

    Given the following input:

    .. code-block:: text

        input.data = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        input.dims = [3, 6]

        index.data = [[0], [1], [2], [5], [4], [3], [2], [1], [3], [2], [5], [4]]
        index.lod =  [[0,        3,                       8,                 12]]

        updates.data = [[0.3], [0.3], [0.4], [0.1], [0.2], [0.3], [0.4], [0.0], [0.2], [0.3], [0.1], [0.4]]
        updates.lod =  [[  0,            3,                                 8,                         12]]

    Then we have the output:

    .. code-block:: text

        out.data = [[1.3, 1.3, 1.4, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.4, 1.3, 1.2, 1.1],
                    [1.0, 1.0, 1.3, 1.2, 1.4, 1.1]]
        out.dims = X.dims = [3, 6]

    Args:
        input (Variable): The source input with rank>=1.
        index (Variable): A LoD Tensor. The index input of sequence scatter op
            where input will be  updated. The index input with rank=1. Its dtype
            should be int32 or int64 as it is used as indexes.
        updates (Variable): A LoD Tensor. The values to scatter to the input
            tensor X, must be a LoDTensor with the same LoD information as index.
        name (str|None): The output variable name. Default None.

    Returns:
        Variable: The output is a tensor with the same shape as input.

    Examples:

        .. code-block:: python
	
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            input = layers.data( name="x", shape=[3, 6], append_batch_size=False, dtype='float32' )
            index = layers.data( name='index', shape=[1], dtype='int32')
            updates = layers.data( name='updates', shape=[1], dtype='float32')
            output = fluid.layers.sequence_scatter(input, index, updates)

    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_scatter', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="sequence_scatter",
        inputs={"X": input,
                "Ids": index,
                "Updates": updates},
        outputs={"Out": out})
    return out


@templatedoc()
def random_crop(x, shape, seed=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        shape(${shape_type}): ${shape_comment}
        seed(int|${seed_type}|None): ${seed_comment} By default, the seed will
            get from `random.randint(-65536, 65535)`.

    Returns:
        ${out_comment}

    Examples:
        >>> import paddle.fluid as fluid
        >>> img = fluid.layers.data("img", [3, 256, 256])
        >>> cropped_img = fluid.layers.random_crop(img, shape=[3, 224, 224])
    """
    helper = LayerHelper("random_crop", **locals())
    dtype = x.dtype
    out = helper.create_variable_for_type_inference(dtype)
    if seed is None:
        seed = np.random.randint(-65536, 65536)
    op_attrs = {"shape": shape}
    if isinstance(seed, int):
        op_attrs["startup_seed"] = seed
        seed = helper.create_variable(
            name=unique_name.generate("random_crop_seed"),
            dtype="int64",
            persistable=True)
    elif not isinstance(seed, Variable):
        raise ValueError("'seed' must be a Variable or an int.")
    helper.append_op(
        type="random_crop",
        inputs={"X": x,
                "Seed": seed},
        outputs={"Out": out,
                 "SeedOut": seed},
        attrs=op_attrs)
    return out


def log(x, name=None):
    """
    Calculates the natural log of the given input tensor, element-wise.

    .. math::

        Out = \\ln(x)

    Args:
        x (Variable): Input tensor.
        name (str|None, default None): A name for this layer If set None,
            the layer will be named automatically.

    Returns:
        Variable: The natural log of the input tensor computed element-wise.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[3, 4], dtype="float32")
            output = fluid.layers.log(x)
    """
    helper = LayerHelper('log', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type="log", inputs={"X": x}, outputs={"Out": out})
    return out


def relu(x, name=None):
    """
    Relu takes one input data (Tensor) and produces one output data (Tensor)
    where the rectified linear function, y = max(0, x), is applied to
    the tensor elementwise.

    .. math::

        Out = \\max(0, x)

    Args:
        x (Variable): The input tensor.
        name (str|None, default None): A name for this layer If set None,
            the layer will be named automatically.

    Returns:
        Variable: The output tensor with the same shape as input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[3, 4], dtype="float32")
            output = fluid.layers.relu(x)
    """
    helper = LayerHelper('relu', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="relu", inputs={"X": helper.input('x')}, outputs={"Out": out})
    return out


@templatedoc()
def selu(x, scale=None, alpha=None, name=None):
    """
    ${comment}

    Args:
        x (Variable): The input tensor.
        scale(float, None): If the scale is not set,
            the default value is 1.0507009873554804934193349852946.
            For more information about this value, please refer
            to: https://arxiv.org/abs/1706.02515.
        alpha(float, None): If the alpha is not set,
            the default value is 1.6732632423543772848170429916717.
            For more information about this value, please refer
            to: https://arxiv.org/abs/1706.02515.
        name (str|None, default None): A name for this layer If set None,
            the layer will be named automatically.

    Returns:
        Variable: The output tensor with the same shape as input.

    Examples:

        .. code-block:: python
             
            import paddle.fluid as fluid
          
            input = fluid.layers.data(
                 name="input", shape=[3, 9, 5], dtype="float32")
            output = fluid.layers.selu(input)
    """
    helper = LayerHelper('selu', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    attrs = {}
    if scale is not None:
        attrs["scale"] = scale
    if alpha is not None:
        attrs["alpha"] = alpha

    helper.append_op(
        type="selu", inputs={"X": x}, outputs={"Out": out}, attrs=attrs)
    return out


def mean_iou(input, label, num_classes):
    """
    Mean Intersection-Over-Union is a common evaluation metric for
    semantic image segmentation, which first computes the IOU for each
    semantic class and then computes the average over classes.
    IOU is defined as follows:

    .. math::

        IOU = \\frac{true\_positive}{(true\_positive + false\_positive + false\_negative)}.

    The predictions are accumulated in a confusion matrix and mean-IOU
    is then calculated from it.


    Args:
        input (Variable): A Tensor of prediction results for semantic labels with type int32 or int64.
        label (Variable): A Tensor of ground truth labels with type int32 or int64.
                           Its shape should be the same as input.
        num_classes (int): The possible number of labels.

    Returns:
        mean_iou (Variable),out_wrong(Variable),out_correct(Variable):

                     Three variables:

                     - mean_iou : A Tensor representing the mean intersection-over-union with shape [1].
                     - out_wrong: A Tensor with shape [num_classes]. The wrong numbers of each class.
                     - out_correct: A Tensor with shape [num_classes]. The correct numbers of each class.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            iou_shape = [32, 32]
            num_classes = 5
            predict = fluid.layers.data(name='predict', shape=iou_shape)
            label = fluid.layers.data(name='label', shape=iou_shape)
            iou, wrongs, corrects = fluid.layers.mean_iou(predict, label,
                                                          num_classes)
    """
    helper = LayerHelper('mean_iou', **locals())
    dtype = helper.input_dtype()
    out_mean_iou = helper.create_variable_for_type_inference(dtype='float32')
    out_wrong = helper.create_variable_for_type_inference(dtype='int32')
    out_correct = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type="mean_iou",
        inputs={"Predictions": input,
                "Labels": label},
        outputs={
            "OutMeanIou": out_mean_iou,
            "OutWrong": out_wrong,
            "OutCorrect": out_correct
        },
        attrs={"num_classes": num_classes})
    return out_mean_iou, out_wrong, out_correct


def crop(x, shape=None, offsets=None, name=None):
    """
    Crop input into output, as specified by offsets and shape.

    **Warning:** THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
    Instructions for updating: Use `fluid.layers.crop_tensor
    <https://www.paddlepaddle.org.cn/documentation/docs/en/api/layers/nn.html#crop_tensor>`_
    instead.

    .. code-block:: text

        * Case 1:
            Given
                X = [[0, 1, 2, 0, 0]
                     [0, 3, 4, 0, 0]
                     [0, 0, 0, 0, 0]],
            and
                shape = [2, 2],
                offsets = [0, 1],
            output is:
                Out = [[1, 2],
                       [3, 4]].
        * Case 2:
            Given
                X = [[0, 1, 2, 5, 0]
                     [0, 3, 4, 6, 0]
                     [0, 0, 0, 0, 0]],
            and shape is tensor
                shape = [[0, 0, 0]
                         [0, 0, 0]]
            and
                offsets = [0, 1],

            output is:
                Out = [[1, 2, 5],
                       [3, 4, 6]].

    Args:
        x (Variable): The input tensor variable.
        shape (Variable|list/tuple of integer): The output shape is specified
            by `shape`, which can be a Variable or a list/tuple of integer.
            If a tensor Variable, it's rank must be the same as `x`. This way
            is suitable for the case that the output shape may be changed each
            iteration. If a list/tuple of integer, it's length must be the same
            as the rank of `x`
        offsets (Variable|list/tuple of integer|None): Specifies the cropping
            offsets at each dimension. It can be a Variable or a list/tuple
            of integers. If a tensor Variable, it's rank must be the same as `x`.
            This way is suitable for the case that the offsets may be changed
            each iteration. If a list/tuple of integer, it's length must be the
            same as the rank of `x`. If None, the offsets are 0 at each
            dimension.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The cropped tensor variable.

    Raises:
        ValueError: If shape is not a list, tuple or Variable.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[3, 5], dtype="float32")
            y = fluid.layers.data(name="y", shape=[2, 3], dtype="float32")
            crop = fluid.layers.crop(x, shape=y)

            # or
            z = fluid.layers.data(name="z", shape=[3, 5], dtype="float32")
            crop = fluid.layers.crop(z, shape=[-1, 2, 3])

    """
    helper = LayerHelper('crop', **locals())

    if not (isinstance(shape, list) or isinstance(shape, tuple) or \
            isinstance(shape, Variable)):
        raise ValueError("The shape should be a list, tuple or Variable.")

    if offsets is None:
        offsets = [0] * len(x.shape)

    out = helper.create_variable_for_type_inference(x.dtype)
    ipts = {'X': x}
    attrs = {}
    if isinstance(shape, Variable):
        ipts['Y'] = shape
    else:
        attrs['shape'] = shape
    if isinstance(offsets, Variable):
        ipts['Offsets'] = offsets
    else:
        attrs['offsets'] = offsets

    helper.append_op(
        type='crop',
        inputs=ipts,
        outputs={'Out': out},
        attrs=None if len(attrs) == 0 else attrs)
    return out


def crop_tensor(x, shape=None, offsets=None, name=None):
    """
    Crop input into output, as specified by offsets and shape.

    .. code-block:: text

        * Case 1:
            Given
                X = [[0, 1, 2, 0, 0]
                     [0, 3, 4, 0, 0]
                     [0, 0, 0, 0, 0]],
            and
                shape = [2, 2],
                offsets = [0, 1],
            output is:
                Out = [[1, 2],
                       [3, 4]].
        * Case 2:
            Given
                X =  [[[0, 1, 2, 3]
                       [0, 5, 6, 7]
                       [0, 0, 0, 0]],

                      [[0, 3, 4, 5]
                       [0, 6, 7, 8]
                       [0, 0, 0, 0]]].
            and
                shape = [2, 2, 3],
                offsets = [0, 0, 1],
            output is:
                Out = [[[1, 2, 3]
                        [5, 6, 7]],

                        [[3, 4, 5]
                         [6, 7, 8]]].

    Args:
        x (Variable): The input tensor variable.
        shape (Variable|list|tuple of integer): The output shape is specified
            by `shape`. It can be a 1-D tensor Variable or a list/tuple. If a 
            1-D tensor Variable, it's rank must be the same as `x`. If a 
            list/tuple, it's length must be the same as the rank of `x`. Each 
            element of list can be an integer or a tensor Variable of shape: [1].
            If Variable contained, it is suitable for the case that the shape may 
            be changed each iteration. Only the first element of list/tuple can be 
            set to -1, it means that the first dimension of the output is the same 
            as the input.
        offsets (Variable|list|tuple of integer|None): Specifies the cropping
            offsets at each dimension. It can be a 1-D tensor Variable or a list/tuple.
            If a 1-D tensor Variable, it's rank must be the same as `x`. If a list/tuple, 
            it's length must be the same as the rank of `x`. Each element of list can be
            an integer or a tensor Variable of shape: [1]. If Variable contained, it is 
            suitable for the case that the offsets may be changed each iteration. If None, 
            the offsets are 0 at each dimension.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The cropped tensor variable.

    Raises:
        ValueError: If shape is not a list, tuple or Variable.
        ValueError: If offsets is not None and not a list, tuple or Variable.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[3, 5], dtype="float32")
            # x.shape = [-1, 3, 5], where -1 indicates batch size, and it will get the exact value in runtime.

            # shape is a 1-D tensor variable
            crop_shape = fluid.layers.data(name="crop_shape", shape=[3], dtype="int32", append_batch_size=False)
            crop0 = fluid.layers.crop_tensor(x, shape=crop_shape)
            # crop0.shape = [-1, -1, -1], it means crop0.shape[0] = x.shape[0] in runtime.

            # or shape is a list in which each element is a constant
            crop1 = fluid.layers.crop_tensor(x, shape=[-1, 2, 3])
            # crop1.shape = [-1, 2, 3]

            # or shape is a list in which each element is a constant or variable
            y = fluid.layers.data(name="y", shape=[3, 8, 8], dtype="float32")
            dim1 = fluid.layers.data(name="dim1", shape=[1], dtype="int32", append_batch_size=False)
            crop2 = fluid.layers.crop_tensor(y, shape=[-1, 3, dim1, 4])
            # crop2.shape = [-1, 3, -1, 4]

            # offsets is a 1-D tensor variable
            crop_offsets = fluid.layers.data(name="crop_offsets", shape=[3], dtype="int32", append_batch_size=False)
            crop3 = fluid.layers.crop_tensor(x, shape=[-1, 2, 3], offsets=crop_offsets)
            # crop3.shape = [-1, 2, 3]

            # offsets is a list in which each element is a constant or variable
            offsets_var =  fluid.layers.data(name="dim1", shape=[1], dtype="int32", append_batch_size=False)
            crop4 = fluid.layers.crop_tensor(x, shape=[-1, 2, 3], offsets=[0, 1, offsets_var])
            # crop4.shape = [-1, 2, 3]

    """
    helper = LayerHelper('crop_tensor', **locals())

    if not (isinstance(shape, list) or isinstance(shape, tuple) or \
            isinstance(shape, Variable)):
        raise ValueError("The shape should be a list, tuple or Variable.")

    if offsets is None:
        offsets = [0] * len(x.shape)

    if not (isinstance(offsets, list) or isinstance(offsets, tuple) or \
            isinstance(offsets, Variable)):
        raise ValueError("The offsets should be a list, tuple or Variable.")

    out = helper.create_variable_for_type_inference(x.dtype)
    ipts = {'X': x}
    attrs = {}

    def contain_var(input_list):
        for ele in input_list:
            if isinstance(ele, Variable):
                return True
        return False

    if isinstance(offsets, Variable):
        offsets.stop_gradient = True
        ipts['Offsets'] = offsets
    elif contain_var(offsets):
        new_offsets_tensor = []
        for dim in offsets:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_offsets_tensor.append(dim)
            else:
                assert (isinstance(dim, int))
                assert dim >= 0, ("offsets should be greater or equal to zero.")
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', dim, force_cpu=True, out=temp_out)
                new_offsets_tensor.append(temp_out)
        ipts['OffsetsTensor'] = new_offsets_tensor
    else:
        attrs['offsets'] = offsets

    unk_dim_idx = -1
    if isinstance(shape, Variable):
        shape.stop_gradient = True
        ipts['Shape'] = shape
    elif contain_var(shape):
        new_shape_tensor = []
        shape_attr = []
        for dim_idx, dim_size in enumerate(shape):
            if isinstance(dim_size, Variable):
                dim_size.stop_gradient = True
                new_shape_tensor.append(dim_size)
                shape_attr.append(-1)
            else:
                assert (isinstance(dim_size, int))
                if dim_size == -1:
                    assert unk_dim_idx == -1, (
                        "Only one element in shape can be unknown.")
                    assert dim_idx == 0, (
                        "Only the first element in shape can be -1.")
                    unk_dim_idx = dim_idx
                else:
                    assert dim_size > 0, (
                        "Each dimension size given in shape must be greater than zero."
                    )
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant(
                    [1], 'int32', dim_size, force_cpu=True, out=temp_out)
                new_shape_tensor.append(temp_out)
                shape_attr.append(dim_size)
        ipts['ShapeTensor'] = new_shape_tensor
        attrs['shape'] = shape_attr
    else:
        attrs['shape'] = shape

    helper.append_op(
        type='crop_tensor',
        inputs=ipts,
        outputs={'Out': out},
        attrs=None if len(attrs) == 0 else attrs)
    return out


def affine_grid(theta, out_shape, name=None):
    """
    It generates a grid of (x,y) coordinates using the parameters of
    the affine transformation that correspond to a set of points where
    the input feature map should be sampled to produce the transformed
    output feature map.

    .. code-block:: text

        * Case 1:

          Given:

              theta = [[[x_11, x_12, x_13]
                        [x_14, x_15, x_16]]
                       [[x_21, x_22, x_23]
                        [x_24, x_25, x_26]]]

              out_shape = [2, 3, 5, 5]

          Step 1:

              Generate normalized coordinates according to out_shape.
              The values of the normalized coordinates are in the interval between -1 and 1.
              The shape of the normalized coordinates is [2, H, W] as below:

              C = [[[-1.  -1.  -1.  -1.  -1. ]
                    [-0.5 -0.5 -0.5 -0.5 -0.5]
                    [ 0.   0.   0.   0.   0. ]
                    [ 0.5  0.5  0.5  0.5  0.5]
                    [ 1.   1.   1.   1.   1. ]]
                   [[-1.  -0.5  0.   0.5  1. ]
                    [-1.  -0.5  0.   0.5  1. ]
                    [-1.  -0.5  0.   0.5  1. ]
                    [-1.  -0.5  0.   0.5  1. ]
                    [-1.  -0.5  0.   0.5  1. ]]]
              C[0] is the coordinates in height axis and  C[1] is the coordinates in width axis.

          Step2:

              Tanspose and reshape C to shape [H * W, 2] and append ones to last dimension. The we get:
              C_ = [[-1.  -1.   1. ]
                    [-0.5 -1.   1. ]
                    [ 0.  -1.   1. ]
                    [ 0.5 -1.   1. ]
                    [ 1.  -1.   1. ]
                    [-1.  -0.5  1. ]
                    [-0.5 -0.5  1. ]
                    [ 0.  -0.5  1. ]
                    [ 0.5 -0.5  1. ]
                    [ 1.  -0.5  1. ]
                    [-1.   0.   1. ]
                    [-0.5  0.   1. ]
                    [ 0.   0.   1. ]
                    [ 0.5  0.   1. ]
                    [ 1.   0.   1. ]
                    [-1.   0.5  1. ]
                    [-0.5  0.5  1. ]
                    [ 0.   0.5  1. ]
                    [ 0.5  0.5  1. ]
                    [ 1.   0.5  1. ]
                    [-1.   1.   1. ]
                    [-0.5  1.   1. ]
                    [ 0.   1.   1. ]
                    [ 0.5  1.   1. ]
                    [ 1.   1.   1. ]]
          Step3:
              Compute output by equation $$Output[i] = C_ * Theta[i]^T$$

    Args:
        theta (Variable): A batch of affine transform parameters with shape [N, 2, 3].
        out_shape (Variable | list | tuple): The shape of target output with format [N, C, H, W].
                                             ``out_shape`` can be a Variable or a list or tuple.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The output with shape [N, H, W, 2].

    Raises:
        ValueError: If the type of arguments is not supported.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            theta = fluid.layers.data(name="x", shape=[2, 3], dtype="float32")
            out_shape = fluid.layers.data(name="y", shape=[-1], dtype="float32")
            data = fluid.layers.affine_grid(theta, out_shape)

            # or
            data = fluid.layers.affine_grid(theta, [5, 3, 28, 28])

    """
    helper = LayerHelper('affine_grid')

    if not (isinstance(out_shape, list) or isinstance(out_shape, tuple) or \
            isinstance(out_shape, Variable)):
        raise ValueError("The out_shape should be a list, tuple or Variable.")

    if not isinstance(theta, Variable):
        raise ValueError("The theta should be a Variable.")

    out = helper.create_variable_for_type_inference(theta.dtype)
    ipts = {'Theta': theta}
    attrs = {}
    if isinstance(out_shape, Variable):
        ipts['OutputShape'] = out_shape
    else:
        attrs['output_shape'] = out_shape

    helper.append_op(
        type='affine_grid',
        inputs=ipts,
        outputs={'Output': out},
        attrs=None if len(attrs) == 0 else attrs)
    return out


def rank_loss(label, left, right, name=None):
    """

    **Rank loss layer for RankNet**

    `RankNet <http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf>`_
    is a pairwise ranking model with a training sample consisting of a pair
    of documents, A and B. Label P indicates whether A is ranked higher than B
    or not:

    P = {0, 1} or {0, 0.5, 1}, where 0.5 means that there is no information
    about the rank of the input pair.

    Rank loss layer takes three inputs: left ( :math:`o_i` ), right ( :math:`o_j` ) and
    label ( :math:`P_{i,j}` ). The inputs respectively represent RankNet's output scores
    for documents A and B and the value of label P. The following equation
    computes rank loss C_{i,j} from the inputs:

    .. math::

      C_{i,j} &= -\\tilde{P_{ij}} * o_{i,j} + \log(1 + e^{o_{i,j}}) \\\\

      o_{i,j} &=  o_i - o_j  \\\\

      \\tilde{P_{i,j}} &= \\left \{0, 0.5, 1 \\right \} \ or \ \\left \{0, 1 \\right \}


    Rank loss layer takes batch inputs with size batch_size (batch_size >= 1).

    Args:
        label (Variable): Indicats whether A ranked higher than B or not.
        left (Variable): RankNet's output score for doc A.
        right (Variable): RankNet's output score for doc B.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        list: The value of rank loss.

    Raises:
        ValueError: Any of label, left, and right is not a variable.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            label = fluid.layers.data(name="label", shape=[-1, 1], dtype="float32")
            left = fluid.layers.data(name="left", shape=[-1, 1], dtype="float32")
            right = fluid.layers.data(name="right", shape=[-1, 1], dtype="float32")
            out = fluid.layers.rank_loss(label, left, right)

    """
    helper = LayerHelper('rank_loss', **locals())

    if not (isinstance(label, Variable)):
        raise ValueError("The label should be a Variable")

    if not (isinstance(left, Variable)):
        raise ValueError("The left should be a Variable")

    if not (isinstance(right, Variable)):
        raise ValueError("The right should be a Variable")

    out = helper.create_variable_for_type_inference("float32")

    helper.append_op(
        type='rank_loss',
        inputs={"Label": label,
                "Left": left,
                "Right": right},
        outputs={'Out': out})
    return out


def margin_rank_loss(label, left, right, margin=0.1, name=None):
    """
    Margin Ranking Loss Layer for ranking problem,
    which compares left score and right score passed in.
    The ranking loss can be defined as following equation:

    .. math::

        rank\_loss = max(0, -label * (left - right) + margin)

    Args:
       label (Variable): Indicates whether the left is ranked higher than the right or not.
       left (Variable): Ranking score for left.
       right (Variable): Ranking score for right.
       margin (float): Indicates the given margin.
       name (str|None): A name for this layer (optional). If set None, the layer
                       will be named automatically.

    Returns:
       Variable: The ranking loss.

    Raises:
       ValueError: Any of label, left, and right is not a Variable.

    Examples:

        .. code-block:: python

           import paddle.fluid as fluid
           label = fluid.layers.data(name="label", shape=[-1, 1], dtype="float32")
           left = fluid.layers.data(name="left", shape=[-1, 1], dtype="float32")
           right = fluid.layers.data(name="right", shape=[-1, 1], dtype="float32")
           out = fluid.layers.margin_rank_loss(label, left, right)
    """
    helper = LayerHelper('margin_rank_loss', **locals())
    if not isinstance(label, Variable):
        raise ValueError("The label should be a Variable.")
    if not isinstance(left, Variable):
        raise ValueError("The left should be a Variable.")
    if not isinstance(right, Variable):
        raise ValueError("The right should be a Variable.")
    out = helper.create_variable_for_type_inference(left.dtype)
    act = helper.create_variable_for_type_inference(left.dtype)
    helper.append_op(
        type='margin_rank_loss',
        inputs={"Label": label,
                "X1": left,
                "X2": right},
        outputs={'Out': out,
                 'Activated': act},
        attrs={'margin': margin})
    return out


def pad2d(input,
          paddings=[0, 0, 0, 0],
          mode='constant',
          pad_value=0.0,
          data_format="NCHW",
          name=None):
    """
    Pad 2-d images accordding to 'paddings' and 'mode'.
    If mode is 'reflect', paddings[0] and paddings[1] must be no greater
    than height-1. And the width dimension has the same condition.

    Example:
        .. code-block:: text

	      Given that X is a channel of image from input:

	      X = [[1, 2, 3],
		   [4, 5, 6]]

	      Case 0:

		paddings = [0, 1, 2, 3],
		mode = 'constant'
		pad_value = 0

		Out = [[0, 0, 1, 2, 3, 0, 0, 0]
		       [0, 0, 4, 5, 6, 0, 0, 0]
		       [0, 0, 0, 0, 0, 0, 0, 0]]

	      Case 1:

		paddings = [0, 1, 2, 1],
		mode = 'reflect'

		Out = [[3, 2, 1, 2, 3, 2]
		       [6, 5, 4, 5, 6, 5]
		       [3, 2, 1, 2, 3, 2]]

	      Case 2:

		paddings = [0, 1, 2, 1],
		mode = 'edge'

		Out = [[1, 1, 1, 2, 3, 3]
		       [4, 4, 4, 5, 6, 6]
		       [4, 4, 4, 5, 6, 6]]


    Args:
        input (Variable): The input image with [N, C, H, W] format or [N, H, W, C] format.
        paddings (tuple|list|Variable): The padding size. If padding is a tuple, it must
            contain four integers, (padding_top, padding_bottom, padding_left, padding_right).
            Default: padding = [0, 0, 0, 0].
        mode (str): Three modes: constant(default), reflect, edge. Default: constant
        pad_value (float32): The value to fill the padded areas in constant mode. Default: 0
        data_format (str): An optional string from: "NHWC", "NCHW". Specify the data format of
                           the input data.
                           Default: "NCHW"
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The tensor variable padded accordding to paddings and mode.


    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='data', shape=[3, 32, 32],
                                   dtype='float32')
          result = fluid.layers.pad2d(input=data, paddings=[1, 2, 3, 4],
                                      mode='reflect')
    """

    helper = LayerHelper('pad2d', **locals())

    assert mode in ['reflect', 'edge', 'constant'
                    ], "mode should be one of constant, reflect, edge."

    dtype = helper.input_dtype(input_param_name='input')
    out = helper.create_variable_for_type_inference(dtype)
    inputs = {'X': input}
    attrs = {'mode': mode, 'pad_value': pad_value, 'data_format': data_format}

    if isinstance(paddings, Variable):
        inputs['Paddings'] = paddings
        attrs['paddings'] = []
    else:
        attrs['paddings'] = paddings

    helper.append_op(
        type='pad2d', inputs=inputs, outputs={"Out": out}, attrs=attrs)

    return out


@templatedoc()
def elu(x, alpha=1.0, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        alpha(${alpha_type}|1.0): ${alpha_comment}
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        output(${out_type}): ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
            y = fluid.layers.elu(x, alpha=0.2)
    """
    helper = LayerHelper('elu', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='elu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'alpha': alpha})
    return out


@templatedoc()
def relu6(x, threshold=6.0, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        threshold(${threshold_type}|6.0): ${threshold_comment}
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        output(${out_type}): ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
            y = fluid.layers.relu6(x, threshold=6.0)
    """
    helper = LayerHelper('relu6', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='relu6',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'threshold': threshold})
    return out


@templatedoc()
def pow(x, factor=1.0, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        factor(float|Variable|1.0): The exponential factor of Pow.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        output(${out_type}): ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")

            # example 1: argument factor is float
            y_1 = fluid.layers.pow(x, factor=2.0)

            # example 2: argument factor is Variable
            factor_tensor = fluid.layers.fill_constant([1], "float32", 3.0)
            y_2 = fluid.layers.pow(x, factor=factor_tensor)
    """
    helper = LayerHelper('pow', **locals())
    inputs = {'X': x}
    attrs = {}
    if isinstance(factor, Variable):
        factor.stop_gradient = True
        inputs['FactorTensor'] = factor
    else:
        attrs['factor'] = factor

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='pow', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


@templatedoc()
def stanh(x, scale_a=2.0 / 3.0, scale_b=1.7159, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        scale_a(${scale_a_type}|2.0 / 3.0): ${scale_a_comment}
        scale_b(${scale_b_type}|1.7159): ${scale_b_comment}
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        output(${out_type}): ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
            y = fluid.layers.stanh(x, scale_a=0.67, scale_b=1.72)
    """
    helper = LayerHelper('stanh', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='stanh',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'scale_a': scale_a,
               'scale_b': scale_b})
    return out


@templatedoc()
def hard_sigmoid(x, slope=0.2, offset=0.5, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        slope(${slope_type}|0.2): ${slope_comment}
        offset(${offset_type}|0.5): ${offset_comment}
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        output(${out_type}): ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
            y = fluid.layers.hard_sigmoid(x, slope=0.3, offset=0.8)
    """
    helper = LayerHelper('hard_sigmoid', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='hard_sigmoid',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'slope': slope,
               'offset': offset})
    return out


@templatedoc()
def swish(x, beta=1.0, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        beta(${beta_type}|1.0): ${beta_comment}
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        output(${out_type}): ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
            y = fluid.layers.swish(x, beta=2.0)
    """
    helper = LayerHelper('swish', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='swish',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'slope': beta})
    return out


def prelu(x, mode, param_attr=None, name=None):
    """
    Equation:

    .. math::
        y = \max(0, x) + \\alpha * \min(0, x)

    There are three modes for the activation:

    .. code-block:: text

        all: All elements share same alpha.
        channel: Elements in same channel share same alpha.
        element: All elements do not share alpha. Each element has its own alpha.

    Args:
        x (Variable): The input tensor.
        mode (string): The mode for weight sharing. 
        param_attr(ParamAttr|None): The parameter attribute for the learnable
          weight (alpha), it can be create by ParamAttr.
        name(str|None): A name for this layer(optional). If set None, the layer
          will be named automatically.

    Returns:
        Variable: The output tensor with the same shape as input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            from paddle.fluid.param_attr import ParamAttr
            x = fluid.layers.data(name="x", shape=[5,10,10], dtype="float32")
            mode = 'channel'
            output = fluid.layers.prelu(
                     x,mode,param_attr=ParamAttr(name='alpha'))

    """
    helper = LayerHelper('prelu', **locals())
    if mode not in ['all', 'channel', 'element']:
        raise ValueError('mode should be one of all, channel, element.')
    alpha_shape = [1]
    if mode == 'channel':
        alpha_shape = [1, x.shape[1], 1, 1]
    elif mode == 'element':
        alpha_shape = x.shape
    dtype = helper.input_dtype(input_param_name='x')
    alpha = helper.create_parameter(
        attr=helper.param_attr,
        shape=alpha_shape,
        dtype='float32',
        is_bias=False,
        default_initializer=Constant(1.0))
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="prelu",
        inputs={"X": x,
                'Alpha': alpha},
        attrs={"mode": mode},
        outputs={"Out": out})
    return out


@templatedoc()
def brelu(x, t_min=0.0, t_max=24.0, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        t_min(${t_min_type}|0.0): ${t_min_comment}
        t_max(${t_max_type}|24.0): ${t_max_comment}
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.
    Returns:
        output(${out_type}): ${out_comment}

    Examples:

    .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[2,3,16,16], dtype="float32")
            y = fluid.layers.brelu(x, t_min=1.0, t_max=20.0)
    """
    helper = LayerHelper('brelu', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='brelu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'t_min': t_min,
               't_max': t_max})
    return out


@templatedoc()
def leaky_relu(x, alpha=0.02, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        alpha(${alpha_type}|0.02): ${alpha_comment}
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.
    Returns:
        output(${out_type}): ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[2,3,16,16], dtype="float32")
            y = fluid.layers.leaky_relu(x, alpha=0.01)
    """
    helper = LayerHelper('leaky_relu', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='leaky_relu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'alpha': alpha})
    return out


@templatedoc()
def soft_relu(x, threshold=40.0, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        threshold(${threshold_type}|40.0): ${threshold_comment}
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.
    Returns:
        output(${out_type}): ${out_comment}

    Examples:

        .. code-block:: python 
 
            import paddle.fluid as fluid
   
            x = fluid.layers.data(name="x", shape=[3,16,16], dtype="float32")
            y = fluid.layers.soft_relu(x, threshold=20.0)
    """
    helper = LayerHelper('soft_relu', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='soft_relu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'threshold': threshold})
    return out


def flatten(x, axis=1, name=None):
    """
    **Flatten layer**
    Flattens the input tensor into a 2D matrix.

    For Example:

    .. code-block:: text

        Case 1:

          Given
            X.shape = (3, 100, 100, 4)

          and
            axis = 2

          We get:
            Out.shape = (3 * 100, 4 * 100)

        Case 2:

          Given
            X.shape = (3, 100, 100, 4)

          and
            axis = 0

          We get:
            Out.shape = (1, 3 * 100 * 100 * 4)

    Args:
        x (Variable): A tensor of rank >= axis.
        axis (int): Indicate up to which input dimensions (exclusive) should
                    be flattened to the outer dimension of the output.
                    The value for axis must be in the range [0, R], where R
                    is the rank of the input tensor. When axis = 0, the shape
                    of the output tensor is (1, (d_0 X d_1 ... d_n), where the
                    shape of the input tensor is (d_0, d_1, ... d_n).
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: A 2D tensor with the contents of the input tensor, with input \
                  dimensions up to axis flattened to the outer dimension of \
                  the output and remaining input dimensions flattened into the \
                  inner dimension of the output.

    Raises:
        ValueError: If x is not a variable.
        ValueError: If axis is not in range [0, rank(x)].

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[4, 4, 3], dtype="float32")
            out = fluid.layers.flatten(x=x, axis=2)
    """
    helper = LayerHelper('flatten', **locals())

    if not (isinstance(x, Variable)):
        raise ValueError("The input x should be a Variable")

    if not (isinstance(axis, int)) or axis > len(x.shape) or axis < 0:
        raise ValueError("The axis should be a int, and in range [0, rank(x)]")

    out = helper.create_variable_for_type_inference(x.dtype)
    x_shape = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='flatten2',
        inputs={"X": x},
        outputs={'Out': out,
                 'XShape': x_shape},
        attrs={"axis": axis})
    return out


def sequence_enumerate(input, win_size, pad_value=0, name=None):
    """
    Generate a new sequence for the input index sequence, which enumerates all the
    sub-sequences with length `win_size` of the input.
    The enumerated sequence has the same 1st dimension with variable `input`, and
    the 2nd dimension is `win_size`, padded by `pad_value` if necessary in generation.

    .. code-block:: text

        Case 1:

          Input:
            X.lod = [[0, 3, 5]]
            X.data = [[1], [2], [3], [4], [5]]
            X.dims = [5, 1]

          Attrs:
            win_size = 2
            pad_value = 0

          Output:
            Out.lod = [[0, 3, 5]]
            Out.data = [[1, 2], [2, 3], [3, 0], [4, 5], [5, 0]]
            Out.dims = [5, 2]

    Args:
        input (Variable): The input variable which is a index sequence.
        win_size (int): The window size for enumerating all sub-sequences.
        pad_value (int): The padding value, default 0.

    Returns:
        Variable: The enumerate sequence variable which is a LoDTensor.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.layers.data(name='x', shape=[-1, 1], dtype='int32', lod_level=1)
            out = fluid.layers.sequence_enumerate(input=x, win_size=3, pad_value=0)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_enumerate', **locals())
    out = helper.create_variable_for_type_inference(
        helper.input_dtype(), stop_gradient=True)
    helper.append_op(
        type='sequence_enumerate',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={'win_size': win_size,
               'pad_value': pad_value})
    return out


def sequence_mask(x, maxlen=None, dtype='int64', name=None):
    """
    **SequenceMask Layer**

    This layer outputs a mask according to the input :code:`x` and
    :code:`maxlen` with data type of :code:`dtype`.

    Supposing :code:`x` is a Tensor with shape [d_1, d_2, ..., d_n], the
    :code:`y` is a mask with shape [d_1, d_2, ..., d_n, maxlen], where:

    .. math::

        y(i_1, i_2,..., i_n, j) = (j < x(i_1, i_2,..., i_n))

    Args:
        x (Variable): Input tensor of sequence_mask layer,
                      whose elements are integers less than :code:`maxlen`.
        maxlen (int|None): Maximum length of the sequence. If :code:`maxlen`
                           is None, it would be replace with :math:`max(x)`.
        dtype (np.dtype|core.VarDesc.VarType|str): Data type of the output.
        name (str|None): A name for this layer(optional). If set None, the
                         layer will be named automatically.

    Returns:
        Variable: The output sequence mask.

    Examples:
        .. code-block:: python
	
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            x = fluid.layers.data(name='x', shape=[10], dtype='float32', lod_level=1)
            mask = layers.sequence_mask(x=x)

    """
    helper = LayerHelper('sequence_mask', **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=dtype, name=name)

    inputs = {'X': [x]}
    attrs = {'out_dtype': out.dtype}
    if maxlen is not None:
        if isinstance(maxlen, Variable):
            inputs['MaxLenTensor'] = maxlen
        else:
            attrs['maxlen'] = maxlen

    helper.append_op(
        type='sequence_mask', inputs=inputs, outputs={'Y': out}, attrs=attrs)

    out.stop_gradient = True
    return out


def stack(x, axis=0):
    """
    **Stack Layer**

    This layer stacks all of the input :code:`x` along axis.

    Input :code:`x` can be a single variable, a :code:`list` of variables,
    or a :code:`tuple` of variables. If :code:`x` is a :code:`list` or
    :code:`tuple`, the shapes of all these variables must be the same.
    Supposing the shape of each input is :math:`[d_0, d_1, ..., d_{n-1}]`,
    the shape of the output variable would be
    :math:`[d_0, d_1, ..., d_{axis}=len(x), ..., d_{n-1}]`.
    If :code:`axis` < 0, it would be replaced with :code:`axis+rank(x[0])+1`.
    If :code:`axis` is None, it would be replaced with 0.

    For Example:

    .. code-block:: text

        Case 1:
          Input:
            x[0].data = [ [1.0 , 2.0 ] ]
            x[0].dims = [1, 2]
            x[1].data = [ [3.0 , 4.0 ] ]
            x[1].dims = [1, 2]
            x[2].data = [ [5.0 , 6.0 ] ]
            x[2].dims = [1, 2]

          Attrs:
            axis = 0

          Output:
            Out.data =[ [ [1.0, 2.0] ],
                        [ [3.0, 4.0] ],
                        [ [5.0, 6.0] ] ]
            Out.dims = [3, 1, 2]

        Case 2:
          Given
            x[0].data = [ [1.0 , 2.0 ] ]
            x[0].dims = [1, 2]
            x[1].data = [ [3.0 , 4.0 ] ]
            x[1].dims = [1, 2]
            x[2].data = [ [5.0 , 6.0 ] ]
            x[2].dims = [1, 2]

          Attrs:
            axis = 1 or axis = -2

          Output:
            Out.data =[ [ [1.0, 2.0]
                          [3.0, 4.0]
                          [5.0, 6.0] ] ]
            Out.dims = [1, 3, 2]

    Args:
        x (Variable|list(Variable)|tuple(Variable)): Input variables.
        axis (int|None): The axis along which all inputs are stacked.

    Returns:
        Variable: The stacked variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            x1 = layers.data(name='x1', shape=[1, 2], dtype='int32')
            x2 = layers.data(name='x2', shape=[1, 2], dtype='int32')
            data = layers.stack([x1,x2])

    """

    helper = LayerHelper('stack', **locals())
    axis = 0 if axis is None else axis

    if not isinstance(x, list) and not isinstance(x, tuple):
        x = [x]

    out = helper.create_variable_for_type_inference(x[0].dtype)
    helper.append_op(
        type='stack', inputs={'X': x}, outputs={'Y': out},
        attrs={'axis': axis})

    return out


@templatedoc(op_type="filter_by_instag")
def filter_by_instag(ins, ins_tag, filter_tag, is_lod):
    """
    **Filter By Instag Layer**
   
    This function filter a batch of ins by instag, 
    There are multiple ins, and every ins belongs to some tags. 
    We can specify some tags we want. So the ins which belongs to that tags
    remains in the output, and others removed.
 
    For example, one batch has 4 ins. Every ins has its tag list. 
     
       | Ins   |   Ins_Tag |
       |:-----:|:------:|
       |  0    |   0, 1 |
       |  1    |   1, 3 |
       |  2    |   0, 3 |
       |  3    |   2, 6 |

    And Lod is [1,1,1,1]

    And the filter tags [1]

    From the definition above, ins which has tag 1 can pass the filter
    So Ins 0 and Ins 1 can pass and be seen in the output,
    Ins 2 and 3 cannot pass because they do not has tag 1.

    Actually, if is_lod is false, it is normal tensor that equals to 
    lod_tensor with all 1, similar to the example above.

    Args:
        ins (Variable): Input Variable (LoDTensor), usually it is 2D tensor
                        And first dimension can have lod info or not.
        ins_tag (Variable): Input Variable (LoDTensor), usually it is 1D list
                        And split them by lod info
        filter_tag (Variable): Input Variable (1D Tensor/List), usually it is 
                        list that holds the tags.
        is_lod (Bool): Boolean value to indicate ins is lod tensor or not.

    Returns:
        Variable: filtered ins (LoDTensor) and loss weight (Tensor)

    Examples:
        .. code-block:: python

          import paddle.fluid.layers as layers
          ins = layers.data(name='Ins', shape=[-1,32], lod_level=0, dtype='float64')
          ins_tag = layers.data(name='Ins_tag', shape=[-1,16], lod_level=0, dtype='int64')
          filter_tag = layers.data(name='Filter_tag', shape=[-1,16], dtype='int64')
          out, loss_weight = layers.filter_by_instag(ins,  ins_tag,  filter_tag, True)
        		
    """
    helper = LayerHelper('filter_by_instag', **locals())

    out = helper.create_variable_for_type_inference(dtype=ins.dtype)
    loss_weight = helper.create_variable_for_type_inference(dtype=np.float64)
    mmap = helper.create_variable_for_type_inference(dtype=ins_tag.dtype)
    helper.append_op(
        type='filter_by_instag',
        inputs={'Ins': ins,
                'Ins_tag': ins_tag,
                'Filter_tag': filter_tag},
        outputs={'Out': out,
                 'LossWeight': loss_weight,
                 'IndexMap': mmap},
        attrs={'is_lod': is_lod})

    return [out, loss_weight]


def unstack(x, axis=0, num=None):
    """
    **UnStack Layer**

    This layer unstacks input :code:`x` into several tensors along axis.

    If :code:`axis` < 0, it would be replaced with :code:`axis+rank(x)`.
    If :code:`num` is None, it would be inferred from :code:`x.shape[axis]`,
    and if :code:`x.shape[axis]` <= 0 or is unknown, :code:`ValueError` is
    raised.

    Args:
        x (Variable): Input variable.
        axis (int): The axis along which the input is unstacked.
        num (int|None): The number of output variables.

    Returns:
        list(Variable): The unstacked variables.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[5, 10], dtype='float32')
            y = fluid.layers.unstack(x, axis=1)
    """

    helper = LayerHelper('unstack', **locals())
    if num is None:
        if axis is None or x.shape[axis] <= 0:
            raise ValueError('unknown unstack number')
        else:
            num = x.shape[axis]

    outs = []
    for _ in range(num):
        outs.append(helper.create_variable_for_type_inference(x.dtype))

    helper.append_op(
        type='unstack',
        inputs={'X': [x]},
        outputs={'Y': outs},
        attrs={'axis': axis,
               'num': num})
    return outs


def expand(x, expand_times, name=None):
    """Expand operator tiles the input by given times number. You should set times
    number for each dimension by providing attribute 'expand_times'. The rank of X
    should be in [1, 6]. Please note that size of 'expand_times' must be the same
    with X's rank. Following is a using case:


    .. code-block:: text

        Input(X) is a 3-D tensor with shape [2, 3, 1]:

                [
                   [[1], [2], [3]],
                   [[4], [5], [6]]
                ]

        Attr(expand_times):  [1, 2, 2]

        Output(Out) is a 3-D tensor with shape [2, 6, 2]:

                [
                    [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
                    [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
                ]

    Args:
        x (Variable): A tensor with rank in [1, 6].
        expand_times (list|tuple|Variable): Expand times number for each dimension.

    Returns:
        Variable: The expanded variable which is a LoDTensor. After expanding, size of each dimension of Output(Out) is equal to ithe size of the corresponding dimension of Input(X) multiplying the corresponding value given by expand_times.


    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            # example 1:
            data_1 = fluid.layers.fill_constant(shape=[2, 3, 1], dtype='int32', value=0)
            expanded_1 = fluid.layers.expand(data_1, expand_times=[1, 2, 2])

            # example 2:
            data_2 = fluid.layers.fill_constant(shape=[12, 14], dtype="int32", value=3)
            expand_times = fluid.layers.fill_constant(shape=[2], dtype="int32", value=4)
            expanded_2 = fluid.layers.expand(data_2, expand_times=expand_times)
    """

    if not isinstance(expand_times, (list, tuple, Variable)):
        raise ValueError(
            "Input expand_times must be an Variable, python list or tuple.")

    helper = LayerHelper('expand', input=x, **locals())
    inputs = {"X": x}
    attrs = {}

    def contain_var(expand_times):
        for ele in expand_times:
            if isinstance(ele, Variable):
                return True
        return False

    def get_attr_expand_times(list_expand_times):
        attrs_expand_times = []
        for idx, times in enumerate(list_expand_times):
            if isinstance(times, Variable):
                attrs_expand_times.append(-1)
            else:
                attrs_expand_times.append(times)
                assert times > 0, (
                    "Each element given in expand_times must not be negtive.")
        return attrs_expand_times

    def get_new_expand_times_tensor(list_expand_times):
        new_expand_times_tensor = []
        for ele in list_expand_times:
            if isinstance(ele, Variable):
                ele.stop_gradient = True
                new_expand_times_tensor.append(ele)
            else:
                assert (isinstance(ele, int))
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', ele, force_cpu=True, out=temp_out)
                new_expand_times_tensor.append(temp_out)
        return new_expand_times_tensor

    if in_dygraph_mode():
        inputs = {'X': x}
        attrs = {'expand_times': expand_times}
    else:
        if isinstance(expand_times, Variable):
            expand_times.stop_gradient = True
            inputs['ExpandTimes'] = expand_times
        elif isinstance(expand_times, (list, tuple)):
            attrs['expand_times'] = get_attr_expand_times(expand_times)
            if contain_var(expand_times):
                inputs['expand_times_tensor'] = get_new_expand_times_tensor(
                    expand_times)

    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='expand', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


from paddle.fluid.framework import convert_np_dtype_to_dtype_


@templatedoc()
def uniform_random_batch_size_like(input,
                                   shape,
                                   dtype='float32',
                                   input_dim_idx=0,
                                   output_dim_idx=0,
                                   min=-1.0,
                                   max=1.0,
                                   seed=0):
    """
    ${comment}

    Args:
        input (Variable): ${input_comment}
        shape (tuple|list): ${shape_comment}
        input_dim_idx (Int): ${input_dim_idx_comment}
        output_dim_idx (Int): ${output_dim_idx_comment}
        min (Float): ${min_comment}
        max (Float): ${max_comment}
        seed (Int): ${seed_comment}
        dtype(np.dtype|core.VarDesc.VarType|str): The type of data : float32, float_16, int etc
    Returns:
        out (Variable): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers 

            input = layers.data(name="input", shape=[13, 11], dtype='float32')
            out = layers.uniform_random_batch_size_like(input, [-1, 11])
    """

    helper = LayerHelper('uniform_random_batch_size_like', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    c_dtype = convert_np_dtype_to_dtype_(dtype)
    helper.append_op(
        type='uniform_random_batch_size_like',
        inputs={'Input': input},
        outputs={'Out': out},
        attrs={
            'shape': shape,
            'input_dim_idx': input_dim_idx,
            'output_dim_idx': output_dim_idx,
            'min': min,
            'max': max,
            'seed': seed,
            'dtype': c_dtype
        })

    return out


@templatedoc()
def gaussian_random(shape, mean=0.0, std=1.0, seed=0, dtype='float32'):
    """
    ${comment}

    Args:
        shape (tuple|list): ${shape_comment}
        mean (Float): ${mean_comment}
        std (Float): ${std_comment}
        seed (Int): ${seed_comment}
        dtype(np.dtype|core.VarDesc.VarType|str): Output data type.

    Returns:
        out (Variable): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            out = layers.gaussian_random(shape=[20, 30])
    """

    helper = LayerHelper('gaussian_random', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    c_dtype = convert_np_dtype_to_dtype_(dtype)
    helper.append_op(
        type='gaussian_random',
        outputs={'Out': out},
        attrs={
            'shape': shape,
            'mean': mean,
            'std': std,
            'seed': seed,
            'dtype': c_dtype,
            'use_mkldnn': False
        })

    return out


@templatedoc()
def sampling_id(x, min=0.0, max=1.0, seed=0, dtype='float32'):
    """
    ${comment}

    Args:
        x (Variable): ${x_comment}
        min (Float): ${min_comment}
        max (Float): ${max_comment}
        seed (Float): ${seed_comment}
        dtype(np.dtype|core.VarDesc.VarType|str): The type of output data : float32, float_16, int etc

    Returns:
        out (Variable): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(
                name="X",
                shape=[13, 11],
                dtype='float32',
                append_batch_size=False)

            out = fluid.layers.sampling_id(x)
    """

    helper = LayerHelper('sampling_id', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='sampling_id',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'min': min,
               'max': max,
               'seed': seed})

    return out


@templatedoc()
def gaussian_random_batch_size_like(input,
                                    shape,
                                    input_dim_idx=0,
                                    output_dim_idx=0,
                                    mean=0.0,
                                    std=1.0,
                                    seed=0,
                                    dtype='float32'):
    """
    ${comment}

    Args:
        input (Variable): ${input_comment}
        shape (tuple|list): ${shape_comment}
        input_dim_idx (Int): ${input_dim_idx_comment}
        output_dim_idx (Int): ${output_dim_idx_comment}
        mean (Float): ${mean_comment}
        std (Float): ${std_comment}
        seed (Int): ${seed_comment}
        dtype(np.dtype|core.VarDesc.VarType|str): The type of output data : float32, float_16, int etc

    Returns:
        out (Variable): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.layers.data(name="input", shape=[13, 11], dtype='float32')

            out = fluid.layers.gaussian_random_batch_size_like(
                input, shape=[-1, 11], mean=1.0, std=2.0)
    """

    helper = LayerHelper('gaussian_random_batch_size_like', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    c_dtype = convert_np_dtype_to_dtype_(dtype)
    helper.append_op(
        type='gaussian_random_batch_size_like',
        inputs={'Input': input},
        outputs={'Out': out},
        attrs={
            'shape': shape,
            'input_dim_idx': input_dim_idx,
            'output_dim_idx': output_dim_idx,
            'mean': mean,
            'std': std,
            'seed': seed,
            'dtype': c_dtype
        })

    return out


@templatedoc()
def sum(x):
    """
    ${comment}

    Args:
        x (Variable): ${x_comment}

    Returns:
        out (Variable): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            input0 = layers.data(name="input0", shape=[13, 11], dtype='float32')
            input1 = layers.data(name="input1", shape=[13, 11], dtype='float32')
            out = layers.sum([input0,input1])
    """

    helper = LayerHelper('sum', **locals())
    out = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('x'))
    helper.append_op(
        type='sum',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'use_mkldnn': False})

    return out


@templatedoc()
def slice(input, axes, starts, ends):
    """
    Slice Operator.

    Produces a slice of the input tensor along multiple axes. Similar to numpy:
    https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    Slice uses `axes`, `starts` and `ends` attributes to specify the start and
    end dimension for each axis in the list of axes, it uses this information
    to slice the input data tensor. If a negative value is passed for any of
    the start or end indices, it represents number of elements before the end
    of that dimension. If the value passed to start or end is larger than
    the n (the number of elements in this dimension), it represents n.
    For slicing to the end of a dimension with unknown size, it is recommended
    to pass in INT_MAX. The size of axes must be equal to starts\' and ends\'.
    Following examples will explain how slice works:

    .. code-block:: text

        Case1:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [1, 0]
                ends = [2, 3]
            Then:
                result = [ [5, 6, 7], ]
        
        Case2:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [0, 1]
                ends = [-1, 1000]
            Then:
                result = [ [2, 3, 4], ]
    Args:
        input (Variable): ${input_comment}.
        axes (List): ${axes_comment}
        starts (List|Variable): ${starts_comment}
        ends (List|Variable): ${ends_comment}

    Returns:
        out (Variable): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            input = fluid.layers.data(
                name="input", shape=[3, 4, 5, 6], dtype='float32')

            # example 1:
            # attr starts is a list which doesn't contain tensor Variable.
            axes = [0, 1, 2]
            starts = [-3, 0, 2]
            ends = [3, 2, 4]
            sliced_1 = fluid.layers.slice(input, axes=axes, starts=starts, ends=ends)

            # example 2:
            # attr starts is a list which contain tensor Variable.
            minus_3 = fluid.layers.fill_constant([1], "int32", -3)
            sliced_2 = fluid.layers.slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends)
    """

    if not isinstance(starts, (list, tuple, Variable)):
        raise ValueError(
            "Input starts must be an Variable, python list or tuple.")
    if not isinstance(ends, (list, tuple, Variable)):
        raise ValueError(
            "Input ends must be an Variable, python list or tuple.")

    helper = LayerHelper('slice', **locals())

    def contain_var(one_list):
        for ele in one_list:
            if isinstance(ele, Variable):
                return True
        return False

    def get_new_list_tensor(old_list):
        new_list_tensor = []
        for dim in old_list:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_list_tensor.append(dim)
            else:
                assert (isinstance(dim, int))
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', dim, force_cpu=True, out=temp_out)
                new_list_tensor.append(temp_out)
        return new_list_tensor

    inputs = {'Input': input}
    attrs = {'axes': axes}
    infer_flags = list(1 for i in range(len(axes)))

    if in_dygraph_mode():
        inputs = {'Input': input}
        attrs = {
            'axes': axes,
            'starts': starts,
            'ends': ends,
            'infer_flags': infer_flags
        }
    else:
        # starts
        if isinstance(starts, Variable):
            starts.stop_gradient = True
            inputs['StartsTensor'] = starts
            infer_flags = list(-1 for i in range(len(axes)))
        elif isinstance(starts, (list, tuple)):
            attrs['starts'] = []
            if not contain_var(starts):
                attrs['starts'] = starts
            else:
                inputs['StartsTensorList'] = get_new_list_tensor(starts)
                for i, dim in enumerate(starts):
                    if isinstance(dim, Variable):
                        attrs['starts'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['starts'].append(dim)

        # ends
        if isinstance(ends, Variable):
            ends.stop_gradient = True
            inputs['EndsTensor'] = ends
            infer_flags = list(-1 for i in range(len(axes)))
        elif isinstance(ends, (list, tuple)):
            attrs['ends'] = []
            if not contain_var(ends):
                attrs['ends'] = ends
            else:
                inputs['EndsTensorList'] = get_new_list_tensor(ends)
                for i, dim in enumerate(ends):
                    if isinstance(dim, Variable):
                        attrs['ends'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['ends'].append(dim)
        # infer_flags
        attrs['infer_flags'] = infer_flags
    out = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('input'))
    helper.append_op(
        type='slice', inputs=inputs, attrs=attrs, outputs={'Out': out})

    return out


@templatedoc()
def strided_slice(input, axes, starts, ends, strides):
    """
    Strided Slice OP

    The conceptualization that really helped me understand this was 
    that this function emulates the indexing behavior of numpy arrays.
    If you're familiar with numpy arrays, you'll know that you can make 
    slices via input[start1:end1:step1, start2:end2:step2, ... startN:endN:stepN]. 
    Basically, a very succinct way of writing for loops to get certain elements of the array.
    strided_slice just allows you to do this fancy indexing without the syntactic sugar. 
    The numpy (#input[start1:end1:step1, start2:end2:step2, ... startN:endN:stepN])
    example from above just becomes fluid.strided_slice(input,[0, 1, ..., N], 
    [start1, start2, ..., startN], [end1, end2, ..., endN], [strides1, strides2, ..., stridesN]),
    the axes which controls the dimension you want to slice makes it more flexible.

    .. code-block:: text

        Case1:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [1, 0]
                ends = [2, 3]
                strides=[1, 1]
            Then:
                result = [ [5, 6, 7], ]
        
        Case2:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [0, 1]
                ends = [-1, 1000]
                strides = [1, 3]
            Then:
                result = [ [2], ]
    Args:
        input (Variable): ${input_comment}.
        axes (List): ${axes_comment}
        starts (List|Variable): ${starts_comment}
        ends (List|Variable): ${ends_comment}

    Returns:
        out (Variable): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            input = fluid.layers.data(
                name="input", shape=[3, 4, 5, 6], dtype='float32')

            # example 1:
            # attr starts is a list which doesn't contain tensor Variable.
            axes = [0, 1, 2]
            starts = [-3, 0, 2]
            ends = [3, 2, 4]
            strides=[1, 1, 1]
            sliced_1 = fluid.layers.strided_slice(input, axes=axes, starts=starts, ends=ends, strides=strides)

            # example 2:
            # attr starts is a list which contain tensor Variable.
            minus_3 = fluid.layers.fill_constant([1], "int32", -3)
            sliced_2 = fluid.layers.strided_slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends, strides=strides)
    """
    if not isinstance(starts, (list, tuple, Variable)):
        raise ValueError(
            "Input starts must be an Variable, python list or tuple.")
    if not isinstance(ends, (list, tuple, Variable)):
        raise ValueError(
            "Input ends must be an Variable, python list or tuple.")
    if not isinstance(strides, (list, tuple, Variable)):
        raise ValueError(
            "Input strides must be an Variable, python list or tuple.")

    helper = LayerHelper('strided_slice', **locals())

    def contain_var(one_list):
        for ele in one_list:
            if isinstance(ele, Variable):
                return True
        return False

    def get_new_list_tensor(old_list):
        new_list_tensor = []
        for dim in old_list:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_list_tensor.append(dim)
            else:
                assert (isinstance(dim, int))
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', dim, force_cpu=True, out=temp_out)
                new_list_tensor.append(temp_out)
        return new_list_tensor

    inputs = {'Input': input}
    attrs = {'axes': axes}
    infer_flags = list(1 for i in range(len(axes)))

    if in_dygraph_mode():
        inputs = {'Input': input}
        attrs = {
            'axes': axes,
            'starts': starts,
            'ends': ends,
            'strides': strides,
            'infer_flags': infer_flags
        }
    else:
        # starts
        if isinstance(starts, Variable):
            starts.stop_gradient = True
            inputs['StartsTensor'] = starts
        elif isinstance(starts, (list, tuple)):
            attrs['starts'] = []
            if not contain_var(starts):
                attrs['starts'] = starts
            else:
                inputs['StartsTensorList'] = get_new_list_tensor(starts)
                for i, dim in enumerate(starts):
                    if isinstance(dim, Variable):
                        attrs['starts'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['starts'].append(dim)

        # ends
        if isinstance(ends, Variable):
            ends.stop_gradient = True
            inputs['EndsTensor'] = ends
        elif isinstance(ends, (list, tuple)):
            attrs['ends'] = []
            if not contain_var(ends):
                attrs['ends'] = ends
            else:
                inputs['EndsTensorList'] = get_new_list_tensor(ends)
                for i, dim in enumerate(ends):
                    if isinstance(dim, Variable):
                        attrs['ends'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['ends'].append(dim)
        # strides
        if isinstance(strides, Variable):
            strides.stop_gradient = True
            inputs['StridesTensor'] = strides
        elif isinstance(strides, (list, tuple)):
            attrs['strides'] = []
            if not contain_var(strides):
                attrs['strides'] = strides
            else:
                inputs['StridesTensorList'] = get_new_list_tensor(strides)
                for i, dim in enumerate(strides):
                    if isinstance(dim, Variable):
                        attrs['strides'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['strides'].append(dim)
        attrs['infer_flags'] = infer_flags
    out = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('input'))
    helper.append_op(
        type='strided_slice', inputs=inputs, attrs=attrs, outputs={'Out': out})

    return out


def shape(input):
    """
    **Shape Layer**

    Get the shape of the input.

    Args:
        input (Variable): The input variable.

    Returns:
        Variable: The shape of the input variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            input = fluid.layers.data(
                name="input", shape=[3, 100, 100], dtype="float32")
            out = fluid.layers.shape(input)
    """

    helper = LayerHelper('shape', **locals())
    out = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type='shape', inputs={'Input': input}, outputs={'Out': out})

    return out


def rank(input):
    """
    **Rank Layer**

    Returns the number of dimensions for a tensor, which is a 0-D int32 Tensor.

    Args:
        input (Variable): The input variable.

    Returns:
        Variable: The rank of the input variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            input = fluid.layers.data(name="input", shape=[3, 100, 100], dtype="float32")
            rank = fluid.layers.rank(input) # 4
    """

    ndims = len(input.shape)
    out = assign(np.array(ndims, 'int32'))

    return out


def size(input):
    """
    **Size Layer**

    Returns the number of elements for a tensor, which is a int64 Tensor with shape [1].

    Args:
        input (Variable): The input variable.

    Returns:
        Variable: The number of elements for the input variable.

    Examples:
        .. code-block:: python

            import paddle.fluid.layers as layers

            input = layers.data(
                name="input", shape=[3, 100], dtype="float32", append_batch_size=False)
            rank = layers.size(input) # 300
    """

    helper = LayerHelper('size', **locals())
    out = helper.create_variable_for_type_inference(dtype='int64')
    helper.append_op(type='size', inputs={'Input': input}, outputs={'Out': out})

    return out


def _elementwise_op(helper):
    op_type = helper.layer_type
    x = helper.kwargs.get('x', None)
    y = helper.kwargs.get('y', None)
    if in_dygraph_mode():
        x = base.to_variable(x)
        y = base.to_variable(y)

    assert x is not None, 'x cannot be None in {}'.format(op_type)
    assert y is not None, 'y cannot be None in {}'.format(op_type)
    axis = helper.kwargs.get('axis', -1)
    use_mkldnn = helper.kwargs.get('use_mkldnn', False)
    name = helper.kwargs.get('name', None)
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type=op_type,
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={'axis': axis,
               'use_mkldnn': use_mkldnn})
    return helper.append_activation(out)


@templatedoc()
def scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        scale(${scale_type}): ${scale_comment}
        bias(${bias_type}): ${bias_comment}
        bias_after_scale(${bias_after_scale_type}): ${bias_after_scale_comment}
        act(basestring|None): Activation applied to the output.
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.layers.data(name="X", shape=[1, 2, 5, 5], dtype='float32')
            y = fluid.layers.scale(x, scale = 2.0, bias = 1.0)
    """

    helper = LayerHelper('scale', **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type='scale',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={
            'scale': float(scale),
            'bias': float(bias),
            'bias_after_scale': bias_after_scale
        })
    return helper.append_activation(out)


def elementwise_add(x, y, axis=-1, act=None, name=None):
    return _elementwise_op(LayerHelper('elementwise_add', **locals()))


def elementwise_div(x, y, axis=-1, act=None, name=None):
    return _elementwise_op(LayerHelper('elementwise_div', **locals()))


def elementwise_sub(x, y, axis=-1, act=None, name=None):
    return _elementwise_op(LayerHelper('elementwise_sub', **locals()))


def elementwise_mul(x, y, axis=-1, act=None, name=None):
    return _elementwise_op(LayerHelper('elementwise_mul', **locals()))


def elementwise_max(x, y, axis=-1, act=None, name=None):
    return _elementwise_op(LayerHelper('elementwise_max', **locals()))


def elementwise_min(x, y, axis=-1, act=None, name=None):
    return _elementwise_op(LayerHelper('elementwise_min', **locals()))


def elementwise_pow(x, y, axis=-1, act=None, name=None):
    return _elementwise_op(LayerHelper('elementwise_pow', **locals()))


def elementwise_mod(x, y, axis=-1, act=None, name=None):
    return _elementwise_op(LayerHelper('elementwise_mod', **locals()))


def elementwise_floordiv(x, y, axis=-1, act=None, name=None):
    return _elementwise_op(LayerHelper('elementwise_floordiv', **locals()))


for func in [
        elementwise_add,
        elementwise_div,
        elementwise_sub,
        elementwise_mul,
        elementwise_max,
        elementwise_min,
        elementwise_pow,
        elementwise_mod,
        elementwise_floordiv,
]:
    op_proto = OpProtoHolder.instance().get_op_proto(func.__name__)
    func.__doc__ = _generate_doc_string_(
        op_proto,
        additional_args_lines=[
            "act (basestring|None): Activation applied to the output.",
            "name (basestring|None): Name of the output."
        ])
    func.__doc__ = func.__doc__ + """

Examples:
  .. code-block:: python
    
    import paddle.fluid as fluid
    # example 1: shape(x) = (2, 3, 4, 5), shape(y) = (2, 3, 4, 5)
    x0 = fluid.layers.data(name="x0", shape=[2, 3, 4, 5], dtype='float32')
    y0 = fluid.layers.data(name="y0", shape=[2, 3, 4, 5], dtype='float32')
    z0 = fluid.layers.%s(x0, y0)

    # example 2: shape(X) = (2, 3, 4, 5), shape(Y) = (5)
    x1 = fluid.layers.data(name="x1", shape=[2, 3, 4, 5], dtype='float32')
    y1 = fluid.layers.data(name="y1", shape=[5], dtype='float32')
    z1 = fluid.layers.%s(x1, y1)

    # example 3: shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
    x2 = fluid.layers.data(name="x2", shape=[2, 3, 4, 5], dtype='float32')
    y2 = fluid.layers.data(name="y2", shape=[4, 5], dtype='float32')
    z2 = fluid.layers.%s(x2, y2, axis=2)

    # example 4: shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
    x3 = fluid.layers.data(name="x3", shape=[2, 3, 4, 5], dtype='float32')
    y3 = fluid.layers.data(name="y3", shape=[3, 4], dtype='float32')
    z3 = fluid.layers.%s(x3, y3, axis=1)

    # example 5: shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
    x4 = fluid.layers.data(name="x4", shape=[2, 3, 4, 5], dtype='float32')
    y4 = fluid.layers.data(name="y4", shape=[2], dtype='float32')
    z4 = fluid.layers.%s(x4, y4, axis=0)

    # example 6: shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
    x5 = fluid.layers.data(name="x5", shape=[2, 3, 4, 5], dtype='float32')
    y5 = fluid.layers.data(name="y5", shape=[2], dtype='float32')
    z5 = fluid.layers.%s(x5, y5, axis=0)
    """ % (func.__name__, func.__name__, func.__name__, func.__name__,
           func.__name__, func.__name__)


def _logical_op(op_name, x, y, out=None, name=None, binary_op=True):
    helper = LayerHelper(op_name, **locals())

    if binary_op:
        assert x.dtype == y.dtype

    if out is None:
        if name is None:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        else:
            out = helper.create_variable(
                name=name, dtype=x.dtype, persistable=False)

    if binary_op:
        helper.append_op(
            type=op_name, inputs={"X": x,
                                  "Y": y}, outputs={"Out": out})
    else:
        helper.append_op(type=op_name, inputs={"X": x}, outputs={"Out": out})

    return out


@templatedoc()
def logical_and(x, y, out=None, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        y(${y_type}): ${y_comment}
        out(Tensor): Output tensor of logical operation.
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            left = fluid.layers.data(
                name='left', shape=[1], dtype='bool')
            right = fluid.layers.data(
                name='right', shape=[1], dtype='bool')
            result = fluid.layers.logical_and(x=left, y=right)
    """

    return _logical_op(
        op_name="logical_and", x=x, y=y, name=name, out=out, binary_op=True)


@templatedoc()
def logical_or(x, y, out=None, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        y(${y_type}): ${y_comment}
        out(Tensor): Output tensor of logical operation.
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            left = fluid.layers.data(
                name='left', shape=[1], dtype='bool')
            right = fluid.layers.data(
                name='right', shape=[1], dtype='bool')
            result = fluid.layers.logical_or(x=left, y=right)
    """

    return _logical_op(
        op_name="logical_or", x=x, y=y, name=name, out=out, binary_op=True)


@templatedoc()
def logical_xor(x, y, out=None, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        y(${y_type}): ${y_comment}
        out(Tensor): Output tensor of logical operation.
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            left = fluid.layers.data(
                name='left', shape=[1], dtype='bool')
            right = fluid.layers.data(
                name='right', shape=[1], dtype='bool')
            result = fluid.layers.logical_xor(x=left, y=right)
    """

    return _logical_op(
        op_name="logical_xor", x=x, y=y, name=name, out=out, binary_op=True)


@templatedoc()
def logical_not(x, out=None, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        out(Tensor): Output tensor of logical operation.
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            left = fluid.layers.data(
                name='left', shape=[1], dtype='bool')
            result = fluid.layers.logical_not(x=left)
    """

    return _logical_op(
        op_name="logical_not", x=x, y=None, name=name, out=out, binary_op=False)


@templatedoc()
def clip(x, min, max, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        min(${min_type}): ${min_comment}
        max(${max_type}): ${max_comment}
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.layers.data(
                name='data', shape=[1], dtype='float32')
            reward = fluid.layers.clip(x=input, min=-1.0, max=1.0)
    """

    helper = LayerHelper("clip", **locals())

    if name is None:
        name = unique_name.generate_with_ignorable_key(".".join(
            [helper.name, 'tmp']))

    out = helper.create_variable(
        type=x.type, name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="clip",
        inputs={"X": x},
        attrs={"min": min,
               "max": max},
        outputs={"Out": out})

    return out


@templatedoc()
def clip_by_norm(x, max_norm, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        max_norm(${max_norm_type}): ${max_norm_comment}
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.layers.data(
                name='data', shape=[1], dtype='float32')
            reward = fluid.layers.clip_by_norm(x=input, max_norm=1.0)
    """

    helper = LayerHelper("clip_by_norm", **locals())

    if name is None:
        name = unique_name.generate_with_ignorable_key(".".join(
            [helper.name, 'tmp']))

    out = helper.create_variable(
        type=x.type, name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="clip_by_norm",
        inputs={"X": x},
        attrs={"max_norm": max_norm},
        outputs={"Out": out})

    return out


@templatedoc()
def mean(x, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.layers.data(
                name='data', shape=[2, 3], dtype='float32')
            mean = fluid.layers.mean(input)
    """

    helper = LayerHelper("mean", **locals())

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="mean", inputs={"X": x}, attrs={}, outputs={"Out": out})

    return out


@templatedoc()
def merge_selected_rows(x, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            b = fluid.default_main_program().global_block()
            var = b.create_var(
                name="X", dtype="float32", persistable=True,
                type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
            y = fluid.layers.merge_selected_rows(var)
    """

    helper = LayerHelper("merge_selected_rows", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="merge_selected_rows",
        inputs={"X": x},
        attrs={},
        outputs={"Out": out})
    return out


@templatedoc()
def mul(x, y, x_num_col_dims=1, y_num_col_dims=1, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        y(${y_type}): ${y_comment}
        x_num_col_dims(${x_num_col_dims_type}): ${x_num_col_dims_comment}
        y_num_col_dims(${y_num_col_dims_type}): ${y_num_col_dims_comment}
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid
            dataX = fluid.layers.data(name="dataX", append_batch_size = False, shape=[2, 5], dtype="float32")
            dataY = fluid.layers.data(name="dataY", append_batch_size = False, shape=[5, 3], dtype="float32")
            output = fluid.layers.mul(dataX, dataY,
                                      x_num_col_dims = 1,
                                      y_num_col_dims = 1)
            

    """

    helper = LayerHelper("mul", **locals())

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="mul",
        inputs={"X": x,
                "Y": y},
        attrs={
            "x_num_col_dims": x_num_col_dims,
            "y_num_col_dims": y_num_col_dims
        },
        outputs={"Out": out})
    return out


@templatedoc()
def sigmoid_cross_entropy_with_logits(x,
                                      label,
                                      ignore_index=kIgnoreIndex,
                                      name=None,
                                      normalize=False):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        label(${label_type}): ${label_comment}
        ignore_index(&{ignore_index}): ${ignore_index_comment}
        name(basestring|None): Name of the output.
        normalize(bool): If true, divide the output by the number of
            targets != ignore_index.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.layers.data(
                name='data', shape=[10], dtype='float32')
            label = fluid.layers.data(
                name='data', shape=[10], dtype='float32')
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=input,
                label=label,
                ignore_index=-1,
                normalize=True) # or False
            # loss = fluid.layers.reduce_sum(loss) # summation of loss
    """

    helper = LayerHelper("sigmoid_cross_entropy_with_logits", **locals())

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="sigmoid_cross_entropy_with_logits",
        inputs={"X": x,
                "Label": label},
        attrs={"ignore_index": ignore_index,
               'normalize': normalize},
        outputs={"Out": out})
    return out


@templatedoc()
def maxout(x, groups, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        groups(${groups_type}): ${groups_comment}
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.layers.data(
                name='data', 
                shape=[256, 32, 32], 
                dtype='float32')
            out = fluid.layers.maxout(input, groups=2)
    """
    helper = LayerHelper("maxout", **locals())

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="maxout",
        inputs={"X": x},
        attrs={"groups": groups},
        outputs={"Out": out})
    return out


def space_to_depth(x, blocksize, name=None):
    """
    Gives a blocksize to space_to_depth the input LoDtensor with Layout: [batch, channel, height, width]

    This op rearranges blocks of spatial data, into depth. More specifically, this op outputs a copy of the
    input LoDtensor where values from the height and width dimensions are moved to the channel dimension.
    The attr blocksize indicates the input block size.

    space_to_depth will reorgnize the elements of input with shape[batch, channel, height, width] according
    to blocksize to construct output with shape [batch, channel * blocksize * blocksize, height/blocksize, width/blocksize]:

    space_to_depth is used to This operation is useful for resizing the activations between convolutions
    (but keeping all data)

    - Non-overlapping blocks of size block_size x block size are rearranged into depth at each location.
    - The depth of the output tensor is block_size * block_size * input channel
    - The Y, X coordinates within each block of the input become the high order component of the output channel index
    - channel should be divisible by square of blocksize
    - height, width should be divsible by blocksize


    Args:
        x(variable): The input LoDtensor.
        blocksize(variable): The blocksize to select the element on each feature map should be > 2

    Returns:
        Variable: The output LoDtensor.

    Raises:
        TypeError: blocksize type must be a long.

    Examples:
        .. code-block:: python
	
            import paddle.fluid as fluid
            import numpy as np

            data = fluid.layers.data(
                name='data', shape=[1, 4, 2, 2], dtype='float32', append_batch_size=False)
            space_to_depthed = fluid.layers.space_to_depth(
                x=data, blocksize=2)

            exe = fluid.Executor(fluid.CPUPlace())
            data_np = np.arange(0,16).reshape((1,4,2,2)).astype('float32')
            out_main = exe.run(fluid.default_main_program(),
                          feed={'data': data_np},
                          fetch_list=[space_to_depthed])

    """

    helper = LayerHelper("space_to_depth", **locals())

    if not (isinstance(blocksize, int)):
        raise ValueError("blocksize must be a python Int")

    if name is None:
        out = helper.create_variable_for_type_inference(
            dtype=x.dtype)  #fix create
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="space_to_depth",
        inputs={"X": x},
        attrs={"blocksize": blocksize},
        outputs={"Out": out})
    return out


@templatedoc()
def sequence_reverse(x, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        name(basestring|None): Name of the output.

    Returns:
        out(${y_type}): ${y_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[2, 6], dtype='float32')
            x_reversed = fluid.layers.sequence_reverse(x)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper("sequence_reverse", **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="sequence_reverse",
        inputs={"X": x},
        outputs={"Y": out},
        attrs=dict())
    return out


def affine_channel(x,
                   scale=None,
                   bias=None,
                   data_layout='NCHW',
                   name=None,
                   act=None):
    """
    Applies a separate affine transformation to each channel of the input.
    Useful for replacing spatial batch norm with its equivalent fixed
    transformation. The input also can be 2D tensor and applies a affine
    transformation in second dimension.

    Args:
        x (Variable): Feature map input can be a 4D tensor with order NCHW
            or NHWC. It also can be a 2D tensor and the affine transformation
            is applied in the second dimension.
        scale (Variable): 1D input of shape (C), the c-th element is the scale
            factor of the affine transformation for the c-th channel of
            the input.
        bias (Variable): 1D input of shape (C), the c-th element is the bias
            of the affine transformation for the c-th channel of the input.
        data_layout (string, default NCHW): NCHW or NHWC. If input is 2D
            tensor, you can ignore data_layout.
        name (str, default None): The name of this layer.
        act (str, default None): Activation to be applied to the output of this layer.

    Returns:
        out (Variable): A tensor of the same shape and data layout with x.

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid
            data = fluid.layers.data(name='data', shape=[3, 32, 32],
                                     dtype='float32')
            input_scale = fluid.layers.create_parameter(shape=[3],
                                     dtype="float32")
            input_bias = fluid.layers.create_parameter(shape=[3],
                                     dtype="float32")
            out = fluid.layers.affine_channel(data,scale=input_scale,
                                     bias=input_bias)

    """
    helper = LayerHelper("affine_channel", **locals())

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="affine_channel",
        inputs={"X": x,
                'Scale': scale,
                'Bias': bias},
        attrs={"data_layout": data_layout},
        outputs={"Out": out})
    return helper.append_activation(out)


def similarity_focus(input, axis, indexes, name=None):
    """
    SimilarityFocus Operator

    Generate a similarity focus mask with the same shape of input using the following method:

    1. Extract the 3-D tensor(here the first dimension is BatchSize) corresponding
       to the axis according to the indexes. For example, if axis=1 and indexes=[a],
       it will get the matrix T=X[:, a, :, :]. In this case, if the shape of input X
       is (BatchSize, A, B, C), the shape of tensor T is (BatchSize, B, C).
    2. For each index, find the largest numbers in the tensor T, so that the same
       row and same column has at most one number(what it means is that if the
       largest number has been found in the i-th row and the j-th column, then
       the numbers in the i-th row or j-th column will be skipped. And then the
       next largest number will be selected from the remaining numbers. Obviously
       there will be min(B, C) numbers), and mark the corresponding position of the
       3-D similarity focus mask as 1, otherwise as 0. Do elementwise-or for
       each index.
    3. Broadcast the 3-D similarity focus mask to the same shape of input X.

    Refer to `Similarity Focus Layer <http://www.aclweb.org/anthology/N16-1108>`_

    .. code-block:: text

        * Example :

            Given a 4-D tensor x with the shape (BatchSize, C, A, B), where C is
            the number of channels and the shape of feature map is (A, B):
                x.shape = (2, 3, 2, 2)
                x.data = [[[[0.8, 0.1],
                            [0.4, 0.5]],

                           [[0.9, 0.7],
                            [0.9, 0.9]],

                           [[0.8, 0.9],
                            [0.1, 0.2]]],


                          [[[0.2, 0.5],
                            [0.3, 0.4]],

                           [[0.9, 0.7],
                            [0.8, 0.4]],

                           [[0.0, 0.2],
                            [0.4, 0.7]]]]

            Given axis: 1 (the axis of the channel)
            Given indexes: [0]

            then we get a 4-D tensor out with the same shape of input x:
                out.shape = (2, 3, 2, 2)
                out.data = [[[[1.0, 0.0],
                              [0.0, 1.0]],

                             [[1.0, 0.0],
                              [0.0, 1.0]],

                             [[1.0, 0.0],
                              [0.0, 1.0]]],

                            [[[0.0, 1.0],
                              [1.0, 0.0]],

                             [[0.0, 1.0],
                              [1.0, 0.0]],

                             [[0.0, 1.0],
                              [1.0, 0.0]]]]

    Args:
        input(Variable): The input tensor variable(default float). It should
            be a 4-D tensor with shape [BatchSize, A, B, C].
        axis(int): Indicating the dimension to be selected. It can only be
            1, 2 or 3.
        indexes(list): Indicating the indexes of the selected dimension.

    Returns:
        Variable: A tensor variable with the same shape and same type \
                  as the input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.layers.data(
                name='data', shape=[-1, 3, 2, 2], dtype='float32')
            fluid.layers.similarity_focus(input=data, axis=1, indexes=[0])
    """
    helper = LayerHelper('similarity_focus', **locals())
    # check attrs
    if isinstance(axis, int) is False:
        raise TypeError("axis must be int type.")
    if isinstance(indexes, list) is False:
        raise TypeError("indexes must be list type.")
    if axis != 1 and axis != 2 and axis != 3:
        raise ValueError("axis must be 1, 2 or 3.")
    if len(indexes) == 0:
        raise ValueError("indexes can not be empty.")

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=input.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=input.dtype, persistable=False)
    helper.append_op(
        type='similarity_focus',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={"axis": axis,
               "indexes": indexes})
    return out


def hash(input, hash_size, num_hash=1, name=None):
    """
    Hash the input to an integer whose value is less than the given hash size.

    The hash algorithm we used was xxHash - Extremely fast hash algorithm
    (https://github.com/Cyan4973/xxHash/tree/v0.6.5)

    A simple example as below:

    .. code-block:: text

        Given:

        # shape [2, 2]
        input.data = 
            [[1, 2],
             [3, 4]]

        hash_size = 10000

        num_hash = 4

        Then:

        Hash op will take all number in input's 2nd dimension as hash algorithm's
        input for each time. Each input will be hashed for 4 times, and get an
        array whose length is 4. Each value in the array ranges from 0 to 9999.

        # shape [2, 4]
        output.data = [
            [[9662, 9217, 1129, 8487],
             [8310, 1327, 1654, 4567]],
        ]

    Args:
        input (Variable): The input variable which is a one-hot word. The
            dimensions of the input variable must be 2. Both Tensor and LoDTensor are supported.
        hash_size (int): The space size for hash algorithm. The output value
            will keep in the range:math:`[0, hash_size - 1]`.
        num_hash (int): The times of hash, default 1.
        name (str, default None): The name of this layer.

    Returns:
       Variable: The hash result variable, which the same variable type as `input`.

    Examples:
       .. code-block:: python

            import paddle.fluid as fluid

            # titles has shape [batch, 1]
            titles = fluid.layers.data(name='titles', shape=[1], dtype='int32', lod_level=0)
            # hash_r has shape [batch, 2]
            hash_r = fluid.layers.hash(name='hash_x', input=titles, num_hash=2, hash_size=1000)


            # titles has shape [batch, 1] and lod information
            titles = fluid.layers.data(name='titles', shape=[1], dtype='int32', lod_level=1)
            # hash_r has shape [batch, 2] and inherits lod information from titles
            hash_r = fluid.layers.hash(name='hash_x', input=titles, num_hash=2, hash_size=1000)
    """
    helper = LayerHelper('hash', **locals())
    out = helper.create_variable_for_type_inference(
        helper.input_dtype(), stop_gradient=True)
    helper.append_op(
        type='hash',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={'num_hash': num_hash,
               'mod_by': hash_size})
    return out


@templatedoc()
def grid_sampler(x, grid, name=None):
    """
    This operation samples input X by using bilinear interpolation based on
    flow field grid, which is usually gennerated by :code:`affine_grid` . The grid of
    shape [N, H, W, 2] is the concatenation of (grid_x, grid_y) coordinates
    with shape [N, H, W] each, where grid_x is indexing the 4th dimension
    (in width dimension) of input data x and grid_y is indexng the 3rd
    dimention (in height dimension), finally results is the bilinear
    interpolation value of 4 nearest corner points.

    .. code-block:: text

        Step 1:
        Get (x, y) grid coordinates and scale to [0, H-1/W-1].

        grid_x = 0.5 * (grid[:, :, :, 0] + 1) * (W - 1)
        grid_y = 0.5 * (grid[:, :, :, 1] + 1) * (H - 1)

        Step 2:
        Indices input data X with grid (x, y) in each [H, W] area, and bilinear
        interpolate point value by 4 nearest points.

          wn ------- y_n ------- en
          |           |           |
          |          d_n          |
          |           |           |
         x_w --d_w-- grid--d_e-- x_e
          |           |           |
          |          d_s          |
          |           |           |
          ws ------- y_s ------- wn

        x_w = floor(x)              // west side x coord
        x_e = x_w + 1               // east side x coord
        y_n = floor(y)              // north side y coord
        y_s = y_s + 1               // south side y coord

        d_w = grid_x - x_w          // distance to west side
        d_e = x_e - grid_x          // distance to east side
        d_n = grid_y - y_n          // distance to north side
        d_s = y_s - grid_y          // distance to south side

        wn = X[:, :, y_n, x_w]      // north-west point value
        en = X[:, :, y_n, x_e]      // north-east point value
        ws = X[:, :, y_s, x_w]      // south-east point value
        es = X[:, :, y_s, x_w]      // north-east point value

        output = wn * d_e * d_s + en * d_w * d_s
               + ws * d_e * d_n + es * d_w * d_n

    Args:
        x(Variable): Input data of shape [N, C, H, W].
        grid(Variable): Input grid tensor of shape [N, H, W, 2].
        name (str, default None): The name of this layer.

    Returns:
        Variable: Output of shape [N, C, H, W] data samples input X
        using bilnear interpolation based on input grid.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.layers.data(name='x', shape=[10, 32, 32], dtype='float32')
            theta = fluid.layers.data(name='theta', shape=[2, 3], dtype='float32')
            grid = fluid.layers.affine_grid(theta=theta, out_shape=[3, 10, 32, 32])
            out = fluid.layers.grid_sampler(x=x, grid=grid)

    """
    helper = LayerHelper("grid_sampler", **locals())

    if not isinstance(x, Variable):
        return ValueError("The x should be a Variable")

    if not isinstance(grid, Variable):
        return ValueError("The grid should be a Variable")

    out = helper.create_variable_for_type_inference(x.dtype)
    ipts = {'X': x, 'Grid': grid}

    helper.append_op(type='grid_sampler', inputs=ipts, outputs={'Output': out})
    return out


def log_loss(input, label, epsilon=1e-4, name=None):
    """
    **Negative Log Loss Layer**

    This layer accepts input predictions and target label and returns the
    negative log loss.

    .. math::

        Out = -label * \\log{(input + \\epsilon)}
              - (1 - label) * \\log{(1 - input + \\epsilon)}

    Args:
        input (Variable|list):  a 2-D tensor with shape [N x 1], where N is the
                                batch size. This input is a probability computed
                                by the previous operator.
        label (Variable|list):  the ground truth which is a 2-D tensor with
                                shape [N x 1], where N is the batch size.
        epsilon (float): epsilon
        name (string): the name of log_loss

    Returns:
        Variable: A 2-D tensor with shape [N x 1], the negative log loss.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          label = fluid.layers.data(name='label', shape=[1], dtype='int64')
          prob = fluid.layers.data(name='prob', shape=[10], dtype='float32')
          cost = fluid.layers.log_loss(input=prob, label=label)
    """
    helper = LayerHelper('log_loss', **locals())

    if name is None:
        loss = helper.create_variable_for_type_inference(dtype=input.dtype)
    else:
        loss = helper.create_variable(
            name=name, dtype=input.dtype, persistable=False)

    helper.append_op(
        type='log_loss',
        inputs={'Predicted': [input],
                'Labels': [label]},
        outputs={'Loss': [loss]},
        attrs={'epsilon': epsilon})
    return loss


def teacher_student_sigmoid_loss(input,
                                 label,
                                 soft_max_up_bound=15.0,
                                 soft_max_lower_bound=-15.0):
    """
    **Teacher Student Log Loss Layer**

    This layer accepts input predictions and target label and returns the
    teacher_student loss.

    .. math::
        loss = max(x, 0) - x * z + log(1 + exp(-abs(x))) + max(x, 0) - x * z' + log(1 + exp(-abs(x)))

    Args:
        input (Variable|list):  a 2-D tensor with shape [N x 1], where N is the
                                batch size. This input is a probability computed
                                by the previous operator.
        label (Variable|list):  the ground truth which is a 2-D tensor with
                                shape [N x 1], where N is the batch size.
        soft_max_up_bound  (float):  if input > soft_max_up_bound, will be bound
        soft_max_lower_bound (float): if input < soft_max_lower_bound, will be bound

    Returns:
        Variable: A 2-D tensor with shape [N x 1], the teacher_student_sigmoid_loss.

    Examples:
        .. code-block:: python
          
          import paddle.fluid as fluid

          batch_size = 64
          label = fluid.layers.data(
                    name="label", shape=[batch_size, 1], dtype="int64", append_batch_size=False)
          similarity = fluid.layers.data(
                    name="similarity", shape=[batch_size, 1], dtype="float32", append_batch_size=False)
          cost = fluid.layers.teacher_student_sigmoid_loss(input=similarity, label=label)

    """
    helper = LayerHelper('teacher_student_sigmoid_loss', **locals())
    out = helper.create_variable(dtype=input.dtype)
    helper.append_op(
        type='teacher_student_sigmoid_loss',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out]},
        attrs={"soft_max_lower_bound": float(soft_max_lower_bound), \
                "soft_max_up_bound": float(soft_max_up_bound)})
    return out


def add_position_encoding(input, alpha, beta, name=None):
    """
    **Add Position Encoding Layer**

    This layer accepts an input 3D-Tensor of shape [N x M x P], and returns an
    output Tensor of shape [N x M x P] with positional encoding value.

    Refer to `Attention Is All You Need <http://arxiv.org/pdf/1706.03762.pdf>`_ .

    .. math::
        PE(pos, 2i) &= \\sin{(pos / 10000^{2i / P})}   \\\\
        PE(pos, 2i + 1) &= \\cos{(pos / 10000^{2i / P})}  \\\\
        Out(:, pos, i) &= \\alpha * input(:, pos, i) + \\beta * PE(pos, i)

    Where:
      - :math:`PE(pos, 2i)` : the increment for the number at even position
      - :math:`PE(pos, 2i + 1)` : the increment for the number at odd position

    Args:
        input (Variable): 3-D input tensor with shape [N x M x P]
        alpha (float): multiple of Input Tensor
        beta (float): multiple of Positional Encoding Tensor
        name (string): the name of position encoding layer

    Returns:
        Variable: A 3-D Tensor of shape [N x M x P] with positional encoding.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid

          tensor = fluid.layers.data(
              name='tensor',
              shape=[32, 64, 512],
              dtype='float32',
              append_batch_size=False)
          position_tensor = fluid.layers.add_position_encoding(
              input=tensor, alpha=1.0, beta=1.0)

    """
    helper = LayerHelper('add_position_encoding', **locals())
    dtype = helper.input_dtype()

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        out = helper.create_variable(name=name, dtype=dtype, persistable=False)

    helper.append_op(
        type="add_position_encoding",
        inputs={"X": input},
        outputs={"Out": out},
        attrs={"alpha": alpha,
               "beta": beta})
    return out


def bilinear_tensor_product(x,
                            y,
                            size,
                            act=None,
                            name=None,
                            param_attr=None,
                            bias_attr=None):
    """
    **Add Bilinear Tensor Product Layer**

    This layer performs bilinear tensor product on two inputs.
    For example:

    .. math::
       out_{i} = x * W_{i} * {y^\mathrm{T}}, i=0,1,...,size-1

    In this formula:
      - :math:`x`: the first input contains M elements, shape is [batch_size, M].
      - :math:`y`: the second input contains N elements, shape is [batch_size, N].
      - :math:`W_{i}`: the i-th learned weight, shape is [M, N]
      - :math:`out_{i}`: the i-th element of out, shape is [batch_size, size].
      - :math:`y^\mathrm{T}`: the transpose of :math:`y_{2}`.

    Args:
        x (Variable): 2-D input tensor with shape [batch_size, M]
        y (Variable): 2-D input tensor with shape [batch_size, N]
        size (int): The dimension of this layer.
        act (str, default None): Activation to be applied to the output of this layer.
        name (str, default None): The name of this layer.
        param_attr (ParamAttr, default None): The parameter attribute for the learnable w.
            parameters/weights of this layer.
        bias_attr (ParamAttr, default None): The parameter attribute for the bias
            of this layer. If it is set to False, no bias will be added to the output units.
            If it is set to None, the bias is initialized zero. Default: None.

    Returns:
        Variable: A 2-D Tensor of shape [batch_size, size].

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          layer1 = fluid.layers.data("t1", shape=[-1, 5], dtype="float32")
          layer2 = fluid.layers.data("t2", shape=[-1, 4], dtype="float32")
          tensor = fluid.layers.bilinear_tensor_product(x=layer1, y=layer2, size=1000)
    """
    helper = LayerHelper('bilinear_tensor_product', **locals())
    dtype = helper.input_dtype('x')

    param_shape = [size, x.shape[1], y.shape[1]]

    w = helper.create_parameter(
        attr=helper.param_attr, shape=param_shape, dtype=dtype, is_bias=False)

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        out = helper.create_variable(name=name, dtype=dtype, persistable=False)

    inputs = {"X": x, "Y": y, "Weight": w}
    if helper.bias_attr:
        bias_size = [1, size]
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)
        inputs["Bias"] = bias
    helper.append_op(
        type="bilinear_tensor_product", inputs=inputs, outputs={"Out": out})

    # add activation
    return helper.append_activation(out)


@templatedoc()
def get_tensor_from_selected_rows(x, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python
	    
            import paddle.fluid as fluid
            b = fluid.default_main_program().global_block()
            input = b.create_var(name="X", dtype="float32", persistable=True, type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
            out = fluid.layers.get_tensor_from_selected_rows(input)
    """

    helper = LayerHelper('get_tensor_from_selected_rows', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='get_tensor_from_selected_rows',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={})
    return out


def shuffle_channel(x, group, name=None):
    """
    **Shuffle Channel Operator**

    This operator shuffles the channels of input x.
    It divide the input channels in each group into :attr:`group` subgroups,
    and obtain a new order by selecting element from every subgroup one by one.

    Please refer to the paper
    https://arxiv.org/pdf/1707.01083.pdf
    
    .. code-block:: text

        Given a 4-D tensor input with the shape (N, C, H, W):
            input.shape = (1, 4, 2, 2)
            input.data =[[[[0.1, 0.2],
                           [0.2, 0.3]],

                          [[0.3, 0.4],
                           [0.4, 0.5]],

                          [[0.5, 0.6],
                           [0.6, 0.7]],

                          [[0.7, 0.8],
                           [0.8, 0.9]]]]
            Given group: 2
            then we get a 4-D tensor out whth the same shape of input:
            out.shape = (1, 4, 2, 2)
            out.data = [[[[0.1, 0.2],
                          [0.2, 0.3]],
                          
                         [[0.5, 0.6],
                          [0.6, 0.7]],
                          
                         [[0.3, 0.4],
                          [0.4, 0.5]],
                          
                         [[0.7, 0.8],
                          [0.8, 0.9]]]]
                        
    Args: 
        x(Variable): The input tensor variable. It should be a 4-D tensor with shape [N, C, H, W]
        group(int): Indicating the conuts of subgroups, It should divide the number of channels.

    Returns:
        out(Variable): the channels shuffling result is a tensor variable with the 
        same shape and same type as the input.

    Raises:
        ValueError: If group is not an int type variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.layers.data(name='input', shape=[4,2,2], dtype='float32')
            out = fluid.layers.shuffle_channel(x=input, group=2)
    """
    helper = LayerHelper("shuffle_channel", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if not isinstance(group, int):
        raise TypeError("group must be int type")

    helper.append_op(
        type="shuffle_channel",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={"group": group})
    return out


@templatedoc()
def temporal_shift(x, seg_num, shift_ratio=0.25, name=None):
    """
    **Temporal Shift Operator**
    
    ${comment}
                        
    Args: 
        x(Variable): ${x_comment}
        seg_num(int): ${seg_num_comment}
        shift_ratio(float): ${shift_ratio_comment}
        name (str, default None): The name of this layer.

    Returns:
        out(Variable): The temporal shifting result is a tensor variable with the 
        same shape and same type as the input.

    Raises:
        TypeError: seg_num must be int type.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.layers.data(name='input', shape=[4,2,2], dtype='float32')
            out = fluid.layers.temporal_shift(x=input, seg_num=2, shift_ratio=0.2)
    """
    helper = LayerHelper("temporal_shift", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if not isinstance(seg_num, int):
        raise TypeError("seg_num must be int type.")

    helper.append_op(
        type="temporal_shift",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={"seg_num": seg_num,
               "shift_ratio": shift_ratio})
    return out


class PyFuncRegistry(object):
    _register_funcs = []

    def __init__(self, func):
        if func is None or not callable(func):
            raise TypeError('func must be a Python function')

        self._func = func
        # find named args using reflection
        args = inspect.getargspec(self._func)
        if len(args[0]) == 0 and args[1] is None and args[2] is None:
            # Function with no inputs
            self._named_args = None
        else:
            self._named_args = args[0]
        self._id = core._append_python_callable_object_and_return_id(self)
        '''
        Why record self here?

        1. For debug usage. Users can call
           :code:`py_func.registered_func(idx)` method
           to find the registered function corresponding
           to :code:`idx`.

        2. For increasing reference count of self.
           It seems that to release Python object
           whose reference count is 1 would cause
           segmentation fault error in C++ side.
           May be lack of Python GC in C++ side?
        '''
        PyFuncRegistry._register_funcs.append(self)

    @classmethod
    def registered_func(cls, idx):
        return cls._register_funcs[idx]._func

    @classmethod
    def registered_func_num(cls):
        return len(cls._register_funcs)

    @property
    def id(self):
        return self._id

    def __call__(self, *args):
        if self._named_args is None:
            func_ret = self._func()
        else:
            kwargs = dict()
            idx = 0
            for arg in self._named_args:
                kwargs[arg] = args[idx]
                idx += 1
            func_ret = self._func(*args[idx:], **kwargs)

        if not isinstance(func_ret, (list, tuple)):
            func_ret = (func_ret, )

        ret = []
        for each_ret in func_ret:
            if each_ret is None or isinstance(each_ret, core.LoDTensor):
                ret.append(each_ret)
                continue

            if not isinstance(each_ret, np.ndarray):
                each_ret = np.array(each_ret)

            tensor = core.LoDTensor()
            tensor.set(each_ret, core.CPUPlace())
            ret.append(tensor)

        return tuple(ret)


@templatedoc()
def py_func(func, x, out, backward_func=None, skip_vars_in_backward_input=None):
    """
    PyFunc Operator.

    User can use :code:`py_func` to register operators in Python side.
    The inputs of :code:`func` is :code:`LoDTensor` and outputs can be
    numpy array or :code:`LoDTensor`. Paddle would call the registered
    :code:`func` in forward part, and call :code:`backward_func` in
    backward part (if :code:`backward_func` is not None).

    User should set the right data type and shape of :code:`out` before
    calling this function. However, data types and shapes of gradients of
    :code:`out` and :code:`x` would be inferred automatically.

    Input orders of :code:`backward_func` would be: forward inputs
    :code:`x`, forward outputs :code:`out` and backward input gradients of
    :code:`out`. If some variables of :code:`out` have no gradient, the input
    tensor would be None in Python side. If some variables of :code:`in` have
    no gradient, users should return None.

    This function can also be used to debug the running network. User can
    add a :code:`py_func` operator without output, and print input
    :code:`x` inside :code:`func`.

    Args:
        func (callable): forward Python function.
        x (Variable|list(Variable)|tuple(Variable)): inputs of :code:`func`.
        out (Variable|list(Variable)|tuple(Variable)): outputs of :code:`func`.
            Paddle cannot infer shapes and data types of :code:`out`. Users
            should create :code:`out` beforehand.
        backward_func (callable|None): backward Python function.
                                       None means no backward. Default None.
        skip_vars_in_backward_input (Variable|list(Variable)|tuple(Variable)):
            Variables that are not needed in :code:`backward_func` inputs.
            These variables must be any of :code:`x` and :code:`out`.
            If set, these vars would not be inputs of :code:`backward_func`,
            Only useful when :code:`backward_func` is not None. Default None.

    Returns:
        out (Variable|list(Variable)|tuple(Variable)): input :code:`out`

    Examples:

        >>> import paddle.fluid as fluid
        >>> import six
        >>>
        >>> def create_tmp_var(name, dtype, shape):
        >>>     return fluid.default_main_program().current_block().create_var(
        >>>         name=name, dtype=dtype, shape=shape)
        >>>
        >>> # tanh activation has been provided by Paddle C++ op
        >>> # Here, we only use tanh to be an example to show the usage
        >>> # of py_func
        >>> def tanh(x):
        >>>     return np.tanh(x)
        >>>
        >>> # forward input x is skipped
        >>> def tanh_grad(y, dy):
        >>>     return np.array(dy) * (1 - np.square(np.array(y)))
        >>>
        >>> def debug_func(x):
        >>>     print(x)
        >>>
        >>> def simple_net(img, label):
        >>>     hidden = img
        >>>     for idx in six.moves.range(4):
        >>>         hidden = fluid.layers.fc(hidden, size=200)
        >>>         new_hidden = create_tmp_var(name='hidden_{}'.format(idx),
        >>>             dtype=hidden.dtype, shape=hidden.shape)
        >>>
        >>>         # user-defined layers with forward and backward
        >>>         hidden = fluid.layers.py_func(func=tanh, x=hidden,
        >>>             out=new_hidden, backward_func=tanh_grad,
        >>>             skip_vars_in_backward_input=hidden)
        >>>
        >>>         # user-defined debug layers to print variables
        >>>         fluid.layers.py_func(func=debug_func, x=hidden, out=None)
        >>>
        >>>     prediction = fluid.layers.fc(hidden, size=10, act='softmax')
        >>>     loss = fluid.layers.cross_entropy(input=prediction, label=label)
        >>>     return fluid.layers.mean(loss)
    """
    helper = LayerHelper('py_func', **locals())
    if x is None:
        x = []
    elif isinstance(x, Variable):
        x = [x]
    elif not isinstance(x, (list, tuple)):
        raise TypeError('Input must be Variable/list(Variable)/tuple(Variable)')

    if out is None:
        out_list = []
    elif isinstance(out, Variable):
        out_list = [out]
    elif isinstance(out, (list, tuple)):
        out_list = out
    else:
        raise TypeError(
            'Output must be Variable/list(Variable)/tuple(Variable)')

    fwd_func_id = PyFuncRegistry(func).id
    bwd_func_id = PyFuncRegistry(
        backward_func).id if backward_func is not None else -1

    for each_out in out_list:
        if len(each_out.shape) == 0:
            raise ValueError(
                'Output shapes of py_func op should be provided by users manually'
            )

    backward_skip_vars = set()
    if backward_func is not None and skip_vars_in_backward_input is not None:
        if isinstance(skip_vars_in_backward_input, Variable):
            skip_vars_in_backward_input = [skip_vars_in_backward_input]

        fwd_in_out = [v.name for v in x]
        fwd_in_out.extend([v.name for v in out_list])
        fwd_in_out = set(fwd_in_out)
        backward_skip_vars = set()
        for v in skip_vars_in_backward_input:
            if not v.name in fwd_in_out:
                raise ValueError(
                    'Variable {} is not found in forward inputs and outputs'
                    .format(v.name))
            backward_skip_vars.add(v.name)

    helper.append_op(
        type='py_func',
        inputs={'X': x},
        outputs={'Out': out_list},
        attrs={
            'forward_callable_id': fwd_func_id,
            'backward_callable_id': bwd_func_id,
            'backward_skip_vars': list(backward_skip_vars)
        })
    return out


# For debug usage
py_func.registered_func = PyFuncRegistry.registered_func
py_func.registered_func_num = PyFuncRegistry.registered_func_num


@templatedoc()
def psroi_pool(input,
               rois,
               output_channels,
               spatial_scale,
               pooled_height,
               pooled_width,
               name=None):
    """
    ${comment}

    Args:
        input (Variable): ${x_comment}
        rois (Variable): ROIs (Regions of Interest) to pool over.It should be
                         a 2-D LoDTensor of shape (num_rois, 4), the lod level
                         is 1. Given as [[x1, y1, x2, y2], ...], (x1, y1) is
                         the top left coordinates, and (x2, y2) is the bottom
                         right coordinates.
        output_channels (integer): ${output_channels_comment}
        spatial_scale (float): ${spatial_scale_comment} Default: 1.0
        pooled_height (integer): ${pooled_height_comment} Default: 1
        pooled_width (integer): ${pooled_width_comment} Default: 1
        name (str, default None): The name of this layer.

    Returns:
        Variable: ${out_comment}.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[490, 28, 28], dtype='float32')
            rois = fluid.layers.data(name='rois', shape=[4], lod_level=1, dtype='float32')
            pool_out = fluid.layers.psroi_pool(x, rois, 10, 1.0, 7, 7)
    """
    helper = LayerHelper('psroi_pool', **locals())
    # check attrs
    if not isinstance(output_channels, int):
        raise TypeError("output_channels must be int type")
    if not isinstance(spatial_scale, float):
        raise TypeError("spatial_scale must be float type")
    if not isinstance(pooled_height, int):
        raise TypeError("pooled_height must be int type")
    if not isinstance(pooled_width, int):
        raise TypeError("pooled_width must be int type")
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='psroi_pool',
        inputs={'X': input,
                'ROIs': rois},
        outputs={'Out': out},
        attrs={
            'output_channels': output_channels,
            'spatial_scale': spatial_scale,
            'pooled_height': pooled_height,
            'pooled_width': pooled_width
        })
    return out


@templatedoc()
def prroi_pool(input,
               rois,
               output_channels,
               spatial_scale=1.0,
               pooled_height=1,
               pooled_width=1,
               name=None):
    """
    The precise roi pooling implementation for paddle?https://arxiv.org/pdf/1807.11590.pdf

    Args:
        input (Variable):The input of Deformable PSROIPooling.The shape of input tensor is
                        [N,C,H,W]. Where N is batch size,C is number of input channels,H
                        is height of the feature, and W is the width of the feature.
        rois (Variable): ROIs (Regions of Interest) to pool over.It should be
                        a 2-D LoDTensor of shape (num_rois, 4), the lod level
                        is 1. Given as [[x1, y1, x2, y2], ...], (x1, y1) is
                        the top left coordinates, and (x2, y2) is the bottom
                        right coordinates.
        output_channels (integer): The output's channel.
        spatial_scale (float): Ratio of input feature map height (or width) to raw image height (or width).
                             Equals the reciprocal of total stride in convolutional layers, Default: 1.0.
        pooled_height (integer): The pooled output height. Default: 1.
        pooled_width (integer): The pooled output width. Default: 1.
        name (str, default None): The name of this operation.

    Returns:
        Variable(Tensor): The shape of the returned Tensor is (num_rois, output_channels, pooled_h, pooled_w), with value type float32,float16..

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[490, 28, 28], dtype='float32')
            rois = fluid.layers.data(name='rois', shape=[4], lod_level=1, dtype='float32')
            pool_out = fluid.layers.prroi_pool(x, rois, 10, 1.0, 7, 7)
    """
    helper = LayerHelper('prroi_pool', **locals())
    # check attrs
    if not isinstance(output_channels, int):
        raise TypeError("output_channels must be int type")
    if not isinstance(spatial_scale, float):
        raise TypeError("spatial_scale must be float type")
    if not isinstance(pooled_height, int):
        raise TypeError("pooled_height must be int type")
    if not isinstance(pooled_width, int):
        raise TypeError("pooled_width must be int type")
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='prroi_pool',
        inputs={'X': input,
                'ROIs': rois},
        outputs={'Out': out},
        attrs={
            'output_channels': output_channels,
            'spatial_scale': spatial_scale,
            'pooled_height': pooled_height,
            'pooled_width': pooled_width
        })
    return out


def huber_loss(input, label, delta):
    """
    Huber loss is a loss function used in robust.
    Huber loss can evaluate the fitness of input to label.
    Different from MSE loss, Huber loss is more robust for outliers.

    When the difference between input and label is large than delta
    .. math::

        huber\_loss = delta * (label - input) - 0.5 * delta * delta

    When the difference between input and label is less than delta
    .. math::

        huber\_loss = 0.5 * (label - input) * (label - input)


    Args:
        input (Variable): This input is a probability computed by the previous operator.
                          The first dimension is batch size, and the last dimension is 1.
        label (Variable): The groud truth whose first dimension is batch size
                          and last dimension is 1.
        delta (float): The parameter of huber loss, which controls
                       the range of outliers

    Returns:
        huber\_loss (Variable): The huber loss with shape [batch_size, 1].

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.layers.data(name='x', shape=[13], dtype='float32')
            predict = fluid.layers.fc(input=x, size=1)
            label = fluid.layers.data(
                name='label', shape=[1], dtype='float32')
            loss = fluid.layers.huber_loss(
                input=predict, label=label, delta=1.0)

    """
    helper = LayerHelper('huber_loss', **locals())
    residual = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='huber_loss',
        inputs={'X': input,
                'Y': label},
        outputs={'Out': out,
                 'Residual': residual},
        attrs={'delta': delta})
    return out


@templatedoc()
def kldiv_loss(x, target, reduction='mean', name=None):
    """
    ${comment}

    Args:
        x (Variable): ${x_comment}
        target (Variable): ${target_comment}
        reduction (Variable): ${reduction_comment}
        name (str, default None): The name of this layer.

    Returns:
        kldiv\_loss (Variable): The KL divergence loss.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[4,2,2], dtype='float32')
            target = fluid.layers.data(name='target', shape=[4,2,2], dtype='float32')
            loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='batchmean')
    """
    helper = LayerHelper('kldiv_loss', **locals())
    loss = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='kldiv_loss',
        inputs={'X': x,
                'Target': target},
        outputs={'Loss': loss},
        attrs={'reduction': reduction})
    return loss


from .ops import square
from .control_flow import equal


def npair_loss(anchor, positive, labels, l2_reg=0.002):
    '''
  **Npair Loss Layer**

  Read `Improved Deep Metric Learning with Multi class N pair Loss Objective <http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf>`_ .

  Npair loss requires paired data. Npair loss has two parts: the first part is L2
  regularizer on the embedding vector; the second part is cross entropy loss which
  takes the similarity matrix of anchor and positive as logits.

  Args:
    anchor(Variable): embedding vector for the anchor image. shape=[batch_size, embedding_dims]
    positive(Variable): embedding vector for the positive image. shape=[batch_size, embedding_dims]
    labels(Variable): 1-D tensor. shape=[batch_size]
    l2_reg(float32): L2 regularization term on embedding vector, default: 0.002

  Returns:
    npair loss(Variable): return npair loss, shape=[1]

  Examples:
    .. code-block:: python

       import paddle.fluid as fluid
       anchor = fluid.layers.data(
                     name = 'anchor', shape = [18, 6], dtype = 'float32', append_batch_size=False)
       positive = fluid.layers.data(
                     name = 'positive', shape = [18, 6], dtype = 'float32', append_batch_size=False)
       labels = fluid.layers.data(
                     name = 'labels', shape = [18], dtype = 'float32', append_batch_size=False)

       npair_loss = fluid.layers.npair_loss(anchor, positive, labels, l2_reg = 0.002)
  '''
    Beta = 0.25
    batch_size = labels.shape[0]

    labels = reshape(labels, shape=[batch_size, 1], inplace=True)
    labels = expand(labels, expand_times=[1, batch_size])

    labels = equal(labels, transpose(labels, perm=[1, 0])).astype('float32')
    labels = labels / reduce_sum(labels, dim=1, keep_dim=True)

    l2loss = reduce_mean(reduce_sum(square(anchor), 1)) \
             + reduce_mean(reduce_sum(square(positive), 1))
    l2loss = l2loss * Beta * l2_reg

    similarity_matrix = matmul(
        anchor, positive, transpose_x=False, transpose_y=True)
    softmax_ce = softmax_with_cross_entropy(
        logits=similarity_matrix, label=labels, soft_label=True)
    cross_entropy = reduce_sum(labels * softmax_ce, 0)
    celoss = reduce_mean(cross_entropy)

    return l2loss + celoss


def pixel_shuffle(x, upscale_factor):
    """

    **Pixel Shuffle Layer**

    This layer rearranges elements in a tensor of shape [N, C, H, W]
    to a tensor of shape [N, C/r**2, H*r, W*r].
    This is useful for implementing efficient sub-pixel convolution
    with a stride of 1/r.
    Please refer to the paper: `Real-Time Single Image and Video Super-Resolution 
    Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ .
    by Shi et. al (2016) for more details.

        .. code-block:: text
        
            Given a 4-D tensor with the shape:
                x.shape = [1, 9, 4, 4]
            Given upscale_factor:
                upscale_factor= 3
            output shape is:
                [1, 1, 12, 12]
    
    Args:

        x(Variable): The input tensor variable.
        upscale_factor(int): factor to increase spatial resolution

    Returns:

        Out(Variable): Reshaped tensor according to the new dimension.

    Raises:

        ValueError: If the square of upscale_factor cannot divide the channels of input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.layers.data(name="input", shape=[9,4,4])
            output = fluid.layers.pixel_shuffle(x=input, upscale_factor=3)

    """

    helper = LayerHelper("pixel_shuffle", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if not isinstance(upscale_factor, int):
        raise TypeError("upscale factor must be int type")

    helper.append_op(
        type="pixel_shuffle",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={"upscale_factor": upscale_factor})
    return out


def fsp_matrix(x, y):
    """

    **FSP matrix op**

    This op is used to calculate the flow of solution procedure (FSP) matrix of two feature maps.
    Given feature map x with shape [x_channel, h, w] and feature map y with shape
    [y_channel, h, w], we can get the fsp matrix of x and y in two steps:

    1. reshape x into matrix with shape [x_channel, h * w] and reshape and
       transpose y into matrix with shape [h * w, y_channel].
    2. multiply x and y to get fsp matrix with shape [x_channel, y_channel].

    The output is a batch of fsp matrices.

    Args:

        x (Variable): A feature map with shape [batch_size, x_channel, height, width].
        y (Variable): A feature map with shape [batch_size, y_channel, height, width].
                      The y_channel can be different with the x_channel of Input(X)
                      while the other dimensions must be the same with Input(X)'s.

    Returns:

        fsp matrix (Variable): The output of FSP op with shape [batch_size, x_channel, y_channel].
        The x_channel is the channel of x and the y_channel is the channel of y.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.layers.data(name='data', shape=[3, 32, 32])
            feature_map_0 = fluid.layers.conv2d(data, num_filters=2,
                                                filter_size=3)
            feature_map_1 = fluid.layers.conv2d(feature_map_0, num_filters=2,
                                                filter_size=1)
            loss = fluid.layers.fsp_matrix(feature_map_0, feature_map_1)

    """
    helper = LayerHelper('fsp_matrix', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype(
        input_param_name='x'))
    helper.append_op(type='fsp', inputs={'X': x, 'Y': y}, outputs={'Out': out})
    return out


def continuous_value_model(input, cvm, use_cvm=True):
    """

    **continuous_value_model layers**

    continuous value model(cvm). Now, it only considers show and click value in CTR project.
    We assume that input is an embedding vector with cvm_feature, whose shape is [N * D] (D is 2 + embedding dim).
    If use_cvm is True, it will log(cvm_feature), and output shape is [N * D].
    If use_cvm is False, it will remove cvm_feature from input, and output shape is [N * (D - 2)].
    
    This layer accepts a tensor named input which is ID after embedded(lod level is 1), cvm is a show_click info.

    Args:

        input (Variable): a 2-D LodTensor with shape [N x D], where N is the batch size, D is 2 + the embedding dim. lod level = 1.
        cvm (Variable):   a 2-D Tensor with shape [N x 2], where N is the batch size, 2 is show and click.
        use_cvm  (bool):  use cvm or not. if use cvm, the output dim is the same as input
                          if don't use cvm, the output dim is input dim - 2(remove show and click)
                          (cvm op is a customized op, which input is a sequence has embed_with_cvm default, so we need an op named cvm to decided whever use it or not.)

    Returns:

        Variable: A 2-D LodTensor with shape [N x D], if use cvm, D is equal to input dim, if don't use cvm, D is equal to input dim - 2. 

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          input = fluid.layers.data(name="input", shape=[-1, 1], lod_level=1, append_batch_size=False, dtype="int64")#, stop_gradient=False)
          label = fluid.layers.data(name="label", shape=[-1, 1], append_batch_size=False, dtype="int64")
          embed = fluid.layers.embedding(
                            input=input,
                            size=[100, 11],
                            dtype='float32')
          ones = fluid.layers.fill_constant_batch_size_like(input=label, shape=[-1, 1], dtype="int64", value=1)
          show_clk = fluid.layers.cast(fluid.layers.concat([ones, label], axis=1), dtype='float32')
          show_clk.stop_gradient = True
          input_with_cvm = fluid.layers.continuous_value_model(embed, show_clk, True)

    """
    helper = LayerHelper('cvm', **locals())
    out = helper.create_variable(dtype=input.dtype)
    helper.append_op(
        type='cvm',
        inputs={'X': [input],
                'CVM': [cvm]},
        outputs={'Y': [out]},
        attrs={"use_cvm": use_cvm})
    return out


def where(condition):
    """
    Return an int64 tensor with rank 2, specifying the coordinate of true element in `condition`.

    Output's first dimension is the number of true element, second dimension is rank(number of dimension) of `condition`.
    If there is zero true element, then an empty tensor will be generated.  

    Args:
        condition(Variable): A bool tensor with rank at least 1.

    Returns:
        Variable: The tensor variable storing a 2-D tensor. 

    Examples:
        .. code-block:: python

             import paddle.fluid as fluid
             import paddle.fluid.layers as layers
             import numpy as np

             # condition is a tensor [True, False, True]
             condition = layers.assign(np.array([1, 0, 1], dtype='int32'))
             condition = layers.cast(condition, 'bool')
             out = layers.where(condition) # [[0], [2]]

             # condition is a tensor [[True, False], [False, True]]
             condition = layers.assign(np.array([[1, 0], [0, 1]], dtype='int32'))
             condition = layers.cast(condition, 'bool')
             out = layers.where(condition) # [[0, 0], [1, 1]]

             # condition is a tensor [False, False, False]
             condition = layers.assign(np.array([0, 0, 0], dtype='int32'))
             condition = layers.cast(condition, 'bool')
             out = layers.where(condition) # [[]]

    """
    helper = LayerHelper("where", **locals())

    out = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.INT64)

    helper.append_op(
        type='where', inputs={'Condition': condition}, outputs={'Out': [out]})
    return out


def sign(x):
    """
    **sign**

    This function returns sign of every element in `x`: 1 for positive, -1 for negative and 0 for zero.

    Args:
        x(Variable|numpy.ndarray): The input tensor.

    Returns:
        Variable: The output sign tensor with identical shape and dtype to `x`.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np

          # [1, 0, -1]
          data = fluid.layers.sign(np.array([3, 0, -2], dtype='int32')) 

    """

    helper = LayerHelper("sign", **locals())

    if not isinstance(x, Variable):
        x = assign(x)

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(type='sign', inputs={'X': [x]}, outputs={'Out': [out]})

    return out


def unique(x, dtype='int32'):
    """
    **unique** 

    Return a unique tensor for `x` and an index tensor pointing to this unique tensor.

    Args:
        x(Variable): A 1-D input tensor.
        dtype(np.dtype|core.VarDesc.VarType|str): The type of index tensor: int32, int64.

    Returns:
        tuple: (out, index). `out` is the unique tensor for `x`, with identical dtype to `x`, and \
            `index` is an index tensor pointing to `out`, by which user can recover the original `x` tensor.

    Examples:
        .. code-block:: python

             import numpy as np
             import paddle.fluid as fluid
             x = fluid.assign(np.array([2, 3, 3, 1, 5, 3], dtype='int32'))
             out, index = fluid.layers.unique(x) # out is [2, 3, 1, 5]; index is [0, 1, 1, 2, 3, 1]
    """

    helper = LayerHelper("unique", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    index = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type='unique',
        inputs={'X': x},
        attrs={'dtype': convert_np_dtype_to_dtype_(dtype)},
        outputs={'Out': [out],
                 'Index': [index]})

    return out, index


def unique_with_counts(x, dtype='int32'):
    """
    **unique** 

    Return a unique tensor for `x` and an index tensor pointing to this unique tensor.

    Args:
        x(Variable): A 1-D input tensor.
        dtype(np.dtype|core.VarDesc.VarType|str): The type of index tensor: int32, int64.

    Returns:
        tuple: (out, index, count). `out` is the unique tensor for `x`, with identical dtype to `x`, and \
            `index` is an index tensor pointing to `out`, by which user can recover the original `x` tensor, \
            `count` is count of unqiue element in the `x`.

    Examples:
        .. code-block:: python

             import numpy as np
             import paddle.fluid as fluid
             x = fluid.layers.assign(np.array([2, 3, 3, 1, 5, 3], dtype='int32'))
             out, index, count = fluid.layers.unique_with_counts(x) # out is [2, 3, 1, 5]; index is [0, 1, 1, 2, 3, 1]
                                                        # count is [1, 3, 1, 1]
    """
    if not (dtype == 'int32' or dtype == 'int64'):
        raise TypeError(
            "Op unique_with_counts, index dtype must be int32 or int64")

    if x is None or len(x.shape) != 1:
        raise ValueError(
            "Op unique_with_counts, x must not be null and size of dim must be 1"
        )

    helper = LayerHelper("unique_with_counts", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    index = helper.create_variable_for_type_inference(dtype)

    count = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type='unique_with_counts',
        inputs={'X': x},
        attrs={'dtype': convert_np_dtype_to_dtype_(dtype)},
        outputs={'Out': [out],
                 'Index': [index],
                 'Count': [count]})

    return out, index, count


def deformable_conv(input,
                    offset,
                    mask,
                    num_filters,
                    filter_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=None,
                    deformable_groups=None,
                    im2col_step=None,
                    param_attr=None,
                    bias_attr=None,
                    modulated=True,
                    name=None):
    """
    **Deformable Convolution Layer**

    Compute 2-D deformable convolution on 4-D input.
    Given input image x, output feature map y, the deformable convolution operation can be expressed as follow:
   
    
    Deformable Convolution v2: 
    
    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}

    Deformable Convolution v1:
    
    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k)}
    
    Where :math:`\Delta p_k` and :math:`\Delta m_k` are the learnable offset and modulation scalar for the k-th location, 
    which :math:`\Delta m_k` is one in deformable convolution v1. Please refer to `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168v2>`_ and `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_.
    
    Example:
        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

          Offset shape: :math:`(N, 2 * deformable\_groups * H_f * H_w, H_{in}, W_{in})`

          Mask shape: :math:`(N, deformable\_groups * H_f * H_w, H_{in}, W_{in})`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        input (Variable): The input image with [N, C, H, W] format.
        offset (Variable): The input coordinate offset of deformable convolution layer.
        Mask (Variable): The input mask of deformable covolution layer.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        stride (int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        padding (int|tuple): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        dilation (int|tuple): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: dilation = 1.
        groups (int): The groups number of the deformable conv layer. According to
            grouped convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        deformable_groups (int): The number of deformable group partitions.
            Default: deformable_groups = 1.
        im2col_step (int): Maximum number of images per im2col computation; 
            The total batch size should be divisable by this value or smaller
            than this value; if you face out of memory problem, you can try
            to use a smaller value here.
            Default: im2col_step = 64.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of deformable conv. If it is set to None or one attribute of ParamAttr,
            deformable conv will create ParamAttr as param_attr.
            If the Initializer of the param_attr is not set, the parameter is
            initialized with :math:`Normal(0.0, std)`, and the 
            :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of
            deformable conv layer. If it is set to False, no bias will be added
            to the output units. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        modulated (bool): Make sure which version should be used between v1 and v2, where v2 is \
            used while True. Default: True.
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None
    Returns:
        Variable: The tensor variable storing the deformable convolution \
                  result.
    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.
    Examples:
        .. code-block:: python

          #deformable conv v2:
         
          import paddle.fluid as fluid
          data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
          offset = fluid.layers.data(name='offset', shape=[18, 32, 32], dtype='float32')
          mask = fluid.layers.data(name='mask', shape=[9, 32, 32], dtype='float32')
          out = fluid.layers.deformable_conv(input=data, offset=offset, mask=mask,
                                             num_filters=2, filter_size=3, padding=1, modulated=True)

          #deformable conv v1:

          import paddle.fluid as fluid
          data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
          offset = fluid.layers.data(name='offset', shape=[18, 32, 32], dtype='float32')
          out = fluid.layers.deformable_conv(input=data, offset=offset, mask=None,
                                             num_filters=2, filter_size=3, padding=1, modulated=False)
    """

    num_channels = input.shape[1]
    assert param_attr is not False, "param_attr should not be False here."

    helper = LayerHelper('deformable_conv', **locals())
    dtype = helper.input_dtype()

    if not isinstance(input, Variable):
        raise TypeError("Input of deformable_conv must be Variable")
    if not isinstance(offset, Variable):
        raise TypeError("Input Offset of deformable_conv must be Variable")

    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        num_filter_channels = num_channels // groups

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    input_shape = input.shape
    filter_shape = [num_filters, int(num_filter_channels)] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * num_channels
        std = (2.0 / filter_elem_num)**0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    pre_bias = helper.create_variable_for_type_inference(dtype)

    if modulated:
        helper.append_op(
            type='deformable_conv',
            inputs={
                'Input': input,
                'Filter': filter_param,
                'Offset': offset,
                'Mask': mask,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            })

    else:
        helper.append_op(
            type='deformable_conv_v1',
            inputs={
                'Input': input,
                'Filter': filter_param,
                'Offset': offset,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            })

    output = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    return output


def unfold(x, kernel_sizes, strides=1, paddings=0, dilations=1, name=None):
    """

    This function returns a col buffer of sliding local blocks of input x, also known
    as im2col for batched 2D image tensors. For each block under the convolution filter,
    all element will be rearranged as a column. While the convolution filter silding over
    the input feature map, a series of such columns will be formed.

    For each input :math:`X` with shape [N, C, H, W], the output shape [N, Cout, Lout]
    can be calculated as following.

    .. math::

        dkernel[0] &= dilations[0] \\times (kernel\_sizes[0] - 1) + 1

        dkernel[1] &= dilations[1] \\times (kernel\_sizes[1] - 1) + 1

        hout &= \\frac{H + paddings[0] + paddings[2] - dkernel[0]}{strides[0]} + 1

        wout &= \\frac{W + paddings[1] + paddings[3] - dkernel[1]}{strides[1]} + 1

        Cout &= C \\times kernel\_sizes[0] \\times kernel\_sizes[1]

        Lout &= hout \\times wout


    Args:
        x(Varaible):              The input tensor of format [N, C, H, W].
        kernel_sizes(int|list):   The size of convolution kernel, should be [k_h, k_w]
                                  or an integer k treated as [k, k].
        strides(int|list):        The strides, should be [stride_h, stride_w]
                                  or an integer stride treated as [sride, stride].
                                  For default, strides will be [1, 1].
        paddings(int|list):       The paddings of each dimension, should be
                                  [padding_top, padding_left, padding_bottom, padding_right]
                                  or [padding_h, padding_w] or an integer padding.
                                  If [padding_h, padding_w] was given, it will expanded to
                                  [padding_h, padding_w, padding_h, padding_w]. If an integer
                                  padding was given, [padding, padding, padding, padding] will
                                  be used. For default, paddings will be [0, 0, 0, 0]
        dilations(int|list):      the dilations of convolution kernel, shold be
                                  [dilation_h, dilation_w], or an integer dialtion treated as
                                  [dilation, dilation]. For default, it will be [1, 1].

    
    Returns:
        Variable: The tensor variable corresponding to the sliding local blocks. The output shape is [N, Cout, Lout] as decribled above. Cout is the  total number of values within each block, and Lout is the total number of such blocks.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name = 'data', shape = [3, 224, 224], dtype = 'float32')
            y = fluid.layers.unfold(x, [3, 3], 1, 1, 1)
    """

    helper = LayerHelper("unfold", **locals())

    assert len(x.shape) == 4, \
            "input should be the format of [N, C, H, W]"

    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes, kernel_sizes]
    else:
        assert isinstance(kernel_sizes, list) and (len(kernel_sizes) == 2), \
            "kernel_sizes should either be an integer or a list of two integers"

    if isinstance(strides, int):
        strides = [strides, strides]
    else:
        assert isinstance(strides, list) and (len(strides) == 2), \
            "strides should either be an integer or a list of two integers"

    if isinstance(dilations, int):
        dilations = [dilations, dilations]
    else:
        assert isinstance(dilations, list) and (len(dilations) == 2), \
            "dilations should either be an integer or a list of two integers"

    if isinstance(paddings, int):
        paddings = [paddings] * 4
    elif isinstance(paddings, list):
        if len(paddings) == 2:
            paddings = paddings * 2
        elif len(paddings) == 4:
            pass
        else:
            raise ValueError(
                "paddings should either be an integer or a list of 2 or 4 integers"
            )
    else:
        raise ValueError(
            "Unexpected type of paddings, it should be either an integer or a list"
            "of 2 or 4 integers")

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="unfold",
        inputs={"X": x},
        outputs={"Y": out},
        attrs={
            "kernel_sizes": kernel_sizes,
            "strides": strides,
            "paddings": paddings,
            "dilations": dilations
        })
    return out


def deformable_roi_pooling(input,
                           rois,
                           trans,
                           no_trans=False,
                           spatial_scale=1.0,
                           group_size=[1, 1],
                           pooled_height=1,
                           pooled_width=1,
                           part_size=None,
                           sample_per_part=1,
                           trans_std=0.1,
                           position_sensitive=False,
                           name=None):
    """
    Deformable ROI Pooling Layer
  
    Performs deformable region-of-interest pooling on inputs. As described
    in `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_, it will get offset for each bin after 
    roi pooling so that pooling at correct region. Batch_size will change to the number of region bounding boxes after deformable_roi_pooling.
  
    The operation has three steps:
    
    1. Dividing each region proposal into equal-sized sections with the pooled_width and pooled_height.
  
    2. Add offset to pixel in ROI to get new location and the new value which are computed directly through
       bilinear interpolation with four nearest pixel.
     
    3. Sample several points in each bin to get average values as output.
  
  
    Args:
        input (Variable):The input of deformable roi pooling and it is tensor which value type is float32. The shape of input is
                         [N, C, H, W]. Where N is batch size, C is number of input channels,
                         H is height of the feature, and W is the width of the feature.
        rois (Variable): ROIs (Regions of Interest) with type float32 to pool over. It should be
                         a 2-D LoDTensor of shape (num_rois, 4), and the lod level
                         is 1. Given as [[x1, y1, x2, y2], ...], (x1, y1) is
                         the top left coordinates, and (x2, y2) is the bottom
                         right coordinates, which value type is float32.
        trans (Variable): Offset of features on ROIs while pooling which value type is float32. The format is [N, C, H, W], where 
                          N is number of ROIs, C is number of channels, which indicate the offset distance 
                          in the x and y directions, H is pooled height, and W is pooled width. 
        no_trans (bool): Whether to add offset to get new value or not while roi pooling, which value with type bool is True or False.
                         If value is True, no offset will be added in operation. Default: False.
        spatial_scale (float): Ratio of input feature map height (or width) to raw image height (or width), which value type is float32.
                         Equals the reciprocal of total stride in convolutional layers, Default: 1.0.
        group_size (list|tuple): The number of groups which input channels are divided and the input is list or tuple, which value type is int32. (eg.number of input channels 
                          is k1 * k2 * (C + 1), which k1 and k2 are group width and height and C+1 is number of output
                          chanels.) eg.(4, 6), which 4 is height of group and 6 is width of group. Default: [1, 1].
        pooled_height (int): The pooled output height which value type is int32. Default: 1.
        pooled_width (int): The pooled output width which value type is int32. Default: 1.
        part_size (list|tuple): The height and width of offset which values in list or tuple is int32, eg.(4, 6), which height is 4 and width is 6, and values always equal to pooled_height \
                         and pooled_width. Default: if None, default value is [pooled_height, pooled_width].
        sample_per_part (int): The number of samples in each bin which value type is int32. If value is bigger, it will consume more performance. Default: 1.
        trans_std (float): Coefficient of offset which value type is float32. It controls weight of offset. Default: 0.1.
        position_sensitive (bool): Whether to choose deformable psroi pooling mode or not, and value type is bool(True or False). If value is False, input dimension equals to output dimension. \
                                   If value is True, input dimension shoule be output dimension * pooled_height * pooled_width. Default: False.
        name (str|None): Name of layer. Default: None.
    Returns:
        Variable: Output of deformable roi pooling is that, if position sensitive is False, input dimension equals to output dimension. If position sensitive is True,\
                  input dimension should be the result of output dimension divided by pooled height and pooled width.

    Examples:
      .. code-block:: python

        # position_sensitive=True
        import paddle.fluid as fluid
        input = fluid.layers.data(name="input",
                                  shape=[2, 192, 64, 64], 
                                  dtype='float32', 
                                  append_batch_size=False)                   
        rois = fluid.layers.data(name="rois",
                                 shape=[4],
                                 dtype='float32', 
                                 lod_level=1)
        trans = fluid.layers.data(name="trans",
                                  shape=[2, 384, 64, 64], 
                                  dtype='float32', 
                                  append_batch_size=False) 
        x = fluid.layers.nn.deformable_roi_pooling(input=input, 
                                                     rois=rois, 
                                                     trans=trans, 
                                                     no_trans=False,
                                                     spatial_scale=1.0, 
                                                     group_size=(1, 1),
                                                     pooled_height=8,
                                                     pooled_width=8,
                                                     part_size=(8, 8),
                                                     sample_per_part=4, 
                                                     trans_std=0.1,
                                                     position_sensitive=True)
  
        # position_sensitive=False
        import paddle.fluid as fluid
        input = fluid.layers.data(name="input",
                                  shape=[2, 192, 64, 64], 
                                  dtype='float32', 
                                  append_batch_size=False)                   
        rois = fluid.layers.data(name="rois",
                                 shape=[4],
                                 dtype='float32', 
                                 lod_level=1)
        trans = fluid.layers.data(name="trans",
                                  shape=[2, 384, 64, 64], 
                                  dtype='float32', 
                                  append_batch_size=False) 
        x = fluid.layers.nn.deformable_roi_pooling(input=input, 
                                                     rois=rois, 
                                                     trans=trans, 
                                                     no_trans=False,
                                                     spatial_scale=1.0, 
                                                     group_size=(1, 1),
                                                     pooled_height=8,
                                                     pooled_width=8,
                                                     part_size=(8, 8),
                                                     sample_per_part=4, 
                                                     trans_std=0.1,
                                                     position_sensitive=False)
    """

    input_channels = input.shape[1]
    if position_sensitive == False:
        output_channels = input_channels
    else:
        output_channels = input_channels / pooled_height / pooled_width

    if part_size is None:
        part_height = pooled_height
        part_width = pooled_width
        part_size = [part_height, part_width]
    part_size = utils.convert_to_list(part_size, 2, 'part_size')
    group_size = utils.convert_to_list(group_size, 2, 'group_size')
    helper = LayerHelper('deformable_psroi_pooling', **locals())
    dtype = helper.input_dtype()
    output = helper.create_variable_for_type_inference(dtype)
    top_count = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type="deformable_psroi_pooling",
        inputs={"Input": input,
                "ROIs": rois,
                "Trans": trans},
        outputs={"Output": output,
                 "TopCount": top_count},
        attrs={
            "no_trans": no_trans,
            "spatial_scale": spatial_scale,
            "output_dim": output_channels,
            "group_size": group_size,
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "part_size": part_size,
            "sample_per_part": sample_per_part,
            "trans_std": trans_std
        })
    return output


def shard_index(input, index_num, nshards, shard_id, ignore_value=-1):
    """
    This function recomputes the `input` indices according to the offset of the
    shard. The length of the indices is evenly divided into N shards, and if
    the `shard_id` matches the shard with the input index inside, the index is
    recomputed on the basis of the shard offset, elsewise it is set to
    `ignore_value`. The detail is as follows:
    :: 
        
        shard_size = (index_num + nshards - 1) // nshards
        y = x % shard_size if x // shard_size == shard_id else ignore_value

    NOTE: If the length of indices cannot be evely divided by the shard number,
    the size of the last shard will be less than the calculated `shard_size`

    Examples:
    ::
    
        Input:
          X.shape = [4, 1]
          X.data = [[1], [6], [12], [19]]
          index_num = 20
          nshards = 2
          ignore_value = -1
        
        if shard_id == 0, we get:
          Out.shape = [4, 1]
          Out.data = [[1], [6], [-1], [-1]]
        
        if shard_id == 1, we get:
          Out.shape = [4, 1]
          Out.data = [[-1], [-1], [2], [9]]
    
    Args:
        - **input** (Variable): Input indices, last dimension must be 1.
        - **index_num** (scalar): An interger defining the range of the index.
        - **nshards** (scalar): The number of shards
        - **shard_id** (scalar): The index of the current shard
        - **ignore_value** (scalar): An ingeter value out of sharded index range

    Returns:
        Variable: The sharded index of input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            shard_label = fluid.layers.shard_index(input=label,
                                                   index_num=20,
                                                   nshards=2,
                                                   shard_id=0)
    """
    op_type = 'shard_index'
    helper = LayerHelper(op_type, **locals())
    if index_num % nshards != 0:
        raise ValueError(
            'The index_num(%d) cannot be evenly divided by nshards(%d)' %
            (index_num, nshards))
    if shard_id < 0 or shard_id >= nshards:
        raise ValueError('The shard_id(%d) should be in [0, %d)' %
                         (shard_id, nshards))

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=op_type,
        inputs={'X': [input]},
        outputs={'Out': out},
        attrs={
            'index_num': index_num,
            'nshards': nshards,
            'shard_id': shard_id,
            'ignore_value': ignore_value
        },
        stop_gradient=True)
    return out


@templatedoc()
def hard_swish(x, threshold=6.0, scale=6.0, offset=3.0, name=None):
    """
    ${comment}
    Args:
        x(Varaible): Input of HardSwish operator.
        threshold(float): The threshold parameter of HardSwish operator. Default:threshold=6.0
        scale(float): The scale parameter of HardSwish operator. Default:scale=6.0
        offset(float): The offset parameter of HardSwish operator. Default:offset=3.0
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The output tensor with the same shape as input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
            y = fluid.layers.hard_swish(x)
    """
    helper = LayerHelper('hard_swish', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='hard_swish',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'threshold': threshold,
               'scale': scale,
               'offset': offset})
    return out


def mse_loss(input, label):
    """
    **Mean square error layer**

    This layer accepts input predications and target label and returns the mean square error.

    The loss can be described as:

    .. math::
        
        Out = mean((X - Y)^2)

    In the above equation:

        * :math:`X`: Input predications, a tensor.
        * :math:`Y`: Input labels, a tensor.
        * :math:`Out`: Output value, same shape with :math:`X`.

    Args:
        input (Variable): Input tensor, has predictions.
        label (Variable): Label tensor, has target labels.

    Returns:
        Variable: The tensor variable storing the mean square error difference of input and label.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            y_predict = fluid.layers.data(name='y_predict', shape=[1], dtype='float32')
            mse = fluid.layers.mse_loss(input=y_predict, label=y)

    """
    return reduce_mean(square_error_cost(input, label))
