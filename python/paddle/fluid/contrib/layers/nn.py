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
"""
Contrib layers just related to the neural network.
"""

from __future__ import print_function

import numpy as np
import six
import os
import inspect
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers import utils

__all__ = [
    'fused_elemwise_activation',
    'sequence_topk_avg_pooling',
    'var_conv_2d',
    'match_matrix_tensor',
    'tree_conv',
]


def fused_elemwise_activation(x,
                              y,
                              functor_list,
                              axis=-1,
                              scale=0.0,
                              save_intermediate_out=True):
    """
    **Fused elementwise_add/mul and activation layers**

    This function computes an elementwise_add/mul cooperated with an activation.

    .. math::

        out = Unary(Binary(x, y))

    or

    .. math::

        out = Binary(x, Unary(y))

    Unary operators can be: `scale`, `relu`, `tanh`. Binary operators can be:
    `elementwise_add`, `elementwise_mul`.

    Args:
        x (Variable): left operation of the binary operator.
        y (Variable): right operator of the binary operator.
        functor_list (list of str): types of operator which will be executed
            by this layer. For example, ['elementwise_add', 'relu']
            (out = elementwise_add(x, relu(y))),
            or ['relu', 'elemmentwise_add'] (out = relu(elementwise_add(x, y))).
        axis (int32, default -1): axis of elementwise op.
        scale (float32, default 0): parameter of scale op.
        save_intermediate_out (bool, default True): whether to save the
            intermediate result, Unary(y) or Binary(x, y).

    Returns:
        Variable: The computation result.
    """
    if isinstance(functor_list, str):
        functor_list = functor_list.split(',')

    if not isinstance(functor_list, list) or len(functor_list) != 2:
        raise ValueError(
            'functor_list should be a list of str, and the length should be 2.')

    helper = LayerHelper('fused_elemwise_activation', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    intermediate_out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='fused_elemwise_activation',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out,
                 'IntermediateOut': intermediate_out},
        attrs={
            'axis': axis,
            'scale': scale,
            'save_intermediate_out': save_intermediate_out,
            'functor_list': functor_list
        })
    return out


def var_conv_2d(input,
                row,
                col,
                input_channel,
                output_channel,
                filter_size,
                stride=1,
                param_attr=None,
                act=None,
                dtype='float32',
                name=None):
    """
    The var_conv_2d layer calculates the output base on the :attr:`input` with variable length,
    row, col, input channel, filter size and strides. Both :attr:`input`, :attr:`row`,
    and :attr:`col` are 1-level LodTensor. The covolution operation is same as conv2d layer with 
    padding. Besides, input.dims[1] should be 1. 

    .. code-block:: text
            
            If input_channel is 2 and given row lodTensor and col lodTensor as follows:
                row.lod = [[5, 4]]
                col.lod = [[6, 7]]
            input is a lodTensor: 
                input.lod = [[60, 56]]	# where 60 = input_channel * 5 * 6
                input.dims = [116, 1]	# where 116 = 60 + 56
            
            If set output_channel is 3, filter_size is [3, 3], stride is [1, 1]:
                output.lod = [[90, 84]] # where 90 = output_channel * [(5-1)/stride + 1] * [(6-1)/stride + 1]
                output.dims = [174, 1]  # where 174 = 90 + 84

    Args:
        input (Variable): The input shoud be 1-level LodTensor with dims[1] equals 1.
        row (Variable): The row shoud be 1-level LodTensor to provide height information.
        col (Variable): The col shoud be 1-level LodTensor to provide width information.
        input_channel (int): The number of input channel.
        output_channel (int): The number of output channel.
        filter_size (int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        stride (int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of var_conv2d. If it is set to None or one attribute of ParamAttr, var_conv2d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None
        dtype ('float32'): The data type of parameter and output.
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None

    Returns:
        Variable: Output variable with LoD specified by this layer.

    Examples:
        .. code-block:: python

            import numpy as np
            from paddle.fluid import layers
            from paddle.fluid import contrib

            x_lod_tensor = layers.data(name='x', shape=[1], lod_level=1)
            row_lod_tensor = layers.data(name='row', shape=[6], lod_level=1)
            col_lod_tensor = layers.data(name='col', shape=[6], lod_level=1)
            out = contrib.var_conv_2d(input=x_lod_tensor, 
                                     row=row_lod_tensor,
                                     col=col_lod_tensor,
                                     input_channel=3,
                                     output_channel=5,
                                     filter_size=[3, 3],
                                     stride=1)
    """
    helper = LayerHelper('var_conv_2d', **locals())
    x_shape = list(input.shape)
    assert len(x_shape) == 2

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')

    filter_shape = [
        int(output_channel),
        int(input_channel) * filter_size[0] * filter_size[1]
    ]
    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype, )

    conv_res = helper.create_variable_for_type_inference(dtype)
    tmp_res = helper.create_variable_for_type_inference(
        dtype, stop_gradient=True)

    helper.append_op(
        type='var_conv_2d',
        inputs={
            'X': input,
            'ROW': row,
            'COLUMN': col,
            'W': filter_param,
        },
        outputs={"Out": conv_res,
                 "Col": tmp_res},
        attrs={
            'InputChannel': input_channel,
            'OutputChannel': output_channel,
            'StrideH': stride[0],
            'StrideW': stride[1],
            'KernelH': filter_size[0],
            'KernelW': filter_size[1],
        })

    return helper.append_activation(conv_res)


def match_matrix_tensor(x,
                        y,
                        channel_num,
                        act=None,
                        param_attr=None,
                        dtype='float32',
                        name=None):
    """
    Calculate the semantic matching matrix of two word sequences with variable length.
    Given a query A of length `n` and a title B of length `m`, the input shape are respectively
    [n, h] and [m, h], which h is hidden_size. If :attr:`channel_num` is set to 3,
    it will generate a learnable parameter matrix W with shape [h, 3, h].
    Then the semantic matching matrix of query A and title B is calculated by 
    A * W * B.T = [n, h]*[h, 3, h]*[h, m] = [n, 3, m]. The learnable parameter matrix `W` 
    is equivalent to a fully connected layer in the calculation process. If :attr:`act` is provided, 
    the corresponding activation function will be applied to output matrix.
    The :attr:`x` and :attr:`y` should be LodTensor and only one level LoD is supported.

    .. code-block:: text

            Given a 1-level LoDTensor x:
                x.lod =  [[2,                     3,                               ]]
                x.data = [[0.3, 0.1], [0.2, 0.3], [0.5, 0.6], [0.7, 0.1], [0.3, 0.4]]
                x.dims = [5, 2]
            y is a Tensor:
                y.lod =  [[3,                                 1,       ]]
                y.data = [[0.1, 0.2], [0.3, 0.7], [0.9, 0.2], [0.4, 0.1]]
                y.dims = [4, 2]
            set channel_num 2, then we get a 1-level LoDTensor:
                out.lod =  [[12, 6]]   # where 12 = channel_num * x.lod[0][0] * y.lod[0][0]
                out.dims = [18, 1]     # where 18 = 12 + 6

    Args:
        x (Variable): Input variable x which should be 1-level LodTensor.
        y (Variable): Input variable y which should be 1-level LodTensor.
        channel_num (int): The channel number of learnable parameter W.
        act (str, default None): Activation to be applied to the output of this layer.
        param_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for learnable
            parameters/weights of this layer.
        dtype ('float32'): The data type of w data.
        name (str|None): A name for this layer(optional). If set None, the layer will be named automatically. Default: None

    Returns:
        Variable: output with LoD specified by this layer.

    Examples:
        .. code-block:: python

            import numpy as np
            from paddle.fluid import layers
            from paddle.fluid import contrib

            x_lod_tensor = layers.data(name='x', shape=[10], lod_level=1)
            y_lod_tensor = layers.data(name='y', shape=[10], lod_level=1)
            out, out_tmp = contrib.match_matrix_tensor(x=x_lod_tensor, y=y_lod_tensor, channel_num=3)
    """
    helper = LayerHelper('match_matrix_tensor', **locals())

    x_shape = list(x.shape)
    y_shape = list(y.shape)
    assert len(x_shape) == 2 and len(y_shape) == 2 and x_shape[-1] == y_shape[
        -1]

    weight_shape = [x_shape[-1], channel_num, y_shape[-1]]
    w = helper.create_parameter(
        attr=helper.param_attr, shape=weight_shape, dtype=dtype, is_bias=False)
    mm_res = helper.create_variable_for_type_inference(dtype)
    tmp_res = helper.create_variable_for_type_inference(
        dtype, stop_gradient=True)
    helper.append_op(
        type='match_matrix_tensor',
        inputs={
            'X': x,
            'Y': y,
            'W': w,
        },
        outputs={"Out": mm_res,
                 "Tmp": tmp_res},
        attrs={'dim_t': channel_num})

    return helper.append_activation(mm_res), tmp_res


def sequence_topk_avg_pooling(input, row, col, topks, channel_num):
    """
    The :attr:`topks` is a list with incremental values in this function. For each topk,
    it will average the topk features as an output feature for each channel of every 
    input sequence. Both :attr:`row` and :attr:`col` are LodTensor, which provide height 
    and width information for :attr:`input` tensor. If feature size of input sequence is less 
    than topk, it will padding 0 at the back.

    .. code-block:: text

            If channel_num is 2 and given row LoDTensor and col LoDTensor as follows:
                row.lod = [[5, 4]]
                col.lod = [[6, 7]]

            input is a LoDTensor with input.lod[0][i] = channel_num * row.lod[0][i] * col.lod[0][i] 
                input.lod = [[60, 56]]  # where 60 = channel_num * 5 * 6
                input.dims = [116, 1]   # where 116 = 60 + 56

            If topks is [1, 3, 5], then we get a 1-level LoDTensor:
                out.lod =  [[5, 4]] 	# share Lod info with row LodTensor
                out.dims = [9, 6]   	# where 6 = len(topks) * channel_num

    Args:
        input (Variable): The input should be 2D LodTensor with dims[1] equals 1.
        row (Variable): The row shoud be 1-level LodTensor to provide the height information
                        of the input tensor data.
        col (Variable): The col shoud be 1-level LodTensor to provide the width information
                        of the input tensor data.
        topks (list): A list of incremental value to average the topk feature.
        channel_num (int): The number of input channel.

    Returns:
        Variable: output LodTensor specified by this layer.

    Examples:

        .. code-block:: python

            import numpy as np
            from paddle.fluid import layers
            from paddle.fluid import contrib

            x_lod_tensor = layers.data(name='x', shape=[1], lod_level=1)
            row_lod_tensor = layers.data(name='row', shape=[6], lod_level=1)
            col_lod_tensor = layers.data(name='col', shape=[6], lod_level=1)
            out = contrib.sequence_topk_avg_pooling(input=x_lod_tensor,
                                                   row=row_lod_tensor,
                                                   col=col_lod_tensor,
                                                   topks=[1, 3, 5],
                                                   channel_num=5)
    """
    helper = LayerHelper('sequence_topk_avg_pooling', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    pos = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype(), stop_gradient=True)
    helper.append_op(
        type='sequence_topk_avg_pooling',
        inputs={'X': input,
                'ROW': row,
                'COLUMN': col},
        outputs={'Out': out,
                 'pos': pos},
        attrs={'topks': topks,
               'channel_num': channel_num})

    return out


def tree_conv(nodes_vector,
              edge_set,
              output_size,
              num_filters=1,
              max_depth=2,
              act='tanh',
              param_attr=None,
              bias_attr=None,
              name=None):
    """ 
    ${comment}
    		
    Args:
        nodes_vector(${nodes_vector_type}): ${nodes_vector_comment}
        edge_set(${edge_set_type}): ${edge_set_comment}
        output_size(int): output feature width
        num_filters(int): number of filters, Default 1
        max_depth(int): max depth of filters, Default 2
        act(str): activation function, Default tanh
        param_attr(ParamAttr): the parameter attribute for the filters, Default None
        bias_attr(ParamAttr): the parameter attribute for the bias of this layer, Default None
        name(str): a name of this layer(optional). If set None, the layer will be named automatically, Default None

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          # 10 for max_node_size of dataset, 5 for vector width
          nodes_vector = fluid.layers.data(name='vectors', shape=[10, 5], dtype='float32')
          # 10 for max_node_size of dataset, 2 for every edge has two nodes
          # edges must be directional
          edge_set = fluid.layers.data(name='edge_set', shape=[10, 2], dtype='float32')
          # the shape of output will be [10, 6, 1],
          # 10 for max_node_size of dataset, 6 for output size, 1 for 1 filter
          out_vector = fluid.layers.tree_conv(nodes_vector, edge_set, 6, 1, 2)
          # After reshape, output tensor could be nodes_vector for next tree convolution
          out_vector = fluid.layers.reshape(out_vector, shape=[-1, 10, 6])
          out_vector_2 = fluid.layers.tree_conv(out_vector, edge_set, 3, 4, 2)
          # also output tensor could be pooling(the pooling in paper called global pooling)
          pooled = fluid.layers.reduce_max(out_vector, dim=2) # global pooling
    """
    helper = LayerHelper("tree_conv", **locals())
    dtype = helper.input_dtype('nodes_vector')
    feature_size = nodes_vector.shape[2]
    W_shape = [feature_size, 3, output_size, num_filters]
    W = helper.create_parameter(
        attr=param_attr, shape=W_shape, dtype=dtype, is_bias=False)
    out = helper.create_variable_for_type_inference(dtype=dtype)
    helper.append_op(
        type='tree_conv',
        inputs={'NodesVector': nodes_vector,
                'EdgeSet': edge_set,
                'Filter': W},
        outputs={'Out': out, },
        attrs={'max_depth': max_depth})
    if helper.bias_attr:
        pre_activation = helper.append_bias_op(out)
    else:
        pre_activation = out
    return helper.append_activation(pre_activation)
