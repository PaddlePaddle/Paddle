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

__all__ = ['fused_elemwise_activation', 'fused_embedding_seq_pool']


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


def fused_embedding_seq_pool(input,
                             size,
                             is_sparse=False,
                             padding_idx=None,
                             combiner='sum',
                             param_attr=None,
                             dtype='float32'):
    """
    **Embedding Sequence pool**

    This layer is used to lookup and pool embeddings of IDs, provided by :attr:`input`, in
    a lookup table. The result of this lookup is the embedding of each ID in the :attr:`input`.
    The combiner type is mentioned in the Args.
    
    Args:
        input (Variable): Input is a Tensor<int64> Variable, which contains the IDs information.
            The value of the input IDs should satisfy :math:`0<= id < size[0]`.
        size (tuple|list): The shape of the look up table parameter. It should
            have two elements which indicate the size of the dictionary of
            embeddings and the size of each embedding vector respectively.
        is_sparse (bool): The flag indicating whether to use sparse update.
            Default: False.
        padding_idx (int|long|None): It will output all-zero padding data whenever
            lookup encounters :math:`padding\_idx` in Ids. If set :attr:`None`, it makes
            no effect to output. If :math:`padding\_idx < 0`, the :math:`padding\_idx`
            will automatically be converted to :math:`size[0] + padding\_idx` to use.
            Default: None.
        combiner (str): The pooling type of sequence_pool, and only support `sum`.
            Default: sum.
        param_attr (ParamAttr): Parameters for this layer.
        dtype (np.dtype|core.VarDesc.VarType|str): The dtype refers to the data type of output
            tensor. It can be float32, float_16, int etc.
    Returns:
        The sequence pooling variable which is a Tensor.
    Examples:
        .. code-block:: python
            import numpy as np
            import paddle.fluid as fluid

            dict_size = 20
            data_t = fluid.layers.data(name='word', shape=[1], dtype='int64', lod_level=1)
            padding_idx = np.random.randint(1, 10)
            out = fluid.contrib.fused_embedding_seq_pool(
                input=data_t,
                size=[dict_size, 32],
                param_attr='w',
                padding_idx=padding_idx,
                is_sparse=False)
    """
    helper = LayerHelper('fused_embedding_seq_pool', **locals())
    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=dtype, is_bias=False)
    out = helper.create_variable_for_type_inference(dtype)
    padding_idx = -1 if padding_idx is None else padding_idx if padding_idx >= 0 else (
        size[0] + padding_idx)
    helper.append_op(
        type='fused_embedding_seq_pool',
        inputs={'Ids': input,
                'W': w},
        outputs={'Out': out},
        attrs={
            'is_sparse': is_sparse,
            'combiner': combiner,
            'padding_idx': padding_idx
        })
    return out
