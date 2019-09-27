#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from .framework import Variable, in_dygraph_mode
from .layer_helper import LayerHelper

__all__ = ['one_hot', 'embedding']


def one_hot(input, depth, allow_out_of_range=False):
    """
    This layer creates the one-hot representations for input indices.

    Args:
        input(Variable): Input indices represent locations, which takes value 1.0
            in indices, while all other locations take value 0.
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
            one_hot_label = fluid.one_hot(input=label, depth=10)
    """
    helper = LayerHelper("one_hot_v2", **locals())

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
        type="one_hot_v2",
        inputs=inputs,
        attrs=attrs,
        outputs={'Out': one_hot_out},
        stop_gradient=True)
    return one_hot_out


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
          # [batch_size, 20]  ->  [batch_size, 20, 64]
          data = fluid.layers.data(name='sequence', shape=[20], dtype='int64')
          emb = fluid.embedding(input=data, size=[128, 64])    
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
        type='lookup_table_v2',
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
