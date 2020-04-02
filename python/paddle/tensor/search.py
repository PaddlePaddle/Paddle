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

# TODO: define searching & indexing functions of a tensor  
# __all__ = ['argmax',
#            'argmin',
#            'argsort',
#            'has_inf',
#            'has_nan',
#            'masked_select',
#            'topk',
#            'where',
#            'index_select',
#            'nonzero',
#            'sort']
from __future__ import print_function

from ..fluid.layer_helper import LayerHelper
from ..fluid.layers import cast, where, slice

__all__ = ['index_select', 'nonzero']


def index_select(input, index, dim=0):
    """
    Returns a new tensor which indexes the `input` tensor
    along dimension `dim` using the entries in `index` which
    is a Tensor.

    The returned tensor has the same number of dimensions
    as the original `input` tensor. The dim-th dimension
    has the same size as the length of `index`; other dimensions
    have the same size as in the `input` tensor. 

        .. code-block:: text
            Given:
                input.shape = [3, 4]
                input.data = [[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0],
                              [9.0, 10.0, 11.0, 12.0]]
                index.shape = [3]
                index.data = [0, 1, 1]
            
            Output when dim = 1:
                out.shape = [3, 3]
                out.data = [[1.0, 2.0, 2.0],
                            [5.0, 6.0, 6.0],
                            [9.0, 10.0, 10.0]]
            
            Output when dim = 0:
                out.shape = [3, 4]
                out.data = [[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [5.0, 6.0, 7.0, 8.0]]

    Args:
        input (Variable): The input tensor variable.
        index (Variable): The 1-D tensor containing the indices to index.
        dim (int): The dimension in which we index.

    Returns:
        A `Tensor` or `LoDTensor`. The data type is same as `input`.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            x = fluid.data(name="data", shape=[None, 32, 32], dtype="float32")
            index = fluid.data(name="index", shape=[5], dtype="int32")
            out = paddle.index_select(x, index, dim=1)
    """
    helper = LayerHelper("index_select", **locals())
    out = helper.create_variable_for_type_inference(input.dtype)
    helper.append_op(
        type='index_select',
        inputs={'X': input,
                'Index': index},
        outputs={'Out': out},
        attrs={'dim': dim})
    return out


def nonzero(input, as_tuple=False):
    """
    Return a tensor containing the indices of all non-zero
    elements of the `input` tensor. If as_tuple is True, return
    a tuple of 1-D tensors, one for each dimension in `input`,
    each containing the indices (in that dimension) of all non-zero
    elements of `input`.

    Given a n-Dimensional `input` tensor with shape [x_1, x_2, ..., x_n],
    If as_tuple is False, we can get a output tensor with shape [z, n],
    where `z` is the number of all non-zero elements in the `input` tensor.
    If as_tuple is True, we can get a 1-D tensor tuple of length `n`, and
    and the shape of each 1-D tensor is [z, 1].

        .. code-block:: text
            Given:
                inputs.shape = [3, 3]
                inputs.data = [[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]]
            
            Output when as_tuple = False:
                out.shape = [3, 2]
                out.data = [[0, 0], [1, 1], [2, 2]]
            
            Output when as_tupe = True:
                out1.shape = [3, 1]
                out1.data = [[0],
                            [1],
                            [2]]
                out2.shape = [3, 1]
                out2.data = [[0],
                            [1],
                            [2]]
                out = tuple(out1, out2)

    Args:
        inputs (Variable): The input tensor variable.
        as_tuple (bool): Return type, Tensor or tuple of Tensor.

    Returns:
        Variable. The data type is int64.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            x = fluid.data(name="data", shape=[None, 32, 32], dtype="float32")
            out = paddle.nonzero(x)
    """
    cast_inputs = cast(input, 'bool')
    outs = where(cast_inputs)
    if as_tuple:
        list_out = []
        shape = input.shape
        rank = len(shape)
        if rank == 1:
            list_out.append(outs)
        else:
            for i in range(rank):
                list_out.append(
                    slice(
                        outs, axes=[rank - 1], starts=[i], ends=[i + 1]))
        return tuple(list_out)
    else:
        return outs
