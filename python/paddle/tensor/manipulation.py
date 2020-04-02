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

# TODO: define functions to manipulate a tensor  
# __all__ = ['cast',
#            'concat',
#            'expand',
#            'expand_as',
#            'flatten',
#            'gather',
#            'gather_nd',
#            'reshape',
#            'reverse',
#            'scatter',
#            'scatter_nd_add',
#            'scatter_nd',
#            'shard_index',
#            'slice',
#            'split',
#            'squeeze',
#            'stack',
#            'strided_slice',
#            'transpose',
#            'unique',
#            'unique_with_counts',
#            'unsqueeze',
#            'unstack',
#            'flip',
#            'unbind',
#            'roll']
from __future__ import print_function

from ..fluid.layer_helper import LayerHelper
from ..fluid.layers import reshape

__all__ = ['roll']


def roll(input, shifts, dims=None):
    """
    Roll the `input` tensor along the given dimension(s).
    Elements that are shifted beyond the last position
    are re-introduced at the first position. If a dimension
    is not specified, the tensor will be flattened before 
    rolling and then restored to the original shape.

        .. code-block:: text
            Given:
                input.shape = [3, 3]
                input.data = [[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]]
            
            Output when shifts = 1, dims = None
                out.shape = [3, 3]
                out.data = [[9.0, 1.0, 2.0],
                            [3.0, 4.0, 5.0],
                            [6.0, 7.0, 8.0]]
            
            Output when shifts = [1], dims = [0]
                out.shape = [3, 3]
                out.data = [[7.0, 8.0, 9.0],
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]]

    Args:
        input (Variable): The input tensor variable.
        shifts (int|list|tuple): The number of places by which the elements
                           of the `input` tensor are shifted.
        dims (int|list|tuple|None): Dimentions along which to roll.

    Returns:
        A `Tensor` or `LoDTensor`. The data type is same as `input`.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            x = fluid.data(name="data_x", shape=[None, 32, 3], dtype="float32")
            out = paddle.roll(x, shifts=[1], dims=[0])
    """
    helper = LayerHelper("roll", **locals())
    out = helper.create_variable_for_type_inference(input.dtype)

    origin_shape = input.shape
    if type(shifts) == int:
        shifts = [shifts]
    if type(dims) == int:
        dims = [dims]
    if dims is None:
        input = reshape(input, shape=[-1, 1])
        dims = [0]

    helper.append_op(
        type='roll',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={'dims': dims,
               'shifts': shifts})
    out = reshape(out, shape=origin_shape, inplace=True)
    return out
