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

# TODO: define functions of linear algebra   
# __all__ = ['matmul', 
#            'dot',
#            'einsum',
#            'morm',
#            'transpose',
#            'dist',
#            't',
#            'cross',
#            'cholesky',
#            'tensordot']
from __future__ import print_function

from ..fluid.layer_helper import LayerHelper

__all__ = ['cross']


def cross(input, other, dim=None):
    """
    Returns the cross product of vectors in dimension `dim` of the
    `input` and `other` tensor. Inputs must have the same shape, 
    and the size of their dim-th dimension should be equla to 3.
    If `dim` is not given, it defaults to the first dimension found
    with the size 3.

        .. code-block:: text
            Given:
                input.shape = [3, 3]
                input.data = [[1.0, 1.0, 1.0],
                              [2.0, 2.0, 2.0],
                              [3.0, 3.0, 3.0]]

                other.shape = [3, 3]
                other.data = [[1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0]]
            
            Output when dim = 0:
                out.shape = [3, 3]
                out.data = [[-1.0, -1.0, -1.0], 
                            [2.0, 2.0, 2.0],
                            [-1.0, -1.0, -1.0]]
            
            Output when dim = 1:
                out.shape = [3, 3]
                out.data = [[0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0]]

    Args:
        input (Variable): The first input tensor variable.
        other (Variable): The second input tensor variable.
        dim (int): The dimension to take the cross-product in.

    Returns:
        A `Tensor` or `LoDTensor`. The shape and data type are same as `input`.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            x = fluid.data(name="data_x", shape=[None, 32, 3], dtype="float32")
            y = fluid.data(name="data_y", shape=[None, 32, 3], dtype="float32")
            out = paddle.cross(x, y, dim=2)
    """
    helper = LayerHelper("cross", **locals())
    out = helper.create_variable_for_type_inference(input.dtype)
    attrs = dict()
    if dim:
        attrs['dim'] = dim
    helper.append_op(
        type='cross',
        inputs={'X': input,
                'Y': other},
        outputs={'Out': out},
        attrs=attrs)
    return out
