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

# TODO: define logic functions of a tensor  
__all__ = [#'equal',
           # 'greater_equal',
           # 'greater_than',
           # 'is_empty',
           # 'isfinite',
           # 'less_equal',
           # 'less_than',
           # 'logical_and',
           # 'logical_not',
           # 'logical_or',
           # 'logical_xor',
           # 'not_equal',
           # 'reduce_all',
           # 'reduce_any',
           # 'allclose',
           'elementwise_equal',
           # 'isnan'
          ]
from paddle.common_ops_import import *


def elementwise_equal(x, y, name=None):
    """
    This layer returns the truth value of :math:`x == y` elementwise.

    Args:
        x(Variable): Tensor, data type is float32, float64, int32, int64.
        y(Variable): Tensor, data type is float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Variable: output Tensor, it's shape is the same as the input's Tensor,
        and the data type is bool. The result of this op is stop_gradient. 

    Examples:
        .. code-block:: python

          import paddle
          import paddle.fluid as fluid
          import numpy as np
          label = fluid.layers.assign(np.array([3, 3], dtype="int32"))
          limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
          out1 = paddle.elementwise_equal(x=label, y=limit) #out1=[True, False]
    """
    helper = LayerHelper("elementwise_equal", **locals())
    out = helper.create_variable_for_type_inference(dtype='bool')
    out.stop_gradient = True

    helper.append_op(
        type='equal',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [out]},
        attrs={'force_cpu': False})
    return out
