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

from paddle.common_ops_import import *
import paddle.fluid as fluid

# TODO: define logic functions of a tensor  
#__all__ = ['equal',
#            'greater_equal',
#            'greater_than',
#            'is_empty',
#            'isfinite',
#            'less_equal',
#            'less_than',
#            'logical_and',
#            'logical_not',
#            'logical_or',
#            'logical_xor',
#            'not_equal',
#            'reduce_all',
#            'reduce_any',
#            'allclose',
#            'elementwise_equal',
#            'isnan']


def equal(x, y, axis=-1, cond=None):
    """
    This layer returns the truth value of :math:`x == y` elementwise.

    Args:
        x(Variable): Tensor, data type is float32, float64, int32, int64.
        y(Variable): Tensor, data type is float32, float64, int32, int64.
		axis(int32, optional): If X.dimension != Y.dimension, Y.dimension
			must be a subsequence of x.dimension. And axis is the start 
			dimension index for broadcasting Y onto X.
        cond(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of *equal*.
            if cond is None, a new Varibale will be created to store the result.

    Returns:
        Variable: output Tensor, data type is bool, value is [False] or [True].

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.tensor as tensor
          import numpy as np

          out_cond =fluid.data(name="input1", shape=[1], dtype='bool')
          label = fluid.layers.assign(np.array([3, 4], dtype="int32"))
		  label_1 = fluid.layers.assign(np.array([1, 2], dtype="int32"))
          limit = fluid.layers.assign(np.array([3, 4], dtype="int32"))
          out1 = tensor.equal(x=label, y=limit) #out1=[True]
          out2 = tensor.equal(x=label_1, y=limit, out=out_cond) #out2=[False] out_cond=[False]

        .. code-block:: python

          import paddle.tensor as tensor
          import paddle.fluid as fluid
          import numpy as np

          def gen_data():
              return {
                    "x": np.ones((2, 3, 4, 5)).astype('float32'),
                    "y": np.zeros((3, 4)).astype('float32')
                }

          x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
          y = fluid.data(name="y", shape=[3,4], dtype='float32')
          out = tensor.equal(x, y, axis=1)
          print(out) # [False]
    """
    helper = LayerHelper("equal_reduce", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True
    attrs = {}
    attrs['axis'] = axis
    helper.append_op(
        type='equal_reduce',
        inputs={'X': [x],
                'Y': [y]},
        attrs=attrs,
        outputs={'Out': [cond]})
    return cond
