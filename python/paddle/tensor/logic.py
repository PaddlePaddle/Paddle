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

from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_type
from ..fluid.layers.layer_function_generator import templatedoc

# TODO: define logic functions of a tensor  
from ..fluid.layers import greater_equal  #DEFINE_ALIAS
from ..fluid.layers import greater_than  #DEFINE_ALIAS
from ..fluid.layers import is_empty  #DEFINE_ALIAS
from ..fluid.layers import isfinite  #DEFINE_ALIAS
from ..fluid.layers import less_equal  #DEFINE_ALIAS
from ..fluid.layers import less_than  #DEFINE_ALIAS
from ..fluid.layers import logical_and  #DEFINE_ALIAS
from ..fluid.layers import logical_not  #DEFINE_ALIAS
from ..fluid.layers import logical_or  #DEFINE_ALIAS
from ..fluid.layers import logical_xor  #DEFINE_ALIAS
from ..fluid.layers import not_equal  #DEFINE_ALIAS
from ..fluid.layers import reduce_all  #DEFINE_ALIAS
from ..fluid.layers import reduce_any  #DEFINE_ALIAS

__all__ = [
    'equal',
    'greater_equal',
    'greater_than',
    'is_empty',
    'isfinite',
    'less_equal',
    'less_than',
    'logical_and',
    'logical_not',
    'logical_or',
    'logical_xor',
    'not_equal',
    'reduce_all',
    'reduce_any',
    'allclose',
    'elementwise_equal',
    #       'isnan'
]


def equal(x, y, axis=-1, name=None):
    """
	:alias_main: paddle.equal
	:alias: paddle.equal,paddle.tensor.equal,paddle.tensor.logic.equal

    This OP returns the truth value of :math:`x == y`. True if two inputs have the same elements, False otherwise.

    **NOTICE**: The output of this OP has no gradient, and this OP supports broadcasting by :attr:`axis`.

    Args:
        x(Variable): Tensor, data type is float32, float64, int32, int64.
        y(Variable): Tensor, data type is float32, float64, int32, int64.
        axis(int32, optional): If X.dimension != Y.dimension, Y.dimension
            must be a subsequence of x.dimension. And axis is the start 
            dimension index for broadcasting Y onto X. For more detail, 
            please refer to OP:`elementwise_add`.
        name(str, optional): Normally there is no need for user to set this property. 
            For more information, please refer to :ref:`api_guide_Name`.Default: None.

    Returns:
        Variable: output Tensor, data type is bool, value is [False] or [True].

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle
          import numpy as np

          label = fluid.layers.assign(np.array([3, 4], dtype="int32"))
          label_1 = fluid.layers.assign(np.array([1, 2], dtype="int32"))
          limit = fluid.layers.assign(np.array([3, 4], dtype="int32"))
          out1 = paddle.equal(x=label, y=limit) #out1=[True]
          out2 = paddle.equal(x=label_1, y=limit) #out2=[False]

        .. code-block:: python

          import paddle.fluid as fluid
          import paddle
          import numpy as np

          def gen_data():
              return {
                    "x": np.ones((2, 3, 4, 5)).astype('float32'),
                    "y": np.zeros((3, 4)).astype('float32')
                }

          x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
          y = fluid.data(name="y", shape=[3,4], dtype='float32')
          out = paddle.equal(x, y, axis=1)
          place = fluid.CPUPlace()
          exe = fluid.Executor(place)

          res = exe.run(feed=gen_data(),
                            fetch_list=[out])
          print(res[0]) #[False]
    """
    helper = LayerHelper("equal_reduce", **locals())
    out = helper.create_variable_for_type_inference(dtype='bool')
    attrs = {}
    attrs['axis'] = axis
    helper.append_op(
        type='equal_reduce',
        inputs={'X': [x],
                'Y': [y]},
        attrs=attrs,
        outputs={'Out': [out]})
    return out


@templatedoc()
def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):
    """
	:alias_main: paddle.allclose
	:alias: paddle.allclose,paddle.tensor.allclose,paddle.tensor.logic.allclose

    ${comment}

    Args:
        input(inputtype):{input_comment}.
        other(othertype):{other_comment}.
        rtol(rtoltype,optional):{rtol_comment}.
        atol(atoltype,optional):{atol_comment}.
        equal_nan(equalnantype,optional):{equal_nan_comment}.
        name(STR, optional): The default value is None.
                        Normally there is no need for user to set this property.
                        For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        ${out_comment}.

    Return Type:
        ${out_type}
        
    Examples:
        .. code-block:: python

          import paddle
          import paddle.fluid as fluid
          import numpy as np

          use_cuda = fluid.core.is_compiled_with_cuda()

          a = fluid.data(name="a", shape=[2], dtype='float32')
          b = fluid.data(name="b", shape=[2], dtype='float32')

          result = paddle.allclose(a, b, rtol=1e-05, atol=1e-08,
                                  equal_nan=False, name="ignore_nan")
          result_nan = paddle.allclose(a, b, rtol=1e-05, atol=1e-08,
                                      equal_nan=True, name="equal_nan")

          place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
          exe = fluid.Executor(place)
          exe.run(fluid.default_startup_program())

          x = np.array([10000., 1e-07]).astype("float32")
          y = np.array([10000.1, 1e-08]).astype("float32")
          result_v, result_nan_v = exe.run(
              feed={'a': x, 'b': y},
              fetch_list=[result, result_nan])
          print(result_v, result_nan_v)
          # Output: (array([False]), array([False]))

          x = np.array([10000., 1e-08]).astype("float32")
          y = np.array([10000.1, 1e-09]).astype("float32")
          result_v, result_nan_v = exe.run(
              feed={'a': x, 'b': y},
              fetch_list=[result, result_nan])
          print(result_v, result_nan_v)
          # Output: (array([ True]), array([ True]))

          x = np.array([1.0, float('nan')]).astype("float32")
          y = np.array([1.0, float('nan')]).astype("float32")
          result_v, result_nan_v = exe.run(
              feed={'a': x, 'b': y},
              fetch_list=[result, result_nan])
          print(result_v, result_nan_v)
          # Output: (array([False]), array([ True]))
    """

    check_type(rtol, 'rtol', float, 'allclose')
    check_type(atol, 'atol', float, 'allclose')
    check_type(equal_nan, 'equal_nan', bool, 'allclose')

    helper = LayerHelper("allclose", **locals())
    out = helper.create_variable_for_type_inference(dtype='bool')

    inputs = {'Input': input, 'Other': other}
    outputs = {'Out': out}
    attrs = {'rtol': rtol, 'atol': atol, 'equal_nan': equal_nan}
    helper.append_op(
        type='allclose', inputs=inputs, outputs=outputs, attrs=attrs)

    return out


def elementwise_equal(x, y, name=None):
    """
	:alias_main: paddle.elementwise_equal
	:alias: paddle.elementwise_equal,paddle.tensor.elementwise_equal,paddle.tensor.logic.elementwise_equal

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
