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

# TODO: define random functions  

import numpy as np

from ..fluid import core
from ..fluid.framework import device_guard, in_dygraph_mode, _varbase_creator, Variable, convert_np_dtype_to_dtype_
from ..fluid.layers.layer_function_generator import templatedoc
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ..fluid.layers import utils, uniform_random, gaussian_random
from ..fluid.layers.tensor import fill_constant

from ..fluid.io import shuffle  #DEFINE_ALIAS

__all__ = [
    #       'gaussin',
    #       'uniform',
    'shuffle',
    'randn',
    'rand',
    'randint',
    'randperm'
]


def randint(low=0, high=None, shape=[1], dtype=None, name=None):
    """
	:alias_main: paddle.randint
	:alias: paddle.tensor.randint, paddle.tensor.random.randint

    This OP returns a Tensor filled with random integers from a discrete uniform
    distribution in the range [``low``, ``high``), with ``shape`` and ``dtype``.
    If ``high`` is None (the default), the range is [0, ``low``).

    Args:
        low(int): The lower bound on the range of random values to generate.
            The ``low`` is included in the range. If ``high`` is None, the
            range is [0, ``low``). Default is 0.
        high(int, optional): The upper bound on the range of random values to
            generate, the ``high`` is excluded in the range. Default is None
            (see above for behavior if high = None). Default is None.
        shape(list|tuple|Tensor): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64). Default is [1].
        dtype(str|np.dtype|core.VarDesc.VarType, optional): The data type of the
            output tensor. Supported data types: int32, int64. If ``dytpe``
            is None, the data type is int64. Default is None.
        name(str, optional): The default value is None.  Normally there is no
            need for user to set this property.  For more information, please
            refer to :ref:`api_guide_Name`.

    Returns: 
        Tensor: A Tensor filled with random integers from a discrete uniform
        distribution in the range [``low``, ``high``), with ``shape`` and ``dtype``.

    Raises:
        TypeError: If ``shape`` is not list, tuple, Tensor.
        TypeError: If ``dtype`` is not int32, int64.
        ValueError: If ``high`` is not greater then ``low``; If ``high`` is 
            None, and ``low`` is not greater than 0.

    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        # example 1:
        # attr shape is a list which doesn't contain Tensor.
        result_1 = paddle.randint(low=-5, high=5, shape=[3])
        # [0, -3, 2]

        # example 2:
        # attr shape is a list which contains Tensor.
        dim_1 = paddle.fill_constant([1], "int64", 2)
        dim_2 = paddle.fill_constant([1], "int32", 3)
        result_2 = paddle.randint(low=-5, high=5, shape=[dim_1, dim_2], dtype="int32")
        # [[0, -1, -3],
        #  [4, -2,  0]]

        # example 3:
        # attr shape is a Tensor
        var_shape = paddle.to_variable(np.array([3]))
        result_3 = paddle.randint(low=-5, high=5, shape=var_shape)
        # [-2, 2, 3]

        # example 4:
        # data type is int32
        result_4 = paddle.randint(low=-5, high=5, shape=[3], dtype='int32')
        # [-5, 4, -4]

        # example 5:
        # Input only one parameter
        # low=0, high=10, shape=[1], dtype='int64'
        result_5 = paddle.randint(10)
        # [7]

    """
    if high is None:
        if low <= 0:
            raise ValueError(
                "If high is None, low must be greater than 0, but received low = {0}.".
                format(low))
        high = low
        low = 0
    if dtype is None:
        dtype = 'int64'
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        shape = utils._convert_shape_to_list(shape)
        return core.ops.randint('shape', shape, 'low', low, 'high', high,
                                'seed', 0, 'dtype', dtype)

    check_type(shape, 'shape', (list, tuple, Variable), 'randint')
    check_dtype(dtype, 'dtype', ['int32', 'int64'], 'randint')
    if low >= high:
        raise ValueError(
            "randint's low must less then high, but received low = {0}, "
            "high = {1}".format(low, high))

    inputs = dict()
    attrs = {'low': low, 'high': high, 'seed': 0, 'dtype': dtype}
    utils._get_shape_tensor_inputs(
        inputs=inputs, attrs=attrs, shape=shape, op_type='randint')

    helper = LayerHelper("randint", **locals())
    out = helper.create_variable_for_type_inference(dtype=dtype)
    helper.append_op(
        type='randint', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


def randn(shape, dtype=None, name=None):
    """
	:alias_main: paddle.randn
	:alias: paddle.tensor.randn, paddle.tensor.random.randn

    This OP returns a Tensor filled with random values sampled from a normal
    distribution with mean 0 and standard deviation 1 (also called the standard
    normal distribution), with ``shape`` and ``dtype``.

    Args:
        shape(list|tuple|Tensor): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64).
        dtype(str|np.dtype|core.VarDesc.VarType, optional): The data type of the
            output tensor. Supported data types: float32, float64. If ``dytpe``
            is None, the data type is float32. Default is None.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a normal
        distribution with mean 0 and standard deviation 1 (also called the
        standard normal distribution), with ``shape`` and ``dtype``.

    Raises:
        TypeError: If ``shape`` is not list, tuple, Tensor.
        TypeError: If ``dtype`` is not float32, float64.

    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        # example 1: attr shape is a list which doesn't contain Tensor.
        result_1 = paddle.randn(shape=[2, 3])
        # [[-2.923464  ,  0.11934398, -0.51249987],
        #  [ 0.39632758,  0.08177969,  0.2692008 ]]

        # example 2: attr shape is a list which contains Tensor.
        dim_1 = paddle.fill_constant([1], "int64", 2)
        dim_2 = paddle.fill_constant([1], "int32", 3)
        result_2 = paddle.randn(shape=[dim_1, dim_2, 2])
        # [[[-2.8852394 , -0.25898588],
        #   [-0.47420555,  0.17683524],
        #   [-0.7989969 ,  0.00754541]],
        #  [[ 0.85201347,  0.32320443],
        #   [ 1.1399018 ,  0.48336947],
        #   [ 0.8086993 ,  0.6868893 ]]]

        # example 3: attr shape is a Tensor, the data type must be int64 or int32.
        var_shape = paddle.to_variable(np.array([2, 3]))
        result_3 = paddle.randn(var_shape)
        # [[-2.878077 ,  0.17099959,  0.05111201]
        #  [-0.3761474, -1.044801  ,  1.1870178 ]]

    """
    if dtype is None:
        dtype = 'float32'

    out = gaussian_random(
        shape=shape, mean=0.0, std=1.0, seed=0, dtype=dtype, name=name)
    out.stop_gradient = True
    return out


@templatedoc()
def randperm(n, dtype="int64", name=None):
    """
	:alias_main: paddle.randperm
	:alias: paddle.tensor.randperm, paddle.tensor.random.randperm

    This OP returns a 1-D Tensor filled with random permutation values from 0
    to n-1, with ``dtype``.

    Args:
        n(int): The upper bound (exclusive), and it should be greater than 0.
        dtype(str|np.dtype|core.VarDesc.VarType, optional): The data type of
            the output Tensor. Supported data types: int32, int64, float32,
            float64. Default is int64.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A 1-D Tensor filled with random permutation values from 0
        to n-1, with ``dtype``.

    Raises:
        ValueError: If ``n`` is not greater than 0.
        TypeError: If ``dtype`` is not int32, int64, float32, float64.

    Examples:
        .. code-block:: python

        import paddle

        paddle.disable_static()

        result_1 = paddle.randperm(5)
        # [4, 1, 2, 3, 0]

        result_2 = paddle.randperm(7, 'int32')
        # [1, 6, 2, 0, 4, 3, 5]
 
    """
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        return core.ops.randperm('n', n, 'seed', 0, 'dtype', dtype)

    if n < 1:
        raise ValueError("The input n should be greater than 0 in randperm op.")
    check_dtype(dtype, 'dtype', ['int64', 'int32', 'float32', 'float64'],
                'randperm')

    helper = LayerHelper("randperm", **locals())
    out = helper.create_variable_for_type_inference(dtype)
    attrs = {'n': n, 'dtype': dtype, 'seed': 0}
    helper.append_op(
        type='randperm', inputs={}, outputs={'Out': out}, attrs=attrs)
    out.stop_gradient = True
    return out


def rand(shape, dtype=None, name=None):
    """
	:alias_main: paddle.rand
	:alias: paddle.tensor.rand, paddle.tensor.random.rand

    This OP returns a Tensor filled with random values sampled from a uniform
    distribution in the range [0, 1), with ``shape`` and ``dtype``.

    Examples:
    ::

        Input:
          shape = [1, 2]

        Output:
          result=[[0.8505902, 0.8397286]]

    Args:
        shape(list|tuple|Tensor): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64).
        dtype(str|np.dtype|core.VarDesc.VarType, optional): The data type of the
            output tensor. Supported data types: float32, float64. If ``dytpe``
            is None, the data type is float32. Default is None.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a uniform
        distribution in the range [0, 1), with ``shape`` and ``dtype``.

    Raises:
        TypeError: If ``shape`` is not list, tuple, Tensor.
        ValueError: If ``dtype`` is not float32, float64.

    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()
        # example 1: attr shape is a list which doesn't contain Tensor.
        result_1 = paddle.rand(shape=[2, 3])
        # [[0.451152  , 0.55825245, 0.403311  ],
        #  [0.22550228, 0.22106001, 0.7877319 ]]

        # example 2: attr shape is a list which contains Tensor.
        dim_1 = paddle.fill_constant([1], "int64", 2)
        dim_2 = paddle.fill_constant([1], "int32", 3)
        result_2 = paddle.rand(shape=[dim_1, dim_2, 2])
        # [[[0.8879919 , 0.25788337],
        #   [0.28826773, 0.9712097 ],
        #   [0.26438272, 0.01796806]],
        #  [[0.33633623, 0.28654453],
        #   [0.79109055, 0.7305809 ],
        #   [0.870881  , 0.2984597 ]]]

        # example 3: attr shape is a Tensor, the data type must be int64 or int32.
        var_shape = paddle.to_variable(np.array([2, 3]))
        result_3 = paddle.rand(var_shape)
        # [[0.22920267, 0.841956  , 0.05981819],
        #  [0.4836288 , 0.24573246, 0.7516129 ]]

    """
    if dtype is None:
        dtype = 'float32'

    out = uniform_random(shape, dtype, min=0.0, max=1.0, name=name)
    out.stop_gradient = True
    return out
