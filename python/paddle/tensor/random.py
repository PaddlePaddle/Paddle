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

import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.base.framework import _current_expected_place
from paddle.base.libpaddle import DataType
from paddle.common_ops_import import Variable
from paddle.framework import (
    in_dynamic_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)

from ..base.data_feeder import (
    check_dtype,
    check_shape,
    check_type,
    check_variable_and_dtype,
)
from ..framework import (
    LayerHelper,
    convert_np_dtype_to_dtype_,
    core,
    dygraph_only,
)

__all__ = []


def bernoulli(x, name=None):
    r"""

    For each element :math:`x_i` in input ``x``, take a sample from the Bernoulli distribution, also called two-point distribution, with success probability :math:`x_i`. The Bernoulli distribution with success probability :math:`x_i` is a discrete probability distribution with probability mass function

    .. math::
        p(y)=\begin{cases}
            x_i,&y=1\\
            1-x_i,&y=0
        \end{cases}.

    Args:
        x (Tensor): The input Tensor, it's data type should be float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: A Tensor filled samples from Bernoulli distribution, whose shape and dtype are same as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.set_device('cpu')  # on CPU device
            >>> paddle.seed(100)

            >>> x = paddle.rand([2,3])
            >>> print(x)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.55355281, 0.20714243, 0.01162981],
             [0.51577556, 0.36369765, 0.26091650]])
            >>> # doctest: -SKIP

            >>> out = paddle.bernoulli(x)
            >>> print(out)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 0., 1.],
             [0., 1., 0.]])
            >>> # doctest: -SKIP

    """

    if in_dynamic_or_pir_mode():
        return _C_ops.bernoulli(x)
    else:
        check_variable_and_dtype(
            x, "x", ["float32", "float64", "float16", "uint16"], "bernoulli"
        )

        helper = LayerHelper("randint", **locals())
        out = helper.create_variable_for_type_inference(
            dtype=x.dtype
        )  # maybe set out to int32 ?
        helper.append_op(
            type='bernoulli', inputs={"X": x}, outputs={'Out': out}, attrs={}
        )
        out.stop_gradient = True
        return out


def poisson(x, name=None):
    r"""
    Returns a tensor filled with random number from a Poisson Distribution.

    .. math::

        out_i \sim Poisson (lambda = x_i)

    Args:
        x(Tensor):  A tensor with rate parameter of poisson Distribution. The data type
            should be bfloat16, float16, float32, float64.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        Tensor: A Tensor filled with random number with the same shape and dtype as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.set_device('cpu')
            >>> paddle.seed(100)

            >>> x = paddle.uniform([2,3], min=1.0, max=5.0)
            >>> out = paddle.poisson(x)
            >>> print(out)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[2., 5., 0.],
             [5., 1., 3.]])
            >>> # doctest: -SKIP
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.poisson(x)
    else:
        check_variable_and_dtype(x, "x", ["float32", "float64"], "poisson")

        helper = LayerHelper("poisson", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='poisson', inputs={'X': x}, outputs={'Out': out}, attrs={}
        )
        return out


def multinomial(x, num_samples=1, replacement=False, name=None):
    """
    Returns a Tensor filled with random values sampled from a Multinomical
    distribution. The input ``x`` is a tensor with probabilities for generating the
    random number. Each element in ``x`` should be larger or equal to 0, but not all
    0. ``replacement`` indicates whether it is a replaceable sample. If ``replacement``
    is True, a category can be sampled more than once.

    Args:
        x(Tensor):  A tensor with probabilities for generating the random number. The data type
            should be float32, float64.
        num_samples(int, optional): Number of samples, default is 1.
        replacement(bool, optional): Whether it is a replaceable sample, default is False.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        Tensor: A Tensor filled with sampled category index after ``num_samples`` times samples.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(100) # on CPU device

            >>> x = paddle.rand([2,4])
            >>> print(x)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.55355281, 0.20714243, 0.01162981, 0.51577556],
             [0.36369765, 0.26091650, 0.18905126, 0.56219709]])
            >>> # doctest: -SKIP

            >>> paddle.seed(200) # on CPU device
            >>> out1 = paddle.multinomial(x, num_samples=5, replacement=True)
            >>> print(out1)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 5], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3, 3, 0, 0, 0],
             [3, 3, 3, 1, 0]])
            >>> # doctest: -SKIP

            >>> # out2 = paddle.multinomial(x, num_samples=5)
            >>> # InvalidArgumentError: When replacement is False, number of samples
            >>> #  should be less than non-zero categories

            >>> paddle.seed(300) # on CPU device
            >>> out3 = paddle.multinomial(x, num_samples=3)
            >>> print(out3)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3, 0, 1],
             [3, 1, 0]])
            >>> # doctest: -SKIP

    """

    if in_dynamic_or_pir_mode():
        return _C_ops.multinomial(x, num_samples, replacement)
    else:
        check_variable_and_dtype(
            x, "x", ["uint16", "float16", "float32", "float64"], "multinomial"
        )

        helper = LayerHelper("multinomial", **locals())
        out = helper.create_variable_for_type_inference(
            dtype=convert_np_dtype_to_dtype_('int64')
        )
        helper.append_op(
            type='multinomial',
            inputs={"X": x},
            outputs={'Out': out},
            attrs={'num_samples': num_samples, 'replacement': replacement},
        )
        out.stop_gradient = True
        return out


def uniform_random_batch_size_like(
    input,
    shape,
    dtype='float32',
    input_dim_idx=0,
    output_dim_idx=0,
    min=-1.0,
    max=1.0,
    seed=0,
):
    """
    This OP initializes a variable with random values sampled from a
    uniform distribution in the range [min, max). The input_dim_idx used to get the input dimension value which will be used to resize the output dimension.
    .. code-block:: text
        *Case 1:
            Given:
                input =[[0.946741  , 0.1357001 , 0.38086128]]    # input.shape=[1,3]
                shape=[2,4]
            result.shape[output_dim_idx] = input.shape[input_dim_idx],
            output_dim_idx = 0,
            input_dim_idx = 0,
            result.shape[0] = input.shape[0],
            then:
                result=[[ 0.3443427 , -0.23056602,  0.3477049 ,  0.06139076]]    # result.shape=[1,4]
       *Case 2:
           Given:
               input =[[0.946741  , 0.1357001 , 0.38086128]]     # input.shape=[1,3]
               shape=[2,4]
               input_dim_idx=1
               output_dim_idx=1
           result.shape[output_dim_idx] = input.shape[input_dim_idx],
           output_dim_idx = 1,
           input_dim_idx = 1,
           result.shape[1] = input.shape[1],
           then:
               result=[[-0.23133647, -0.84195036,  0.21441269],
                       [-0.08774924,  0.25605237, -0.09403259]]    # result.shape=[2,3]
    Args:
        input (Variable): A Tensor. Supported data types: float32, float64.
        shape (tuple|list): A python list or python tuple. The shape of the output Tensor, the data type is int.
        input_dim_idx (int, optional): An index used to get the input dimension value which will be used to resize the output dimension. Default  0.
        output_dim_idx (int, optional): An index used to indicate the specific dimension that will be replaced by corresponding input dimension value. Default 0.
        min (float, optional): The lower bound on the range of random values to generate, the min is included in the range. Default -1.0.
        max (float, optional): The upper bound on the range of random values to generate, the max is excluded in the range. Default 1.0.
        seed (int, optional):  Random seed used for generating samples. 0 means use a seed generated by the system.Note that if seed is not 0, this operator will always generate the same random numbers every time.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The data type of output Tensor. Supported data types: float32, float64. Default float32.
    Returns:
        Variable: A Tensor of the specified shape filled with uniform_random values. The shape of the Tensor is determined by the shape parameter and the specified dimension of the input Tensor.
    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.base as base
            >>> from paddle.tensor import random
            >>> paddle.enable_static()
            >>> # example 1:
            >>> input = paddle.static.data(name="input", shape=[1, 3], dtype='float32')
            >>> out_1 = random.uniform_random_batch_size_like(input, [2, 4])
            >>> print(out_1.shape)
            [1, 4]

            >>> # example 2:
            >>> out_2 = random.uniform_random_batch_size_like(input, [2, 4], input_dim_idx=1, output_dim_idx=1)
            >>> print(out_2.shape)
            [2, 3]
    """
    check_variable_and_dtype(
        input,
        'Input',
        ("float32", 'float64', "uint16"),
        'uniform_random_batch_size_like',
    )
    check_type(shape, 'shape', (list, tuple), 'uniform_random_batch_size_like')
    check_dtype(
        dtype,
        'dtype',
        ('float32', 'float64', "uint16"),
        'uniform_random_batch_size_like',
    )

    helper = LayerHelper('uniform_random_batch_size_like', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    c_dtype = convert_np_dtype_to_dtype_(dtype)
    helper.append_op(
        type='uniform_random_batch_size_like',
        inputs={'Input': input},
        outputs={'Out': out},
        attrs={
            'shape': shape,
            'input_dim_idx': input_dim_idx,
            'output_dim_idx': output_dim_idx,
            'min': min,
            'max': max,
            'seed': seed,
            'dtype': c_dtype,
        },
    )

    return out


def gaussian(shape, mean=0.0, std=1.0, seed=0, dtype=None, name=None):
    """
    Returns a Tensor filled with random values sampled from a Gaussian
    distribution, with ``shape`` and ``dtype``.

    Args:
        shape (tuple|list|Tensor): Shape of the Tensor to be created. The data type is ``int32`` or ``int64`` .
            If ``shape`` is a list or tuple, each element of it should be integer or 0-D Tensor with shape [].
            If ``shape`` is an Tensor, it should be an 1-D Tensor which represents a list.
        mean (float|int, optional): Mean of the output tensor, default is 0.0.
        std (float|int, optional): Standard deviation of the output tensor, default
            is 1.0.
        seed (int, optional): Random seed of generator.
        dtype (str|np.dtype, optional): The data type of the output Tensor.
            Supported data types: float32, float64.
            Default is None, use global default dtype (see ``get_default_dtype``
            for details).
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a Gaussian
        distribution, with ``shape`` and ``dtype``.
    """
    op_type_for_check = 'gaussian/standard_normal/randn/normal'
    supported_dtypes = ['float32', 'float64', 'float16', 'uint16', 'bfloat16']

    if dtype is None:
        dtype = paddle.framework.get_default_dtype()
        if dtype not in supported_dtypes:
            raise TypeError(
                "{} only supports {}, but the default dtype is {}".format(
                    op_type_for_check, supported_dtypes, dtype
                )
            )
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_or_pir_mode():
        shape = paddle.utils.convert_shape_to_list(shape)
        place = _current_expected_place()
        return _C_ops.gaussian(
            shape, float(mean), float(std), seed, dtype, place
        )
    else:
        check_shape(shape, op_type_for_check)
        check_dtype(dtype, 'dtype', supported_dtypes, op_type_for_check)

        inputs = {}
        attrs = {
            'mean': mean,
            'std': std,
            'seed': seed,
            'dtype': dtype,
            'use_mkldnn': False,
        }
        paddle.utils.get_shape_tensor_inputs(
            inputs=inputs, attrs=attrs, shape=shape, op_type=op_type_for_check
        )

        helper = LayerHelper('gaussian', **locals())
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='gaussian_random',
            inputs=inputs,
            outputs={'Out': out},
            attrs=attrs,
        )
        out.stop_gradient = True
        return out


@dygraph_only
def gaussian_(x, mean=0.0, std=1.0, seed=0, name=None):
    """
    This is the inplace version of OP ``gaussian``, which returns a Tensor filled
    with random values sampled from a gaussian distribution. The output Tensor will
    be inplaced with input ``x``. Please refer to :ref:`api_tensor_gaussian`.

    Args:
        x(Tensor): The input tensor to be filled with random values.
        mean (float|int, optional): Mean of the output tensor, default is 0.0.
        std (float|int, optional): Standard deviation of the output tensor, default
            is 1.0.
        seed (int, optional): Random seed of generator.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        Tensor: The input tensor x filled with random values sampled from a gaussian
        distribution.
    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.randn([3, 4])
            >>> paddle.tensor.random.gaussian_(x)
            >>> print(x)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
                [[ 0.86384124,  0.67328387,  0.21874231, -0.12615913],
                [ 0.69844258,  0.42084831, -0.42476156, -0.00072985],
                [ 1.72819555,  1.87785017,  0.48915744,  0.09235018]])
    """
    return _C_ops.gaussian_inplace_(x, float(mean), float(std), int(seed))


def standard_normal(shape, dtype=None, name=None):
    """
    Returns a Tensor filled with random values sampled from a standard
    normal distribution with mean 0 and standard deviation 1, with ``shape``
    and ``dtype``.

    Args:
        shape (tuple|list|Tensor): Shape of the Tensor to be created. The data type is ``int32`` or ``int64`` .
            If ``shape`` is a list or tuple, each element of it should be integer or 0-D Tensor with shape [].
            If ``shape`` is an Tensor, it should be an 1-D Tensor which represents a list.
        dtype (str|np.dtype, optional): The data type of the output Tensor.
            Supported data types: float32, float64.
            Default is None, use global default dtype (see ``get_default_dtype``
            for details).
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a standard
        normal distribution with mean 0 and standard deviation 1, with
        ``shape`` and ``dtype``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # doctest: +SKIP("Random output")
            >>> # example 1: attr shape is a list which doesn't contain Tensor.
            >>> out1 = paddle.standard_normal(shape=[2, 3])
            >>> print(out1)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.33719197, -0.25688133, -0.42868865],
             [-0.27804616, -0.25058213, -0.28209466]])
            >>> # doctest: -SKIP

            >>> # example 2: attr shape is a list which contains Tensor.
            >>> dim1 = paddle.to_tensor(2, 'int64')
            >>> dim2 = paddle.to_tensor(3, 'int32')
            >>> out2 = paddle.standard_normal(shape=[dim1, dim2, 2])
            >>> print(out2)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[ 0.81888396, -0.64831746],
              [ 1.28911388, -1.88154876],
              [-0.03271919, -0.32410008]],
             [[-0.20224631,  0.46683890],
              [ 1.91947734,  0.71657443],
              [ 0.33410960, -0.64256823]]])
            >>> # doctest: -SKIP

            >>> # example 3: attr shape is a Tensor, the data type must be int64 or int32.
            >>> shape_tensor = paddle.to_tensor([2, 3])
            >>> out3 = paddle.standard_normal(shape_tensor)
            >>> print(out3)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.01182475, -0.44895259, -1.79227340],
             [ 1.52022707, -0.83830303,  0.05261501]])
            >>> # doctest: -SKIP

    """
    return gaussian(shape=shape, mean=0.0, std=1.0, dtype=dtype, name=name)


def randn(shape, dtype=None, name=None):
    """
    Returns a Tensor filled with random values sampled from a standard
    normal distribution with mean 0 and standard deviation 1, with ``shape``
    and ``dtype``.

    Args:
        shape (tuple|list|Tensor): Shape of the Tensor to be created. The data type is ``int32`` or ``int64`` .
            If ``shape`` is a list or tuple, each element of it should be integer or 0-D Tensor with shape [].
            If ``shape`` is an Tensor, it should be an 1-D Tensor which represents a list.
        dtype (str|np.dtype, optional): The data type of the output Tensor.
            Supported data types: float32, float64.
            Default is None, use global default dtype (see ``get_default_dtype``
            for details).
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a standard
        normal distribution with mean 0 and standard deviation 1, with
        ``shape`` and ``dtype``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # example 1: attr shape is a list which doesn't contain Tensor.
            >>> out1 = paddle.randn(shape=[2, 3])
            >>> print(out1)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.29270014, -0.02925120, -1.07807338],
             [ 1.19966674, -0.46673676, -0.18050613]])
            >>> # doctest: -SKIP

            >>> # example 2: attr shape is a list which contains Tensor.
            >>> dim1 = paddle.to_tensor(2, 'int64')
            >>> dim2 = paddle.to_tensor(3, 'int32')
            >>> out2 = paddle.randn(shape=[dim1, dim2, 2])
            >>> print(out2)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[-0.26019713,  0.54994684],
              [ 0.46403214, -1.41178775],
              [-0.15682915, -0.26639181]],
             [[ 0.01364388, -2.81676364],
              [ 0.86996621,  0.07524570],
              [ 0.21443737,  0.90938759]]])
            >>> # doctest: -SKIP

            >>> # example 3: attr shape is a Tensor, the data type must be int64 or int32.
            >>> shape_tensor = paddle.to_tensor([2, 3])
            >>> out3 = paddle.randn(shape_tensor)
            >>> print(out3)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.57575506, -1.60349274, -0.27124876],
             [ 1.08381045,  0.81270242, -0.26763600]])
            >>> # doctest: -SKIP
    """
    return standard_normal(shape, dtype, name)


def normal(mean=0.0, std=1.0, shape=None, name=None):
    """
    Returns a Tensor filled with random values sampled from a normal
    distribution with ``mean`` and ``std`` (standard deviation) .

    If ``mean`` is a Tensor, the output Tensor has the same shape and data type as ``mean``.
    If ``mean`` is not a Tensor and ``std`` is a Tensor, the output Tensor has the same shape and data type as ``std``.
    If ``mean`` and ``std`` are not a Tensor, the output Tensor has the same shape as ``shape``, with data type float32.

    If ``mean`` and ``std`` are Tensor, the num of elements of ``mean`` and ``std`` should be the same.

    Args:
        mean (float|Tensor, optional): The mean of the output Tensor's normal distribution.
            If ``mean`` is float, all elements of the output Tensor shared the same mean.
            If ``mean`` is a Tensor(data type supports float32, float64), it has per-element means.
            Default is 0.0
        std (float|Tensor, optional): The  standard deviation of the output Tensor's normal distribution.
            If ``std`` is float, all elements of the output Tensor shared the same standard deviation.
            If ``std`` is a Tensor(data type supports float32, float64), it has per-element standard deviations.
            Defaule is 1.0
        shape (tuple|list|Tensor): Shape of the Tensor to be created. The data type is ``int32`` or ``int64`` .
            If ``shape`` is a list or tuple, each element of it should be integer or 0-D Tensor with shape [].
            If ``shape`` is an Tensor, it should be an 1-D Tensor which represents a list. If ``mean`` or ``std``
            is a Tensor, the shape of the output Tensor is the same as ``mean`` or ``std`` , attr ``shape`` is ignored.
            Default is None
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor filled with random values sampled from a normal distribution with ``mean`` and ``std`` .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> out1 = paddle.normal(shape=[2, 3])
            >>> print(out1)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.85107994, -0.85490644, -1.35941815],
             [-0.55500370,  0.20964541,  2.24193954]])
            >>> # doctest: -SKIP

            >>> mean_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
            >>> out2 = paddle.normal(mean=mean_tensor)
            >>> print(out2)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.05411839, 3.71514320, 3.42665267])
            >>> # doctest: -SKIP

            >>> std_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
            >>> out3 = paddle.normal(mean=mean_tensor, std=std_tensor)
            >>> print(out3)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.48646951, 0.00815189, 3.74022293])
            >>> # doctest: -SKIP
    """
    if not in_dynamic_or_pir_mode():
        check_type(mean, 'mean', (int, float, Variable), 'normal')
        check_type(std, 'std', (int, float, Variable), 'normal')
        if isinstance(mean, Variable):
            check_dtype(
                mean.dtype,
                'mean',
                ['float32', 'float64'],
                'normal',
                "If mean is Tensor, it's data type only support float32, float64.",
            )
        if isinstance(std, Variable):
            check_dtype(
                std.dtype,
                'std',
                ['float32', 'float64'],
                'normal',
                "If std is Tensor, it's data type only support float32, float64.",
            )
        if shape is not None:
            check_shape(shape, 'normal')

    if isinstance(mean, Variable):
        if isinstance(std, Variable):
            if std.dtype != mean.dtype:
                std = paddle.cast(std, mean.dtype)
            mean_shape = paddle.shape(mean)
            std = paddle.reshape(std, mean_shape)
        else:
            std = float(std)
        out = standard_normal(paddle.shape(mean), mean.dtype, name)
    elif isinstance(std, Variable):
        mean = float(mean)
        out = standard_normal(paddle.shape(std), std.dtype, name)
    else:
        return gaussian(shape=shape, mean=mean, std=std, name=name)

    out = out * std + mean
    if not in_dynamic_or_pir_mode():
        out.stop_grediant = True
    return out


@dygraph_only
def normal_(x, mean=0.0, std=1.0, name=None):
    """
    This is the inplace version of api ``normal``, which returns a Tensor filled
    with random values sampled from a normal distribution. The output Tensor will
    be inplaced with input ``x``. Please refer to :ref:`api_tensor_noraml`.

    Args:
        x(Tensor): The input tensor to be filled with random values.
        mean (float|Tensor, optional): The mean of the output Tensor's normal distribution.
            If ``mean`` is float, all elements of the output Tensor shared the same mean.
            If ``mean`` is a Tensor(data type supports float32, float64), it has per-element means.
            Default is 0.0
        std (float|Tensor, optional): The  standard deviation of the output Tensor's normal distribution.
            If ``std`` is float, all elements of the output Tensor shared the same standard deviation.
            If ``std`` is a Tensor(data type supports float32, float64), it has per-element standard deviations.
            Defaule is 1.0
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        A Tensor filled with random values sampled from a normal distribution with ``mean`` and ``std`` .
    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.randn([3, 4])
            >>> x.normal_()
            >>> # doctest: +SKIP('random check')
            >>> print(x)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.06132207,  1.11349595,  0.41906244, -0.24858207],
             [-1.85169315, -1.50370061,  1.73954511,  0.13331604],
             [ 1.66359663, -0.55764782, -0.59911072, -0.57773495]])

    """
    return gaussian_(x, mean=mean, std=std)


def uniform(shape, dtype=None, min=-1.0, max=1.0, seed=0, name=None):
    """
    Returns a Tensor filled with random values sampled from a uniform
    distribution in the range [``min``, ``max``), with ``shape`` and ``dtype``.

    Examples:

    .. code-block:: text

        Input:
          shape = [1, 2]
        Output:
          result=[[0.8505902, 0.8397286]]

    Args:
        shape (tuple|list|Tensor): Shape of the Tensor to be created. The data type is ``int32`` or ``int64`` .
            If ``shape`` is a list or tuple, each element of it should be integer or 0-D Tensor with shape [].
            If ``shape`` is an Tensor, it should be an 1-D Tensor which represents a list.
        dtype(str|np.dtype, optional): The data type of the output Tensor.
            Supported data types: float32, float64.
            Default is None, use global default dtype (see ``get_default_dtype``
            for details).
        min(float|int, optional): The lower bound on the range of random values
            to generate, ``min`` is included in the range. Default is -1.0.
        max(float|int, optional): The upper bound on the range of random values
            to generate, ``max`` is excluded in the range. Default is 1.0.
        seed(int, optional): Random seed used for generating samples. If seed is 0,
            it will use the seed of the global default generator (which can be set by paddle.seed).
            Note that if seed is not 0, this operator will always generate the same random numbers every
            time. Default is 0.
        name(str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a uniform
        distribution in the range [``min``, ``max``), with ``shape`` and ``dtype``.

    Examples:
        .. code-block:: python
            :name: code-example1

            >>> import paddle

            >>> # example 1:
            >>> # attr shape is a list which doesn't contain Tensor.
            >>> out1 = paddle.uniform(shape=[3, 4])
            >>> print(out1)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.38170254, -0.47945309,  0.39794648, -0.94233936],
             [-0.85296679, -0.76094693,  0.10565400,  0.59155810],
             [ 0.11681318, -0.42144555, -0.81596589,  0.62113667]])
            >>> # doctest: -SKIP

            >>> # example 2:
            >>> # attr shape is a list which contains Tensor.
            >>> dim1 = paddle.to_tensor(2, 'int64')
            >>> dim2 = paddle.to_tensor(3, 'int32')
            >>> out2 = paddle.uniform(shape=[dim1, dim2])
            >>> print(out2)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.00294012, -0.07210171, -0.44236207],
             [ 0.70089281,  0.21500075, -0.22084606]])
            >>> # doctest: -SKIP

            >>> # example 3:
            >>> # attr shape is a Tensor, the data type must be int64 or int32.
            >>> shape_tensor = paddle.to_tensor([2, 3])
            >>> out3 = paddle.uniform(shape_tensor)
            >>> print(out3)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.60801756,  0.32448411,  0.90269291],
             [-0.66421294, -0.95218551, -0.51022208]])
            >>> # doctest: -SKIP
    """
    supported_dtypes = ['float32', 'float64', 'float16', 'uint16']
    if dtype is None:
        dtype = paddle.framework.get_default_dtype()
        if dtype not in supported_dtypes:
            raise TypeError(
                "uniform/rand only supports {}, but the default dtype is {}".format(
                    supported_dtypes, dtype
                )
            )

    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_mode():
        shape = paddle.utils.convert_shape_to_list(shape)
        return _C_ops.uniform(
            shape,
            dtype,
            float(min),
            float(max),
            seed,
            _current_expected_place(),
        )
    elif in_pir_mode():
        check_type(
            shape, 'shape', (list, tuple, paddle.pir.OpResult), 'uniform/rand'
        )
        check_dtype(dtype, 'dtype', supported_dtypes, 'uniform/rand')
        check_type(
            min, 'min', (float, int, paddle.pir.OpResult), 'uniform/rand'
        )
        check_type(
            max, 'max', (float, int, paddle.pir.OpResult), 'uniform/rand'
        )
        if paddle.utils._contain_var(shape):
            shape = paddle.utils.get_int_tensor_list(
                shape, _current_expected_place()
            )
        return _C_ops.uniform(
            shape,
            dtype,
            float(min),
            float(max),
            seed,
            _current_expected_place(),
        )
    else:
        check_type(shape, 'shape', (list, tuple, Variable), 'uniform/rand')
        check_dtype(dtype, 'dtype', supported_dtypes, 'uniform/rand')
        check_type(min, 'min', (float, int, Variable), 'uniform/rand')
        check_type(max, 'max', (float, int, Variable), 'uniform/rand')

        inputs = {}
        attrs = {'seed': seed, 'min': min, 'max': max, 'dtype': dtype}
        paddle.utils.get_shape_tensor_inputs(
            inputs=inputs, attrs=attrs, shape=shape, op_type='uniform/rand'
        )

        helper = LayerHelper("uniform", **locals())
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="uniform_random",
            inputs=inputs,
            attrs=attrs,
            outputs={"Out": out},
        )
        out.stop_gradient = True
        return out


@dygraph_only
def uniform_(x, min=-1.0, max=1.0, seed=0, name=None):
    """
    This is the inplace version of OP ``uniform``, which returns a Tensor filled
    with random values sampled from a uniform distribution. The output Tensor will
    be inplaced with input ``x``. Please refer to :ref:`api_paddle_uniform`.

    Args:
        x(Tensor): The input tensor to be filled with random values.
        min(float|int, optional): The lower bound on the range of random values
            to generate, ``min`` is included in the range. Default is -1.0.
        max(float|int, optional): The upper bound on the range of random values
            to generate, ``max`` is excluded in the range. Default is 1.0.
        seed(int, optional): Random seed used for generating samples. If seed is 0,
            it will use the seed of the global default generator (which can be set by paddle.seed).
            Note that if seed is not 0, this operator will always generate the same random numbers every
            time. Default is 0.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        Tensor: The input tensor x filled with random values sampled from a uniform
        distribution in the range [``min``, ``max``).
    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # example:
            >>> x = paddle.ones(shape=[3, 4])
            >>> x.uniform_()
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.50484276,  0.49580324,  0.33357990, -0.93924278],
             [ 0.39779735,  0.87677515, -0.24377221,  0.06212139],
             [-0.92499518, -0.96244860,  0.79210341, -0.78228098]])
            >>> # doctest: -SKIP
    """
    return _C_ops.uniform_inplace_(x, min, max, seed, 0, 0, 1.0)


def randint(low=0, high=None, shape=[1], dtype=None, name=None):
    """
    Returns a Tensor filled with random integers from a discrete uniform
    distribution in the range [``low``, ``high``), with ``shape`` and ``dtype``.
    If ``high`` is None (the default), the range is [0, ``low``).

    Args:
        low (int, optional): The lower bound on the range of random values to generate.
            The ``low`` is included in the range. If ``high`` is None, the
            range is [0, ``low``). Default is 0.
        high (int, optional): The upper bound on the range of random values to
            generate, the ``high`` is excluded in the range. Default is None
            (see above for behavior if high = None). Default is None.
        shape (tuple|list|Tensor): Shape of the Tensor to be created. The data type is ``int32`` or ``int64`` .
            If ``shape`` is a list or tuple, each element of it should be integer or 0-D Tensor with shape [].
            If ``shape`` is an Tensor, it should be an 1-D Tensor which represents a list. Default is [1].
        dtype (str|np.dtype, optional): The data type of the
            output tensor. Supported data types: int32, int64. If ``dytpe``
            is None, the data type is int64. Default is None.
        name (str, optional): The default value is None.  Normally there is no
            need for user to set this property.  For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random integers from a discrete uniform
        distribution in the range [``low``, ``high``), with ``shape`` and ``dtype``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # example 1:
            >>> # attr shape is a list which doesn't contain Tensor.
            >>> out1 = paddle.randint(low=-5, high=5, shape=[2, 3])
            >>> print(out1)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[-1,  4,  4],
             [-2, -5, -2]])
            >>> # doctest: -SKIP

            >>> # example 2:
            >>> # attr shape is a list which contains Tensor.
            >>> dim1 = paddle.to_tensor(2, 'int64')
            >>> dim2 = paddle.to_tensor(3, 'int32')
            >>> out2 = paddle.randint(low=-5, high=5, shape=[dim1, dim2])
            >>> print(out2)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[-4, -4,  2],
             [-3, -1, -5]])
            >>> # doctest: -SKIP

            >>> # example 3:
            >>> # attr shape is a Tensor
            >>> shape_tensor = paddle.to_tensor([2, 3])
            >>> out3 = paddle.randint(low=-5, high=5, shape=shape_tensor)
            >>> print(out3)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[-1,  4, -3],
             [ 1,  2, -1]])
            >>> # doctest: -SKIP

            >>> # example 4:
            >>> # data type is int32
            >>> out4 = paddle.randint(low=-5, high=5, shape=[3], dtype='int32')
            >>> print(out4)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[3], dtype=int32, place=Place(cpu), stop_gradient=True,
            [4, 4, 0])
            >>> # doctest: -SKIP

            >>> # example 5:
            >>> # Input only one parameter
            >>> # low=0, high=10, shape=[1], dtype='int64'
            >>> out5 = paddle.randint(10)
            >>> print(out5)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [7])
            >>> # doctest: -SKIP

    """
    if high is None:
        if low <= 0:
            raise ValueError(
                f"If high is None, low must be greater than 0, but received low = {low}."
            )
        high = low
        low = 0
    if dtype is None:
        dtype = core.VarDesc.VarType.INT64
        if in_pir_mode():
            dtype = DataType.INT64
    elif not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_mode():
        shape = paddle.utils.convert_shape_to_list(shape)
        return _C_ops.randint(
            low, high, shape, dtype, _current_expected_place()
        )
    elif in_pir_mode():
        check_type(
            shape, 'shape', (list, tuple, paddle.pir.OpResult), 'randint'
        )
        check_dtype(dtype, 'dtype', ['int32', 'int64'], 'randint')
        if paddle.utils._contain_var(shape):
            shape = paddle.utils.get_int_tensor_list(
                shape, _current_expected_place()
            )
        return _C_ops.randint(
            low, high, shape, dtype, _current_expected_place()
        )
    else:
        check_shape(shape, 'randint')
        check_dtype(dtype, 'dtype', ['int32', 'int64'], 'randint')
        if low >= high:
            raise ValueError(
                f"randint's low must less then high, but received low = {low}, "
                f"high = {high}"
            )

        inputs = {}
        attrs = {'low': low, 'high': high, 'seed': 0, 'dtype': dtype}
        paddle.utils.get_shape_tensor_inputs(
            inputs=inputs, attrs=attrs, shape=shape, op_type='randint'
        )

        helper = LayerHelper("randint", **locals())
        out = helper.create_variable_for_type_inference(dtype=dtype)
        helper.append_op(
            type='randint', inputs=inputs, outputs={'Out': out}, attrs=attrs
        )
        out.stop_gradient = True
        return out


def randint_like(x, low=0, high=None, dtype=None, name=None):
    """
    Returns a Tensor filled with random integers from a discrete uniform
    distribution in the range [``low``, ``high``), with the same shape as ``x``.
    (use ``dtype`` if ``dtype`` is not None)
    If ``high`` is None (the default), the range is [0, ``low``).

    Args:
        x (Tensor): The input multi-dimensional tensor which specifies shape. The dtype of ``x``
            can be bool, int32, int64, float16, float32, float64.
        low (int, optional): The lower bound on the range of random values to generate.
            The ``low`` is included in the range. If ``high`` is None, the
            range is [0, ``low``). Default is 0.
        high (int, optional): The upper bound on the range of random values to
            generate, the ``high`` is excluded in the range. Default is None.
            If ``high`` is None, the range is [0, ``low``).
        dtype (str|np.dtype, optional): The data type of the
            output tensor. Supported data types: bool, int32, int64, float16,
            float32, float64. If ``dytpe`` is None, the data type is the
            same as x's data type. Default is None.
        name (str, optional): The default value is None.  Normally there is no
            need for user to set this property.  For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random integers from a discrete uniform
        distribution in the range [``low``, ``high``), with ``shape`` and ``dtype``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # example 1:
            >>> # dtype is None and the dtype of x is float32
            >>> x = paddle.zeros((1,2)).astype("float32")
            >>> out2 = paddle.randint_like(x, low=-5, high=5)
            >>> print(out2)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[1, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 0.]])
            >>> # doctest: -SKIP
            >>> print(out2.dtype)
            paddle.float32

            >>> # example 2:
            >>> # dtype is None and the dtype of x is float64
            >>> x = paddle.zeros((1,2)).astype("float64")
            >>> out2 = paddle.randint_like(x, low=-5, high=5)
            >>> print(out2)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[1, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[ 4., -5.]])
            >>> # doctest: -SKIP
            >>> print(out2.dtype)
            paddle.float64

            >>> # example 3:
            >>> # dtype is None and the dtype of x is int32
            >>> x = paddle.zeros((1,2)).astype("int32")
            >>> out3 = paddle.randint_like(x, low=-5, high=5)
            >>> print(out3)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[1, 2], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[ 0, -4]])
            >>> # doctest: -SKIP
            >>> print(out3.dtype)
            paddle.int32

            >>> # example 4:
            >>> # dtype is None and the dtype of x is int64
            >>> x = paddle.zeros((1,2)).astype("int64")
            >>> out4 = paddle.randint_like(x, low=-5, high=5)
            >>> print(out4)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[1, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[ 4, -3]])
            >>> # doctest: -SKIP
            >>> print(out4.dtype)
            paddle.int64

            >>> # example 5:
            >>> # dtype is float64 and the dtype of x is float32
            >>> x = paddle.zeros((1,2)).astype("float32")
            >>> out5 = paddle.randint_like(x, low=-5, high=5, dtype="float64")
            >>> print(out5)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[1, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[3., 1.]])
            >>> # doctest: -SKIP
            >>> print(out5.dtype)
            paddle.float64

            >>> # example 6:
            >>> # dtype is bool and the dtype of x is float32
            >>> x = paddle.zeros((1,2)).astype("float32")
            >>> out6 = paddle.randint_like(x, low=-5, high=5, dtype="bool")
            >>> print(out6)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[1, 2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [[False, True ]])
            >>> # doctest: -SKIP
            >>> print(out6.dtype)
            paddle.bool

            >>> # example 7:
            >>> # dtype is int32 and the dtype of x is float32
            >>> x = paddle.zeros((1,2)).astype("float32")
            >>> out7 = paddle.randint_like(x, low=-5, high=5, dtype="int32")
            >>> print(out7)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[1, 2], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[-2, -2]])
            >>> # doctest: -SKIP
            >>> print(out7.dtype)
            paddle.int32

            >>> # example 8:
            >>> # dtype is int64 and the dtype of x is float32
            >>> x = paddle.zeros((1,2)).astype("float32")
            >>> out8 = paddle.randint_like(x, low=-5, high=5, dtype="int64")
            >>> print(out8)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[1, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[-5,  4]])
            >>> # doctest: -SKIP
            >>> print(out8.dtype)
            paddle.int64

            >>> # example 9:
            >>> # dtype is int64 and the dtype of x is bool
            >>> x = paddle.zeros((1,2)).astype("bool")
            >>> out9 = paddle.randint_like(x, low=-5, high=5, dtype="int64")
            >>> print(out9)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[1, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[ 1, -2]])
            >>> # doctest: -SKIP
            >>> print(out9.dtype)
            paddle.int64

    """
    if high is None:
        if low <= 0:
            raise ValueError(
                f"If high is None, low must be greater than 0, but received low = {low}."
            )
        high = low
        low = 0
    if dtype is None:
        dtype = x.dtype
    else:
        if not isinstance(dtype, (core.VarDesc.VarType, core.DataType)):
            dtype = convert_np_dtype_to_dtype_(dtype)
    shape = paddle.shape(x)

    if low >= high:
        raise ValueError(
            f"randint_like's low must less then high, but received low = {low}, "
            f"high = {high}"
        )

    if in_dynamic_or_pir_mode():
        if in_dynamic_mode():
            shape = paddle.utils.convert_shape_to_list(shape)
            out = _legacy_C_ops.randint(
                'shape',
                shape,
                'low',
                low,
                'high',
                high,
                'seed',
                0,
                'dtype',
                core.VarDesc.VarType.INT64,
            )
        else:
            check_type(
                shape,
                'shape',
                (list, tuple, paddle.pir.OpResult),
                'randint_like',
            )
            check_dtype(
                dtype,
                'dtype',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'randint_like',
            )
            if paddle.utils._contain_var(shape):
                shape = paddle.utils.get_int_tensor_list(
                    shape, _current_expected_place()
                )
            out = _C_ops.randint(
                low, high, shape, DataType.INT64, _current_expected_place()
            )
        out = paddle.cast(out, dtype)
        return out
    else:
        check_shape(shape, 'randint_like')
        check_dtype(
            dtype,
            'dtype',
            ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
            'randint_like',
        )

        inputs = {"ShapeTensor": shape}
        attrs = {
            'low': low,
            'high': high,
            'seed': 0,
            'dtype': core.VarDesc.VarType.INT64,
        }

        helper = LayerHelper("randint", **locals())
        out = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.INT64
        )
        helper.append_op(
            type='randint', inputs=inputs, outputs={'Out': out}, attrs=attrs
        )
        out.stop_gradient = True
        out = paddle.cast(out, dtype)
        return out


def randperm(n, dtype="int64", name=None):
    """
    Returns a 1-D Tensor filled with random permutation values from 0
    to n-1, with ``dtype``.

    Args:
        n (int): The upper bound (exclusive), and it should be greater than 0.
        dtype (str|np.dtype, optional): The data type of
            the output Tensor. Supported data types: int32, int64, float32,
            float64. Default is int64.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A 1-D Tensor filled with random permutation values from 0
        to n-1, with ``dtype``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> out1 = paddle.randperm(5)
            >>> print(out1)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[5], dtype=int64, place=Place(cpu), stop_gradient=True,
            [3, 0, 1, 4, 2])
            >>> #doctest: -SKIP

            >>> out2 = paddle.randperm(7, 'int32')
            >>> print(out2)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[7], dtype=int32, place=Place(cpu), stop_gradient=True,
            [3, 2, 0, 6, 5, 4, 1])
            >>> #doctest: -SKIP

    """
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_or_pir_mode():
        return _C_ops.randperm(n, dtype, _current_expected_place())
    else:
        if n < 1:
            raise ValueError(
                "The input n should be greater than 0 in randperm op."
            )
        check_dtype(
            dtype, 'dtype', ['int64', 'int32', 'float32', 'float64'], 'randperm'
        )

        helper = LayerHelper("randperm", **locals())
        out = helper.create_variable_for_type_inference(dtype)
        attrs = {'n': n, 'dtype': dtype, 'seed': 0}
        helper.append_op(
            type='randperm', inputs={}, outputs={'Out': out}, attrs=attrs
        )
        out.stop_gradient = True
        return out


def rand(shape, dtype=None, name=None):
    """
    Returns a Tensor filled with random values sampled from a uniform
    distribution in the range [0, 1), with ``shape`` and ``dtype``.

    Args:
        shape (tuple|list|Tensor): Shape of the Tensor to be created. The data type is ``int32`` or ``int64`` .
            If ``shape`` is a list or tuple, each element of it should be integer or 0-D Tensor with shape [].
            If ``shape`` is an Tensor, it should be an 1-D Tensor which represents a list.
        dtype (str|np.dtype, optional): The data type of the output Tensor.
            Supported data types: float32, float64.
            Default is None, use global default dtype (see :ref:`get_default_dtype`
            for details).
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a uniform
        distribution in the range [0, 1), with ``shape`` and ``dtype``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # example 1: attr shape is a list which doesn't contain Tensor.
            >>> out1 = paddle.rand(shape=[2, 3])
            >>> print(out1)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.68532258, 0.69431782, 0.44835982],
             [0.13204314, 0.48128194, 0.36574543]])
            >>> # doctest: -SKIP

            >>> # example 2: attr shape is a list which contains Tensor.
            >>> dim1 = paddle.to_tensor(2, 'int64')
            >>> dim2 = paddle.to_tensor(3, 'int32')
            >>> out2 = paddle.rand(shape=[dim1, dim2, 2])
            >>> print(out2)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0.62102991, 0.45255184],
              [0.81386960, 0.22463219],
              [0.87946558, 0.28097662]],
             [[0.36565998, 0.63203937],
              [0.58640617, 0.92696166],
              [0.85060406, 0.38138932]]])
            >>> # doctest: -SKIP

            >>> # example 3: attr shape is a Tensor, the data type must be int64 or int32.
            >>> shape_tensor = paddle.to_tensor([2, 3])
            >>> out3 = paddle.rand(shape_tensor)
            >>> print(out3)
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.77650446, 0.12870903, 0.05153799],
             [0.27029657, 0.03963696, 0.42487794]])
            >>> # doctest: -SKIP
    """
    return uniform(shape, dtype, min=0.0, max=1.0, name=name)


def exponential_(x, lam=1.0, name=None):
    r"""
    This inplace OP fill input Tensor ``x`` with random number from a Exponential Distribution.

    ``lam`` is :math:`\lambda` parameter of Exponential Distribution.

    .. math::

        f(x) = \lambda e^{-\lambda x}

    Args:
        x(Tensor):  Input tensor. The data type should be float32, float64.
        lam(float, optional): :math:`\lambda` parameter of Exponential Distribution. Default, 1.0.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        Tensor: Input Tensor ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.set_device('cpu')
            >>> paddle.seed(100)

            >>> x = paddle.empty([2,3])
            >>> x.exponential_()
            >>> # doctest: +SKIP("Random output")
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.80643415, 0.23211166, 0.01169797],
             [0.72520679, 0.45208144, 0.30234432]])
            >>> # doctest: -SKIP

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.exponential_(x, lam)
    else:
        check_variable_and_dtype(
            x, "x", ["float16", "float32", "float64", "uint16"], "exponential"
        )

        helper = LayerHelper("exponential", **locals())
        helper.append_op(
            type='exponential',
            inputs={"X": x},
            outputs={'Out': x},
            attrs={"lambda": lam},
        )
        return x
