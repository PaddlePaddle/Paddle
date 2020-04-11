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

# __all__ = ['gaussin', 
#            'uniform', 
#            'shuffle',
#            'randn',
#            'rand',
#            'randint']

from ..fluid import core
from ..fluid.framework import device_guard, in_dygraph_mode, _varbase_creator, Variable
from ..fluid.layers.layer_function_generator import templatedoc
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ..fluid.layers import uniform_random, utils
from ..fluid.layers.tensor import fill_constant

__all__ = ['randperm', 'randint', 'rand']


def randint(low,
            high=None,
            shape=None,
            out=None,
            dtype=None,
            device=None,
            stop_gradient=False,
            name=None):
    """
    This function returns a Tensor filled with random integers from the "discrete uniform" distribution of the
    specified data type in the interval [low, high). If high is None (the default), then results are from [0, low).

    Args:
        low (int): The lower bound on the range of random values to generate, the low is included in the range.
            (unless high=None, in which case this parameter is one above the highest such integer).
        high (int, optional): The upper bound on the range of random values to generate, the high is excluded 
            in the range. Default None(see above for behavior if high=None).
        shape (list|tuple|Variable, optional): The shape of the output Tensor,  if the shape is a list or tuple, 
                                     its elements can be an integer
                                     or a Tensor with the shape [1], and the type of the Tensor must be int32 or int64. 
                                     If the shape is a Variable, it is a 1-D Tensor, and the type of the Tensor must be 
                                     int32 or int64. Default is None, in which case the shape is [1].
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): Data type of the output Tensor
            which can be int32, int64, if dytpe is `None`, the data
            type of created Tensor is `int64`
        device(str, optional): This parameter specifies that the Tensor is created 
            on the GPU or CPU.
        stop_gradient(bool, optional): Indicating if we stop gradient from current(out) Variable,
            default value is False.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns: 
        Variable: A Tensor of the specified shape filled with random integers.

    Raises:
        TypeError: Randint's low must less then high.

    Examples:
        .. code-block:: python
            import paddle
            import paddle.tensor as tensor

            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            result_1 = paddle.randint(low=-5, high=5, shape=[3, 4], dtype="int64")

            # example 2:
            # attr shape is a list which contains tensor Variable.
            dim_1 = fluid.layers.fill_constant([1],"int64",3)
            dim_2 = fluid.layers.fill_constant([1],"int32",5)
            result_2 = paddle.randint(low=-5, high=5, shape=[dim_1, dim_2], dtype="int32")

            # example 3:
            # attr shape is a Variable, the data type must be int64 or int32.
            var_shape = fluid.data(name='var_shape', shape=[2], dtype="int64")
            result_3 = padddle.randint(low=-5, high=5, shape=var_shape, dtype="int32")
            var_shape_int32 = fluid.data(name='var_shape_int32', shape=[2], dtype="int32")
            result_4 = paddle.randint(low=-5, high=5, shape=var_shape_int32, dtype="int64")

            # example 4:
            # Input only one parameter
            # low=0, high=10, shape=[1], dtype='int64'
            result_4 = paddle.randint(10)
     """

    def get_new_shape_tensor(list_shape):
        new_shape_tensor = []
        for dim in list_shape:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_shape_tensor.append(dim)
            else:
                assert isinstance(dim, int) or isinstance(dim, long)
                temp_out = helper.create_variable_for_type_inference('int64')
                fill_constant([1], 'int64', dim, force_cpu=True, out=temp_out)
                new_shape_tensor.append(temp_out)
        return new_shape_tensor

    def get_attr_shape(list_shape):
        unk_dim_idx = -1
        attrs_shape = []
        for dim_idx, dim_size in enumerate(list_shape):
            if isinstance(dim_size, Variable):
                attrs_shape.append(-1)
            else:
                attrs_shape.append(dim_size)
                assert dim_size > 0, (
                    "Each dimension size given in shape must not be negative "
                    "except one unknown dimension.")
        return attrs_shape

    if dtype is None:
        dtype = 'int64'
    check_dtype(dtype, 'dtype', ['int32', 'int64'], 'randint')

    inputs = dict()
    attrs = dict()

    if shape is None:
        shape = [1]
        assert len(shape) > 0, ("The size of argument(shape) can't be zero.")

    helper = LayerHelper("randint", **locals())

    if in_dygraph_mode():
        attrs['shape'] = shape
    else:
        if isinstance(shape, Variable):
            shape.stop_gradient = True
            inputs["ShapeTensor"] = shape
        elif isinstance(shape, (list, tuple)):
            assert len(shape) > 0, (
                "The size of argument(shape) can't be zero.")
            if utils._contain_var(shape):
                inputs['ShapeTensorList'] = get_new_shape_tensor(shape)
            else:
                attrs["shape"] = get_attr_shape(shape)
    check_type(shape, 'shape', (list, tuple, Variable), 'randint')

    if high is None:
        high = low
        low = 0
    attrs['low'] = low
    attrs['high'] = high
    if (low >= high):
        raise ValueError(
            "randint's low must less then high, but received low = {0}, "
            "high = {1}".format(low, high))

    if out is None:
        if name is None:
            out = helper.create_variable_for_type_inference(dtype=dtype)
        else:
            out = helper.create_variable(
                name=name, dtype=dtype, persistable=False)
    else:
        check_dtype(dtype, 'dtype',
                    convert_dtype(out.dtype), 'randint',
                    "(The dtype in randint must be the same with out's dtype.)")
    attrs['dtype'] = out.dtype
    out.stop_gradient = stop_gradient

    if device is None:
        helper.append_op(
            type='randint', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    else:
        with device_guard(device):
            helper.append_op(
                type='randint',
                inputs=inputs,
                outputs={'Out': out},
                attrs=attrs)
    return out


@templatedoc()
def randperm(n,
             out=None,
             dtype="int64",
             device=None,
             stop_gradient=True,
             seed=0):
    """
    ${comment}

    Args:
        n (int): The upper bound (exclusive), and it should be greater than 0.
        out (Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            If out is None, a new Varibale will be create to store the result. 
            Default: None.
        dtype (np.dtype|core.VarDesc.VarType|str, optional): The type of the 
            output Tensor. Supported data types: int64, int32. Default: int32.
        device (str, optional): Specific the output variable to be saved in cpu
            or gpu memory. Supported None, 'cpu', 'gpu'. If it is None, the output
            variable will be automatically assigned devices.
            Default: None.
        stop_gradient (bool, optional): Whether grad should record operations 
            on the returned tensor. Default: True.
        seed (int, optional): Random seed used for permute samples. If seed is 
            equal to 0, it means use a seed generated by the system. Note that 
            if seed is not 0, this operator will always generate the same random 
            permutation every time. Default: 0.

    Returns:
        ${out_comment}.

    Return Type:
        ${out_type}

    Examples:
        .. code-block:: python

	    import paddle
	    import paddle.fluid as fluid

	    num = 6
	    is_use_gpu = False

	    data_1 = paddle.randperm(num)
	    fluid.layers.Print(data_1)

	    data_2 = paddle.randperm(num, dtype="int32", seed=1)
	    fluid.layers.Print(data_2)

	    data_3 = paddle.randperm(num, stop_gradient=False, device="cpu")
	    fluid.layers.Print(data_3)

	    paddle.randperm(num, out=data_3)
	    fluid.layers.Print(data_3)

	    place = fluid.CUDAPlace(0) if is_use_gpu else fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())
	    exe.run()
 
    """

    if n < 1:
        raise ValueError("The input n should be greater than 0 in randperm op.")
    check_dtype(dtype, 'dtype', ['int64', 'int32'], 'randperm')
    dtype = convert_dtype(dtype)
    if device not in [None, 'cpu', 'gpu']:
        raise ValueError("The input device should in [None, 'cpu', 'gpu'].")
    check_type(stop_gradient, 'stop_gradient', bool, 'randperm')

    helper = LayerHelper("randperm", **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        check_variable_and_dtype(out, 'out', [dtype], 'randperm')
    if stop_gradient:
        out.stop_gradient = True
    inputs = dict()
    outputs = {'Out': [out]}
    attrs = {'n': n, 'dtype': out.dtype, 'seed': seed}
    with device_guard(device):
        helper.append_op(
            type='randperm', inputs=inputs, outputs=outputs, attrs=attrs)
    return out


def rand(shape, out=None, dtype=None, device=None, stop_gradient=True):
    """
    This OP initializes a variable with random values sampled from a
    uniform distribution in the range [0, 1).

    Examples:
    ::

        Input:
          shape = [1, 2]

        Output:
          result=[[0.8505902, 0.8397286]]

    Args:
        shape(list|tuple|Variable): Shape of the Tensor to be created.
                The data type is ``int32`` or ``int64`` . If ``shape`` is a list or tuple,
                the elements of it should be integers or Tensors with shape [1].
                If ``shape`` is an Variable, it should be an 1-D Tensor .
        out(Variable, optional): Optional output which can be any created
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): Data type of the output tensor
            which can be float32, float64, if dytpe is `None`, the data
            type of created tensor is `float32`
        device(str, optional): This parameter specifies that the Tensor is created
            on the GPU or CPU.
        stop_gradient(bool, optional): Indicating if we stop gradient from current(out) Variable,
            default value is True.
    Returns:
        Variable: A Tensor of the specified shape filled with random numbers from a uniform distribution on the interval [0, 1).

    Raises:
        TypeError: The shape type should be list or tupple or Variable.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            result_1 = paddle.rand(shape=[3, 4])

            # example 2:
            # attr shape is a list which contains tensor Variable.
            dim_1 = fluid.layers.fill_constant([1],"int64",3)
            dim_2 = fluid.layers.fill_constant([1],"int32",5)
            result_2 = paddle.rand(shape=[dim_1, dim_2])

            # example 3:
            # attr shape is a Variable, the data type must be int64 or int32.
            var_shape = fluid.data(name='var_shape', shape=[2], dtype="int64")
            result_3 = paddle.rand(var_shape)
            var_shape_int32 = fluid.data(name='var_shape_int32', shape=[2], dtype="int32")
            result_4 = paddle.rand(var_shape_int32)
    """

    if dtype is None:
        dtype = 'float32'

    check_dtype(dtype, 'create data type', ['float32', 'float64'], 'rand')
    check_type(shape, 'shape', (Variable, list, tuple), 'rand')
    if device not in [None, 'cpu', 'gpu']:
        raise ValueError("The input device should in [None, 'cpu', 'gpu'].")

    helper = LayerHelper("rand", **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)

    out.stop_gradient = stop_gradient

    with device_guard(device):
        out = uniform_random(shape, dtype, min=0., max=1.0)
    return out
