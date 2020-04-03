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

# TODO: define functions to get create a tensor  


# TODO: define functions to get create a tensor  
__all__ = [
    #            'create_tensor',
    #            'create_lod_tensor', 
    #            'create_random_int_lodtensor',
    #            'crop_tensor', 
    #            'diag', 'eye', 
    #            'fill_constant', 
    #            'get_tensor_from_selected_rows', 
    'linspace',
    #            'ones', 
    #            'ones_like', 
    #            'range', 
    #            'zeros', 
    #            'zeros_like', 
    'arange',
    #            'eye',
    'full',
    #            'full_like',
    #            'triu',
    #            'tril',
    #            'meshgrid'
]


def linspace(start, stop, num, dtype, out=None, device=None, name=None):
    """
    This OP return fixed number of evenly spaced values within a given interval.
    
    **NOTICE**: The output of this OP has no gradient.

    Args:
        start(float|Variable): The input :attr:`start` is start variable of range. It is a float scalar, \
            or a tensor of shape [1] with input data type float32, float64.
        stop(float|Variable): The input :attr:`stop` is start variable of range. It is a float scalar, \
            or a tensor of shape [1] with input data type float32, float64.
        num(int|Variable): The input :attr:`num` is given num of the sequence. It is an int scalar, \
            or a tensor of shape [1] with type int32.
        dtype(string): The data type of output tensor, it could be 'float32' and 'float64'.
        out (Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result. Default: None.
        device (string, optional): Which device to run the operator. The :attr:`device` must be
            None, 'cpu', 'gpu'. If :attr:`device` is None, it will be choose the device that the user set in 
            the paddle program. Default: None.
        name(str, optional): Normally there is no need for user to set this property. 
            For more information, please refer to :ref:`api_guide_Name`.Default: None.

    Returns:
        Variable, the output data type will be float32, float64.: The 1-D tensor with fixed number of evenly spaced values, \
        the data shape of this tensor is :math:`[num]` . If the :attr:`num` is set 1, the output tensor just has \
        the value with input :attr:`start`. 

    Examples:
        .. code-block:: python

             import paddle
             data = paddle.linspace(0, 10, 5, dtype='float32') # [0.0,  2.5,  5.0,  7.5, 10.0]
             data = paddle.linspace(0, 10, 1, dtype='float32') # [0.0]

    """
    helper = LayerHelper("linspace", **locals())

    if not isinstance(start, Variable):
        start = fill_constant([1], dtype, start)
    if not isinstance(stop, Variable):
        stop = fill_constant([1], dtype, stop)
    if not isinstance(num, Variable):
        num = fill_constant([1], 'int32', num)

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=start.dtype)
    else:
        check_dtype(
            out.dtype, out.name,
            convert_dtype(start.dtype), 'linspace',
            "The out data type '%s' in linspace must be the same with '%s' seted by parameter 'dtype'."
            % (out.dtype, dtype))
        if name:
            warning.warn(
                "The output Variable name of the paddle.tensor.linspace operation can only be given by parameter out or name.\
                When parameter out and name are set at the same time, out has a higher priority than name. \
                Finally, the output Variable name is same as the out name %s." %
                out.name,
                category=UserWarning,
                stacklevel=2)

    if device is not None:
        if device not in ['cpu', 'gpu']:
            raise ValueError(
                "The value of 'device' in linspace operation must be cpu or gpu, but received %s."
                % (device))
        else:
            with device_guard(device):
                helper.append_op(
                    type='linspace',
                    inputs={'Start': start,
                            'Stop': stop,
                            'Num': num},
                    outputs={'Out': [out]})
    else:
        helper.append_op(
            type='linspace',
            inputs={'Start': start,
                    'Stop': stop,
                    'Num': num},
            outputs={'Out': [out]})

    return out


def full(shape,
         fill_value,
         out=None,
         dtype=None,
         device=None,
         stop_gradient=True,
         name=None):
    """
    This function return a Tensor with the `fill_value` which size is same as `shape`
    
    Args:
        shape(list|tuple|Variable): Shape of the Tensor to be created.
                The data type is ``int32`` or ``int64`` . If ``shape`` is a list or tuple,
                the elements of it should be integers or Tensors with shape [1].
                If ``shape`` is an Variable, it should be an 1-D Tensor .
        value(float): The constant value used to initialize the Tensor to be created.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): Data type of the output tensor
            which can be float16, float32, float64, int32, int64, if dytpe is `None`, the data
            type of created tensor is `float32`
        device(str, optional): This parameter specifies that the Tensor is created 
            on the GPU or CPU.
        stop_gradient(bool, optional): Indicating if we stop gradient from current(out) Variable,
            default value is True.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.
    
    Examples:
        .. code-block:: python

          import paddle.tensor as tensor
          import paddle.fluid as fluid

          data1 = tensor.full(shape=[2,1], full_value=0, dtype='int64') # data1=[[0],[0]]
          data2 = tensor.full(shape=[2,1], full_value=5, dtype='int64', device='gpu') # data2=[[5],[5]]

          # attr shape is a list which contains Variable Tensor.
          positive_2 = fluid.layers.fill_constant([1], "int32", 2)
          data3 = tensor.full(shape=[1, positive_2], dtype='float32', full_value=1.5) # data3=[1.5, 1.5]

          # attr shape is an Variable Tensor.
          shape = fluid.layers.fill_constant([1,2], "int32", 2) # shape=[2,2]
          data4 = tensor.full(shape=shape, dtype='bool', full_value=True) # data4=[[True,True],[True,True]]
    """

    helper = LayerHelper("full", **locals())

    if dtype is None:
        dtype = 'float32'

    check_dtype(dtype, 'create data type',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'full')
    check_type(shape, 'shape', (Variable, list, tuple), 'full')

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)

    out.stop_gradient = stop_gradient

    with device_guard(device):
        out = fill_constant(shape=shape, dtype=dtype, value=fill_value, out=out)
    return out


def arange(start, end, step, dtype):
    """
    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval [start, stop) (in other words,
    the interval including start but excluding stop).

    Parameters:
        start(float32 | float64 | int32 | int64 | Variable): Start of interval. The interval includes this value.
            when start is Variable, it is a 1-D Tensor with shape [1].
        end(float32 | float64 | int32 | int64 | Variable): End of interval. The interval does not include this
                                 value, except in some cases where step is not an integer
                                 and floating point round-off affects the length of out. When end is Variable,
                                 it is a 1-D Tensor with shape [1].
        step(float32 | float64 | int32 | int64 | Variable): Spacing between values. For any output out, this is the
                                  distance between two adjacent values, out[i+1] - out[i].
        dtype(str|core.VarDesc.VarType): the data type of the output tensor, can be float32, float64, int32, int64.

    Returns: a 1-D Tensor which is evenly spaced values within a given interval. Its data type is set by dtype.
    
    Return type: Variable

    examples:

        .. code-block:: python

             import paddle
             data = paddle.arange(0, 10, 2, 'int32')

    """
    helper = LayerHelper("range", **locals())

    check_dtype(dtype, 'create data type',
                ['float32', 'float64', 'int32', 'int64'], 'range')

    dtype = convert_dtype(dtype)
    if not isinstance(start, Variable):
        start = fill_constant([1], dtype, start)
    elif convert_dtype(start.dtype) != dtype:
        # make sure that start, end, step has the same dtype as
        # `dtype`
        start = cast(x=start, dtype=dtype)

    if not isinstance(end, Variable):
        end = fill_constant([1], dtype, end)
    elif convert_dtype(end.dtype) != dtype:
        end = cast(x=end, dtype=dtype)

    return out

