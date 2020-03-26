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

# TODO: define functions to get create a tensor  
__all__ = [#'create_tensor', 
            #'create_lod_tensor', 
            #'create_random_int_lodtensor',
            #'crop_tensor', 
            #'diag', 'eye', 
            'fill_constant',
            #'get_tensor_from_selected_rows', 
            #'linspace', 
            'ones',
            'ones_like',
            #'range', 
            'zeros',
            'zeros_like',
            #'arrange',
            #'eye',
            #'full',
            #'linspace',
            #'full_like',
            #'triu',
            #'tril',
            #'meshgrid'
            ]

from paddle.common_ops_import import *
import warnings


def fill_constant(shape, dtype, value, force_cpu=False, out=None):
    """
    This OP creates a Tensor with specified `shape` and `dtype`, and
    initializes it with a constant specified by `value`.

    Args:
        shape(list|tuple|Variable): Shape of the Tensor to be created.
                The data type is ``int32`` or ``int64`` . If ``shape`` is a list or tuple,
                the elements of it should be integers or Tensors with shape [1].
                If ``shape`` is an Variable, it should be an 1-D Tensor .
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of the output tensor which can
            be float16, float32, float64, int32, int64.
        value(float): The constant value used to initialize the Tensor to be created.
        force_cpu(True): data should be on CPU if it's true, default value is False.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.

    Returns:
        Variable: Tensor which is created according to shape and dtype.

    Raise:
        TypeError: The dtype must be one of bool, float16, float32, float64, int32 and int64
        and the data type of out Tensor must be the same as the dtype. 

    Examples:
        .. code-block:: python

          import paddle
          # attr shape is a list which doesn't contain Variable Tensor.
          data1 = paddle.tensor.fill_constant(shape=[2,1], value=0, dtype='int64') # data1=[[0],[0]]
          data2 = paddle.tensor.fill_constant(shape=[2,1], value=5, dtype='int64', out=data1)
          data3 = paddle.tensor.fill_constant(shape=[2,1], value=6, dtype='int64', out=data1)
          # data1=[[0], [0]] data2=[[5], [5]] data3=[[6], [6]]

          # attr shape is a list which contains Variable Tensor.
          positive_2 = paddle.tensor.fill_constant([1], "int32", 2)
          data3 = paddle.tensor.fill_constant(shape=[1, positive_2], dtype='float32', value=1.5) # data3=[1.5, 1.5]

          # attr shape is an Variable Tensor.
          shape = paddle.tensor.fill_constant([1,2], "int32", 2) # shape=[2,2]
          data4 = paddle.tensor.fill_constant(shape=shape, dtype='bool', value=True) # data4=[[True,True],[True,True]]
    """
    attrs = {'value': float(value), 'force_cpu': force_cpu}

    if convert_dtype(dtype) in ['int64', 'int32']:
        attrs['str_value'] = str(int(value))
    else:
        attrs['str_value'] = str(float(value))

    if in_dygraph_mode():
        if isinstance(shape, (list, tuple)):
            shape = list(
                map(lambda x: x.numpy()[0] if isinstance(x, Variable) else x,
                    shape))
        else:
            shape = list(shape.numpy().astype(int))
        attrs['shape'] = shape
        if out is None:
            out = _varbase_creator(dtype=dtype)
        attrs['dtype'] = out.dtype
        outputs = {'Out': [out]}
        outs = core.ops.fill_constant({}, attrs, outputs)
        out.stop_gradient = True
        return out

    helper = LayerHelper("fill_constant", **locals())
    check_dtype(dtype, 'create data type',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'fill_constant')
    check_type(shape, 'shape', (Variable, list, tuple), 'fill_constant')
    inputs = {}
    attrs = {'value': float(value), 'force_cpu': force_cpu}

    def _get_attr_shape(list_shape):
        attr_shape = []
        for idx, dim in enumerate(list_shape):
            if isinstance(dim, Variable):
                attr_shape.append(-1)
            else:
                attr_shape.append(dim)
        return attr_shape

    def _get_shape_tensor(list_shape):
        new_shape_tensor = []
        for idx, dim in enumerate(list_shape):
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                check_dtype(
                    dim.dtype, 'shape[' + str(idx) + ']', ['int32', 'int64'],
                    'fill_constant',
                    '(When type of shape in fill_constant is list or tuple.)')
                if convert_dtype(dim.dtype) == 'int64':
                    dim = cast(x=dim, dtype='int32')
                new_shape_tensor.append(dim)
            else:
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', dim, force_cpu=True, out=temp_out)
                new_shape_tensor.append(temp_out)
        return new_shape_tensor

    if isinstance(shape, Variable):
        shape.stop_gradient = True
        check_dtype(shape.dtype, 'shape', ['int32', 'int64'], 'fill_constant',
                    '(When type of shape in fill_constant is Variable.)')
        if (convert_dtype(shape.dtype) == 'int64'):
            shape = cast(shape, 'int32')
        inputs["ShapeTensor"] = shape
    elif isinstance(shape, (list, tuple)):
        assert len(shape) > 0, (
            "The size of 'shape' in fill_constant can't be zero, "
            "but received %s." % len(shape))
        attrs["shape"] = _get_attr_shape(shape)
        if utils._contain_var(shape):
            inputs['ShapeTensorList'] = _get_shape_tensor(shape)

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        check_dtype(
            dtype, 'create data type',
            convert_dtype(out.dtype), 'fill_constant',
            '(The create data type in fill_constant must be the same with out data type.)'
        )
    attrs['dtype'] = out.dtype
    helper.append_op(
        type='fill_constant',
        inputs=inputs,
        outputs={'Out': [out]},
        attrs=attrs,
        stop_gradient=True)
    out.stop_gradient = True
    return out


def ones(shape, dtype=None, out=None, device=None):
    """
    The OP creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 1.

    Parameters:
        shape (tuple|list): Shape of output tensor.
        dtype (np.dtype|core.VarDesc.VarType|str): Data type of output tensor, it supports
            bool, float16, float32, float64, int32 and int64.
        out (Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        device (str, optional): Which device to run the operator. The :attr:`device` must be
        None, 'cpu', 'gpu'. If :attr:`device` is None, it will be choose the device that the user set in 
        the paddle program. Default: False.

    Returns:
        Variable: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 1.

    Examples:
        .. code-block:: python

          import paddle
          data = paddle.tensor.ones(shape=[3, 2], dtype='float32') # [[1., 1.], [1., 1.], [1., 1.]]
          data = paddle.tensor.ones(shape=[2, 2], dtype='float32', device='cpu') # [[1., 1.], [1., 0.]]
    """
    check_dtype(dtype, 'create data type',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'zeros')

    if device is not None:
        if device not in ['cpu', 'gpu']:
            raise ValueError(
                "The value of 'device' in zeros_op must be cpu or gpu, but received %s."
                % (device))
        with fluid.device_guard(device):
            return fill_constant(value=1.0, shape=shape, dtype=dtype, out=out)
    return fill_constant(value=1.0, shape=shape, dtype=dtype, out=out)


def ones_like(input, dtype=None, device=None, name=None):
    """
    **ones_like**

    This function creates a ones tensor which has identical shape and dtype 
    with `input`.

    Args:
        input(Variable): The input tensor which specifies shape and dtype.The dtype of input can be 
            float32, float64, int32, int64.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The data type can be set bool, float32, float64, int32, int64. 
            The default value is None, the dtype is the same as input.
        device(str, optional): Which device to run the operator. The :attr:`device` must be
            None, 'cpu', 'gpu'. If :attr:`device` is None, it will be choose the device that the user set in 
            the paddle program. Default: False.
        name(str, optional): The name of output variable, normally there is no need for user to set this this property. 
            Default value is None, the framework set the name of output variable.  
    Returns:
        out(Variable): The tensor variable storing the output.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid

          x = fluid.layers.data(name='x', dtype='float32', shape=[3], append_batch_size=False)
          y = fluid.layers.data(name='y', dtype='float32', shape=[3], append_batch_size=False)
          data = paddle.tensor.ones_like(x) # data=[1.0, 1.0, 1.0]
          data1 = paddle.tensor.ones_like(input=x, out=y) # data1=[1.0, 1.0, 1.0]
          data2 = paddle.tensor.ones_like(input=x, out=y, device="gpu") data2=[1.0, 1.0. 1.0]

    """

    helper = LayerHelper("zeros_like", **locals())

    attrs = {"value": 0.0}
    var_dtype = None
    if dtype is not None:
        check_dtype(
            dtype, 'create data type',
            ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
            'zeros_like')
        var_dtype = convert_np_dtype_to_dtype_(dtype)
        attrs["dtype"] = var_dtype
    else:
        var_dtype = input.dtype

    out = helper.create_variable_for_type_inference(dtype=var_dtype)

    if device is not None:
        if device not in ['cpu', 'gpu']:
            raise ValueError(
                "The value of 'device' in zeros_op must be cpu or gpu, but received %s."
                % (device))
        with fluid.device_guard(device):
            helper.append_op(
                type='fill_any_like',
                inputs={'X': [input]},
                attrs=attrs,
                outputs={'Out': [out]})
            return out
    helper.append_op(
        type='fill_any_like',
        inputs={'X': [input]},
        attrs=attrs,
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out


def zeros(shape, dtype, out=None, device=None):
    """
    The OP creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 0.
    Parameters:
        shape (tuple|list): Shape of output tensor.
        dtype (np.dtype|core.VarDesc.VarType|str): Data type of output tensor, it supports
            bool, float16, float32, float64, int32 and int64.
        out (Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        device (str, optional): Which device to run the operator. The :attr:`device` must be
            None, 'cpu', 'gpu'. If :attr:`device` is None, it will be choose the device that the user set in 
            the paddle program. Default: False.

    Returns:
        Variable: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 0.

    Examples:
        .. code-block:: python

          import paddle
          data = paddle.tensor.zeros(shape=[3, 2], dtype='float32') # [[0., 0.], [0., 0.], [0., 0.]]
          data = paddle.tensor.zeros(shape=[2, 2], dtype='float32', device='cpu') # [[0., 0.], [0., 0.]]
    """
    check_dtype(dtype, 'create data type',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'zeros')
    if device is not None:
        if device not in ['cpu', 'gpu']:
            raise ValueError(
                "The value of 'device' in zeros_op must be cpu or gpu, but received %s."
                % (device))
        with fluid.device_guard(device):
            return fill_constant(value=0.0, shape=shape, dtype=dtype, out=out)

    return fill_constant(value=0.0, shape=shape, dtype=dtype, out=out)


def zeros_like(input, dtype=None, device=None, name=None):
    """
    **zeros_like**

    This function creates a ozeros tensor which has identical shape and dtype 
    with `input`.

    Args:
        input(Variable): The input tensor which specifies shape and dtype.The dtype of input can be 
            bool, float32, float64, int32, int64.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The data type can be set bool, float32, float64, int32, int64. 
            The default value is None, the dtype is the same as input.
        device(str, optional): Which device to run the operator. The :attr:`device` must be
            None, 'cpu', 'gpu'. If :attr:`device` is None, it will be choose the device that the user set in 
            the paddle program. Default: False.
        name(str, optional): The name of output variable, normally there is no need for user to set this this property. 
            Default value is None, the framework set the name of output variable.  

    Returns:
        out(Variable): The tensor variable storing the output.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid

          x = fluid.layers.data(name='x', dtype='float32', shape=[3], append_batch_size=False)
          y = fluid.layers.data(name='y', dtype='float32', shape=[3], append_batch_size=False)
          data = paddle.tensor.ones_like(x) # data=[1.0, 1.0, 1.0]
          data1 = paddle.tensor.zeros_like(input=x, out=y) # data1=[1.0, 1.0, 1.0]
          data2 = paddle.tensor.zeros_like(input=x, out=y, device="gpu") data2=[1.0, 1.0. 1.0]

    """

    helper = LayerHelper("zeros_like", **locals())

    attrs = {"value": 1.0}
    var_dtype = None
    if dtype is not None:
        check_dtype(dtype, 'create data type',
                    ['bool', 'float32', 'float64', 'int32', 'int64'],
                    'zeros_like')
        var_dtype = convert_np_dtype_to_dtype_(dtype)
        attrs["dtype"] = var_dtype
    else:
        var_dtype = input.dtype

    out = helper.create_variable_for_type_inference(dtype=var_dtype)

    if device is not None:
        if device not in ['cpu', 'gpu']:
            raise ValueError(
                "The value of 'device' in zeros_op must be cpu or gpu, but received %s."
                % (device))
        with fluid.device_guard(device):
            helper.append_op(
                type='fill_any_like',
                inputs={'X': [input]},
                attrs=attrs,
                outputs={'Out': [out]})
            return out
    helper.append_op(
        type='fill_any_like',
        inputs={'X': [input]},
        attrs=attrs,
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out
