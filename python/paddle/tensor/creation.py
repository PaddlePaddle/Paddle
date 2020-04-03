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

__all__ = [
    'create_tensor',
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
    #            'arrange',
    #            'eye',
    'full',
    #            'full_like',
    'triu',
    'tril',
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
 

def _tril_triu_op(helper):
    """base op of tril_op and triu_op
    """
    op_type = helper.layer_type
    x = helper.kwargs.get('input', none)

    assert x is not none, 'x cannot be none in {}'.format(op_type)
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             op_type)
    if len(x.shape) < 2:
        raise valueerror("input shape in {} must be at least 2-d".format(
            op_type))
    diagonal = helper.kwargs.get('diagonal', 0)
    if not isinstance(diagonal, (int, )):
        raise typeerror("diagonal in {} must be a python int".format(op_type))
    name = helper.kwargs.get('name', none)

    if name is none:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=false)

    helper.append_op(
        type="tril_triu",
        inputs={"x": x},
        attrs={
            "diagonal": diagonal,
            "lower": true if op_type == 'tril' else false,
        },
        outputs={"out": out}, )

    return out


def tril(input, diagonal=0, name=none):
    """
    this op returns the lower triangular part of the matrix (2-d tensor) or batch of matrices
    :attr:`input`, the other elements of the result tensor are set to 0.
    the lower triangular part of the matrix is defined as the elements on and
    below the diagonal.

    args:
        input (variable): the input variable which is a tensor. 
            support data types: float64, float32, in32, int64.
        diagonal (int, optional): the diagonal to consider, default value is 0. 
            if :attr:`diagonal` = 0, all elements on and below the main diagonal are
            retained. a positive value includes just as many diagonals above the main
            diagonal, and similarly a negative value excludes just as many diagonals below
            the main diagonal. the main diagonal are the set of indices
            :math:`{ (i, i) }` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
            :math:`d_{1}, d_{2}` are the dimensions of the matrix.
        name (str, optional): the default value is none. normally there is no need for
            user to set this property. for more information, please refer to :ref:`api_guide_name`.

    returns:
        variable: tensor, results of upper triangular operation by the specified diagonal of input tensor,
        it's data type is the same as input's tensor.

    raises:
        typeerror: diagonal is not a int type.
        valueerror: dimension of :attr:`input` is less than 2.

     example:
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            data = np.arange(1, 13, dtype="int64").reshape(3,-1)
            # array([[ 1,  2,  3,  4],
            #        [ 5,  6,  7,  8],
            #        [ 9, 10, 11, 12]])        
            x = fluid.data(shape=(-1, 4), dtype='int64', name='x')
            exe = fluid.executor(fluid.cpuplace())

            # example 1, default diagonal
            tril = fluid.layers.tril(x)
            tril_out, = exe.run(fluid.default_main_program(), feed={"x": data}, 
                fetch_list=[tril], return_numpy=true)
            # array([[ 1,  0,  0,  0],
            #        [ 5,  6,  0,  0],
            #        [ 9, 10, 11,  0]])
            
            # example 2, positive diagonal value
            tril = fluid.layers.tril(x, diagonal=2)
            tril_out, = exe.run(fluid.default_main_program(), feed={"x": data}, 
                fetch_list=[tril], return_numpy=true)
            # array([[ 1,  2,  3,  0],
            #        [ 5,  6,  7,  8],
            #        [ 9, 10, 11, 12]])
            
            # example 3, negative diagonal value
            tril = fluid.layers.tril(x, diagonal=-1)
            tril_out, = exe.run(fluid.default_main_program(), feed={"x": data}, 
                fetch_list=[tril], return_numpy=true)
            # array([[ 0,  0,  0,  0],
            #        [ 5,  0,  0,  0],
            #        [ 9, 10,  0,  0]])
        
   """

    return _tril_triu_op(layerhelper('tril', **locals()))


def triu(input, diagonal=0, name=none):
    """
    this op returns the upper triangular part of a matrix (2-d tensor) or batch 
    of matrices :attr:`input`, the other elements of the result tensor are set to 0.
    the upper triangular part of the matrix is defined as the elements on and
    above the diagonal.

    args:
        input (variable): the input variable which is a tensor. 
            support data types: float64, float32, in32, int64.
        diagonal (int, optional): the diagonal to consider, default value is 0. 
            if :attr:`diagonal` = 0, all elements on and above the main diagonal are
            retained. a positive value excludes just as many diagonals above the main
            diagonal, and similarly a negative value includes just as many diagonals below
            the main diagonal. the main diagonal are the set of indices
            :math:`{(i, i)}` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
            :math:`d_{1}, d_{2}` are the dimensions of the matrix.
        name (str, optional): the default value is none. normally there is no need for
            user to set this property. for more information, please refer to :ref:`api_guide_name`.

    returns:
        variable: tensor, results of upper triangular operation by the specified diagonal of input tensor,
        it's data type is the same as input's tensor.

    raises:
        typeerror: diagonal is not int.
        valueerror: dimension of :attr:`input` is less than 2.

     example:
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            data = np.arange(1, 13, dtype="int64").reshape(3,-1)
            # array([[ 1,  2,  3,  4],
            #        [ 5,  6,  7,  8],
            #        [ 9, 10, 11, 12]])        
            x = fluid.data(shape=(-1, 4), dtype='int64', name='x')
            exe = fluid.executor(fluid.cpuplace())

        .. code-block:: python

            # example 1, default diagonal
            triu = fluid.layers.triu(x)
            triu_out, = exe.run(fluid.default_main_program(), feed={"x": data}, 
                fetch_list=[triu], return_numpy=true)
            # array([[ 1,  2,  3,  4],
            #        [ 0,  6,  7,  8],
            #        [ 0,  0, 11, 12]])        
            
        .. code-block:: python

            # example 2, positive diagonal value
            triu = fluid.layers.triu(x, diagonal=2)
            triu_out, = exe.run(fluid.default_main_program(), feed={"x": data}, 
                fetch_list=[triu], return_numpy=true)
            # array([[0, 0, 3, 4],
            #        [0, 0, 0, 8],
            #        [0, 0, 0, 0]])        
            
        .. code-block:: python

            # example 3, negative diagonal value
            triu = fluid.layers.triu(x, diagonal=-1)
            triu_out, = exe.run(fluid.default_main_program(), feed={"x": data}, 
                fetch_list=[triu], return_numpy=true)
            # array([[ 1,  2,  3,  4],
            #        [ 5,  6,  7,  8],
            #        [ 0, 10, 11, 12]])        
        
    """

    return _tril_triu_op(layerhelper('triu', **locals()))

