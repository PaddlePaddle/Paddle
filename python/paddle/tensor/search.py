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

from __future__ import print_function

from ..fluid.layer_helper import LayerHelper
from ..fluid.layers import cast, where, slice
from paddle.common_ops_import import *

# TODO: define searching & indexing functions of a tensor  
__all__ = [
    'argmax',
    #            'argmin',
    #            'argsort',
    #            'has_inf',
    #            'has_nan',
    #            'masked_select',
    #            'topk',
    #            'where',
    'index_select',
    'nonzero',
    'sort'
]


def argmax(input, axis=None, dtype=None, out=None, keepdims=False, name=None):
    """
    This OP computes the indices of the max elements of the input tensor's
    element along the provided axis.

    Args:
        input(Variable): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(input). when axis<0, it works the same way
            as axis+R. Default is None, it will use the last dim to select indices of max value.
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of the output tensor which can
                    be int32, int64. The default value is None, and it will
                    return the int64 indices.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result. Defalut is None.
        keepdims(bool, optional): Keep the axis that do the select max.
        name(str, optional): The name of output variable, normally there is no need for user to set this this property. 
            Default value is None, the framework set the name of output variable.  


    Returns:
        Variable: A Tensor with data type int64.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            in1 = np.array([[[5,8,9,5],
                            [0,0,1,7],
                            [6,9,2,4]],
                            [[5,2,4,2],
                            [4,7,7,9],
                            [1,7,0,6]]])
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(in1)
                out1 = paddle.argmax(input=x, axis=-1)
                out2 = paddle.argmax(input=x, axis=0)
                out3 = paddle.argmax(input=x, axis=1)
                out4 = paddle.argmax(input=x, axis=2)
                out5 = paddle.argmax(input=x, axis=2, keepdims=True)
                print(out1.numpy())
                # [[2 3 1]
                #  [0 3 1]]
                print(out2.numpy())
                # [[0 0 0 0]
                #  [1 1 1 1]
                #  [0 0 0 1]]
                print(out3.numpy())
                # [[2 2 0 1]
                #  [0 1 1 1]]
                print(out4.numpy())
                # [[2 3 1]
                #  [0 3 1]]
                print(out5.numpy())
                #array([[[2],
                #        [3],
                #        [1]],
                #       [[0],
                #        [3],
                #        [1]]])
    """
    helper = LayerHelper("arg_max", **locals())
    var_dtype = None
    attrs = {}
    if dtype is not None:
        check_dtype(dtype, 'create data type', ['int32', 'int64'], 'arg_max')
        var_dtype = convert_np_dtype_to_dtype_(dtype)
        attrs["dtype"] = var_dtype
    else:
        var_dtype = VarDesc.VarType.INT64
    if out is None:
        out = helper.create_variable_for_type_inference(var_dtype)
    if axis is None:
        axis = -1
    attrs['keepdims'] = keepdims
    attrs['axis'] = axis
    helper.append_op(
        type='arg_max',
        inputs={'X': input},
        outputs={'Out': [out]},
        attrs=attrs)
    out.stop_gradient = True
    return out


def index_select(input, index, dim=0):
    """
    Returns a new tensor which indexes the `input` tensor
    along dimension `dim` using the entries in `index` which
    is a Tensor.
    The returned tensor has the same number of dimensions
    as the original `input` tensor. The dim-th dimension
    has the same size as the length of `index`; other dimensions
    have the same size as in the `input` tensor. 
        .. code-block:: text
            Given:
                input.shape = [3, 4]
                input.data = [[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0],
                              [9.0, 10.0, 11.0, 12.0]]
                index.shape = [3]
                index.data = [0, 1, 1]
            
            Output when dim = 1:
                out.shape = [3, 3]
                out.data = [[1.0, 2.0, 2.0],
                            [5.0, 6.0, 6.0],
                            [9.0, 10.0, 10.0]]
            
            Output when dim = 0:
                out.shape = [3, 4]
                out.data = [[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [5.0, 6.0, 7.0, 8.0]]
    Args:
        input (Variable): The input tensor variable.
        index (Variable): The 1-D tensor containing the indices to index.
        dim (int): The dimension in which we index.
    Returns:
        A `Tensor` or `LoDTensor`. The data type is same as `input`.
    Examples:
        .. code-block:: python
            import paddle
            import paddle.fluid as fluid
            x = fluid.data(name="data", shape=[None, 32, 32], dtype="float32")
            index = fluid.data(name="index", shape=[5], dtype="int32")
            out = paddle.index_select(x, index, dim=1)
    """
    helper = LayerHelper("index_select", **locals())
    out = helper.create_variable_for_type_inference(input.dtype)
    helper.append_op(
        type='index_select',
        inputs={'X': input,
                'Index': index},
        outputs={'Out': out},
        attrs={'dim': dim})
    return out


def nonzero(input, as_tuple=False):
    """
    Return a tensor containing the indices of all non-zero
    elements of the `input` tensor. If as_tuple is True, return
    a tuple of 1-D tensors, one for each dimension in `input`,
    each containing the indices (in that dimension) of all non-zero
    elements of `input`.
    Given a n-Dimensional `input` tensor with shape [x_1, x_2, ..., x_n],
    If as_tuple is False, we can get a output tensor with shape [z, n],
    where `z` is the number of all non-zero elements in the `input` tensor.
    If as_tuple is True, we can get a 1-D tensor tuple of length `n`, and
    and the shape of each 1-D tensor is [z, 1].
        .. code-block:: text
            Given:
                inputs.shape = [3, 3]
                inputs.data = [[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]]
            
            Output when as_tuple = False:
                out.shape = [3, 2]
                out.data = [[0, 0], [1, 1], [2, 2]]
            
            Output when as_tupe = True:
                out1.shape = [3, 1]
                out1.data = [[0],
                            [1],
                            [2]]
                out2.shape = [3, 1]
                out2.data = [[0],
                            [1],
                            [2]]
                out = tuple(out1, out2)
    Args:
        inputs (Variable): The input tensor variable.
        as_tuple (bool): Return type, Tensor or tuple of Tensor.
    Returns:
        Variable. The data type is int64.
    Examples:
        .. code-block:: python
            import paddle
            import paddle.fluid as fluid
            x = fluid.data(name="data", shape=[None, 32, 32], dtype="float32")
            out = paddle.nonzero(x)
    """
    cast_inputs = cast(input, 'bool')
    outs = where(cast_inputs)
    if as_tuple:
        list_out = []
        shape = input.shape
        rank = len(shape)
        if rank == 1:
            list_out.append(outs)
        else:
            for i in range(rank):
                list_out.append(
                    slice(
                        outs, axes=[rank - 1], starts=[i], ends=[i + 1]))
        return tuple(list_out)
    else:
        return outs


def sort(input, axis=-1, descending=False, out=None, name=None):
    """
    This OP sorts the input along the given axis, and returns sorted output
    data Varibale and its corresponding index Variable with the same shape as
    :attr:`input`.
    
    **NOTICE**: The Variable in the output of this OP has gradient. You could\
        set Variable :attr:`stop_gradient`.
    Args:
        input(Variable): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is 0.
        descending(bool, optional) : Descending is a flag, if set to true,
            algorithm will sort by descending order, else sort by
            ascending order. Default is false.
        out(Variable, optional): The default value is None. Optional output 
            which can be any created Variable that meets the requirements to
            store the result of operation. if out is None, a new Varibale will
            be create to store the result.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        tuple: A tuple of sorted data Variable(with the same shape and data
        type as input) and the sorted indices(with the same shape as input's
        and with data type int64).
    Examples:
        .. code-block:: python
            import paddle
            import paddle.fluid as fluid
            import numpy as np
            in1 = np.array([[[5,8,9,5],
                            [0,0,1,7],
                            [6,9,2,4]],
                            [[5,2,4,2],
                            [4,7,7,9],
                            [1,7,0,6]]]).astype(np.float32)
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(in1)
                out1 = paddle.sort(input=x, axis=-1)
                out2 = paddle.sort(input=x, axis=0)
                out3 = paddle.sort(input=x, axis=1)
                print(out1[0].numpy())
                # [[[5. 5. 8. 9.]
                #   [0. 0. 1. 7.]
                #   [2. 4. 6. 9.]]
                #  [[2. 2. 4. 5.]
                #   [4. 7. 7. 9.]
                #   [0. 1. 6. 7.]]]
                print(out1[1].numpy())
                # [[[0 3 1 2]
                #   [0 1 2 3]
                #   [2 3 0 1]]
                #  [[1 3 2 0]
                #   [0 1 2 3]
                #   [2 0 3 1]]]
                print(out2[0].numpy())
                # [[[5. 2. 4. 2.]
                #   [0. 0. 1. 7.]
                #   [1. 7. 0. 4.]]
                #  [[5. 8. 9. 5.]
                #   [4. 7. 7. 9.]
                #   [6. 9. 2. 6.]]]
                print(out3[0].numpy())
                # [[[0. 0. 1. 4.]
                #   [5. 8. 2. 5.]
                #   [6. 9. 9. 7.]]
                #  [[1. 2. 0. 2.]
                #   [4. 7. 4. 6.]
                #   [5. 7. 7. 9.]]]
    """
    helper = LayerHelper("sort", **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(
            dtype=input.dtype, stop_gradient=False)
    ids = helper.create_variable_for_type_inference(
        VarDesc.VarType.INT64, stop_gradient=True)
    helper.append_op(
        type='argsort',
        inputs={'X': input},
        outputs={'Out': out,
                 'Indices': ids},
        attrs={'axis': axis,
               'descending': descending})
    return out, ids
