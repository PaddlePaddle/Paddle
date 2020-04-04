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
    #            'index_select',
    #            'nonzero',
    'sort'
]

from paddle.common_ops_import import *


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
