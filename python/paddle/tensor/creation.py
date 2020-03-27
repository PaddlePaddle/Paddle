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

# TODO: define functions to get create a tensor  
# __all__ = ['create_tensor', 
#            'create_lod_tensor', 
#            'create_random_int_lodtensor',
#            'crop_tensor', 
#            'diag', 'eye', 
#            'fill_constant', 
#            'get_tensor_from_selected_rows', 
#            'linspace', 
#            'ones', 
#            'ones_like', 
#            'range', 
#            'zeros', 
#            'zeros_like', 
#            'arrange',
#            'eye',
#            'full',
#            'linspace',
#            'full_like',
#            'triu',
#            'tril',
#            'meshgrid']


def linspace(start, stop, num, dtype, out=None, device=None):
    """
    This OP return fixed number of evenly spaced values within a given interval.

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
            if out is None, a new Varibale will be create to store the result.
        device (string, optional): Which device to run the operator. The :attr:`device` must be
        	None, 'cpu', 'gpu'. If :attr:`device` is None, it will be choose the device that the user set in 
        	the paddle program. Default: None.
    Returns:
        Variable, the output data type will be float32, float64.: The 1-D tensor with fixed number of evenly spaced values, \
        the data shape of this tensor is :math:`[num]` . If the :attr:`num` is set 1, the output tensor just has \
        the value with input :attr:`start`. 

    Examples:
        .. code-block:: python

             import paddle
             data = paddle.tensor.linspace(0, 10, 5, 'float32') # [0.0,  2.5,  5.0,  7.5, 10.0]
             data = paddle.tensor.linspace(0, 10, 1, 'float32') # [0.0]

    """
    helper = LayerHelper("linspace", **locals())

    if not isinstance(start, Variable):
        start = fluid.layers.fill_constant([1], dtype, start)
    if not isinstance(stop, Variable):
        stop = fluid.layers.fill_constant([1], dtype, stop)
    if not isinstance(num, Variable):
        num = fluid.layers.fill_constant([1], 'int32', num)

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=start.dtype)
    else:
        check_dtype(
            dtype, 'create data type',
            convert_dtype(out.dtype), 'linspace',
            '(The create data type in linspace must be the same with out data type.)'
        )

    if device is not None:
        if device not in ['cpu', 'gpu']:
            raise ValueError(
                "The value of 'device' in linspace_op must be cpu or gpu, but received %s."
                % (device))
        else:
            with fluid.device_guard(device):
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
