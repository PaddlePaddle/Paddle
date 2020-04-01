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
from ..fluid.framework import Variable, OpProtoHolder, in_dygraph_mode, convert_np_dtype_to_dtype_
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype

# TODO: define functions to get create a tensor  
__all__ = [
    'create_tensor',
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
    'full_like',
    #            'triu',
    #            'tril',
    #            'meshgrid',
]


def full_like(input, fill_value=0.0, out=None):
    """
    **full_like**
    This function creates a tensor filled with `fill_value` which has identical shape and dtype 
    with `input`.
    Args:
        input(Variable): The input tensor which specifies shape and dtype.
        fill_value: The value to fill the tensor with. Data type can be bool, float32, float64, int32, int64. Default value is 0.
        out(Variable): The output tensor.
    Returns:
        out(Variable): The tensor variable storing the output.
    Examples:
        .. code-block:: python
          import paddle
          import paddle.fluid as fluid
          import numpy as np

          input = fluid.data(name='input', dtype='float32', shape=[2, 3])
          output = paddle.tensor.full_like(input, 2.0)
          exe = fluid.Executor(fluid.CPUPlace())
          exe.run(fluid.default_startup_program())
          img=np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
          res = exe.run(fluid.default_main_program(), feed={'input':img}, fetch_list=[output])
          print(res) # [array([[2., 2., 2.], [2., 2., 2.]], dtype=float32)]
    """

    helper = LayerHelper("full_like", **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='fill_any_like',
        inputs={'X': [input]},
        attrs={'value': fill_value},
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out
