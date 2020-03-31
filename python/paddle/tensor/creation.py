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
    #            'meshgrid'
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

          import paddle.fluid as fluid

          input = fluid.data(name='input', dtype='float32', shape=[3])
          data = fluid.layers.full_like(input, 2.0) # [2.0, 2.0, 2.0]

    """

    helper = LayerHelper("full_like", **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='full_like',
        inputs={'X': [input]},
        attrs={'value': fill_value},
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out
