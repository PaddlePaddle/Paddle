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
__all__ = ['create_tensor', 
           'create_lod_tensor', 
           'create_random_int_lodtensor',
           'crop_tensor', 
           'diag', 'eye', 
           'fill_constant', 
           'get_tensor_from_selected_rows', 
           'linspace', 
           'ones', 
           'ones_like', 
           'range', 
           'zeros', 
           'zeros_like', 
           'arange',
           'eye',
           'full',
           'linspace',
           'full_like',
           'triu',
           'tril',
           'meshgrid']


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

             import paddle.fluid as fluid
             data = fluid.tensor.arange(0, 10, 2, 'int32')

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

    if not isinstance(step, Variable):
        step = fill_constant([1], dtype, step)
    elif convert_dtype(step.dtype) != dtype:
        step = cast(x=step, dtype=dtype)

    out = helper.create_variable_for_type_inference(dtype=start.dtype)

    helper.append_op(
        type='range',
        inputs={'Start': start,
                'End': end,
                'Step': step},
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out
