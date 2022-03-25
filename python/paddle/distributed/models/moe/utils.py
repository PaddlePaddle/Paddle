# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import _non_static_mode


def _number_count(numbers, upper_range):
    """
    calculate the expert count according to the gate index.
    Args:
        numbers (Tensor): Tensor. The input gate index whose data type should be int32 or int64.
        upper_range (int): The number of the experts.
    Returns:
        out (Tensor): The output expert count.
    Examples:
        .. code-block:: python
            # required: distributed
            import paddle

            numbers = [
                [0, 2],
                [0, 2]
            ]
            upper_range = 6
            numbers = paddle.to_tensor(numbers, dtype="int32")
            number_count = paddle.distributed.utils.number_count(numbers, upper_range)
            print(number_count) # the result: [2, 0, 2, 0, 0, 0]
    """
    if _non_static_mode():
        return core.ops.number_count(numbers, 'upper_range', upper_range)

    else:
        op_type = 'number_count'

        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=numbers.dtype)

        helper.append_op(
            type=op_type,
            inputs={'numbers': numbers},
            outputs={'Out': out},
            attrs={'upper_range': upper_range})
        return out


def _assign_pos(x, cum_count):
    """
    Assign pos decides which tokens should be fetched belong to 
    specially expert orderingly.
    
    Args:
        x (Tensor): Tensor. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32 or int64.
        cum_count (Tensor): The cumulative sum tokens of counters. Every element in the list must be a Tensor whose 
            data type should be int64.
  
    Returns:
        out (Tensor): Assemble numbers in the order of counters. 
    
    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            number_count = [2, 0, 2, 0]
            numbers = [
                [0, 2],
                [0, 2]
            ]
            number_count = paddle.to_tensor(number_count)
            numbers = paddle.to_tensor(numbers, dtype="int32")
            num_cum = paddle.cumsum(number_count)
            pos = paddle.distributed.utils.assign_pos(x=numbers, cum_count=num_cum)
            print(pos) # the result: (2, 0, 3, 1)
    """
    if _non_static_mode():
        return core.ops.assign_pos(x, cum_count, cum_count[-1])
    else:
        op_type = 'assign_pos'

        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=cum_count.dtype)

        helper.append_op(
            type=op_type,
            inputs={
                'X': [x],
                'cum_count': [cum_count],
                "eff_num_len": [cum_count[-1]]
            },
            outputs={'Out': [out]})
        return out
