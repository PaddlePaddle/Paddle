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
from paddle.fluid.framework import in_dygraph_mode


def _number_count(gate_idx, upper_range):
    """
    calculate the expert count according to the gate index.
    Args:
        gate_idx (Tensor): Tensor. The input gate index whose data type should be int32 or int64.
        upper_range (int): The number of the experts.
    Returns:
        out (Tensor): The output expert count.
    Examples:
        .. code-block:: python
            # required: distributed
            import paddle

            gate_idx = [
                [0, 2],
                [0, 2]
            ]
            upper_range = 6
            gate_idx = paddle.to_tensor(gate_idx, dtype="int32")
            number_count = paddle.distributed.utils.number_count(gate_idx, upper_range)
            print(number_count) # the result: [2, 0, 2, 0, 0, 0]
    """
    if in_dygraph_mode():
        return core.ops.number_count(gate_idx, 'upper_range', upper_range)
    else:
        op_type = 'number_count'

        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=gate_idx.dtype)

        helper.append_op(
            type=op_type,
            inputs={'gate_idx': gate_idx},
            outputs={'Out': out},
            attrs={'upper_range': upper_range})
        return out


def _random_routing(topk_idx, topk_value, prob, topk=2):
    r"""
        random routing topk gate idx
        ```
            out = topk_idx
            for i in len(topk_idx):
                if topk * value[i][topk-1] < prob[i]:
                    out[i][topk-1] = -1
        ```
        Args:
            topk_idx: gate idx, shape=(N, topk)
            topk_value: values, shape = topk_idx.shape
            prob: random prob, shape=(topk_idx.shape[0],)
    """
    if topk == 2:
        if in_dygraph_mode():
            return core.ops.random_routing(prob, topk_value, topk_idx)
        else:
            raise RuntimeError("Not supporting static mode now")
    else:
        raise RuntimeError("only topk=2 is supported now")
