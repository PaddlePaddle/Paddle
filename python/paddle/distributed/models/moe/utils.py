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

__all__ = [  #noqa
    'expert_count'
]


def expert_count(gate_idx, n_expert):
    """
    calculate the expert count according to the gate index.
    Args:
        gate_idx (Tensor): Tensor. The input gate index whose data type should be int32 or int64.
        n_expert (int): The number of the experts.
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
            n_expert = 6
            gate_idx = paddle.to_tensor(gate_idx, dtype="int32")
            expert_count = paddle.distributed.utils.expert_count(gate_idx, n_expert)
            print(expert_count) # the result: [2, 0, 2, 0, 0, 0]
    """
    if in_dygraph_mode():
        return core.ops.expert_count(gate_idx, 'n_expert', n_expert)
    else:
        op_type = 'expert_count'

        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=gate_idx.dtype)

        helper.append_op(
            type=op_type,
            inputs={'gate_idx': gate_idx},
            outputs={'Out': out},
            attrs={'n_expert': n_expert})
        return out
