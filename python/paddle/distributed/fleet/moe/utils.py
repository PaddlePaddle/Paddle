# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#
# The file has been adapted from the file:
#     https://github.com/laekov/fastmoe/blob/master/fmoe/functions.py
#     Git commit hash: 295a615aacce7e54a37e7935274ba15e901c78e4
# We retain the following license from the original files:
#     Copyright 2021, Jiaao He. All rights reserved.
#   Licensed under the Apache License, Version 2.0 (the "License").

import paddle
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import _non_static_mode, _in_legacy_dygraph, in_dygraph_mode
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle import _C_ops


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
    if in_dygraph_mode():
        return _C_ops.number_count(numbers, 'upper_range', upper_range)
    elif _in_legacy_dygraph():
        return core.ops.number_count(numbers, 'upper_range', upper_range)
    else:
        op_type = 'number_count'

        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=numbers.dtype)

        helper.append_op(type=op_type,
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
    if in_dygraph_mode():
        return _C_ops.assign_pos(x, cum_count, cum_count[-1])
    elif _in_legacy_dygraph():
        return core.ops.assign_pos(x, cum_count, cum_count[-1])
    else:
        op_type = 'assign_pos'

        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=cum_count.dtype)

        helper.append_op(type=op_type,
                         inputs={
                             'X': [x],
                             'cum_count': [cum_count],
                             "eff_num_len": [cum_count[-1]]
                         },
                         outputs={'Out': [out]})
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
            return _C_ops.random_routing(prob, topk_value, topk_idx)
        elif _in_legacy_dygraph():
            return core.ops.random_routing(prob, topk_value, topk_idx)
        else:
            raise RuntimeError("Not supporting static mode now")
    else:
        raise RuntimeError("only topk=2 is supported now")


def _limit_by_capacity(expert_count, capacity, n_worker):
    """
    limit the expert count by capacity.
    Args:
        expert_count (Tensor): Tensor. The input expert count whose data type should be int32 or int64.
        capacity (Tensor): Tensor. The input capacity whose data type should be int32 or int64 and the elements of capacity should be the same with expert_count.numel()/n_work.
        n_work (int): The number of the works.
    Returns:
        out (Tensor): The output expert count limit by capacity.
    Examples:
        .. code-block:: python
            # required: distributed
            import paddle
            expert_count = [1, 2, 2, 8, 3, 6]
            capacity = [5, 5, 5]
            n_work = 2
            expert_count = paddle.to_tensor(expert_count, dtype="int32")
            capacity = paddle.to_tensor(capacity, dtype="int32")
            out = paddle.distributed.utils.limit_by_capacity(expert_count, capacity, n_work)
            print(out) # the result: [1, 2, 2, 4, 3, 3]
    """
    if in_dygraph_mode():
        return _C_ops.limit_by_capacity(expert_count, capacity, 'n_worker',
                                        n_worker)
    elif _in_legacy_dygraph():
        return core.ops.limit_by_capacity(expert_count, capacity, 'n_worker',
                                          n_worker)
    else:
        op_type = 'limit_by_capacity'

        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(
            dtype=expert_count.dtype)

        helper.append_op(type=op_type,
                         inputs={
                             'expert_count': expert_count,
                             'capacity': capacity
                         },
                         outputs={'Out': out},
                         attrs={'n_worker': n_worker})
        return out


def _prune_gate_by_capacity(gate_idx, expert_count, n_expert, n_worker):
    """
    prune gate by capacity(only support CUDA)

    Args:
        gate_idx (Tensor): Represents the gate_id sequence corresponding to the input data with type int32, int64.
        expert_count (Tensor): The quantity value counted on the gate_id sequence of the input data with type int32, int64.
        n_worker(int，optional): The number of workers on the trainer with type int64.
  
    Returns:
        new_gate_idx (Tensor): The gate_id sequence corresponding to the new input data after passing through prune.
    
    Examples:
        .. code-block:: python

            import paddle
            gate_idx = paddle.to_tensor([1, 3, 3, 3, 3, 2, 1, 1], dtype='int32')
            expert_count = paddle.to_tensor([0, 3, 1, 3, 0, 0, 0, 0], dtype='int32')
            n_worker = 1
            new_gate_id = paddle.distributed.utils.prune_gate_by_capacity(gate_idx, expert_count, n_expert, n_worker)
            print(new_gate_id)
            # Tensor(shape=[8], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
              [1, 3, 3, 3, -1, 2, 1, 1])
    """
    if in_dygraph_mode():
        return _C_ops.prune_gate_by_capacity(gate_idx, expert_count, "n_expert",
                                             n_expert, "n_worker", n_worker)
    elif _in_legacy_dygraph():
        return core.ops.prune_gate_by_capacity(gate_idx, expert_count,
                                               "n_expert", n_expert, "n_worker",
                                               n_worker)
    check_variable_and_dtype(gate_idx, 'GateIdx', ['int32', 'int64'],
                             'paddle.distributed.utils.prune_gate_by_capacity')
    check_variable_and_dtype(expert_count, 'ExpertCount', ['int32', 'int64'],
                             'paddle.distributed.utils.prune_gate_by_capacity')

    helper = LayerHelper('prune_gate_by_capacity', **locals())
    new_gate_idx = helper.create_variable_for_type_inference(
        dtype=gate_idx.dtype)
    helper.append_op(type='prune_gate_by_capacity',
                     inputs={
                         'GateIdx': gate_idx,
                         "ExpertCount": expert_count
                     },
                     outputs={'NewGateIdx': new_gate_idx},
                     attrs={
                         "n_expert": n_expert,
                         "n_worker": n_worker
                     })

    return new_gate_idx


def _alltoall(in_tensor_list, group=None, use_calc_stream=True):
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        group = paddle.distributed.collective._get_default_group(
        ) if group is None else group
        out = paddle.empty(in_tensor_list.shape, in_tensor_list.dtype)
        task = group.process_group.alltoall(in_tensor_list, out)
        task.wait()
        return out
    else:
        ring_id = 0 if group is None else group.id
        return paddle._C_ops.alltoall(in_tensor_list, 'use_calc_stream',
                                      use_calc_stream, 'ring_id', ring_id)


def count_by_gate(gate, num_expert, world_size, require_pos=True, group=None):
    total_expert_count = num_expert * world_size
    with paddle.no_grad():
        local_expert_count = _number_count(gate, total_expert_count)

        if world_size > 1:
            global_expert_count = _alltoall(local_expert_count, group=group)
        else:
            global_expert_count = local_expert_count
        if not require_pos:
            pos = None
        else:
            lec_cum = paddle.cumsum(local_expert_count, axis=0)
            pos = _assign_pos(gate, lec_cum)
    return pos, local_expert_count, global_expert_count


def limit_by_capacity(topk_idx, num_expert, world_size, capacity, group=None):
    with paddle.no_grad():
        capacity = paddle.ones(shape=[num_expert],
                               dtype=paddle.int64) * capacity
        pos, lec, gec = count_by_gate(topk_idx,
                                      num_expert,
                                      world_size,
                                      require_pos=False,
                                      group=group)
        new_gec = _limit_by_capacity(gec, capacity, world_size)
        if world_size > 1:
            assert group.nranks == world_size
            new_lec = _alltoall(new_gec, group=group)
        else:
            new_lec = new_gec

        topk_idx = _prune_gate_by_capacity(topk_idx, new_lec, num_expert,
                                           world_size)

    return new_lec, new_gec, topk_idx
