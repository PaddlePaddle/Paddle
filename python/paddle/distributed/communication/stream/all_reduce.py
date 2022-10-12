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

import paddle.fluid.framework as framework
import paddle.fluid.data_feeder as data_feeder
import paddle.fluid.layer_helper as layer_helper
from paddle.distributed.communication.reduce import _get_reduce_op, ReduceOp
from paddle.distributed.communication.group import _get_global_group


def _all_reduce_in_dygraph(tensor, op, group, sync_op, use_calc_stream):
    op_type = _get_reduce_op(op, "all_reduce")

    group = _get_global_group() if group is None else group
    if use_calc_stream:
        return group.process_group.allreduce_on_calc_stream(tensor, op_type)

    task = group.process_group.allreduce(tensor, op_type, sync_op)
    if sync_op:
        task.wait()

    return task


def _all_reduce_in_static_mode(tensor, op, group, sync_op, use_calc_stream):
    data_feeder.check_variable_and_dtype(tensor, 'tensor', [
        'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8',
        'bool'
    ], 'all_reduce')

    op_type = _get_reduce_op(op, "all_reduce")
    ring_id = 0 if group is None else group.id

    if not isinstance(ring_id, int):
        raise ValueError("The type of 'ring_id' for all_reduce should be int.")

    # TODO: Support task and use task.wait in static mode
    #       Use use_calc_stream rather than sync_op
    helper = layer_helper.LayerHelper(op_type, **locals())
    helper.append_op(type=op_type,
                     inputs={'X': [tensor]},
                     outputs={'Out': [tensor]},
                     attrs={
                         'ring_id': ring_id,
                         'use_calc_stream': sync_op
                     })

    return None


def all_reduce(tensor,
               op=ReduceOp.SUM,
               group=None,
               sync_op=True,
               use_calc_stream=False):
    """

    Perform specific reduction (for example, sum, max) on inputs across devices.

    Args:
        tensor (Tensor): The input tensor on each rank. The result will overwrite this tenor after communication. Support
            float16, float32, float64, int32, int64, int8, uint8 or bool as the input data type.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD, optional): The reduction used. If none is given, use ReduceOp.SUM as default.
        group (Group, optional): Communicate in which group. If none is given, use the global group as default.
        sync_op (bool, optional): Indicate whether the communication is sync or not. If none is given, use true as default.
        use_calc_stream (bool, optional): Indicate whether the communication is done on calculation stream. If none is given, use false as default. This
            option is designed for high performance demand, be careful to turn it on except you are clearly know its meaning.

    Returns:
        Return a task object.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            local_rank = dist.get_rank()
            data = None
            if local_rank == 0:
                data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
            else:
                data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            task = dist.stream.all_reduce(data, sync_op=False)
            task.wait()
            out = data.numpy()
            # [[5, 7, 9], [5, 7, 9]]
    """
    if group is not None and not group.is_member():
        raise RuntimeError(
            "The group should not be None and all ranks which invoke this operation should be the member of this group."
        )

    if not sync_op and use_calc_stream:
        raise RuntimeError(
            "use_calc_stream can only be true in sync op behavior.")

    if framework.in_dygraph_mode():
        return _all_reduce_in_dygraph(tensor, op, group, sync_op,
                                      use_calc_stream)
    else:
        return _all_reduce_in_static_mode(tensor, op, group, sync_op,
                                          use_calc_stream)
