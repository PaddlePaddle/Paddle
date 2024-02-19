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

from paddle import framework
from paddle.base import data_feeder
from paddle.distributed.communication.group import (
    _get_global_group,
    _get_or_throw_group_rank,
    _warn_cur_rank_not_in_group,
)
from paddle.distributed.communication.reduce import ReduceOp, _get_reduce_op


def _reduce_in_dygraph(
    tensor, dst_rank_in_group, op, group, sync_op, use_calc_stream
):
    op_type = _get_reduce_op(op, "reduce")
    if use_calc_stream:
        return group.process_group.reduce_on_calc_stream(
            tensor, dst_rank_in_group, op_type
        )

    task = group.process_group.reduce(
        tensor, dst_rank_in_group, op_type, sync_op
    )
    if sync_op:
        task.wait()

    return task


def _reduce_in_static_mode(
    tensor, dst_rank_in_group, op, group, sync_op, use_calc_stream
):
    data_feeder.check_variable_and_dtype(
        tensor,
        'tensor',
        [
            'float16',
            'float32',
            'float64',
            'int32',
            'int64',
            'int8',
            'uint8',
            'bool',
        ],
        'reduce',
    )

    op_type = _get_reduce_op(op, "reduce")
    ring_id = 0 if group is None else group.id

    helper = framework.LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={
            'ring_id': ring_id,
            'use_calc_stream': sync_op,
            'root_id': dst_rank_in_group,
        },
    )


def reduce(
    tensor,
    dst=0,
    op=ReduceOp.SUM,
    group=None,
    sync_op=True,
    use_calc_stream=False,
):
    """

    Perform specific reduction (for example, sum, max) on a tensor across devices and send to the destination device.

    Args:
        tensor (Tensor): The input tensor on each rank. The result will overwrite this tenor after communication. Support
            float16, float32, float64, int32, int64, int8, uint8 or bool as the input data type.
        dst (int, optional): Rank of the destination device. If none is given, use `0` as default.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD, optional): The reduction used. If none is given, use ReduceOp.SUM as default.
        group (Group, optional): Communicate in which group. If none is given, use the global group as default.
        sync_op (bool, optional): Indicate whether the communication is sync or not. If none is given, use true as default.
        use_calc_stream (bool, optional): Indicate whether the communication is done on calculation stream. If none is given, use false as default. This
            option is designed for high performance demand, be careful to turn it on except you are clearly know its meaning.

    Returns:
        Return a task object.

    Warning:
        This API only supports the dygraph mode now.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> local_rank = dist.get_rank()
            >>> if local_rank == 0:
            ...     data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
            >>> else:
            ...     data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            >>> task = dist.stream.reduce(data, dst=0, sync_op=False)
            >>> task.wait()
            >>> out = data.numpy()
            >>> print(out)
            >>> # [[5, 7, 9], [5, 7, 9]] (2 GPUs, out for rank 0)
            >>> # [[1, 2, 3], [1, 2, 3]] (2 GPUs, out for rank 1)
    """
    if _warn_cur_rank_not_in_group(group):
        return

    if not sync_op and use_calc_stream:
        raise RuntimeError(
            "use_calc_stream can only be true in sync op behavior."
        )

    if framework.in_dynamic_mode():
        group = _get_global_group() if group is None else group
        dst_rank_in_group = _get_or_throw_group_rank(dst, group)
        return _reduce_in_dygraph(
            tensor, dst_rank_in_group, op, group, sync_op, use_calc_stream
        )
    else:
        assert (
            group is None
        ), "Group can not be used in static graph mode for now."
        return _reduce_in_static_mode(
            tensor, dst, op, group, sync_op, use_calc_stream
        )
