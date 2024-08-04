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

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle
import paddle.distributed as dist
from paddle import framework
from paddle.base import data_feeder
from paddle.distributed.communication.group import (
    _get_global_group,
    _warn_cur_rank_not_in_group,
)
from paddle.distributed.communication.reduce import ReduceOp, _get_reduce_op

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor
    from paddle.base.core import task
    from paddle.distributed.communication.group import Group
    from paddle.distributed.communication.reduce import _ReduceOp


def _reduce_scatter_tensor_in_dygraph(
    out_tensor,
    in_tensor,
    op,
    group,
    sync_op,
    use_calc_stream,
    caller="reduce_scatter",
):
    op_type = _get_reduce_op(op, caller)

    if use_calc_stream:
        return group.process_group.reduce_scatter_tensor_on_calc_stream(
            out_tensor, in_tensor, op_type
        )

    task = group.process_group.reduce_scatter_tensor(
        out_tensor, in_tensor, op_type, sync_op
    )
    if sync_op:
        task.wait()

    return task


def _reduce_scatter_in_dygraph(
    tensor, tensor_list, op, group, sync_op, use_calc_stream
):
    op_type = _get_reduce_op(op, "reduce_scatter")

    if use_calc_stream:
        return group.process_group.reduce_scatter_on_calc_stream(
            tensor, tensor_list, op_type
        )

    task = group.process_group.reduce_scatter(
        tensor, tensor_list, op_type, sync_op
    )
    if sync_op:
        task.wait()

    return task


def _reduce_scatter_in_static_mode(tensor, tensor_or_tensor_list, group):
    op_type = 'reduce_scatter'
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
            'uint16',
        ],
        op_type,
    )

    helper = framework.LayerHelper(op_type, **locals())
    ring_id = 0 if group is None else group.id
    nranks = dist.get_world_size()

    helper.append_op(
        type=op_type,
        inputs={'x': [tensor_or_tensor_list]},
        outputs={'out': [tensor]},
        attrs={
            'ring_id': ring_id,
            'nranks': nranks,
        },
    )


def reduce_scatter(
    tensor: Tensor,
    tensor_or_tensor_list: Tensor | Sequence[Tensor],
    op: _ReduceOp = ReduceOp.SUM,
    group: Group | None = None,
    sync_op: bool = True,
    use_calc_stream: bool = False,
) -> task | None:
    """

    Reduce, then scatter a tensor (or a tensor list) across devices.

    Args:
        tensor (Tensor): The output tensor on each rank. The result will overwrite this tenor after communication. Support
            float16, float32, float64, int32, int64, int8, uint8 or bool as the input data type.
        tensor_or_tensor_list (Union[Tensor, List[Tensor]]): The input to scatter.
            If it is a tensor, it should be correctly-sized. If it is a list, it should contain correctly-sized tensors.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD, optional): The reduction used. If none is given, use ReduceOp.SUM as default.
        group (Group|None, optional): Communicate in which group. If none is given, use the global group as default.
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
            >>> if dist.get_rank() == 0:
            ...     data1 = paddle.to_tensor([0, 1])
            ...     data2 = paddle.to_tensor([2, 3])
            >>> else:
            ...     data1 = paddle.to_tensor([4, 5])
            ...     data2 = paddle.to_tensor([6, 7])
            >>> dist.stream.reduce_scatter(data1, [data1, data2])
            >>> out = data1.numpy()
            >>> print(out)
            >>> # [4, 6]  (2 GPUs, out for rank 0)
            >>> # [8, 10] (2 GPUs, out for rank 1)
    """
    if _warn_cur_rank_not_in_group(group):
        return

    if not sync_op and use_calc_stream:
        raise RuntimeError(
            "use_calc_stream can only be true in sync op behavior."
        )

    if framework.in_dynamic_mode():
        group = _get_global_group() if group is None else group
        if paddle.is_tensor(tensor_or_tensor_list):
            return _reduce_scatter_tensor_in_dygraph(
                tensor,
                tensor_or_tensor_list,
                op,
                group,
                sync_op,
                use_calc_stream,
            )
        else:
            return _reduce_scatter_in_dygraph(
                tensor,
                tensor_or_tensor_list,
                op,
                group,
                sync_op,
                use_calc_stream,
            )
    else:
        assert (
            group is None
        ), "Group can not be used in static graph mode for now."
        return _reduce_scatter_in_static_mode(
            tensor, tensor_or_tensor_list, group
        )


def _reduce_scatter_base(
    out_tensor,
    in_tensor,
    op=ReduceOp.SUM,
    group=None,
    sync_op=True,
    use_calc_stream=False,
):
    """

    Reduce, then scatter a flattened tensor across devices.

    Args:
        out_tensor (Tensor): The output tensor on each rank. The result will overwrite this tenor after communication. Support
            float16, float32, float64, int32 or int64 as the input data type.
        in_tensor (Tensor): The input tensor to reduce and scatter.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD, optional): The reduction used. If none is given, use ReduceOp.SUM as default.
        group (Group, optional): Communicate in which group. If none is given, use the global group as default.
        sync_op (bool, optional): Indicate whether the communication is sync or not. If none is given, use true as default.
        use_calc_stream (bool, optional): Indicate whether the communication is done on calculation stream. If none is given, use false as default. This
            option is designed for high performance demand, be careful to turn it on except you are clearly know its meaning.

    Returns:
        Return a task object.

    Warning:
        This API will be deprecated in the future, and only supports the dygraph mode now.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> if dist.get_rank() == 0:
            ...     data1 = paddle.to_tensor([7, 8, 9])
            ...     data2 = paddle.to_tensor([10, 11, 12])
            ...     dist.stream.scatter(data1, src=1)
            >>> else:
            ...     data1 = paddle.to_tensor([1, 2, 3])
            ...     data2 = paddle.to_tensor([4, 5, 6])
            ...     dist.stream.scatter(data1, [data1, data2], src=1)
            >>> out = data1.numpy()
            >>> print(out)
            >>> # [1, 2, 3] (2 GPUs, out for rank 0)
            >>> # [4, 5, 6] (2 GPUs, out for rank 1)
    """
    if _warn_cur_rank_not_in_group(group):
        return

    if not sync_op and use_calc_stream:
        raise RuntimeError(
            "use_calc_stream can only be true in sync op behavior."
        )

    if framework.in_dynamic_mode():
        group = _get_global_group() if group is None else group
        return _reduce_scatter_tensor_in_dygraph(
            out_tensor,
            in_tensor,
            op,
            group,
            sync_op,
            use_calc_stream,
            "_reduce_scatter_base",
        )

    raise RuntimeError(
        "paddle.distributed.stream._reduce_scatter_base is only supported in dygraph mode now."
    )
