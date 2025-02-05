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

import warnings
from typing import TYPE_CHECKING

import paddle
import paddle.distributed as dist
from paddle import framework
from paddle.base import data_feeder
from paddle.distributed.communication.group import (
    _get_global_group,
    _get_or_throw_group_rank,
    _warn_cur_rank_not_in_group,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor
    from paddle.base.core import task
    from paddle.distributed.communication.group import Group


def _scatter_tensor_in_dygraph(
    out_tensor, in_tensor, src_rank_in_group, group, sync_op, use_calc_stream
):
    nranks = group.nranks

    if use_calc_stream:
        return group.process_group.scatter_tensor_on_calc_stream(
            out_tensor, in_tensor, src_rank_in_group
        )

    task = group.process_group.scatter_tensor(
        out_tensor, in_tensor, src_rank_in_group, sync_op
    )
    if sync_op:
        task.wait()

    return task


def _scatter_in_dygraph(
    tensor, tensor_list, src_rank_in_group, group, sync_op, use_calc_stream
):
    nranks = group.nranks
    if group.rank == src_rank_in_group:
        if len(tensor_list) == 0:
            raise RuntimeError(
                "The tensor_list should not be empty on src rank."
            )
    else:
        tensor_list = [tensor for _ in range(nranks)]

    if use_calc_stream:
        return group.process_group.scatter_on_calc_stream(
            tensor, tensor_list, src_rank_in_group
        )

    task = group.process_group.scatter(
        tensor, tensor_list, src_rank_in_group, sync_op
    )
    if sync_op:
        task.wait()

    return task


def _scatter_in_static_mode(
    tensor,
    tensor_or_tensor_list,
    src_rank_in_group,
    group,
    sync_op,
    use_calc_stream,
):
    nranks = dist.get_world_size() if group is None else group.nranks
    rank = dist.get_rank()

    input_tensor = tensor_or_tensor_list
    if isinstance(tensor_or_tensor_list, list):
        tensor_list = tensor_or_tensor_list
        if rank == src_rank_in_group:
            if len(tensor_list) == 0:
                raise RuntimeError(
                    "The tensor_list should not be empty on src rank."
                )
        else:
            tensor_list = [tensor for _ in range(nranks)]
        # 0-D use stack/unstack while others use concat/split
        if len(tensor_list[0].shape) == 0:
            input_tensor = paddle.stack(tensor_list, axis=0)
        else:
            input_tensor = paddle.concat(tensor_list, axis=0)

    ring_id = 0 if group is None else group.id

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
        'scatter',
    )

    op_type = 'c_scatter'
    helper = framework.LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [input_tensor]},
        outputs={'Out': [tensor]},
        attrs={
            'ring_id': ring_id,
            'root': src_rank_in_group,
            'use_calc_stream': sync_op,
            'nranks': nranks,
        },
    )


def scatter(
    tensor: Tensor,
    tensor_or_tensor_list: Tensor | Sequence[Tensor] | None = None,
    src: int = 0,
    group: Group | None = None,
    sync_op: bool = True,
    use_calc_stream: bool = False,
) -> task | None:
    """

    Scatter a tensor (or a tensor list) across devices.

    Args:
        tensor (Tensor): The output tensor on each rank. The result will overwrite this tenor after communication. Support
            float16, float32, float64, int32, int64, int8, uint8 or bool as the input data type.
        tensor_or_tensor_list (Union[Tensor, List[Tensor]]): The input to scatter (default is `None`, must be specified on the source rank).
            If it is a tensor, it should be correctly-sized. If it is a list, it should contain correctly-sized tensors.
        src (int, optional): Rank of the source device. If none is given, use `0` as default.
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

    # NOTE(liyurui): Only the source rank needs to specific the tensor_or_tensor_list argument.
    # Other ranks which pass this argument in will be ignored with a warning.
    # If a tensor_list passed in, we need to concat it to a tensor before invoke C++ API.
    # If a tensor passed in, concat is not needed.
    # The passed in type for non-src rank is meaningless, for it will be ignored.
    if src != dist.get_rank():
        if tensor_or_tensor_list is not None:
            warnings.warn(
                "Specific `tensor_or_tensor_list` is meaningless for rank which is not src."
            )
        tensor_or_tensor_list = []

    if framework.in_dynamic_mode():
        group = _get_global_group() if group is None else group
        src_rank_in_group = _get_or_throw_group_rank(src, group)
        if paddle.is_tensor(tensor_or_tensor_list):
            return _scatter_tensor_in_dygraph(
                tensor,
                tensor_or_tensor_list,
                src_rank_in_group,
                group,
                sync_op,
                use_calc_stream,
            )
        else:
            return _scatter_in_dygraph(
                tensor,
                tensor_or_tensor_list,
                src_rank_in_group,
                group,
                sync_op,
                use_calc_stream,
            )
    else:
        assert (
            group is None
        ), "Group can not be used in static graph mode for now."

        return _scatter_in_static_mode(
            tensor,
            tensor_or_tensor_list,
            src,
            group,
            sync_op,
            use_calc_stream,
        )
