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

from typing import TYPE_CHECKING, Any

import paddle
from paddle.distributed.communication import stream
from paddle.distributed.communication.group import (
    _get_global_group,
    _warn_cur_rank_not_in_group,
)
from paddle.distributed.communication.serialization_utils import (
    convert_object_to_tensor,
)

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle.base.core import task
    from paddle.distributed.communication.group import Group


def send(
    tensor: Tensor,
    dst: int = 0,
    group: Group | None = None,
    sync_op: bool = True,
) -> task | None:
    """
    Send a tensor to the receiver.

    Args:
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16.
        dst (int): The destination rank id.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
        sync_op (bool, optional): Whether this op is a sync op. The default value is True.

    Returns:
        Return a task object.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> if dist.get_rank() == 0:
            ...     data = paddle.to_tensor([7, 8, 9])
            ...     dist.send(data, dst=1)
            >>> else:
            ...     data = paddle.to_tensor([1, 2, 3])
            ...     dist.recv(data, src=0)
            >>> print(data)
            >>> # [7, 8, 9] (2 GPUs)
    """
    return stream.send(
        tensor, dst=dst, group=group, sync_op=sync_op, use_calc_stream=False
    )


def isend(tensor: Tensor, dst: int, group: Group | None = None) -> task | None:
    """
    Send tensor asynchronously

    Args:
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16.
        dst (int): The destination rank.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.

    Returns:
        Return a task object.

    Warning:
        This API only supports the dygraph mode.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> if dist.get_rank() == 0:
            ...     data = paddle.to_tensor([7, 8, 9])
            ...     task = dist.isend(data, dst=1)
            >>> else:
            ...     data = paddle.to_tensor([1, 2, 3])
            ...     task = dist.irecv(data, src=0)
            >>> task.wait()  # type: ignore[union-attr]
            >>> print(data)
            >>> # [7, 8, 9] (2 GPUs)

    """
    return send(tensor, dst, group, sync_op=False)


def send_object_list(
    object_list: list[Any],
    dst: int | None = None,
    group: Group | None = None,
    dst_in_group: int | None = None,
):
    """
    Send a list of Python objects to the receiver.

    Args:
        object_list (list): The list of Python objects to send.
        dst (int, optional): The destination rank id. Default: 0.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
        dst_in_group (int, optional): The destination rank within the group. Cannot be specified together with dst. Default: None.

    Returns:
        This function does not return any value.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> if dist.get_rank() == 0:
            ...     data = ["hello", {"key": 100}, [1, 2, 3]]
            ...     dist.send_object_list(data, dst=1)
            >>> else:
            ...     data = [None] * 3  # type: ignore
            ...     dist.recv_object_list(data, src=0)
            >>> print(data)
            >>> # ["hello", {"key": 100}, [1, 2, 3]] (2 GPUs)
    """
    if object_list is None or len(object_list) == 0:
        raise ValueError("object_list cannot be None or empty")

    group = _get_global_group() if group is None else group
    if _warn_cur_rank_not_in_group(group):
        return

    if dst_in_group is not None:
        if dst is not None:
            raise ValueError(
                "Cannot specify both 'dst' and 'dst_in_group' arguments."
            )
        dst = group.get_global_rank(dst_in_group)
    else:
        dst = 0 if dst is None else dst

    # Convert objects to tensors and get their sizes
    tensor_list, size_list = zip(
        *[convert_object_to_tensor(obj) for obj in object_list]
    )
    size_list_values = [size.item() for size in size_list]

    # Send sizes first
    object_sizes_tensor = paddle.to_tensor(size_list_values, dtype='int64')
    send(object_sizes_tensor, dst=dst, group=group)

    # Send object data
    if len(tensor_list) == 1:
        object_tensor = tensor_list[0]
    else:
        object_tensor = paddle.concat(tensor_list)
    send(object_tensor, dst=dst, group=group)
