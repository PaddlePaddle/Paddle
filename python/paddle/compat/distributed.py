# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle.base.core import task
    from paddle.distributed.communication.group import Group

__all__ = [
    'all_to_all_single',
]


def all_to_all_single(
    output: Tensor,
    input: Tensor,
    output_split_sizes: list[int] | None = None,
    input_split_sizes: list[int] | None = None,
    group: Group | None = None,
    async_op: bool = False,
) -> task | None:
    """
    Split ``input`` and scatter the splits to all ranks in ``group``, then
    concatenate the splits received from all ranks into ``output``.

    Aligned with ``torch.distributed.all_to_all_single``. Compared with
    :func:`paddle.distributed.alltoall_single`, the split sizes are given in
    ``(output_split_sizes, input_split_sizes)`` order, and ``async_op``
    replaces ``sync_op``.

    Args:
        output (Tensor): The output tensor.
        input (Tensor): The input tensor to scatter.
        output_split_sizes (list[int]|None, optional): Sizes of the dim-0 splits
            received from each rank. If None, ``output`` is split equally.
            Default: None.
        input_split_sizes (list[int]|None, optional): Sizes of the dim-0 splits
            sent to each rank. If None, ``input`` is split equally. Default: None.
        group (Group|None, optional): The group to work on. If None, the default
            group is used. Default: None.
        async_op (bool, optional): Whether to run asynchronously. Default: False.

    Returns:
        A task object if ``async_op`` is True, otherwise None.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> rank = dist.get_rank()
            >>> size = dist.get_world_size()
            >>> data = paddle.arange(2, dtype='int64') + rank * 2
            >>> output = paddle.empty([2], dtype='int64')
            >>> paddle.compat.distributed.all_to_all_single(output, data)
            >>> print(output)
            >>> # [0, 2] (2 GPUs, out for rank 0)
            >>> # [1, 3] (2 GPUs, out for rank 1)
    """
    from paddle.distributed import alltoall_single

    task = alltoall_single(
        output,
        input,
        in_split_sizes=input_split_sizes,
        out_split_sizes=output_split_sizes,
        group=group,
        sync_op=not async_op,
    )
    return task if async_op else None
