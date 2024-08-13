# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import framework
from paddle.distributed.communication import stream

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle.base.core import task
    from paddle.distributed.communication.group import Group


def gather(
    tensor: Tensor,
    gather_list: list[Tensor] | None = None,
    dst: int = 0,
    group: Group | None = None,
    sync_op: bool = True,
) -> task | None:
    """

    Gather tensors from all participators.

    Args:
        tensor (Tensor): The input Tensor. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16.
        gather_list (list): A list of Tensors to hold the gathered tensors. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16. Default value is None.
        dst (int): The dst rank id. Default value is 0.
        group (Group, optional): The group instance return by new_group or None for global default group.
        sync_op (bool, optional): Whether this op is a sync op. The default value is True.

    Returns:
        Async work handle,which can be wait on, if async_op is set to True.
        None, if not async_op

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> gather_list = [] # type: ignore
            >>> if dist.get_rank() == 0:
            ...     data = paddle.to_tensor([1, 2, 3])
            ...     dist.gather(data, gather_list, dst=0)
            >>> else:
            ...     data = paddle.to_tensor([4, 5, 6])
            ...     dist.gather(data, gather_list, dst=0)
            >>> print(gather_list)
            >>> # [[1, 2, 3], [4, 5, 6]] (2 GPUs, out for rank 0)
            >>> # [] (2 GPUs, out for rank 1)
    """
    assert (
        framework.in_dynamic_mode()
    ), "gather doesn't support static graph mode yet."
    return stream.gather(tensor, gather_list, dst, group, sync_op)
