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

import warnings

import paddle
import paddle.distributed as dist
from paddle import framework
from paddle.distributed.communication.group import (
    _get_global_group,
    _get_or_throw_group_rank,
    _warn_cur_rank_not_in_group,
)


def _gather_in_dygraph(
    tensor, gather_list, dst_rank_in_group, group, sync_op, use_calc_stream
):
    nranks = group.nranks
    if group.rank == dst_rank_in_group:
        if len(gather_list) == 0:
            gather_list += [paddle.empty_like(tensor) for _ in range(nranks)]
    else:
        gather_list = [tensor for _ in range(nranks)]

    assert (
        len(gather_list) == nranks
    ), f" gather_list length {len(gather_list)} and nrankd {nranks} not equal"

    task = group.process_group.gather(
        tensor, gather_list, dst_rank_in_group, sync_op, use_calc_stream
    )

    if sync_op:
        task.wait()

    return task


def gather(
    tensor,
    gather_list=None,
    dst=0,
    group=None,
    sync_op=True,
    use_calc_stream=False,
):
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
        use_calc_stream (bool, optional): Indicate whether the communication is done on calculation stream. If none is given, use false as default. This
            option is designed for high performance demand, be careful to turn it on except you are clearly know its meaning.

    Returns:
        Async work handle,which can be wait on, if async_op is set to True.
        None, if not async_op

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> gather_list = []
            >>> if dist.get_rank() == 0:
            ...     data = paddle.to_tensor([1, 2, 3])
            ...     dist.stream.gather(data, gather_list, dst=0)
            >>> else:
            ...     data = paddle.to_tensor([4, 5, 6])
            ...     dist.stream.gather(data1, gather_list, dst=0)
            >>> print(gather_list)
            >>> # [[1, 2, 3], [4, 5, 6]] (2 GPUs, out for rank 0)
            >>> # [] (2 GPUs, out for rank 1)
    """

    assert (
        framework.in_dynamic_mode()
    ), "gather doesn't support static graph mode yet."

    if _warn_cur_rank_not_in_group(group):
        return

    if not sync_op and use_calc_stream:
        raise RuntimeError(
            "use_calc_stream can only be true in sync op behavior."
        )

    # NOTE(liuzhenhai): Only the dst rank needs to specific the gather_list argument.
    # Other ranks which pass this argument in will be ignored with a warning.
    # The passed in type for non-dst rank is meaningless, for it will be ignored.
    if dst != dist.get_rank():
        if gather_list is not None:
            warnings.warn(
                "Specific `gather_list` is meaningless for rank which is not dst."
            )
        gather_list = []
    else:
        assert (
            gather_list is not None
        ), "gather_list must not be none for dst rank"

    group = _get_global_group() if group is None else group
    dst_rank_in_group = _get_or_throw_group_rank(dst, group)
    return _gather_in_dygraph(
        tensor, gather_list, dst_rank_in_group, group, sync_op, use_calc_stream
    )
