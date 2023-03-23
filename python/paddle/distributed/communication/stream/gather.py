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
import paddle.framework as framework
from paddle.distributed.communication.group import (
    _get_global_group,
    _get_or_throw_group_rank,
    _warn_cur_rank_not_in_group,
)


def _gather_in_dygraph(
        tensor, tensor_list, dst_rank_in_group, group, sync_op, use_calc_stream
):
    nranks = group.nranks
    if group.rank == dst_rank_in_group:
        if len(tensor_list) == 0:
            raise RuntimeError(
                "The tensor_list should not be empty on dst rank."
            )
    else:
        tensor_list = [tensor for _ in range(nranks)]

    if use_calc_stream:
        #TODO(liuzhenhai): to implement
        raise RuntimeError(
            "yet to be implemented."
        )

    task = group.process_group.gather(
        tensor, tensor_list, dst_rank_in_group, sync_op
    )
    if sync_op:
        task.wait()

    return task


def gather(
        tensor,
        tensor_list=None,
        dst=0,
        group=None,
        sync_op=True,
        use_calc_stream=False,
):

    # TODO(liuzhehhai):fill doc later

    assert (
        framework.in_dygraph_mode()
    ), "gather doesn't support static graph mode yet."



    if _warn_cur_rank_not_in_group(group):
        return

    if not sync_op and use_calc_stream:
        raise RuntimeError(
            "use_calc_stream can only be true in sync op behavior."
        )

    # NOTE(liuzhenhai): Only the dst rank needs to specific the tensor_list argument.
    # Other ranks which pass this argument in will be ignored with a warning.
    # The passed in type for non-dst rank is meaningless, for it will be ignored.
    if dst != dist.get_rank():
        if tensor_list is not None:
            warnings.warn(
                "Specific `tensor_list` is meaningless for rank which is not dst."
            )
        tensor_list = []
    else:
        assert (
            tensor_list is not None
        ), "tensor_list must not be none for dst rank"

    group = _get_global_group() if group is None else group
    dst_rank_in_group = _get_or_throw_group_rank(dst, group)
    return _gather_in_dygraph(
        tensor,
        tensor_list,
        dst_rank_in_group,
        group,
        sync_op,
        use_calc_stream)
