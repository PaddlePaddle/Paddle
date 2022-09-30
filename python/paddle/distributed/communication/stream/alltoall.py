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

import paddle
import paddle.fluid.framework as framework
from paddle.distributed import collective


def _check_tensor_shape(tensor, shape, nranks=1):
    if tensor.shape != shape:
        raise RuntimeError('The tensor for alltoall is not correctly-sized.')


def _check_tensor_list_shape(tensor_list, shape, nranks=1):
    if len(tensor_list) != nranks:
        raise RuntimeError(
            'The tensor_list for alltoall is not correctly-sized.')
    for tensor in tensor_list:
        if tensor.shape != shape:
            raise RuntimeError(
                'The tensor_list for alltoall is not correctly-sized.')


def _alltoall_base_in_dygraph(out_tensor, in_tensor, group, sync_op,
                              use_calc_stream):
    group = collective._get_default_group() if group is None else group

    _check_tensor_shape(out_tensor, in_tensor.shape, group.nranks)

    if use_calc_stream:
        return group.process_group.alltoall_base_on_calc_stream(
            in_tensor, out_tensor)

    task = group.process_group.alltoall_base(in_tensor, out_tensor, sync_op)
    if sync_op:
        task.wait()

    return task


def _alltoall_in_dygraph(out_tensor_list, in_tensor_list, group, sync_op,
                         use_calc_stream):
    group = collective._get_default_group() if group is None else group

    if len(in_tensor_list) == 0:
        raise RuntimeError("The input tensor_list should not be empty.")

    if len(out_tensor_list) == 0:
        out_tensor_list += [
            paddle.empty_like(tensor) for tensor in in_tensor_list
        ]
    else:
        _check_tensor_list_shape(out_tensor_list, in_tensor_list[0].shape,
                                 group.nranks)

    if use_calc_stream:
        return group.process_group.alltoall_on_calc_stream(
            in_tensor_list, out_tensor_list)

    task = group.process_group.alltoall(in_tensor_list, out_tensor_list,
                                        sync_op)
    if sync_op:
        task.wait()

    return task


def alltoall(out_tensor_or_tensor_list,
             in_tensor_or_tensor_list,
             group=None,
             sync_op=True,
             use_calc_stream=False):
    """

    Perform specific reduction (for example, sum, max) on inputs across devices.

    Args:
        out_tensor_or_tensor_list (Union[Tensor, List[Tensor]]):
        in_tensor_or_tensor_list (Union[Tensor, List[Tensor]]):
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
        out_is_tensor = paddle.is_tensor(out_tensor_or_tensor_list)
        in_is_tensor = paddle.is_tensor(in_tensor_or_tensor_list)
        if out_is_tensor and in_is_tensor:
            return _alltoall_base_in_dygraph(out_tensor_or_tensor_list,
                                             in_tensor_or_tensor_list, group,
                                             sync_op, use_calc_stream)
        elif not out_is_tensor and not in_is_tensor:
            return _alltoall_in_dygraph(out_tensor_or_tensor_list,
                                        in_tensor_or_tensor_list, group,
                                        sync_op, use_calc_stream)
        else:
            raise RuntimeError(
                "The output and input should be both tensor or tensor list.")

    raise RuntimeError(
        "paddle.distributed.stream.alltoall is only supported in dygraph mode now."
    )
