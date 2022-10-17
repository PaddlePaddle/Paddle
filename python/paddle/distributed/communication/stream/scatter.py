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
import paddle.distributed as dist
import paddle.fluid.framework as framework
from paddle.distributed import collective


def _check_tensor_shape(tensor, shape, nranks=1):
    expect_shape = list(shape)
    expect_shape[0] //= nranks
    if list(tensor.shape) != expect_shape:
        raise RuntimeError("The in_tensor for scatter is not correctly-sized.")


def _check_tensor_list_shape(tensor_list, shape, nranks=1):
    if len(tensor_list) != nranks:
        raise RuntimeError(
            "The tensor_list for scatter is not correctly-sized.")
    for tensor in tensor_list:
        if tensor.shape != shape:
            raise RuntimeError(
                "The tensor_list for scatter is not correctly-sized.")


def _scatter_tensor_in_dygraph(out_tensor, in_tensor, src, group, sync_op,
                               use_calc_stream):
    group = collective._get_default_group() if group is None else group

    src_rank = group.get_group_rank(src)
    if src_rank == -1:
        raise RuntimeError("Src rank out of group.")

    nranks = group.nranks
    rank = dist.get_rank()
    if rank == src_rank:
        _check_tensor_shape(out_tensor, in_tensor.shape, nranks)

    if use_calc_stream:
        return group.process_group.scatter_tensor_on_calc_stream(
            in_tensor, out_tensor, src)

    task = group.process_group.scatter_tensor(in_tensor, out_tensor, src,
                                              sync_op)
    if sync_op:
        task.wait()

    return task


def _scatter_in_dygraph(tensor, tensor_list, src, group, sync_op,
                        use_calc_stream):
    group = collective._get_default_group() if group is None else group

    src_rank = group.get_group_rank(src)
    if src_rank == -1:
        raise RuntimeError("Src rank out of group.")

    nranks = group.nranks
    rank = dist.get_rank()
    if rank == src_rank:
        if len(tensor_list) == 0:
            raise RuntimeError(
                "The tensor_list should not be empty on src rank.")
        _check_tensor_list_shape(tensor_list, tensor.shape, nranks)
    else:
        tensor_list = [tensor for _ in range(nranks)]

    if use_calc_stream:
        return group.process_group.scatter_on_calc_stream(
            tensor_list, tensor, src)

    task = group.process_group.scatter(tensor_list, tensor, src, sync_op)
    if sync_op:
        task.wait()

    return task


def scatter(tensor,
            tensor_or_tensor_list=None,
            src=0,
            group=None,
            sync_op=True,
            use_calc_stream=False):
    """

    Scatter a tensor (or a tensor list) across devices.

    Args:
        tensor (Tensor): The output tensor on each rank. The result will overwrite this tenor after communication. Support
            float16, float32, float64, int32, int64, int8, uint8 or bool as the input data type.
        tensor_or_tensor_list (Union[Tensor, List[Tensor]]): The input to scatter (default is `None`, must be specified on the source rank).
            If it is a tensor, it should be correctly-sized. If it is a list, it should contain correctly-sized tensors.
        src (int, optional): Rank of the source device. If none is given, use `0` as default.
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
            if dist.get_rank() == 0:
                data1 = paddle.to_tensor([7, 8, 9])
                data2 = paddle.to_tensor([10, 11, 12])
                dist.stream.scatter(data1, src=1)
            else:
                data1 = paddle.to_tensor([1, 2, 3])
                data2 = paddle.to_tensor([4, 5, 6])
                dist.stream.scatter(data1, [data1, data2], src=1)
            out = data1.numpy()
            # [1, 2, 3] (2 GPUs, out for rank 0)
            # [4, 5, 6] (2 GPUs, out for rank 1)
    """
    if group is not None and not group.is_member():
        raise RuntimeError(
            "The group should not be None and all ranks which invoke this operation should be the member of this group."
        )

    if not sync_op and use_calc_stream:
        raise RuntimeError(
            "use_calc_stream can only be true in sync op behavior.")

    if tensor_or_tensor_list is None:
        raise RuntimeError("The input should be specified.")

    if framework.in_dygraph_mode():
        if paddle.is_tensor(tensor_or_tensor_list):
            return _scatter_tensor_in_dygraph(tensor, tensor_or_tensor_list,
                                              src, group, sync_op,
                                              use_calc_stream)
        else:
            return _scatter_in_dygraph(tensor, tensor_or_tensor_list, src,
                                       group, sync_op, use_calc_stream)

    raise RuntimeError(
        "paddle.distributed.stream.scatter is only supported in dygraph mode now."
    )
