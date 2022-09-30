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

import paddle.fluid.framework as framework
from paddle.distributed import collective


def _check_tensor_shape(tensor, shape, nranks=1):
    if tensor.shape != shape:
        raise RuntimeError(
            'The tensor for alltoall_single is not correctly-sized.')


def _alltoall_single_in_dygraph(out_tensor, in_tensor, out_split_sizes,
                                in_split_sizes, group, sync_op,
                                use_calc_stream):
    group = collective._get_default_group() if group is None else group

    _check_tensor_shape(out_tensor, in_tensor.shape, group.nranks)

    if out_split_sizes is None:
        out_split_sizes = []
    if in_split_sizes is None:
        in_split_sizes = []

    if use_calc_stream:
        return group.process_group.alltoall_single_on_calc_stream(
            in_tensor, out_tensor, out_split_sizes, in_split_sizes)

    task = group.process_group.alltoall_single(in_tensor, out_tensor,
                                               out_split_sizes, in_split_sizes,
                                               sync_op)
    if sync_op:
        task.wait()

    return task


def alltoall_single(out_tensor,
                    in_tensor,
                    out_split_sizes=None,
                    in_split_sizes=None,
                    group=None,
                    sync_op=True,
                    use_calc_stream=False):
    """

    Perform specific reduction (for example, sum, max) on inputs across devices.

    Args:
        out_tensor(Tensor): The output tensor. The data type should be the same as the data type of the input.
        in_tensor (Tensor): The input tensor. The data type should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        out_split_sizes (List[int], optional): Split sizes of out_tensor for dim[0]. If not given, dim[0] of out_tensor must be divisible
            by group size and out_tensor will be gathered averagely from all participators. If none is given, use a empty list as default.
        in_split_sizes (List[int], optional): Split sizes of in_tensor for dim[0]. If not given, dim[0] of in_tensor must be divisible
        by group size and in_tensor will be scattered averagely to all participators. If none is given, use a empty list as default.
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

            output = paddle.empty([2], dtype="int64")
            if local_rank == 0:
                data = paddle.to_tensor([0, 1])
            else:
                data = paddle.to_tensor([2, 3])
            task = dist.stream.alltoall_single(output, data, sync_op=False)
            task.wait()
            out = output.numpy()
            # [0, 2] (2 GPUs, out for rank 0)
            # [1, 3] (2 GPUs, out for rank 1)
    """
    if group is not None and not group.is_member():
        raise RuntimeError(
            "The group should not be None and all ranks which invoke this operation should be the member of this group."
        )

    if not sync_op and use_calc_stream:
        raise RuntimeError(
            "use_calc_stream can only be true in sync op behavior.")

    if framework.in_dygraph_mode():
        return _alltoall_single_in_dygraph(out_tensor, in_tensor,
                                           out_split_sizes, in_split_sizes,
                                           group, sync_op, use_calc_stream)

    raise RuntimeError(
        "paddle.distributed.stream.alltoall_single is only supported in dygraph mode now."
    )
