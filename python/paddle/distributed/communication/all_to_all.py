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

from paddle.distributed.communication import stream


def alltoall(in_tensor_list, out_tensor_list, group=None, sync_op=True):
    """
    Scatter tensors in in_tensor_list to all participators averagely and gather the result tensors in out_tensor_list.
    As shown below, the in_tensor_list in GPU0 includes 0_0 and 0_1, and GPU1 includes 1_0 and 1_1.
    Through alltoall operator, the 0_0 in GPU0 will be sent to GPU0 and 0_1 to GPU1, 1_0 in GPU1 sent to GPU0 and 1_1 to GPU1.
    Finally the out_tensor_list in GPU0 includes 0_0 and 1_0, and GPU1 includes 0_1 and 1_1.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/alltoall.png
        :width: 800
        :alt: alltoall
        :align: center

    Args:
        in_tensor_list (List[Tensor]): List of tensors to scatter one per rank. The data type of each tensor
            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16.
        out_tensor_list (List[Tensor]): List of tensors to be gathered one per rank. The data type of each tensor should be the same as the input tensors.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
        sync_op (bool, optional): Whether this op is a sync op. The default value is True.

    Returns:
        Return a task object.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            out_tensor_list = []
            if dist.get_rank() == 0:
                data1 = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
                data2 = paddle.to_tensor([[7, 8, 9], [10, 11, 12]])
            else:
                data1 = paddle.to_tensor([[13, 14, 15], [16, 17, 18]])
                data2 = paddle.to_tensor([[19, 20, 21], [22, 23, 24]])
            dist.alltoall([data1, data2], out_tensor_list)
            print(out_tensor_list)
            # [[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]] (2 GPUs, out for rank 0)
            # [[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]] (2 GPUs, out for rank 1)
    """
    return stream.alltoall(
        out_tensor_list, in_tensor_list, group, sync_op, False
    )


def alltoall_single(
    in_tensor,
    out_tensor,
    in_split_sizes=None,
    out_split_sizes=None,
    group=None,
    sync_op=True,
):
    """
    Scatter a single input tensor to all participators and gather the received tensors in out_tensor.

    Note:
        ``alltoall_single`` is only supported in eager mode.

    Args:
        in_tensor (Tensor): Input tensor. The data type should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16.
        out_tensor (Tensor): Output Tensor. The data type should be the same as the data type of the input Tensor.
        in_split_sizes (list[int], optional): Split sizes of ``in_tensor`` for dim[0]. If not given, dim[0] of ``in_tensor``
            must be divisible by group size and ``in_tensor`` will be scattered averagely to all participators. Default: None.
        out_split_sizes (list[int], optional): Split sizes of ``out_tensor`` for dim[0]. If not given, dim[0] of ``out_tensor``
            must be divisible by group size and ``out_tensor`` will be gathered averagely from all participators. Default: None.
        group (Group, optional): The group instance return by ``new_group`` or None for global default group. Default: None.
        sync_op (bool, optional): Whether this op is a sync op. The default value is True.

    Returns:
        Return a task object.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            rank = dist.get_rank()
            size = dist.get_world_size()

            # case 1 (2 GPUs)
            data = paddle.arange(2, dtype='int64') + rank * 2
            # data for rank 0: [0, 1]
            # data for rank 1: [2, 3]
            output = paddle.empty([2], dtype='int64')
            dist.alltoall_single(data, output)
            print(output)
            # output for rank 0: [0, 2]
            # output for rank 1: [1, 3]

            # case 2 (2 GPUs)
            in_split_sizes = [i + 1 for i in range(size)]
            # in_split_sizes for rank 0: [1, 2]
            # in_split_sizes for rank 1: [1, 2]
            out_split_sizes = [rank + 1 for i in range(size)]
            # out_split_sizes for rank 0: [1, 1]
            # out_split_sizes for rank 1: [2, 2]
            data = paddle.ones([sum(in_split_sizes), size], dtype='float32') * rank
            # data for rank 0: [[0., 0.], [0., 0.], [0., 0.]]
            # data for rank 1: [[1., 1.], [1., 1.], [1., 1.]]
            output = paddle.empty([(rank + 1) * size, size], dtype='float32')
            group = dist.new_group([0, 1])
            task = dist.alltoall_single(data,
                                        output,
                                        in_split_sizes,
                                        out_split_sizes,
                                        sync_op=False,
                                        group=group)
            task.wait()
            print(output)
            # output for rank 0: [[0., 0.], [1., 1.]]
            # output for rank 1: [[0., 0.], [0., 0.], [1., 1.], [1., 1.]]

    """
    return stream.alltoall_single(
        out_tensor,
        in_tensor,
        out_split_sizes,
        in_split_sizes,
        group,
        sync_op,
        False,
    )
