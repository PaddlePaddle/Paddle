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
from paddle.distributed.communication.reduce import ReduceOp
from paddle.distributed.communication.stream.reduce_scatter import (
    _reduce_scatter_base as _reduce_scatter_base_stream,
)


def reduce_scatter(
    tensor, tensor_list, op=ReduceOp.SUM, group=None, sync_op=True
):
    """
    Reduces, then scatters a list of tensors to all processes in a group

    Args:
        tensor (Tensor): The output tensor on each rank. The result will overwrite this tenor after communication. Support
            float16, float32, float64, int32, int64, int8, uint8 or bool as the input data type.
        tensor_list (List[Tensor]]): List of tensors to reduce and scatter. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD, optional): The reduction used. If none is given, use ReduceOp.SUM as default.
        group (Group, optional): Communicate in which group. If none is given, use the global group as default.
        sync_op (bool, optional): Indicate whether the communication is sync or not. If none is given, use true as default.

    Returns:
        Return a task object.

    Warning:
        This API only supports the dygraph mode.


    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            if dist.get_rank() == 0:
                data1 = paddle.to_tensor([0, 1])
                data2 = paddle.to_tensor([2, 3])
            else:
                data1 = paddle.to_tensor([4, 5])
                data2 = paddle.to_tensor([6, 7])
            dist.reduce_scatter(data1, [data1, data2])
            print(data1)
            # [4, 6] (2 GPUs, out for rank 0)
            # [8, 10] (2 GPUs, out for rank 1)

    """
    return stream.reduce_scatter(
        tensor,
        tensor_list,
        op=op,
        group=group,
        sync_op=sync_op,
        use_calc_stream=False,
    )


def _reduce_scatter_base(
    output, input, op=ReduceOp.SUM, group=None, sync_op=True
):
    """
    Reduces, then scatters a flattened tensor to all processes in a group.

    Args:
        output (Tensor): Output tensor. Its data type should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16.
        input (Tensor): Input tensor that is of size output tensor size times world size. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD): Optional. The operation used. Default: ReduceOp.SUM.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        sync_op (bool, optional): Whether this op is a sync op. The default value is True.

    Returns:
        Async task handle, if sync_op is set to False.
        None, if sync_op or if not part of the group.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            rank = dist.get_rank()
            data = paddle.arange(4) + rank
            # [0, 1, 2, 3] (2 GPUs, for rank 0)
            # [1, 2, 3, 4] (2 GPUs, for rank 1)
            output = paddle.empty(shape=[2], dtype=data.dtype)
            dist.collective._reduce_scatter_base(output, data)
            print(output)
            # [1, 3] (2 GPUs, out for rank 0)
            # [5, 7] (2 GPUs, out for rank 1)

    """
    return _reduce_scatter_base_stream(
        output,
        input,
        op=op,
        group=group,
        sync_op=sync_op,
        use_calc_stream=False,
    )
