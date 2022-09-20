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

import paddle.distributed.collective as collective
import paddle.fluid.framework as framework


def _recv_in_dygraph(tensor, src, group, sync_op, use_calc_stream):
    group = collective._get_default_group() if group is None else group
    if use_calc_stream:
        return group.process_group.recv_on_calc_stream(tensor, src)

    task = group.process_group.recv(tensor, src, sync_op)
    if sync_op:
        task.wait()

    return task


def recv(tensor, src=0, group=None, sync_op=True, use_calc_stream=False):
    """

    Receive a tensor from the source device.

    Args:
        tensor (Tensor): The tensor to receive. Support float16, float32, float64, int32, int64, int8, uint8 or bool as its data type.
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
            local_rank = dist.get_rank()
            if local_rank == 0:
                data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
                task = dist.stream.send(data, dst=1, sync_op=False)
            else:
                data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
                task = dist.stream.recv(data, src=0, sync_op=False)
            task.wait()
            out = data.numpy()
            # [[4, 5, 6], [4, 5, 6]
    """
    if group is not None and not group.is_member():
        raise RuntimeError(
            "The group should not be None and all ranks which invoke this operation should be the member of this group."
        )

    if not sync_op and use_calc_stream:
        raise RuntimeError(
            "use_calc_stream can only be True in sync op behavior.")

    if framework.in_dygraph_mode():
        return _recv_in_dygraph(tensor, src, group, sync_op, use_calc_stream)

    raise RuntimeError(
        "paddle.distributed.stream.recv is only supported in dygraph mode now.")
