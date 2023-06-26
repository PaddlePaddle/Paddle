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
from paddle import framework
from paddle.distributed.communication import stream

from .serialization_utils import (
    convert_object_to_tensor,
    convert_tensor_to_object,
)


def broadcast(tensor, src, group=None, sync_op=True):
    """

    Broadcast a tensor from the source to all others.
    As shown below, one process is started with a GPU and GPU0 owns data 0. Through broadcast operator,
    data 0 will be sent to all GPUs from GPU0.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/broadcast.png
        :width: 800
        :alt: broadcast
        :align: center

    Args:
        tensor (Tensor): The tensor to send if current rank is the source, or the tensor to receive otherwise. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16.
        src (int): The source rank in global view.
        group (Group, optional): The group instance return by new_group or None for global default group.
        sync_op (bool, optional): Whether this op is a sync op. The default value is True.

    Returns:
        Return a task object.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            if dist.get_rank() == 0:
                data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
            else:
                data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            dist.broadcast(data, src=1)
            print(data)
            # [[1, 2, 3], [1, 2, 3]] (2 GPUs)
    """
    return stream.broadcast(
        tensor,
        src,
        group=group,
        sync_op=sync_op,
        use_calc_stream=False,
    )


def broadcast_object_list(object_list, src, group=None):
    """

    Broadcast picklable objects from the source to all others. Similiar to broadcast(), but python object can be passed in.

    Args:
        object_list (list): The list of objects to send if current rank is the source, or the list of objects to receive otherwise.
        src (int): The source rank in global view.
        group (Group): The group instance return by new_group or None for global default group.

    Returns:
        None.

    Warning:
        This API only supports the dygraph mode.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle.distributed as dist

            dist.init_parallel_env()
            if dist.get_rank() == 0:
                object_list = [{"foo": [1, 2, 3]}]
            else:
                object_list = [{"bar": [4, 5, 6]}]
            dist.broadcast_object_list(object_list, src=1)
            print(object_list)
            # [{"bar": [4, 5, 6]}] (2 GPUs)
    """
    assert (
        framework.in_dynamic_mode()
    ), "broadcast_object_list doesn't support static graph mode."

    rank = dist.get_rank()
    obj_tensors = []
    obj_nums = len(object_list)

    if rank == src:
        obj_sizes = []
        for obj in object_list:
            obj_tensor, obj_size = convert_object_to_tensor(obj)
            obj_tensors.append(obj_tensor)
            obj_sizes.append(obj_size)
        obj_size_tensor = paddle.stack(obj_sizes)
    else:
        obj_size_tensor = paddle.empty([obj_nums], dtype="int64")
    broadcast(obj_size_tensor, src, group)

    if rank == src:
        # cast to uint8 to keep the same dtype
        obj_data_tensor = paddle.concat(obj_tensors).cast("uint8")
    else:
        data_len = paddle.sum(obj_size_tensor).item()
        obj_data_tensor = paddle.empty([data_len], dtype="uint8")
    broadcast(obj_data_tensor, src, group)

    offset = 0
    for i in range(obj_nums):
        data_len = obj_size_tensor[i]
        object_list[i] = convert_tensor_to_object(
            obj_data_tensor[offset : offset + data_len], data_len
        )
        offset += data_len
