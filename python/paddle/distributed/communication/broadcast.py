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

import paddle.distributed.communication.stream as stream


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
