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


def scatter(tensor, tensor_list=None, src=0, group=None, sync_op=True):
    """

    Scatter a tensor to all participators. As shown below, one process is started with a GPU and the source of the scatter
    is GPU0. Through scatter operator, the data in GPU0 will be sent to all GPUs averagely.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/scatter.png
        :width: 800
        :alt: scatter
        :align: center

    Args:
        tensor (Tensor): The output Tensor. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16.
        tensor_list (list|tuple): A list/tuple of Tensors to scatter. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16. Default value is None.
        src (int): The source rank id. Default value is 0.
        group (Group, optional): The group instance return by new_group or None for global default group.
        sync_op (bool, optional): Whether this op is a sync op. The default value is True.

    Returns:
        None.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            if dist.get_rank() == 0:
                data1 = paddle.to_tensor([7, 8, 9])
                data2 = paddle.to_tensor([10, 11, 12])
                dist.scatter(data1, src=1)
            else:
                data1 = paddle.to_tensor([1, 2, 3])
                data2 = paddle.to_tensor([4, 5, 6])
                dist.scatter(data1, tensor_list=[data1, data2], src=1)
            print(data1, data2)
            # [1, 2, 3] [10, 11, 12] (2 GPUs, out for rank 0)
            # [4, 5, 6] [4, 5, 6] (2 GPUs, out for rank 1)
    """
    return stream.scatter(tensor, tensor_list, src, group, sync_op)
