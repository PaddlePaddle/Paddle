#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

import paddle
from paddle import framework
from paddle.distributed.communication import stream

from .serialization_utils import (
    convert_object_to_tensor,
    convert_tensor_to_object,
)

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle.base.core import task
    from paddle.distributed.communication.group import Group

    _T = TypeVar("_T")


def all_gather(
    tensor_list: list[Tensor],
    tensor: Tensor,
    group: Group | None = None,
    sync_op: bool = True,
) -> task | None:
    """

    Gather tensors from all participators and all get the result. As shown
    below, one process is started with a GPU and the data of this process is represented
    by its group rank. Through the all_gather operator, each GPU will have data
    from all GPUs.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/allgather.png
        :width: 800
        :alt: all_gather
        :align: center

    Args:
        tensor_list (list): A list of output Tensors. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool, bfloat16, complex64 or complex128.
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool, bfloat16, complex64 or complex128.
        group (Group|None, optional): The group instance return by new_group or None for global default group.
        sync_op (bool, optional): Whether this op is a sync op. The default value is True.

    Returns:
        None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> tensor_list = [] # type: ignore
            >>> if dist.get_rank() == 0:
            ...     data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
            >>> else:
            ...     data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            >>> dist.all_gather(tensor_list, data)
            >>> print(tensor_list)
            >>> # [[[4, 5, 6], [4, 5, 6]], [[1, 2, 3], [1, 2, 3]]] (2 GPUs)
    """
    return stream.all_gather(tensor_list, tensor, group, sync_op)


def all_gather_object(
    object_list: list[_T], obj: _T, group: Group = None
) -> None:
    """

    Gather picklable objects from all participators and all get the result. Similar to all_gather(), but python object can be passed in.

    Args:
        object_list (list): A list of output object. The datatype of every element in the list is same as the input obj.
        obj (Any): The picklable object to send.
        group (Group): The group instance return by new_group or None for global default group.

    Returns:
        None.

    Warning:
        This API only supports the dygraph mode.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> object_list = [] # type: ignore
            >>> if dist.get_rank() == 0:
            ...     obj = {"foo": [1, 2, 3]}
            >>> else:
            ...     obj = {"bar": [4, 5, 6]}
            >>> dist.all_gather_object(object_list, obj)
            >>> print(object_list)
            >>> # [{'foo': [1, 2, 3]}, {'bar': [4, 5, 6]}] (2 GPUs)
    """
    assert (
        framework.in_dynamic_mode()
    ), "all_gather_object doesn't support static graph mode."

    tensor, len_of_tensor = convert_object_to_tensor(obj)

    # gather len_of_tensor from all ranks
    list_len_of_tensor = []
    all_gather(list_len_of_tensor, len_of_tensor, group)
    # get the max length from list
    max_len_of_tensor = int(max(list_len_of_tensor).item())
    # resize the input tensor to max length avoid hang in all gather
    # Note(liyurui): Maybe we should support various length all_gather?
    # Now this operation is efficient for we don't support resize in python.
    numpy_data = tensor.numpy()
    numpy_data = np.resize(numpy_data, [max_len_of_tensor])
    input_tensor = paddle.to_tensor(numpy_data)

    tensor_list = []
    all_gather(tensor_list, input_tensor, group)
    for i, tensor in enumerate(tensor_list):
        object_list.append(
            convert_tensor_to_object(tensor, list_len_of_tensor[i])
        )
