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
from paddle.base import data_feeder
from paddle.distributed.communication.group import _get_global_group


def _all_gather_into_tensor_in_dygraph(
    out_tensor, in_tensor, group, sync_op, use_calc_stream
):
    group = _get_global_group() if group is None else group

    if use_calc_stream:
        return group.process_group.all_gather_into_tensor_on_calc_stream(
            out_tensor,
            in_tensor,
        )

    task = group.process_group.all_gather_into_tensor(
        out_tensor, in_tensor, sync_op
    )
    if sync_op:
        task.wait()

    return task


def _all_gather_in_dygraph(
    tensor_list, tensor, group, sync_op, use_calc_stream
):
    group = _get_global_group() if group is None else group

    if len(tensor_list) == 0:
        tensor_list += [paddle.empty_like(tensor) for _ in range(group.nranks)]

    if use_calc_stream:
        return group.process_group.all_gather_on_calc_stream(
            tensor_list, tensor
        )

    task = group.process_group.all_gather(tensor_list, tensor, sync_op)
    if sync_op:
        task.wait()

    return task


def _all_gather_in_static_mode(tensor_list, tensor, group, sync_op):
    op_type = 'c_allgather'
    helper = framework.LayerHelper(op_type, **locals())
    out = helper.create_variable_for_type_inference(dtype=tensor.dtype)
    for elem in tensor_list:
        data_feeder.check_variable_and_dtype(
            elem,
            'tensor_list',
            [
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'bool',
                'int8',
                'uint8',
                'complex64',
                'complex128',
            ],
            'all_gather',
        )
    data_feeder.check_variable_and_dtype(
        tensor,
        'tensor',
        [
            'float16',
            'float32',
            'float64',
            'int32',
            'int64',
            'bool',
            'int8',
            'uint8',
            'complex64',
            'complex128',
        ],
        'all_gather',
    )

    ring_id = 0 if group is None else group.id
    nranks = dist.get_world_size()
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [out]},
        attrs={
            'ring_id': ring_id,
            'use_calc_stream': sync_op,
            'nranks': nranks,
        },
    )
    tensor_list.clear()
    # 0-D use stack/unstack while others use concat/split
    if len(tensor.shape) == 0:
        tensor_list.extend(paddle.unstack(out, 0))
    else:
        tensor_list.extend(paddle.split(out, nranks, 0))


def all_gather(
    tensor_or_tensor_list,
    tensor,
    group=None,
    sync_op=True,
    use_calc_stream=False,
):
    """

    Gather tensors across devices to a correctly-sized tensor or a tensor list.

    Args:
        tensor_or_tensor_list (Union[Tensor, List[Tensor]]): The output. If it is a tensor, it should be correctly-sized. If it is a list, it
            should be empty or contain correctly-sized tensors.
        tensor (Tensor): The input tensor on each rank. The result will overwrite this tenor after communication. Support
            float16, float32, float64, int32, int64, int8, uint, bool, complex64 or complex128 as the input data type.
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

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> local_rank = dist.get_rank()
            >>> tensor_list = []
            >>> if local_rank == 0:
            ...     data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
            >>> else:
            ...     data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            >>> task = dist.stream.all_gather(tensor_list, data, sync_op=False)
            >>> task.wait()
            >>> print(tensor_list)
            [[[4, 5, 6], [4, 5, 6]], [[1, 2, 3], [1, 2, 3]]] (2 GPUs)
    """
    if group is not None and not group.is_member():
        raise RuntimeError(
            "The group should not be None and all ranks which invoke this operation should be the member of this group."
        )

    if not sync_op and use_calc_stream:
        raise RuntimeError(
            "use_calc_stream can only be true in sync op behavior."
        )

    if framework.in_dynamic_mode():
        if paddle.is_tensor(tensor_or_tensor_list):
            return _all_gather_into_tensor_in_dygraph(
                tensor_or_tensor_list, tensor, group, sync_op, use_calc_stream
            )
        else:
            return _all_gather_in_dygraph(
                tensor_or_tensor_list, tensor, group, sync_op, use_calc_stream
            )
    else:
        assert (
            group is None
        ), "Group can not be used in static graph mode for now."
        if paddle.is_tensor(tensor_or_tensor_list):
            raise RuntimeError(
                "Only support passing a tensor list to `all_gather` in static graph mode now."
            )
        else:
            return _all_gather_in_static_mode(
                tensor_or_tensor_list, tensor, group, sync_op
            )
