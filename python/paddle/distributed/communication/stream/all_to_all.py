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
from paddle.distributed.communication.group import (
    _get_global_group,
    _warn_cur_rank_not_in_group,
)
from paddle.fluid import data_feeder


def _all_to_all_tensor_in_dygraph(
    out_tensor, in_tensor, group, sync_op, use_calc_stream
):
    if use_calc_stream:
        return group.process_group.all_to_all_tensor_on_calc_stream(
            in_tensor, out_tensor
        )

    task = group.process_group.all_to_all_tensor(in_tensor, out_tensor, sync_op)
    if sync_op:
        task.wait()

    return task


def _all_to_all_in_dygraph(
    out_tensor_list, in_tensor_list, group, sync_op, use_calc_stream
):
    if len(in_tensor_list) == 0:
        raise RuntimeError("The input tensor_list should not be empty.")

    if len(out_tensor_list) == 0:
        out_tensor_list += [
            paddle.empty_like(tensor) for tensor in in_tensor_list
        ]

    if use_calc_stream:
        return group.process_group.all_to_all_on_calc_stream(
            out_tensor_list, in_tensor_list
        )

    task = group.process_group.all_to_all(
        out_tensor_list, in_tensor_list, sync_op
    )
    if sync_op:
        task.wait()

    return task


def _all_to_all_in_static_mode(
    out_tensor_or_tensor_list,
    in_tensor_or_tensor_list,
    group,
    sync_op,
    use_calc_stream,
):
    op_type = 'alltoall'
    ring_id = 0 if group is None else group.id
    nranks = dist.get_world_size()
    helper = framework.LayerHelper(op_type, **locals())

    in_tensor = in_tensor_or_tensor_list
    if isinstance(in_tensor_or_tensor_list, list):
        if len(in_tensor_or_tensor_list) == 0:
            raise RuntimeError("The input tensor_list should not be empty.")
        # 0-D use stack/unstack while others use concat/split
        if len(in_tensor_or_tensor_list[0].shape) == 0:
            in_tensor = paddle.stack(in_tensor_or_tensor_list, axis=0)
        else:
            in_tensor = paddle.concat(in_tensor_or_tensor_list, axis=0)

    out_tensor = out_tensor_or_tensor_list
    if isinstance(out_tensor_or_tensor_list, list):
        if len(out_tensor_or_tensor_list) != 0:
            raise ValueError(
                "The 'out_tensor_list' for all_to_all " "must be an empty list."
            )
        out_tensor = helper.create_variable_for_type_inference(
            dtype=in_tensor.dtype
        )

    data_feeder.check_variable_and_dtype(
        in_tensor,
        'in_tensor',
        ['float16', 'float32', 'float64', 'int32', 'int64'],
        'all_to_all',
    )
    helper.append_op(
        type=op_type,
        inputs={'X': [in_tensor]},
        outputs={'Out': [out_tensor]},
        attrs={
            'ring_id': ring_id,
            'use_calc_stream': sync_op,
        },
    )
    # NOTE(liyurui): If the argument `out_tensor_or_tensor_list` is a tensor_list,
    # we need to split the result. So we should wait the result of all_to_all
    # before split if the communication is not on calc stream.
    if isinstance(out_tensor_or_tensor_list, list):
        if not sync_op:
            dist.wait(out_tensor, use_calc_stream=False)
        # 0-D use stack/unstack while others use concat/split
        if len(in_tensor_or_tensor_list[0].shape) == 0:
            out_tensor_or_tensor_list.extend(paddle.unstack(out_tensor, 0))
        else:
            out_tensor_or_tensor_list.extend(
                paddle.split(out_tensor, nranks, 0)
            )

    return None


def alltoall(
    out_tensor_or_tensor_list,
    in_tensor_or_tensor_list,
    group=None,
    sync_op=True,
    use_calc_stream=False,
):
    """

    Scatter a tensor (or a tensor list) across devices and gather outputs to another tensor (or a tensor list, respectively).

    Args:
        out_tensor_or_tensor_list (Union[Tensor, List[Tensor]]): The output. If it is a tensor, it should be correctly-sized.
        If it is a list, it should be empty or contain correctly-sized tensors. Its data type should be the same as the input.
        in_tensor_or_tensor_list (Union[Tensor, List[Tensor]]): The input to scatter (must be specified on the source rank).
            If it is a tensor, it should be correctly-sized. If it is a list, it should contain correctly-sized tensors. Support
            float16, float32, float64, int32, int64, int8, uint8 or bool as the input data type.
        group (Group, optional): Communicate in which group. If none is given, use the global group as default.
        sync_op (bool, optional): Indicate whether the communication is sync or not. If none is given, use true as default.
        use_calc_stream (bool, optional): Indicate whether the communication is done on calculation stream. If none is given, use false as default. This
            option is designed for high performance demand, be careful to turn it on except you are clearly know its meaning.

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
            task = dist.stream.alltoall(out_tensor_list, [data1, data2], sync_op=False)
            task.wait()
            print(out_tensor_list)
            # [[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]]    (2 GPUs, out for rank 0)
            # [[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]] (2 GPUs, out for rank 1)
    """
    if _warn_cur_rank_not_in_group(group):
        return

    if not sync_op and use_calc_stream:
        raise RuntimeError(
            "use_calc_stream can only be true in sync op behavior."
        )

    if out_tensor_or_tensor_list is None:
        raise RuntimeError("The output should be specified.")
    if in_tensor_or_tensor_list is None:
        raise RuntimeError("The input should be specified.")

    if framework.in_dynamic_mode():
        group = _get_global_group() if group is None else group
        out_is_tensor = paddle.is_tensor(out_tensor_or_tensor_list)
        in_is_tensor = paddle.is_tensor(in_tensor_or_tensor_list)
        if out_is_tensor and in_is_tensor:
            return _all_to_all_tensor_in_dygraph(
                out_tensor_or_tensor_list,
                in_tensor_or_tensor_list,
                group,
                sync_op,
                use_calc_stream,
            )
        elif not out_is_tensor and not in_is_tensor:
            return _all_to_all_in_dygraph(
                out_tensor_or_tensor_list,
                in_tensor_or_tensor_list,
                group,
                sync_op,
                use_calc_stream,
            )
        else:
            raise RuntimeError(
                "The output and input should be both tensor or tensor list."
            )
    else:
        assert (
            group is None
        ), "Group can not be used in static graph mode for now."
        return _all_to_all_in_static_mode(
            out_tensor_or_tensor_list,
            in_tensor_or_tensor_list,
            group,
            sync_op,
            use_calc_stream,
        )


def _alltoall_single_in_dygraph(
    out_tensor,
    in_tensor,
    out_split_sizes,
    in_split_sizes,
    group,
    sync_op,
    use_calc_stream,
):
    world_size = dist.get_world_size(group)
    if out_split_sizes is None:
        out_split_sizes = [
            out_tensor.shape[0] // world_size for _ in range(world_size)
        ]
    if in_split_sizes is None:
        in_split_sizes = [
            in_tensor.shape[0] // world_size for _ in range(world_size)
        ]

    if use_calc_stream:
        return group.process_group.all_to_all_single_on_calc_stream(
            out_tensor, in_tensor, out_split_sizes, in_split_sizes
        )

    task = group.process_group.all_to_all_single(
        out_tensor, in_tensor, out_split_sizes, in_split_sizes, sync_op
    )
    if sync_op:
        task.wait()

    return task


def alltoall_single(
    out_tensor,
    in_tensor,
    out_split_sizes=None,
    in_split_sizes=None,
    group=None,
    sync_op=True,
    use_calc_stream=False,
):
    """

    Split and Scatter the splitted input tensor to the out tensor across devices.

    Args:
        out_tensor(Tensor): The output tensor. Its data type should be the same as the input.
        in_tensor (Tensor): The input tensor. Its data type should be float16, float32, float64, int32, int64, int8, uint8 or bool.
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

            # case 1
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

            # case 2
            size = dist.get_world_size()
            output = paddle.empty([(local_rank + 1) * size, size], dtype='float32')
            if local_rank == 0:
                data = paddle.to_tensor([[0., 0.], [0., 0.], [0., 0.]])
            else:
                data = paddle.to_tensor([[1., 1.], [1., 1.], [1., 1.]])
            out_split_sizes = [local_rank + 1 for i in range(size)]
            in_split_sizes = [i + 1 for i in range(size)]
            task = dist.stream.alltoall_single(output,
                                            data,
                                            out_split_sizes,
                                            in_split_sizes,
                                            sync_op=False)
            task.wait()
            out = output.numpy()
            # [[0., 0.], [1., 1.]]                     (2 GPUs, out for rank 0)
            # [[0., 0.], [0., 0.], [1., 1.], [1., 1.]] (2 GPUs, out for rank 1)
    """
    if _warn_cur_rank_not_in_group(group):
        return

    if not sync_op and use_calc_stream:
        raise RuntimeError(
            "use_calc_stream can only be true in sync op behavior."
        )

    if framework.in_dynamic_mode():
        group = _get_global_group() if group is None else group
        return _alltoall_single_in_dygraph(
            out_tensor,
            in_tensor,
            out_split_sizes,
            in_split_sizes,
            group,
            sync_op,
            use_calc_stream,
        )

    raise RuntimeError(
        "paddle.distributed.stream.alltoall_single is only supported in dygraph mode now."
    )
