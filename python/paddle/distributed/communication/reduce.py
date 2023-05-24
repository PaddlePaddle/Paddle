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
from paddle import framework
from paddle.distributed.communication import stream


class ReduceOp:
    """

    Specify the type of operation used for element-wise reductions.
    It should be one of the following values:

        ReduceOp.SUM

        ReduceOp.MAX

        ReduceOp.MIN

        ReduceOp.PROD

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
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
            print(data)
            # [[5, 7, 9], [5, 7, 9]] (2 GPUs)
    """

    SUM = 0
    MAX = 1
    MIN = 2
    PROD = 3
    AVG = 4


def _get_reduce_op(reduce_op, func_name):
    if framework.in_dynamic_mode():
        if reduce_op == ReduceOp.SUM:
            return framework.core.ReduceOp.SUM
        elif reduce_op == ReduceOp.MAX:
            return framework.core.ReduceOp.MAX
        elif reduce_op == ReduceOp.MIN:
            return framework.core.ReduceOp.MIN
        elif reduce_op == ReduceOp.PROD:
            return framework.core.ReduceOp.PRODUCT
    else:
        if reduce_op == ReduceOp.SUM:
            return f'c_{func_name}_sum'
        elif reduce_op == ReduceOp.MAX:
            return f'c_{func_name}_max'
        elif reduce_op == ReduceOp.MIN:
            return f'c_{func_name}_min'
        elif reduce_op == ReduceOp.PROD:
            return f'c_{func_name}_prod'
        else:
            return f'c_{func_name}'

    raise ValueError(f"Unknown reduce_op type for {func_name}.")


def reduce(tensor, dst, op=ReduceOp.SUM, group=None, sync_op=True):
    """

    Reduce a tensor to the destination from all others. As shown below, one process is started with a GPU and the data of this process is represented
    by its group rank. The destination of the reduce operator is GPU0 and the process is sum. Through reduce operator,
    the GPU0 will owns the sum of all data from all GPUs.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/reduce.png
        :width: 800
        :alt: reduce
        :align: center

    Args:
        tensor (Tensor): The output Tensor for the destination and the input Tensor otherwise. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16.
        dst (int): The destination rank id.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD, optional): The operation used. Default value is ReduceOp.SUM.
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
            dist.reduce(data, dst=0)
            print(data)
            # [[5, 7, 9], [5, 7, 9]] (2 GPUs, out for rank 0)
            # [[1, 2, 3], [1, 2, 3]] (2 GPUs, out for rank 1)
    """
    return stream.reduce(
        tensor,
        dst=dst,
        op=op,
        group=group,
        sync_op=sync_op,
        use_calc_stream=False,
    )

    # code below will be removed after we remove the old dygraph
    if group is not None and not group.is_member():
        return
    use_calc_stream = sync_op
    ring_id = 0 if group is None else group.id
    gdst = dst if group is None else group.get_group_rank(dst)
    assert gdst >= 0, "dst rank out of group, need global rank"

    if op == ReduceOp.SUM:
        return paddle._legacy_C_ops.c_reduce_sum(
            tensor,
            tensor,
            'use_calc_stream',
            use_calc_stream,
            'ring_id',
            ring_id,
            'root_id',
            gdst,
        )
    elif op == ReduceOp.MAX:
        return paddle._legacy_C_ops.c_reduce_max(
            tensor,
            tensor,
            'use_calc_stream',
            use_calc_stream,
            'ring_id',
            ring_id,
            'root_id',
            gdst,
        )
    elif op == ReduceOp.MIN:
        return paddle._legacy_C_ops.c_reduce_min(
            tensor,
            tensor,
            'use_calc_stream',
            use_calc_stream,
            'ring_id',
            ring_id,
            'root_id',
            gdst,
        )
    elif op == ReduceOp.PROD:
        return paddle._legacy_C_ops.c_reduce_prod(
            tensor,
            tensor,
            'use_calc_stream',
            use_calc_stream,
            'ring_id',
            ring_id,
            'root_id',
            gdst,
        )
    else:
        raise ValueError(f"Unknown parameter: {op}.")
