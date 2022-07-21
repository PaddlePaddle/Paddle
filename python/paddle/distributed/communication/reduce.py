#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import _C_ops
from .group import _get_default_group
from paddle.fluid.framework import _non_static_mode
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype

__all__ = ["reduce", "ReduceOp", "_get_reduce_op"]


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

            import numpy as np
            import paddle
            from paddle.distributed import ReduceOp
            from paddle.distributed import init_parallel_env

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            if paddle.distributed.ParallelEnv().local_rank == 0:
                np_data = np.array([[4, 5, 6], [4, 5, 6]])
            else:
                np_data = np.array([[1, 2, 3], [1, 2, 3]])
            data = paddle.to_tensor(np_data)
            paddle.distributed.all_reduce(data, op=ReduceOp.SUM)
            out = data.numpy()
            # [[5, 7, 9], [5, 7, 9]]
    """
    SUM = 0
    MAX = 1
    MIN = 2
    PROD = 3
    AVG = 4


def _get_reduce_op(reduce_op, func_name):
    if reduce_op == ReduceOp.SUM:
        return core.ReduceOp.SUM
    elif reduce_op == ReduceOp.MAX:
        return core.ReduceOp.MAX
    elif reduce_op == ReduceOp.MIN:
        return core.ReduceOp.MIN
    elif reduce_op == ReduceOp.PROD:
        return core.ReduceOp.PRODUCT
    else:
        raise ValueError("Unknown reduce_op type for {}.".format(func_name))


def _dygraph_reduce(tensor,
                    dst,
                    op=ReduceOp.SUM,
                    group=None,
                    use_calc_stream=True):
    op_type = _get_reduce_op(op, "reduce")
    group = _get_default_group() if group is None else group
    gdst = group.get_group_rank(dst)
    assert gdst >= 0, ("dst rank out of group, need global rank")
    task = group.process_group.reduce(tensor, gdst, op_type)
    if use_calc_stream:
        task.wait()
        return None
    else:
        return task


def _static_reduce(tensor,
                   dst,
                   op=ReduceOp.SUM,
                   group=None,
                   use_calc_stream=True):
    ring_id = 0 if group is None else group.id
    gdst = dst if group is None else group.get_group_rank(dst)
    assert gdst >= 0, ("dst rank out of group, need global rank")

    if _non_static_mode():
        if op == ReduceOp.SUM:
            return _C_ops.c_reduce_sum(tensor, tensor, 'use_calc_stream',
                                       use_calc_stream, 'ring_id', ring_id,
                                       'root_id', gdst)
        elif op == ReduceOp.MAX:
            return _C_ops.c_reduce_max(tensor, tensor, 'use_calc_stream',
                                       use_calc_stream, 'ring_id', ring_id,
                                       'root_id', gdst)
        elif op == ReduceOp.MIN:
            return _C_ops.c_reduce_min(tensor, tensor, 'use_calc_stream',
                                       use_calc_stream, 'ring_id', ring_id,
                                       'root_id', gdst)
        elif op == ReduceOp.PROD:
            return _C_ops.c_reduce_prod(tensor, tensor, 'use_calc_stream',
                                        use_calc_stream, 'ring_id', ring_id,
                                        'root_id', gdst)
        else:
            raise ValueError("Unknown parameter: {}.".format(op))

    op_type = 'c_reduce'
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'all_reduce')

    if op == ReduceOp.SUM:
        op_type = 'c_reduce_sum'
    elif op == ReduceOp.MAX:
        op_type = 'c_reduce_max'
    elif op == ReduceOp.MIN:
        op_type = 'c_reduce_min'
    elif op == ReduceOp.PROD:
        op_type = 'c_reduce_prod'

    helper = LayerHelper(op_type, **locals())
    helper.append_op(type=op_type,
                     inputs={'X': [tensor]},
                     outputs={'Out': [tensor]},
                     attrs={
                         'ring_id': ring_id,
                         'use_calc_stream': use_calc_stream,
                         'root_id': gdst,
                     })


def reduce(tensor, dst, op=ReduceOp.SUM, group=None, use_calc_stream=True):
    """

    Reduce a tensor to the destination from all others. As shown below, 4 GPUs each start 4 processes and the data on each GPU is respresnted
    by the GPU number. The destination of the reduce operator is GPU0 and the process is sum. Through reduce operator,
    the GPU0 will owns the sum of all data from all GPUs.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/reduce.png
        :width: 800
        :alt: reduce
        :align: center

    Args:
        tensor (Tensor): The output Tensor for the destination and the input Tensor otherwise. Its data type
            should be float16, float32, float64, int32 or int64.
        dst (int): The destination rank id.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD): Optional. The operation used. Default value is ReduceOp.SUM.
        group (Group): The group instance return by new_group or None for global default group.
        use_calc_stream (bool): Wether to use calculation stream (True) or communication stream (False).
            Default to True.

    Returns:
        None.

    Examples:
        .. code-block:: python

            # required: distributed
            import numpy as np
            import paddle
            from paddle.distributed import init_parallel_env

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            if paddle.distributed.ParallelEnv().local_rank == 0:
                np_data = np.array([[4, 5, 6], [4, 5, 6]])
            else:
                np_data = np.array([[1, 2, 3], [1, 2, 3]])
            data = paddle.to_tensor(np_data)
            paddle.distributed.reduce(data, 0)
            out = data.numpy()
            # [[5, 7, 9], [5, 7, 9]]
    """
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        return _dygraph_reduce(tensor, dst, op, group, use_calc_stream)

    _static_reduce(tensor, dst, op, group, use_calc_stream)
