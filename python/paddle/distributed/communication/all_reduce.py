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
from paddle.fluid.framework import _non_static_mode
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype
from .reduce import ReduceOp
from .reduce import _get_reduce_op
from .group import _get_default_group

__all__ = ["all_reduce"]


def _dygraph_all_reduce(tensor, op=ReduceOp.SUM, group=None, use_calc_stream=True):
    op_type = _get_reduce_op(op, "all_reduce")
    group = _get_default_group() if group is None else group
    task = group.process_group.allreduce(tensor, op_type)
    if use_calc_stream:
        task.wait()
        return None
    else:
        return task

def _static_all_reduce(tensor, op=ReduceOp.SUM, group=None, use_calc_stream=True):
    ring_id = 0 if group is None else group.id
    if _non_static_mode():
        if op == ReduceOp.SUM:
            return _C_ops.c_allreduce_sum_(tensor, 'use_calc_stream',
                                           use_calc_stream, 'ring_id', ring_id)
        elif op == ReduceOp.MAX:
            return _C_ops.c_allreduce_max_(tensor, 'use_calc_stream',
                                           use_calc_stream, 'ring_id', ring_id)
        elif op == ReduceOp.MIN:
            return _C_ops.c_allreduce_min_(tensor, 'use_calc_stream',
                                           use_calc_stream, 'ring_id', ring_id)
        elif op == ReduceOp.PROD:
            return _C_ops.c_allreduce_prod_(tensor, 'use_calc_stream',
                                            use_calc_stream, 'ring_id', ring_id)
        else:
            raise ValueError("Unknown parameter: {}.".format(op))

    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'all_reduce')
    if op == ReduceOp.SUM:
        op_type = 'c_allreduce_sum'
    elif op == ReduceOp.MAX:
        op_type = 'c_allreduce_max'
    elif op == ReduceOp.MIN:
        op_type = 'c_allreduce_min'
    elif op == ReduceOp.PROD:
        op_type = 'c_allreduce_prod'
    if not isinstance(ring_id, int):
        raise ValueError("The type of 'ring_id' for all_reduce should be int.")
    helper = LayerHelper(op_type, **locals())
    helper.append_op(type=op_type,
                     inputs={'X': [tensor]},
                     outputs={'Out': [tensor]},
                     attrs={
                         'ring_id': ring_id,
                         'use_calc_stream': use_calc_stream
                     })

def all_reduce(tensor, op=ReduceOp.SUM, group=None, use_calc_stream=True):
    """

    Reduce a tensor over all ranks so that all get the result.
    As shown below, 4 GPUs each start 4 processes and the data on each GPU is represnted
    by the GPU number. The reduce operator is sum. Through all_reduce operator, 
    each GPU will have the sum of the data from all GPUs.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/allreduce.png
        :width: 800
        :alt: all_reduce
        :align: center

    Args:
        tensor (Tensor): The input Tensor. It also works as the output Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
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
            from paddle.distributed import ReduceOp
            from paddle.distributed import init_parallel_env

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            if paddle.distributed.ParallelEnv().local_rank == 0:
                np_data = np.array([[4, 5, 6], [4, 5, 6]])
            else:
                np_data = np.array([[1, 2, 3], [1, 2, 3]])
            data = paddle.to_tensor(np_data)
            paddle.distributed.all_reduce(data)
            out = data.numpy()
            # [[5, 7, 9], [5, 7, 9]]
    """
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        return _dygraph_all_reduce(tensor, op, group, use_calc_stream)

    _static_all_reduce(tensor, op, group, use_calc_stream)
