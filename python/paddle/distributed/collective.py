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

from __future__ import print_function

import numpy as np
from ..fluid.layer_helper import LayerHelper
from ..fluid.framework import Variable, OpProtoHolder, in_dygraph_mode, convert_np_dtype_to_dtype_
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ..fluid.layers.tensor import fill_constant
from ..fluid.layers import utils
from ..fluid.dygraph.parallel import prepare_context
import paddle
import paddle.fluid as fluid

__all__ = [
    'broadcast',
    'all_reduce',
    'reduce',
    'ReduceOp',
    'init_process_group',
]

_default_backend = None


class ReduceOp:
    SUM = 0
    MAX = 1
    MIN = 2
    PROD = 3


def init_process_group(backend,
                       timeout,
                       rank_num,
                       rank,
                       store=None,
                       group_name=''):
    """

    Initialize the default distributed environment.

    Args:
        backend (str): The backend to use, one of 'nccl' or 'gloo'.
        rank_num (int): Number of processes in the distributed process group.
        rank (int): Rank of the current process starting from 0.
        timeout (int): 
        group_name (str): Name of the group.

    Returns:
        None

    Examples:
        .. code-block:: python

        import paddle
        import paddle.fluid as fluid
        place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)

        with fluid.dygraph.guard(place=place):
            paddle.distributed.init_process_group('nccl', 100, 2, 1)
    """
    global _default_backend
    if not backend in ['nccl', 'gloo']:
        raise ValueError("backend must be on of 'nccl' or 'gloo' in lowcase, "
                         "but the given one is %s." % backend)
    if _default_backend:
        raise RuntimeError("The default process group has been initialized.")

    _default_backend = backend

    if rank_num < 2:
        raise ValueError(
            "At least 2 ranks are required to use distributed training.")

    if rank >= rank_num or rank < 0:
        raise ValueError("The value of rank must be in [0, rank_num)")

    if backend == 'nccl':
        prepare_context()
    elif backend == 'gloo':
        gloo = fluid.core.Gloo()
        gloo.set_rank(rank)
        gloo.set_size(rank_num)
        #gloo.set_http_store()
    else:
        raise ValueError("Unknow backend: %s" % backend)


def broadcast(tensor, src, group=0, async_op=False):
    """

    Broadcast a tensor from the source to all others.

    Args:
        tensor (Tensor): The Tensor to send if current rank is the source, or the tensor to receive otherwise. Its data type
            should be float16, float32, float64, int32 or int64.
        src (int): The source rank.
        group (int): The process group to work on. It is Optional.
        async_op (bool): Whether the op is sync or async. It is Optional.

    Returns:
        None.

    Examples:
        .. code-block:: python

        import paddle

        paddle.disable_static()
        place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
        with fluid.dygraph.guard(place=place):
             paddle.distributed.init_process_group('nccl', 1000, 2, 1)
             if fluid.dygraph.ParallelEnv().local_rank == 0:
                 np_data = np.array([[4, 5, 6], [4, 5, 6]])
             else:
                 np_data = np.array([[1, 2, 3], [1, 2, 3]])
             data = paddle.to_tensor(np_data)
             paddle.distributed.broadcast(data, 1)
             out = data.numpy()
             # [[1, 2, 3], [1, 2, 3]]
    """
    op_type = 'c_broadcast'
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'broadcast')
    if not isinstance(src, int) or not isinstance(group, int):
        raise ValueError("Both the type of 'src' and 'group' for broadcast "
                         "should be int.")
    if not isinstance(async_op, bool):
        raise ValueError("The type of 'async_op' for broadcast should be bool.")

    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={
            'root': src,
            'ring_id': group,
            'use_calc_stream': False if async_op else True
        })


def all_reduce(tensor, op=ReduceOp.SUM, group=0, async_op=False):
    """

    Reduce a tensor over all ranks so that all get the result.

    Args:
        tensor (Tensor): The input Tensor. It also works as the output Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD): Optional. The operation used.
        group (int): Optional. The process group to work on.
        async_op (bool): Optional. Whether the op is sync or async.

    Returns:
        None.

    Examples:
        .. code-block:: python

        import paddle
        from paddle.distributed import ReduceOp

        paddle.disable_static()
        place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
        with fluid.dygraph.guard(place=place):
             paddle.distributed.init_process_group('nccl', 1000, 2, 1)
             if fluid.dygraph.ParallelEnv().local_rank == 0:
                 np_data = np.array([[4, 5, 6], [4, 5, 6]])
             else:
                 np_data = np.array([[1, 2, 3], [1, 2, 3]])
             data = paddle.to_tensor(np_data)
             paddle.distributed.all_reduce(data)
             out = data.numpy()
             # [[5, 7, 9], [5, 7, 9]]
    """
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'all_reduce')
    if not op in [ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN, ReduceOp.PROD]:
        raise ValueError("The op for all_reduce must be one of educeOp.PROD, "
                         "ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN.")
    if op == ReduceOp.SUM:
        op_type = 'c_allreduce_sum'
    elif op == ReduceOp.MAX:
        op_type = 'c_allreduce_max'
    elif op == ReduceOp.MIN:
        op_type = 'c_allreduce_min'
    elif op == ReduceOp.PROD:
        op_type = 'c_allreduce_prod'
    if not isinstance(group, int):
        raise ValueError("The type of 'group' for all_reduce should be int.")
    if not isinstance(async_op, bool):
        raise ValueError("The type of 'async_op' for broadcast should be bool.")
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={
            'ring_id': group,
            'use_calc_stream': False if async_op else True
        })


def reduce(tensor, dst, op=ReduceOp.SUM, group=0, async_op=False):
    """

    Reduce a tensor to the destination from all others.

    Args:
        tensor (Tensor): The output Tensor for the destination and the input Tensor otherwise. Its data type
            should be float16, float32, float64, int32 or int64.
        destination (int): The destination rank id.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD): Optional. The operation used.
        group (int): The id of the process group to work on.
        async_op (bool): Whether the op is sync or async.

    Returns:
        None.

    Examples:
    """
    op_type = 'c_reduce'
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'all_reduce')
    if not op in [ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN, ReduceOp.PROD]:
        raise ValueError("The op for reduce must be one of educeOp.PROD, "
                         "ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN.")
    if not isinstance(dst, int) or not isinstance(group, int):
        raise ValueError("Both the type of 'dst' and 'group' for reduce "
                         "should be int.")
    if not isinstance(async_op, bool):
        raise ValueError("The type of 'async_op' for reduce should be bool.")
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={
            'ring_id': group,
            'root_id': dst,
            'use_calc_stream': False if async_op else True
        })


def all_gather(tensor_list, tensor, group=0, async_op=False):
    """

    Gather tensors from all participators and all get the result.

    Args:
        tensor_list (list): A list of output Tensors. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32 or int64.
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32 or int64.
        group (int): The id of the process group to work on.
        async_op (bool): Whether the op is sync or async.

    Returns:
        None.

    Examples:
    """
    op_type = 'c_allgather'
    if not isinstance(tensor_list, list):
        raise ValueError("The type of 'tensor_list' for all_gather "
                         "should be list.")
    for elem in tensor_list:
        check_variable_and_dtype(
            elem, 'tensor_list',
            ['float16', 'float32', 'float64', 'int32', 'int64'], 'all_gather')
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'all_reduce')
    if not isinstance(group, int):
        raise ValueError("The type of 'group' for all_gather " "should be int.")
    if not isinstance(async_op, bool):
        raise ValueError(
            "The type of 'async_op' for all_gather should be bool.")
    helper = LayerHelper(op_type, **locals())
    temp = paddle.concat(tensor_list)
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [temp]},
        attrs={
            'ring_id': group,
            'use_calc_stream': False if async_op else True
        })
    temp = paddle.split(temp, len(tensor_list), 0)
    for i in range(len(temp)):
        tensor_list[i] = temp[i]


def scatter(tensor, tensor_list=None, src=0, group=0, async_op=False):
    """

    Scatter a tensor to all participators.

    Args:
        tensor (Tensor): The output Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        tensor_list (list): A list of Tensors to scatter. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32 or int64.
        src (int): The source rank id.
        group (int): The id of the process group to work on.
        async_op (bool): Whether the op is sync or async.

    Returns:
        None.

    Examples:
    """
    op_type = 'c_scatter'
    if not isinstance(tensor_list, list):
        raise ValueError("The type of 'tensor_list' for all_gather "
                         "should be list.")
    for elem in tensor_list:
        check_variable_and_dtype(
            elem, 'tensor_list',
            ['float16', 'float32', 'float64', 'int32', 'int64'], 'all_gather')
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'all_reduce')
    if not isinstance(group, int) or not isinstance(src, int):
        raise ValueError("Both the type of 'src' and 'group' for scatter "
                         "should be int.")
    if not isinstance(async_op, bool):
        raise ValueError(
            "The type of 'async_op' for all_gather should be bool.")
    helper = LayerHelper(op_type, **locals())
    temp = paddle.concat(tensor_list)
    helper.append_op(
        type=op_type,
        inputs={'X': [temp]},
        outputs={'Out': [tensor]},
        attrs={
            'ring_id': group,
            'root_id': src,
            'use_calc_stream': False if async_op else True
        })
