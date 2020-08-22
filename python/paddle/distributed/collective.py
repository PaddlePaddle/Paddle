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
    'all_gather',
    'scatter',
    'ReduceOp',
    'init_process_group',
]


class ReduceOp:
    """Reduce Operation"""
    SUM = 0
    MAX = 1
    MIN = 2
    PROD = 3


class Group():
    """The abstract representation of group."""

    def __init__(self, rank, rank_num):
        self.rank = rank
        self.nranks = rank_num


_default_group = Group(0, 1)
_default_backend = None


def init_process_group(backend,
                       rank_num,
                       rank,
                       timeout=999999,
                       group_name='',
                       fs_path="",
                       fs_name="",
                       fs_ugi=""):
    """

    Initialize the default distributed environment.

    Args:
        backend (str): The backend to use, one of 'nccl' or 'gloo'.
        rank_num (int): Number of processes in the group.
        rank (int): Rank of the current process starting from 0.
        timeout (int): Timeout in seconds for gloo only. 
        group_name (str): Name of the group.
        fs_path (str): A file system path used to initialized gloo.
        fs_name (str): A file system name used to initialized gloo.
        fs_ugi (str): A file system ugi (name and password) used to 
                      initialized gloo.

    Returns:
        None

    Examples:
        .. code-block:: python

        import paddle
        place = paddle.fluid.CUDAPlace(0)
        paddle.distributed.init_process_group('nccl', 2, 1)
    """
    global _default_backend
    if not backend in ['nccl', 'gloo']:
        raise ValueError("backend must be on of 'nccl' or 'gloo' in lowcase, "
                         "but the given one is %s." % backend)
    if _default_backend:
        raise RuntimeError("The default process group has been initialized.")

    _default_backend = backend
    _default_group.rank = rank
    _default_group.nranks = rank_num

    if rank_num < 2:
        raise ValueError(
            "At least 2 ranks are required to use distributed training.")

    if rank >= rank_num or rank < 0:
        raise ValueError("The value of rank must be in [0, rank_num)")

    if backend == 'nccl':
        prepare_context()
    elif backend == 'gloo':
        strategy = fluid.core.GlooParallelStrategy()
        strategy.rank = rank
        strategy.rank_num = rank_num
        strategy.prefix = ""
        strategy.iface = "lo"
        strategy.init_seconds = timeout
        strategy.run_seconds = timeout
        strategy.path = '/tmp/tmp0'
        strategy.fs_name = ""
        strategy.fs_ugi = ""
        gloo = fluid.core.GlooParallelContext(strategy)
        gloo.init()
    else:
        raise ValueError("Unknow backend: %s" % backend)


def broadcast(tensor, src, group=0):
    """

    Broadcast a tensor from the source to all others.

    Args:
        tensor (Tensor): The Tensor to send if current rank is the source, or the tensor to receive otherwise. Its data type
            should be float16, float32, float64, int32 or int64.
        src (int): The source rank.
        group (int): The process group to work on. It is Optional.

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

    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={
            'root': src,
            'ring_id': group,
        })


def all_reduce(tensor, op=ReduceOp.SUM, group=0):
    """

    Reduce a tensor over all ranks so that all get the result.

    Args:
        tensor (Tensor): The input Tensor. It also works as the output Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD): Optional. The operation used.
        group (int): Optional. The process group to work on.

    Returns:
        None.

    Examples:
        .. code-block:: python

        import paddle
        import paddle.fluid as fluid
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
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={'ring_id': group, })


def reduce(tensor, dst, op=ReduceOp.SUM, group=0):
    """

    Reduce a tensor to the destination from all others.

    Args:
        tensor (Tensor): The output Tensor for the destination and the input Tensor otherwise. Its data type
            should be float16, float32, float64, int32 or int64.
        dst (int): The destination rank id.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD): Optional. The operation used.
        group (int): The id of the process group to work on.

    Returns:
        None.

    Examples:
        .. code-block:: python

        import paddle
        import paddle.fluid as fluid

        paddle.disable_static()
        place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
        with fluid.dygraph.guard(place=place):
             paddle.distributed.init_process_group('nccl', 1000, 2, 1)
             if fluid.dygraph.ParallelEnv().local_rank == 0:
                 np_data = np.array([[4, 5, 6], [4, 5, 6]])
             else:
                 np_data = np.array([[1, 2, 3], [1, 2, 3]])
             data = paddle.to_tensor(np_data)
             paddle.distributed.reduce(data, 0)
             out = data.numpy()
             # [[5, 7, 9], [5, 7, 9]]
    """
    op_type = 'c_reduce'
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'all_reduce')
    if not op in [ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN, ReduceOp.PROD]:
        raise ValueError("The op for reduce must be one of educeOp.PROD, "
                         "ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN.")
    if op == ReduceOp.SUM:
        op_type = 'c_reduce_sum'
    elif op == ReduceOp.MAX:
        op_type = 'c_reduce_max'
    elif op == ReduceOp.MIN:
        op_type = 'c_reduce_min'
    elif op == ReduceOp.PROD:
        op_type = 'c_reduce_prod'

    if not isinstance(dst, int) or not isinstance(group, int):
        raise ValueError("Both the type of 'dst' and 'group' for reduce "
                         "should be int.")
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={
            'ring_id': group,
            'root_id': dst,
        })


def all_gather(tensor_list, tensor, group=0):
    """

    Gather tensors from all participators and all get the result.

    Args:
        tensor_list (list): A list of output Tensors. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32 or int64.
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32 or int64.
        group (int): The id of the process group to work on.

    Returns:
        None.

    Examples:
        .. code-block:: python

        import paddle
        import paddle.fluid as fluid

        paddle.disable_static()
        place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
        with fluid.dygraph.guard(place=place):
             paddle.distributed.init_process_group('nccl', 1000, 2, 1)
             tensor_list = []
             if fluid.dygraph.ParallelEnv().local_rank == 0:
                 np_data1 = np.array([[4, 5, 6], [4, 5, 6]])
                 np_data2 = np.array([[4, 5, 6], [4, 5, 6]])
                 data1 = paddle.to_tensor(np_data1)
                 data2 = paddle.to_tensor(np_data2)
                 paddle.distributed.all_gather(tensor_list, data1)
             else:
                 np_data1 = np.array([[1, 2, 3], [1, 2, 3]])
                 np_data2 = np.array([[1, 2, 3], [1, 2, 3]])
                 data1 = paddle.to_tensor(np_data1)
                 data2 = paddle.to_tensor(np_data2)
                 out = paddle.distributed.all_gather(tensor_list, data2)
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
    helper = LayerHelper(op_type, **locals())
    out = helper.create_variable_for_type_inference(dtype=tensor.dtype)
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [out]},
        attrs={'ring_id': group,
               'nranks': _default_group.nranks})

    tensor_list.extend(paddle.split(out, _default_group.nranks, 0))


def scatter(tensor, tensor_list=None, src=0, group=0):
    """

    Scatter a tensor to all participators.

    Args:
        tensor (Tensor): The output Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        tensor_list (list): A list of Tensors to scatter. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32 or int64.
        src (int): The source rank id.
        group (int): The id of the process group to work on.

    Returns:
        None.

    Examples:
        .. code-block:: python

        import paddle
        import paddle.fluid as fluid

        paddle.disable_static()
        place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
        with fluid.dygraph.guard(place=place):
             paddle.distributed.init_process_group('nccl', 1000, 2, 1)
             if fluid.dygraph.ParallelEnv().local_rank == 0:
                 np_data1 = np.array([7, 8, 9])
                 np_data2 = np.array([10, 11, 12])
             else:
                 np_data1 = np.array([1, 2, 3])
                 np_data2 = np.array([4, 5, 6])
             data1 = paddle.to_tensor(np_data1)
             data2 = paddle.to_tensor(np_data2)
             if fluid.dygraph.ParallelEnv().local_rank == 0:
                 paddle.distributed.scatter(data1, src=1)
             else:
                 paddle.distributed.scatter(data1, tensor_list=[data1, data2], src=1)
             out = data1.numpy()
    """
    op_type = 'c_scatter'
    global _default_group
    rank = _default_group.rank
    nranks = _default_group.nranks
    if rank == src:
        if not isinstance(tensor_list, list):
            raise ValueError("The type of 'tensor_list' for all_gather "
                             "should be list for src.")
    else:
        if tensor_list:
            raise ValueError("'tensor_list' for all_gather "
                             "should be None for others.")
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'all_reduce')
    if not isinstance(group, int) or not isinstance(src, int):
        raise ValueError("Both the type of 'src' and 'group' for scatter "
                         "should be int.")
    helper = LayerHelper(op_type, **locals())
    if rank != src:
        tensor_list = []
        for _ in range(nranks):
            tensor_list.append(tensor)
    temp = paddle.concat(tensor_list)
    helper.append_op(
        type=op_type,
        inputs={'X': [temp]},
        outputs={'Out': [tensor]},
        attrs={
            'ring_id': group,
            'root': src,
            'nranks': nranks,
        })
