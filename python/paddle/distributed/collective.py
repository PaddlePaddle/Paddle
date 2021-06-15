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
import os
from ..fluid.layer_helper import LayerHelper
from ..fluid.framework import Variable
from ..fluid.framework import OpProtoHolder
from ..fluid.framework import in_dygraph_mode
from ..fluid.framework import convert_np_dtype_to_dtype_
from ..fluid.data_feeder import convert_dtype
from ..fluid.data_feeder import check_variable_and_dtype
from ..fluid.data_feeder import check_type
from ..fluid.data_feeder import check_dtype
from ..fluid.layers.tensor import fill_constant
from ..fluid.layers import utils
from ..fluid.dygraph.parallel import prepare_context
import paddle
from .fleet import fleet
import paddle.fluid as fluid
import paddle.fluid.core as core

__all__ = []


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


class Group():
    """
    The abstract representation of group.
    """

    def __init__(self, rank, rank_num, id=0, ranks=[]):
        self.rank = rank
        self.nranks = rank_num
        self.id = id
        self.ranks = ranks

    def is_member(self):
        if self.rank < 0:
            return False
        if self.nranks < 2:
            return False
        return True

    def get_group_rank(self, rank):
        if self.id == 0:
            return rank
        if self.is_member() and rank in self.ranks:
            return self.ranks.index(rank)
        else:
            return -1

    def __repr__(self):
        debug_str = "rank: {}, nranks: {}, id: {}, ranks: ".format(
            self.rank, self.nranks, self.id)
        debug_str += ", ".join(map(str, self.ranks))
        debug_str += ". "
        return debug_str


_global_env = None


def _get_global_env():
    global _global_env
    if not _global_env:
        _global_env = paddle.distributed.ParallelEnv()
    return _global_env


# group map : the map of all group, 0 for GlobalGroup
# Dict[int, Group]
_group_map = {}


def _get_group_map():
    global _group_map
    if not _group_map:
        genv = _get_global_env()
        _group_map[0] = Group(genv.rank, genv.world_size, 0)
    return _group_map


def _get_global_group():
    return _get_group_map()[0]


def _new_ring_id():
    return len(_get_group_map()) + max(_get_global_env().nrings, 9)


def get_group(id=0):
    """

    Get group instance by group id.

    Args:
        id (int): the group id. Default value is 0.

    Returns:
        Group: the group instance.

    Examples:
        .. code-block:: python

            ...
            gid = paddle.distributed.new_group([2,4,6])
            paddle.distributed.get_group(gid.id)

    """

    gm = _get_group_map()
    return gm[group] if group in gm else None


def barrier(group=None):
    """

    Barrier among all participators in the group.

    Args:
        group (Group): The group instance return by new_group or None for global default group.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.distributed import init_parallel_env

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            paddle.distributed.barrier()
    """
    if group is not None and not group.is_member():
        return

    ring_id = 0 if group is None else group.id

    op_type = 'barrier'
    temp = fill_constant([1], dtype="int32", value="1")
    if in_dygraph_mode():
        return core.ops.barrier(temp, temp, 'ring_id', ring_id)
    if not isinstance(ring_id, int):
        raise ValueError("The type of 'group' for barrier must be int.")
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [temp]},
        outputs={'Out': [temp]},
        attrs={'ring_id': ring_id})


def new_group(ranks=None, backend=None):
    """

    Creates a new distributed communication group.

    Args:
        ranks (list): The global ranks of group members.
        backend (str): The backend used to create group, only nccl is supported now.

    Returns:
        Group: The group instance.

    Examples:
        .. code-block:: python

            import paddle

            paddle.distributed.init_parallel_env()
            tindata = paddle.randn(shape=[2, 3])
            gp = paddle.distributed.new_group([2,4,6])
            paddle.distributed.all_reduce(tindata, group=gp, use_calc_stream=False)

    """

    if not backend:
        backend = 'nccl'
    assert backend == 'nccl', ("backend other than nccl is not supported yet")

    genv = _get_global_env()
    global_rank = genv.rank

    ring_id = _new_ring_id()

    global _group_map
    if global_rank not in ranks:
        gp = Group(-1, -1, ring_id, ranks)
        _group_map[ring_id] = gp
    else:
        ranks = sorted(ranks)
        group_rank = ranks.index(global_rank)
        group_size = len(ranks)
        gp = Group(group_rank, group_size, ring_id, ranks)
        _group_map[ring_id] = gp

        if group_size >= 2:
            strategy = core.ParallelStrategy()
            strategy.nranks = group_size
            strategy.local_rank = group_rank
            strategy.trainer_endpoints = [
                genv.trainer_endpoints[i] for i in ranks
            ]
            strategy.current_endpoint = genv.current_endpoint
            strategy.nrings = 1

            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(genv.device_id)
                core.NCCLParallelContext(strategy,
                                         place).init_with_ring_id(ring_id)
            else:
                assert False, ("no cuda device found")
        else:
            return gp

    # TODO(shenliang03): This is a temporary solution to solve the problem of 
    # hang caused by cross-creation of new_group
    tmp = paddle.to_tensor(
        [1], dtype="int32") if in_dygraph_mode() else fill_constant(
            [0], dtype="int32", value="1")
    paddle.distributed.all_reduce(tmp, use_calc_stream=True)
    paddle.distributed.wait(tmp)
    return gp


def wait(tensor, group=None, use_calc_stream=True):
    """

    wait to sync stream for group.

    Args:
        tensor (Tensor): The Tensor used before sync.
        group (Group): The Group instance to perform sync.
        use_calc_stream (bool): Wether to use calculation stream (True) or communication stream (False).
            Default to True.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle

            paddle.distributed.init_parallel_env()
            tindata = paddle.randn(shape=[2, 3])
            paddle.distributed.all_reduce(tindata, use_calc_stream=True)
            paddle.distributed.wait(tindata)

    """

    if group is not None and not group.is_member():
        return

    ring_id = 0 if group is None else group.id

    if use_calc_stream:
        _sync_calc_stream(tensor)
    else:
        _sync_comm_stream(tensor, ring_id)


def _sync_calc_stream(tensor):

    if in_dygraph_mode():
        return core.ops.c_sync_calc_stream(tensor, tensor)

    op_type = 'c_sync_calc_stream'

    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]}, )


def _sync_comm_stream(tensor, ring_id=0):

    if in_dygraph_mode():
        return core.ops.c_sync_comm_stream([tensor], [tensor], 'ring_id',
                                           ring_id)

    op_type = 'c_sync_comm_stream'

    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={'ring_id': ring_id}, )


def broadcast(tensor, src, group=None, use_calc_stream=True):
    """

    Broadcast a tensor from the source to all others.

    Args:
        tensor (Tensor): The Tensor to send if current rank is the source, or the tensor to receive otherwise. Its data type
            should be float16, float32, float64, int32 or int64.
        src (int): The source rank.
        group (Group): The group instance return by new_group or None for global default group.
        use_calc_stream (bool): Wether to use calculation stream (True) or communication stream (False).
            Default to True.

    Returns:
        None.

    Examples:
        .. code-block:: python

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
            paddle.distributed.broadcast(data, 1)
            out = data.numpy()
            # [[1, 2, 3], [1, 2, 3]]
    """

    if group is not None and not group.is_member():
        return

    if not isinstance(src, int):
        raise ValueError("src should be int.")

    ring_id = 0 if group is None else group.id
    gsrc = src if group is None else group.get_group_rank(src)
    assert gsrc >= 0, ("src rank out of group, need global rank")

    if in_dygraph_mode():
        return core.ops.c_broadcast(tensor, tensor, 'root', gsrc,
                                    'use_calc_stream', use_calc_stream,
                                    'ring_id', ring_id)

    op_type = 'c_broadcast'
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'broadcast')

    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={
            'root': gsrc,
            'use_calc_stream': use_calc_stream,
            'ring_id': ring_id,
        })


def all_reduce(tensor, op=ReduceOp.SUM, group=None, use_calc_stream=True):
    """

    Reduce a tensor over all ranks so that all get the result.

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

    ring_id = 0 if group is None else group.id
    if in_dygraph_mode():
        if op == ReduceOp.SUM:
            return core.ops.c_allreduce_sum_(
                tensor, 'use_calc_stream', use_calc_stream, 'ring_id', ring_id)
        elif op == ReduceOp.MAX:
            return core.ops.c_allreduce_max_(
                tensor, 'use_calc_stream', use_calc_stream, 'ring_id', ring_id)
        elif op == ReduceOp.MIN:
            return core.ops.c_allreduce_min_(
                tensor, 'use_calc_stream', use_calc_stream, 'ring_id', ring_id)
        elif op == ReduceOp.PROD:
            return core.ops.c_allreduce_prod_(
                tensor, 'use_calc_stream', use_calc_stream, 'ring_id', ring_id)
        else:
            raise ValueError("Unknown parameter: {}.".format(op))
        return out

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
    if not isinstance(ring_id, int):
        raise ValueError("The type of 'ring_id' for all_reduce should be int.")
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={'ring_id': ring_id,
               'use_calc_stream': use_calc_stream})


def reduce(tensor, dst, op=ReduceOp.SUM, group=None, use_calc_stream=True):
    """

    Reduce a tensor to the destination from all others.

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

    if not isinstance(dst, int):
        raise ValueError("dst should be int.")

    ring_id = 0 if group is None else group.id
    gdst = dst if group is None else group.get_group_rank(dst)
    assert gdst >= 0, ("dst rank out of group, need global rank")

    if in_dygraph_mode():
        if op == ReduceOp.SUM:
            return core.ops.c_reduce_sum(tensor, tensor, 'use_calc_stream',
                                         use_calc_stream, 'ring_id', ring_id,
                                         'root_id', gdst)
        elif op == ReduceOp.MAX:
            return core.ops.c_reduce_max(tensor, tensor, 'use_calc_stream',
                                         use_calc_stream, 'ring_id', ring_id,
                                         'root_id', gdst)
        elif op == ReduceOp.MIN:
            return core.ops.c_reduce_min(tensor, tensor, 'use_calc_stream',
                                         use_calc_stream, 'ring_id', ring_id,
                                         'root_id', gdst)
        elif op == ReduceOp.PROD:
            return core.ops.c_reduce_prod(tensor, tensor, 'use_calc_stream',
                                          use_calc_stream, 'ring_id', ring_id,
                                          'root_id', gdst)
        else:
            raise ValueError("Unknown parameter: {}.".format(op))

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

    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={
            'ring_id': ring_id,
            'use_calc_stream': use_calc_stream,
            'root_id': gdst,
        })


def all_gather(tensor_list, tensor, group=None, use_calc_stream=True):
    """

    Gather tensors from all participators and all get the result.

    Args:
        tensor_list (list): A list of output Tensors. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32 or int64.
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32 or int64.
        group (Group): The group instance return by new_group or None for global default group.
        use_calc_stream (bool): Wether to use calculation stream (True) or communication stream (False).
            Default to True.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            from paddle.distributed import init_parallel_env

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            tensor_list = []
            if paddle.distributed.ParallelEnv().local_rank == 0:
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
                paddle.distributed.all_gather(tensor_list, data2)
    """
    if group is not None and not group.is_member():
        return

    ring_id = 0 if group is None else group.id
    nranks = _get_global_group().nranks if group is None else group.nranks

    op_type = 'c_allgather'
    helper = LayerHelper(op_type, **locals())
    out = helper.create_variable_for_type_inference(dtype=tensor.dtype)

    if in_dygraph_mode():
        core.ops.c_allgather(tensor, out, 'use_calc_stream', use_calc_stream,
                             'ring_id', ring_id, 'nranks', nranks)
    else:
        if not isinstance(tensor_list, list):
            raise ValueError("The type of 'tensor_list' for all_gather "
                             "should be list.")
        for elem in tensor_list:
            check_variable_and_dtype(
                elem, 'tensor_list',
                ['float16', 'float32', 'float64', 'int32', 'int64'],
                'all_gather')
        check_variable_and_dtype(
            tensor, 'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64'], 'all_gather')
        helper.append_op(
            type=op_type,
            inputs={'X': [tensor]},
            outputs={'Out': [out]},
            attrs={
                'ring_id': ring_id,
                'use_calc_stream': use_calc_stream,
                'nranks': nranks
            })

    tensor_list.extend(paddle.split(out, nranks, 0))


def scatter(tensor, tensor_list=None, src=0, group=None, use_calc_stream=True):
    """

    Scatter a tensor to all participators.

    Args:
        tensor (Tensor): The output Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        tensor_list (list|tuple): A list/tuple of Tensors to scatter. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32 or int64. Default value is None.
        src (int): The source rank id. Default value is 0.
        group (Group): The group instance return by new_group or None for global default group.
        use_calc_stream (bool): Wether to use calculation stream (True) or communication stream (False).
            Default to True.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            from paddle.distributed import init_parallel_env

            # required: gpu

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            if paddle.distributed.ParallelEnv().local_rank == 0:
                np_data1 = np.array([7, 8, 9])
                np_data2 = np.array([10, 11, 12])
            else:
                np_data1 = np.array([1, 2, 3])
                np_data2 = np.array([4, 5, 6])
            data1 = paddle.to_tensor(np_data1)
            data2 = paddle.to_tensor(np_data2)
            if paddle.distributed.ParallelEnv().local_rank == 0:
                paddle.distributed.scatter(data1, src=1)
            else:
                paddle.distributed.scatter(data1, tensor_list=[data1, data2], src=1)
            out = data1.numpy()
    """
    if group is not None and not group.is_member():
        return

    if not isinstance(src, int):
        raise ValueError("src should be int.")

    ring_id = 0 if group is None else group.id
    gsrc = src if group is None else group.get_group_rank(src)
    assert gsrc >= 0, ("src rank out of group, need global rank")
    rank = _get_global_group().rank if group is None else group.rank
    nranks = _get_global_group().nranks if group is None else group.nranks

    op_type = 'c_scatter'

    if rank != gsrc:
        tensor_list = []
        for _ in range(nranks):
            tensor_list.append(tensor)
    temp = paddle.concat(tensor_list, axis=0)
    if in_dygraph_mode():
        return core.ops.c_scatter(temp, tensor, 'use_calc_stream',
                                  use_calc_stream, 'ring_id', ring_id, 'nranks',
                                  nranks, 'root', gsrc)
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'scatter')
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [temp]},
        outputs={'Out': [tensor]},
        attrs={
            'ring_id': ring_id,
            'root': gsrc,
            'use_calc_stream': use_calc_stream,
            'nranks': nranks,
        })


def _c_identity(tensor, group=None):
    """
    Return a copy of the tensor, mainly used with model parallel.

    Args:
        tensor (Tensor): The input Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        group (int): The id of the process group to work on.

    Returns:
        Tensor.
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    if in_dygraph_mode():
        return core.ops.c_identity(tensor, 'use_calc_stream', True, 'ring_id',
                                   ring_id, 'use_model_parallel', True)
    op_type = 'c_identity'
    helper = LayerHelper(op_type, **locals())
    out = helper.create_variable_for_type_inference(dtype=tensor.dtype)

    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        '_c_identity')

    helper.append_op(
        type=op_type,
        inputs={'X': tensor},
        outputs={'Out': out},
        attrs={
            'ring_id': ring_id,
            'use_calc_stream': True,
            'use_model_parallel': True,
        })
    return out


def _c_concat(tensor, nranks, group=None):
    """
    Return allgather of the tensor, mainly used with model parallel.

    Args:
        tensor (Tensor): The input Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        group (int): The id of the process group to work on.

    Returns:
        Tensor.
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    if in_dygraph_mode():
        return core.ops.c_concat(tensor, 'ring_id', ring_id, 'use_calc_stream',
                                 True, 'nranks', nranks, 'use_model_parallel',
                                 True)

    op_type = 'c_concat'
    helper = LayerHelper(op_type, **locals())
    out = helper.create_variable_for_type_inference(dtype=tensor.dtype)

    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        '_c_concat')

    helper.append_op(
        type=op_type,
        inputs={'X': tensor},
        outputs={'Out': out},
        attrs={
            'ring_id': ring_id,
            'use_calc_stream': True,
            'use_model_parallel': True,
            'nranks': nranks
        })
    return out


def _c_split(tensor, rank, nranks, group=None):
    """
    Split tensor evenly among all members, mainly used with model parallel.

    Args:
        tensor (Tensor): The input Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        rank (int): The rank of the current process.
        group (int): The id of the process group to work on.

    Returns:
        Tensor.
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    if in_dygraph_mode():
        return core.ops.c_split(tensor, 'use_calc_stream', True, 'ring_id',
                                ring_id, 'rank', rank, 'nranks', nranks,
                                'use_model_parallel', True)

    op_type = 'c_split'
    helper = LayerHelper(op_type, **locals())
    out = helper.create_variable_for_type_inference(dtype=tensor.dtype)

    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        '_c_split')

    helper.append_op(
        type=op_type,
        inputs={'X': tensor},
        outputs={'Out': out},
        attrs={
            'ring_id': ring_id,
            'use_calc_stream': True,
            'rank': rank,
            'nranks': nranks,
            'use_model_parallel': True,
        })
    return out


def _mp_allreduce(tensor,
                  op=ReduceOp.SUM,
                  group=None,
                  use_calc_stream=True,
                  use_model_parallel=True):
    """[it is same as allreduce above, but it suuports model parallel. And it support inplace startegy]
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    if in_dygraph_mode():
        if op == ReduceOp.SUM:
            return core.ops.c_allreduce_sum_(
                tensor, 'use_calc_stream', use_calc_stream, 'ring_id', ring_id,
                "use_model_parallel", use_model_parallel)
        else:
            raise ValueError("Unknown parameter: {}.".format(op))
    else:
        raise NotImplementedError("No support _mp_allreduce in dygraph mode.")


def _parallel_linear(x,
                     num_rows,
                     num_cols,
                     axis,
                     param_attr,
                     bias_attr,
                     gather_out,
                     inner_rank,
                     nranks,
                     split_tensor,
                     name,
                     group=None):
    """
    Parallel Linear
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    if axis == 0:
        if split_tensor:
            x = _c_split(x, inner_rank, nranks, group=group)
    else:
        x = _c_identity(x, group=group)

    linear = paddle.nn.Linear(
        num_rows,
        num_cols,
        weight_attr=param_attr,
        bias_attr=bias_attr,
        name=name)

    linear_out = linear(x)
    startup_block = paddle.static.default_startup_program().global_block()
    main_block = paddle.static.default_main_program().global_block()
    startup_block.vars[linear.weight.name].is_distributed = True
    main_block.vars[linear.weight.name].is_distributed = True

    if not gather_out: return linear_out

    op_type = 'c_allreduce_sum' if axis == 0 else 'c_concat'
    out_shape = list(linear_out.shape)
    out_shape[0] *= 1 if axis == 0 else nranks
    out = main_block.create_var(
        shape=out_shape,
        dtype=linear_out.dtype,
        type=linear_out.type,
        lod_level=linear_out.lod_level,
        persistable=False,
        is_data=False,
        need_check_feed=linear_out.desc.need_check_feed())
    if axis == 0:
        main_block.append_op(
            type='c_allreduce_sum',
            inputs={'X': linear_out},
            outputs={'Out': out},
            attrs={
                'ring_id': ring_id,
                'use_calc_stream': True,
                'use_model_parallel': True
            })
    else:
        main_block.append_op(
            type='c_concat',
            inputs={'X': linear_out},
            outputs={'Out': out},
            attrs={
                'ring_id': ring_id,
                'nranks': nranks,
                'use_calc_stream': True,
                'use_model_parallel': True
            })
    return out


def _parallel_embedding(x,
                        per_part_embeddings,
                        origin_size,
                        param_attr,
                        inner_rank,
                        num_partitions,
                        name,
                        group=None):
    """
    Parallel Embedding
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    origin_num_embeddings = origin_size[0]
    embedding = paddle.nn.Embedding(
        per_part_embeddings,
        origin_size[1],
        padding_idx=per_part_embeddings - 1,
        sparse=False,
        weight_attr=param_attr,
        name=name)

    origin_input_shape = x.shape
    if len(origin_input_shape) == 2:
        x = paddle.unsqueeze(x, axis=-1)
    else:
        assert origin_input_shape[-1] == 1, (
            "The last dimension size of x must be 1.")
    x_shard = paddle.shard_index(x, origin_num_embeddings, num_partitions,
                                 inner_rank, per_part_embeddings - 1)
    if len(origin_input_shape) == 2:
        x_shard = paddle.squeeze(x_shard, axis=-1)
    emb_out = embedding(x_shard)
    startup_block = paddle.static.default_startup_program().global_block()
    main_block = paddle.static.default_main_program().global_block()
    startup_block.vars[embedding.weight.name].is_distributed = True
    main_block.vars[embedding.weight.name].is_distributed = True
    out = main_block.create_var(
        shape=emb_out.shape,
        dtype=emb_out.dtype,
        type=emb_out.type,
        lod_level=emb_out.lod_level,
        persistable=False,
        is_data=False,
        need_check_feed=emb_out.desc.need_check_feed())
    main_block.append_op(
        type='c_allreduce_sum',
        inputs={'X': emb_out},
        outputs={'Out': out},
        attrs={
            'ring_id': ring_id,
            'use_calc_stream': True,
            'use_model_parallel': True
        })
    return out


def split(x,
          size,
          operation,
          axis=0,
          num_partitions=1,
          gather_out=True,
          weight_attr=None,
          bias_attr=None,
          name=None):
    """

    Split the weight of the specified operation into multiple devices
    and do the computation in parallel.

    Now the following three cases are supported.

    Case 1: Parallel Embedding
        The weight of the embedding operation is a NxM matrix with N rows and M columns.
        With parallel embedding, the weight is split into num_partitions partitions, each
        of which is a matrix with (N/num_partitions + 1) rows and M column where the last
        row as the padding idx.

        Suppose we split the NxM weight into two partitons on device_0 and device_1
        respectively. Then, one each device, the final weight has (N/2 + 1) rows with the
        index range from 0 to N/2. On device_0, all values in the input within [0, N/2 -1]
        keep unchanged and all other values are changed to N/2 which is the padding index and
        are mapped to all zeros after embedding. In the same way, on device_1, the value V in the
        input within [N/2, N-1] will be changed to (V - N/2), and all other values are changed
        to N/2 and are mapped to all zeros after embedding. Finally, the results on the two
        devices are sum-reduced.

    Case 2: Row Parallel Linear
        The weight of the linear operation is a NxM matrix with N rows and M columns.
        With row parallel linear, the weight is split into num_partitions partitions, each
        of which is a matrix with N/num_partitions rows and M column.

    Case 3: Column Parallel Linear
        The weight of the linear operation is a NxM matrix with N rows and M columns.
        With column parallel linear, the weight is split into num_paratitions partitions, each
        of which is a matrix with N rows and M/num_partitions column.

    Args:
        x (Tensor): Input tensor. It's data type should be float16, float32, float64, int32 or int64.
        size (list|tuple): A list or tuple with two elements indicating the shape of the weight.
        operation (str): The name of the operation. The supported operations are 'linear' and 'embedding'.
        axis (int, Optional): Indicate along which axis to split the weight. Default: 0.
        num_partitions (int, Optional): How many parts the weight is partitioned. Default: 1.
        gather_out (bool, Optional): Whether to gather the output after computation. By default, the output
            on each partitions will be gathered after computation. Default: True.
        weight_attr (ParamAttr, Optional): The parameter attribute for the learnable
            weights(Parameter) of the specified operation. Default: None.
        bias_attr (ParamAttr, Optional): The parameter attribute for the bias
            of the specified operation. Default: None.
        name (str, Optional): The default value is None. Normally there is no need for user to set this
            property. Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.distributed import init_parallel_env

            # required: gpu

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            data = paddle.randint(0, 8, shape=[10,4])
            emb_out = paddle.distributed.split(
                data,
                (8, 8),
                operation="embedding",
                num_partitions=2)
    """
    assert isinstance(size, (list, tuple)), (
        "The type of size for "
        "paddle.distributed.split must be list or tuple.")
    assert len(size) == 2, ("Number of elements in size of "
                            "paddle.distributed.split must be two.")
    assert isinstance(operation, str), ("The type of operation for "
                                        "paddle.distributed.split must be str.")
    supported_operations = [
        'linear',
        'embedding',
    ]
    assert operation in supported_operations, (
        "The operation for "
        "paddle.distributed.split must be one of {}.".format(
            supported_operations))
    if in_dygraph_mode():
        raise ValueError(
            "paddle.distributed.split cannot be used in dynamic "
            "graph mode, plese use ParallelEmbedding, ParallelRowLinear, "
            "ParallelColumnLinear instead.")
    else:
        assert fleet._role_maker, ("To use paddle.distributed.split, "
                                   "you must call fleet.init() firstly.")
        rank = fleet.worker_index()
        nranks = fleet.worker_num()

    # rank within a model parallel group
    inner_rank = rank % num_partitions

    if operation == "embedding":
        assert axis == 0, ("We only support to split the weight of embedding "
                           "along the first axis now.")
        per_part_size = (size[0] + num_partitions - 1) // num_partitions
        last_part_size = size[0] - per_part_size * (num_partitions - 1)
        if inner_rank == num_partitions - 1: per_part_size = last_part_size
        per_part_size += 1  # make the last row as the padding index

        emb_out = _parallel_embedding(
            x,
            per_part_size,
            size,
            weight_attr,
            inner_rank,
            num_partitions,
            name,
            group=None)
        return emb_out
    else:
        should_split = False
        if axis == 0:
            assert size[0] % num_partitions == 0, (
                "Number of rows of the weight for linear ({}) must be"
                " divisible by num_partitions ({})".format(size[0],
                                                           num_partitions))
            per_part_size = size[0] // num_partitions
            linear_size = (per_part_size, size[1])
            if x.shape[-1] == size[0]: should_split = True

        elif axis == 1:
            assert size[1] % num_partitions == 0, (
                "Number of column of the weight for linear ({}) must be"
                " divisible by num_partitions ({})".format(size[1],
                                                           num_partitions))
            per_part_size = size[1] // num_partitions
            linear_size = (size[0], per_part_size)
        else:
            raise ValueError("The value of axis must be 0 or 1, but the value "
                             "given is {}.".format(axis))

        linear_out = _parallel_linear(
            x,
            linear_size[0],
            linear_size[1],
            axis,
            weight_attr,
            bias_attr,
            gather_out,
            inner_rank,
            num_partitions,
            should_split,
            name=name,
            group=None)
        return linear_out


def send(tensor, dst=0, group=None, use_calc_stream=True):
    """
    Send a tensor to the receiver.

    Args:
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32 or int64.
        dst (int): The destination rank id.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
        use_calc_stream (bool, optional): Whether to use calculate stream or communication stream. Default: True.
    Returns:
        None.

    Examples:
        .. code-block:: python
            # required: distributed
            import paddle
            from paddle.distributed import init_parallel_env
            init_parallel_env()
            if paddle.distributed.ParallelEnv().rank == 0:
                data = paddle.to_tensor([7, 8, 9])
                paddle.distributed.send(data, dst=1)
            else:
                data = paddle.to_tensor([1,2,3])
                paddle.distributed.recv(data, src=0)
            out = data.numpy()
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    op_type = 'send_v2'
    if in_dygraph_mode():
        return core.ops.send_v2(tensor, 'use_calc_stream', use_calc_stream,
                                'ring_id', ring_id, 'peer', dst)
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'send')

    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        attrs={
            'ring_id': ring_id,
            'peer': dst,
            'use_calc_stream': use_calc_stream,
        })


def recv(tensor, src=0, group=None, use_calc_stream=True):
    """
    Receive a tensor to the sender.

    Args:
        tensor (Tensor): The Tensor to receive. Its data type
            should be float16, float32, float64, int32 or int64.
        src (int): The source rank id.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
        use_calc_stream (bool, optional): Whether to use calculate stream or communication stream. Default: True.
    Returns:
        None.

    Examples:
        .. code-block:: python
            # required: distributed
            import paddle
            from paddle.distributed import init_parallel_env
            init_parallel_env()
            if paddle.distributed.ParallelEnv().rank == 0:
                data = paddle.to_tensor([7, 8, 9])
                paddle.distributed.send(data, dst=1)
            else:
                data = paddle.to_tensor([1,2,3])
                paddle.distributed.recv(data, src=0)
            out = data.numpy()
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    op_type = 'recv_v2'
    if in_dygraph_mode():
        return core.ops.recv_v2(tensor, 'use_calc_stream', use_calc_stream,
                                'ring_id', ring_id, 'peer', src, 'dtype',
                                tensor.dtype, 'out_shape', tensor.shape)
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'recv')
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        outputs={'Out': [tensor]},
        attrs={
            'ring_id': ring_id,
            'peer': src,
            'out_shape': tensor.shape,
            'dtype': tensor.dtype,
            'use_calc_stream': use_calc_stream,
        })
