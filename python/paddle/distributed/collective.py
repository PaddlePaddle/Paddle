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
import pickle
import io
import datetime
import time
from ..fluid.layer_helper import LayerHelper
from ..fluid.framework import Variable
from ..fluid.framework import in_dygraph_mode
from ..fluid.framework import OpProtoHolder
from ..fluid.framework import _non_static_mode
from ..fluid.framework import _in_legacy_dygraph
from ..fluid.framework import convert_np_dtype_to_dtype_
from ..fluid.framework import _varbase_creator
from ..fluid.data_feeder import convert_dtype
from ..fluid.data_feeder import check_variable_and_dtype
from ..fluid.data_feeder import check_type
from ..fluid.data_feeder import check_dtype
from ..fluid.layers.tensor import fill_constant
from ..fluid.layers import utils
from ..fluid.dygraph import layers
from ..fluid.dygraph.parallel import prepare_context
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle import _C_ops, _legacy_C_ops
import paddle.fluid.dygraph_utils as dygraph_utils
import contextlib
from .fleet.layers.mpu.mp_ops import split
from .fleet.layers.mpu.mp_ops import _c_identity
from .fleet.layers.mpu.mp_ops import _c_concat
from .fleet.layers.mpu.mp_ops import _c_split
from .fleet.layers.mpu.mp_ops import _mp_allreduce
from .fleet.layers.mpu.mp_ops import _c_lookup_table
from .fleet.layers.mpu.mp_ops import _Linear
from .fleet.layers.mpu.mp_ops import _set_var_distributed
from .fleet.layers.mpu.mp_ops import _c_softmax_with_cross_entropy
from .fleet.layers.mpu.mp_ops import _linear
from .fleet.layers.mpu.mp_ops import _parallel_linear
from .fleet.layers.mpu.mp_ops import _parallel_embedding
from .communication.comm_utils import ReduceOp

__all__ = []


class Group():
    """
    The abstract representation of group.
    """

    def __init__(self, rank, rank_num, id=0, ranks=[], pg=None, name=None):
        self.rank = rank
        self.nranks = rank_num
        self.id = id
        self.ranks = ranks
        self.pg = pg
        self.name = name

    def is_member(self):
        if self.rank < 0:
            return False
        if self.nranks < 2:
            return False
        return True

    def get_group_rank(self, rank):
        if self.is_member() and rank in self.ranks:
            return self.ranks.index(rank)
        else:
            return -1

    @property
    def process_group(self):
        return self.pg

    @property
    def world_size(self):
        return self.nranks if self.rank >= 0 else -1

    def __repr__(self):
        debug_str = "rank: {}, nranks: {}, id: {}, ranks: ".format(
            self.rank, self.nranks, self.id)
        debug_str += ", ".join(map(str, self.ranks))
        debug_str += "; name: "
        debug_str += self.name if self.name else "None"
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
_global_env_gid = 0

# group map by name : the map of all groups from their names
# Dict[name, Group]
_group_map_by_name = {}

# backend map by group : the map of all backend from their groups
# Dict[group, backend]
_group_map_backend = {}

# Name of the default group for init_parallel_env
_default_group_name = "_default_pg"

_valid_backend_list = ['nccl', 'gloo', 'hccl', 'heter', 'xccl']
_default_store = None  # the default tcp store
_default_backend = None
_default_timeout = datetime.timedelta(seconds=1800)
_start_ring_id = 0


def _set_default_backend(backend):
    global _default_backend
    _default_backend = backend


def _set_default_store(store):
    global _default_store
    _default_store = store


def _get_group_map():
    global _group_map
    if _global_env_gid not in _group_map:
        genv = _get_global_env()
        _group_map[_global_env_gid] = Group(genv.rank,
                                            genv.world_size,
                                            ranks=list(range(genv.world_size)))
    return _group_map


def _get_global_group():
    return _get_group_map()[_global_env_gid]


def _get_group_map_by_name():
    global _group_map_by_name
    return _group_map_by_name


def _get_default_group():
    global _group_map_by_name
    assert is_initialized(), ("Call paddle.distributed.init_parallel_env first "
                              "to initialize the distributed environment.")
    return _get_group_map_by_name()[_default_group_name]


def _set_group_map(gid, group):
    global _group_map
    assert gid not in _group_map
    _group_map[gid] = group


def _set_group_map_by_name(name, group):
    global _group_map_by_name
    assert name not in _group_map_by_name
    _group_map_by_name[name] = group


def _set_group_map_backend(group, backend):
    global _group_map_backend
    assert group not in _group_map_backend
    _group_map_backend[group] = backend


def _new_ring_id():
    # NOTE(liyurui): For compatible reason, auto parallel and eager mode relay on previous syntax.
    if in_dygraph_mode():
        global _start_ring_id
        _start_ring_id += 1
        return _start_ring_id + max(_get_global_env().nrings, 9)
    else:
        return len(_get_group_map()) + max(_get_global_env().nrings, 9)


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
    return gm[id] if id in gm else None


def _new_process_group_impl(backend,
                            store,
                            rank,
                            world_size,
                            group_name,
                            pg_options,
                            group_id=0,
                            src_rank=None,
                            dst_rank=None):
    pg = None
    genv = _get_global_env()
    if backend != 'heter':
        assert src_rank is None and dst_rank is None, (
            "src_rank and dst_rank "
            "can only be set for heter backend.")
    assert backend in _valid_backend_list, "Unsupported backend: %s." % backend
    if backend == "gloo":
        place = core.CPUPlace()
        pg = core.ProcessGroupGloo(store, rank, world_size, place, group_id)
    elif backend == "nccl":
        place = core.CUDAPlace(genv.device_id)
        pg = core.ProcessGroupNCCL(store, rank, world_size, place, group_id)
    elif backend == "hccl":
        place = core.NPUPlace(genv.device_id)
        pg = core.ProcessGroupHCCL(store, rank, world_size, place, group_id)
    elif backend == "xccl":
        place = core.CustomPlace(genv.device_type, genv.device_id)
        pg = core.ProcessGroupCustom(store, rank, world_size, place, group_id)
    elif backend == "heter":
        place = None
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(genv.device_id)
        elif core.is_compiled_with_npu():
            place = core.NPUPlace(genv.device_id)
        cluster_id = int(os.getenv("CLUSTER_ID", "-1"))
        assert cluster_id >= 0, "please set the CLUSTER_ID variable."
        cluster_size = os.getenv("CLUSTER_SIZE", None)
        assert cluster_size, "please set the CLUSTER_SIZE variable."
        cluster_size = cluster_size.split(",")
        cluster_size = [int(s) for s in cluster_size]
        switch_ep = os.getenv("CLUSTER_SWITCH", None)
        assert switch_ep, "please set the CLUSTER_SWITCH variable."
        cluster_size_cumsum = np.cumsum(cluster_size)
        cluster_offset = 0 if cluster_id == 0 else cluster_size_cumsum[
            cluster_id - 1]
        global_rank = cluster_offset + rank
        global_world_size = cluster_size_cumsum[-1]
        global_rank, global_world_size = _get_global_config(backend, rank)
        pg = core.ProcessGroupHeter(store,
                                    rank=global_rank,
                                    world_size=global_world_size,
                                    place=place,
                                    gid=group_id,
                                    local_rank=rank,
                                    local_size=world_size,
                                    gloo_rank=cluster_id,
                                    gloo_size=len(cluster_size),
                                    with_switch=True,
                                    switch_endpoint=switch_ep,
                                    src_rank=src_rank,
                                    dst_rank=dst_rank)

    return pg


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

    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        task = group.process_group.barrier()
        task.wait()
        return

    ring_id = 0 if group is None else group.id

    temp = fill_constant([1], dtype="int32", value="1")
    if _non_static_mode():
        return _legacy_C_ops.barrier(temp, temp, 'ring_id', ring_id)

    op_type = 'barrier'

    if not isinstance(ring_id, int):
        raise ValueError("The type of 'group' for barrier must be int.")
    helper = LayerHelper(op_type, **locals())
    helper.append_op(type=op_type,
                     inputs={'X': [temp]},
                     outputs={'Out': [temp]},
                     attrs={'ring_id': ring_id})


# _custom_gid provides a way for users to
# set the group id, which is usually useful
# to be compatible with the static mode.
_custom_gid = None


def _set_custom_gid(gid):
    global _custom_gid
    _custom_gid = gid


def _barrier_by_tcp_store(group_name, store, timeout):
    global_rank = paddle.distributed.get_rank()
    global_world_size = paddle.distributed.get_world_size()

    if global_world_size < 2:
        return

    barrier_prefix = "Barrier/" + group_name + "/"
    is_master = (global_rank == 0)

    def _check_keys_ready(wait_keys):
        start_time = time.time()
        while len(wait_keys) > 0:
            time.sleep(0.1)
            elapse_time = time.time() - start_time
            if datetime.timedelta(seconds=elapse_time) > timeout:
                raise RuntimeError(
                    "Timeout while initializing process group {}."
                    "Keys {} are not ready sinck rank {} is waiting them."
                    "Two reason may cause this error:\n 1. The create process group api should be called by all ranks.\n"
                    " 2. Try to increase the waiting time.\n".format(
                        group_name, wait_keys, global_rank))
            wait_keys = list(
                filter(lambda key: int(store.get(key)) != 1, wait_keys))

    # all the workers set their exiting key and exit
    # the master will wait for all workers' exiting key, ensure to exit in the end
    if is_master:
        wait_keys = [
            barrier_prefix + str(rank) for rank in range(1, global_world_size)
        ]
        _check_keys_ready(wait_keys)
    else:
        store.add(barrier_prefix + str(global_rank), 1)


def new_group(ranks=None, backend=None, timeout=_default_timeout):
    """

    Creates a new distributed communication group.

    Args:
        ranks (list): The global ranks of group members.
        backend (str): The backend used to create group, only nccl is supported now.
        timeout (datetime.timedelta, optional): The waiting timeout for store relevant options, default is 30 minutes.

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
    global _custom_gid
    global _group_map
    if in_dygraph_mode():
        global _default_group_name
        gid = _custom_gid if _custom_gid else _new_ring_id()
        group_name = _default_group_name + str(gid)
        if backend != 'heter' and (ranks is None or len(ranks) > 1):
            global_group = _get_default_group()
            global_rank = global_group.rank
            global_ranks = global_group.ranks
            backend = _default_backend if backend is None else backend
            if ranks is None:
                ranks = global_ranks
            assert len(ranks) <= len(global_ranks), (
                "Size of new group must be less than or "
                "equal to that of the default global group.")
        size = len(ranks)
        ranks = sorted(ranks)
        if backend == 'heter' or (size > 1 and global_rank in ranks):
            rank = 0 if backend == 'heter' else ranks.index(global_rank)
            src_rank = ranks[0] if backend == 'heter' else None
            dst_rank = ranks[1] if backend == 'heter' else None
            pg = _new_process_group_impl(backend,
                                         _default_store,
                                         rank,
                                         size,
                                         group_name,
                                         pg_options=None,
                                         group_id=gid,
                                         src_rank=src_rank,
                                         dst_rank=dst_rank)
        else:
            rank = -1
            pg = None
        group = Group(rank, size, id=gid, ranks=ranks, pg=pg, name=group_name)
        _group_map_by_name[group_name] = group
        _group_map[gid] = group
        _group_map_backend[group] = backend

        # TODO(shenliang03): This is a temporary solution to solve the problem of
        # hang caused by tcp
        paddle.distributed.barrier(group=group)
        # NOTE(liyurui): All processors should hang and wait using tcp store, in case master exit before sub-group is created.
        if backend != 'heter':
            _barrier_by_tcp_store(group_name, _default_store, timeout)
        else:
            print("Warning: store barrier is not supported for heter backend.")
        return group

    if not backend:
        backend = 'nccl'
    assert backend == 'nccl', ("backend other than nccl is not supported yet")

    genv = _get_global_env()
    global_rank = genv.rank

    ring_id = _new_ring_id()

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
            elif core.is_compiled_with_npu():
                place = core.NPUPlace(genv.device_id)
                core.HCCLParallelContext(strategy,
                                         place).init_with_ring_id(ring_id)
            elif core.is_compiled_with_mlu():
                place = core.MLUPlace(genv.device_id)
                core.CNCLParallelContext(strategy,
                                         place).init_with_ring_id(ring_id)
            elif core.is_compiled_with_xpu():
                place = core.XPUPlace(genv.device_id)
                core.BKCLParallelContext(strategy,
                                         place).init_with_ring_id(ring_id)
            else:
                assert False, ("no cuda device found")
        else:
            return gp

    # TODO(shenliang03): This is a temporary solution to solve the problem of
    # hang caused by cross-creation of new_group
    tmp = paddle.to_tensor(
        [1], dtype="int32") if _non_static_mode() else fill_constant(
            [0], dtype="int32", value="1")
    paddle.distributed.all_reduce(tmp, use_calc_stream=True)
    paddle.distributed.wait(tmp)
    return gp


def is_initialized():
    """

    Check whether the distributed environment has been initialized

    Returns (bool): `True` if distributed environment has been initialized, otherwise `False`.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle

            print(paddle.distributed.is_initialized())
            # False

            paddle.distributed.init_parallel_env()
            print(paddle.distributed.is_initialized())
            # True

    """
    global _group_map_by_name
    return _default_group_name in _group_map_by_name


def destroy_process_group(group=None):
    """
    Destroy a given group for communication

    Args:
        group (ProcessGroup, optional): The group to be destroyed. All of process groups, including
                                        the default group, will be destroyed and the distributed
                                        environment will be deinitialized.

    Returns : None

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            group = dist.new_group([0, 1])

            dist.destroy_process_group(group)
            print(dist.is_initialized())
            # True
            dist.destroy_process_group()
            print(dist.is_initialized())
            # False

    """
    global _group_map
    global _group_map_by_name

    pg = _get_default_group() if group is None else group
    assert _group_map.get(pg.id, None) is not None, "Invalid group."

    if group is None:
        _group_map.clear()
        _group_map_by_name.clear()
        _group_map_backend.clear()
    else:
        del _group_map[pg.id]
        del _group_map_by_name[pg.name]
        del _group_map_backend[pg]


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

    if _non_static_mode():
        return _legacy_C_ops.c_sync_calc_stream(tensor, tensor)

    op_type = 'c_sync_calc_stream'

    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
    )


def _sync_comm_stream(tensor, ring_id=0):

    if _non_static_mode():
        return _legacy_C_ops.c_sync_comm_stream([tensor], [tensor], 'ring_id',
                                                ring_id)

    op_type = 'c_sync_comm_stream'

    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={'ring_id': ring_id},
    )


def broadcast(tensor, src, group=None, use_calc_stream=True):
    """

    Broadcast a tensor from the source to all others.
    As shown below, one process is started with a GPU and GPU0 owns data 0. Through broadcast operator,
    data 0 will be sent to all GPUs from GPU0.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/broadcast.png
        :width: 800
        :alt: broadcast
        :align: center

    Args:
        tensor (Tensor): The Tensor to send if current rank is the source, or the Tensor to receive otherwise. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        src (int): The source rank.
        group (Group): The group instance return by new_group or None for global default group.
        use_calc_stream (bool): Wether to use calculation stream (True) or communication stream (False).
            Default to True.

    Returns:
        None.

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
            dist.broadcast(data, src=1)
            print(data)
            # [[1, 2, 3], [1, 2, 3]] (2 GPUs)
    """

    if group is not None and not group.is_member():
        return

    if not isinstance(src, int):
        raise ValueError("src should be int.")

    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        gsrc = group.get_group_rank(src)
        assert gsrc >= 0, ("src rank out of group, need global rank")
        task = group.process_group.broadcast(tensor, gsrc)
        if use_calc_stream:
            task.wait()
            return None
        else:
            return task

    ring_id = ring_id = 0 if group is None else group.id
    gsrc = src if group is None else group.get_group_rank(src)
    assert gsrc >= 0, ("src rank out of group, need global rank")

    if _non_static_mode():
        return _legacy_C_ops.c_broadcast(tensor, tensor, 'root', gsrc,
                                         'use_calc_stream', use_calc_stream,
                                         'ring_id', ring_id)

    op_type = 'c_broadcast'
    check_variable_and_dtype(tensor, 'tensor', [
        'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8',
        'bool'
    ], 'broadcast')

    helper = LayerHelper(op_type, **locals())
    helper.append_op(type=op_type,
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
    As shown below, one process is started with a GPU and the data of this process is represented
    by its group rank. The reduce operator is sum. Through all_reduce operator,
    each GPU will have the sum of the data from all GPUs.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/allreduce.png
        :width: 800
        :alt: all_reduce
        :align: center

    Args:
        tensor (Tensor): The input Tensor. It also works as the output Tensor. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD): Optional. The operation used. Default value is ReduceOp.SUM.
        group (Group): The group instance return by new_group or None for global default group.
        use_calc_stream (bool): Wether to use calculation stream (True) or communication stream (False).
            Default to True.

    Returns:
        None.

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
            dist.all_reduce(data)
            print(data)
            # [[5, 7, 9], [5, 7, 9]] (2 GPUs)
    """
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        op_type = _get_reduce_op(op, "all_reduce")
        group = _get_default_group() if group is None else group
        task = group.process_group.allreduce(tensor, op_type)
        if use_calc_stream:
            task.wait()
            return None
        else:
            return task

    ring_id = 0 if group is None else group.id
    if _non_static_mode():
        if op == ReduceOp.SUM:
            return _legacy_C_ops.c_allreduce_sum_(tensor, 'use_calc_stream',
                                                  use_calc_stream, 'ring_id',
                                                  ring_id)
        elif op == ReduceOp.MAX:
            return _legacy_C_ops.c_allreduce_max_(tensor, 'use_calc_stream',
                                                  use_calc_stream, 'ring_id',
                                                  ring_id)
        elif op == ReduceOp.MIN:
            return _legacy_C_ops.c_allreduce_min_(tensor, 'use_calc_stream',
                                                  use_calc_stream, 'ring_id',
                                                  ring_id)
        elif op == ReduceOp.PROD:
            return _legacy_C_ops.c_allreduce_prod_(tensor, 'use_calc_stream',
                                                   use_calc_stream, 'ring_id',
                                                   ring_id)
        else:
            raise ValueError("Unknown parameter: {}.".format(op))

    check_variable_and_dtype(tensor, 'tensor', [
        'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8',
        'bool'
    ], 'all_reduce')
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


def reduce(tensor, dst, op=ReduceOp.SUM, group=None, use_calc_stream=True):
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
            should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        dst (int): The destination rank id.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD): Optional. The operation used. Default value is ReduceOp.SUM.
        group (Group): The group instance return by new_group or None for global default group.
        use_calc_stream (bool): Wether to use calculation stream (True) or communication stream (False).
            Default to True.

    Returns:
        None.

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
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
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

    ring_id = 0 if group is None else group.id
    gdst = dst if group is None else group.get_group_rank(dst)
    assert gdst >= 0, ("dst rank out of group, need global rank")

    if _non_static_mode():
        if op == ReduceOp.SUM:
            return _legacy_C_ops.c_reduce_sum(tensor, tensor, 'use_calc_stream',
                                              use_calc_stream, 'ring_id',
                                              ring_id, 'root_id', gdst)
        elif op == ReduceOp.MAX:
            return _legacy_C_ops.c_reduce_max(tensor, tensor, 'use_calc_stream',
                                              use_calc_stream, 'ring_id',
                                              ring_id, 'root_id', gdst)
        elif op == ReduceOp.MIN:
            return _legacy_C_ops.c_reduce_min(tensor, tensor, 'use_calc_stream',
                                              use_calc_stream, 'ring_id',
                                              ring_id, 'root_id', gdst)
        elif op == ReduceOp.PROD:
            return _legacy_C_ops.c_reduce_prod(tensor, tensor,
                                               'use_calc_stream',
                                               use_calc_stream, 'ring_id',
                                               ring_id, 'root_id', gdst)
        else:
            raise ValueError("Unknown parameter: {}.".format(op))

    op_type = 'c_reduce'
    check_variable_and_dtype(tensor, 'tensor', [
        'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8',
        'bool'
    ], 'reduce')

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


def all_gather(tensor_list, tensor, group=None, use_calc_stream=True):
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
            should be float16, float32, float64, int32, int64, int8, uint8, bool, complex64 or complex128.
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool, complex64 or complex128.
        group (Group): The group instance return by new_group or None for global default group.
        use_calc_stream (bool): Wether to use calculation stream (True) or communication stream (False).
            Default to True.

    Returns:
        None.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            tensor_list = []
            if dist.get_rank() == 0:
                data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
            else:
                data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            dist.all_gather(tensor_list, data)
            print(tensor_list)
            # [[[4, 5, 6], [4, 5, 6]], [[1, 2, 3], [1, 2, 3]]] (2 GPUs)
    """
    if group is not None and not group.is_member():
        return

    def convert_to_complex(list_of_tensor):
        list_of_complex = []
        for tensor in list_of_tensor:
            list_of_complex.append(paddle.as_complex(tensor))
        return list_of_complex

    is_input_complex = (tensor.dtype == paddle.complex64
                        or tensor.dtype == paddle.complex128)
    if is_input_complex:
        tensor = paddle.as_real(tensor)

    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        if len(tensor_list) == 0:
            tensor_shape = list(tensor.shape)
            tensor_shape[0] *= group.nranks
            out = paddle.empty(tensor_shape, tensor.dtype)
        else:
            out = paddle.concat(tensor_list, axis=0)
        task = group.process_group.all_gather(tensor, out)
        task.wait()
        tensor_list.clear()
        list_of_tensor = paddle.split(out, group.nranks, 0)
        if is_input_complex:
            tensor_list.extend(convert_to_complex(list_of_tensor))
        else:
            tensor_list.extend(list_of_tensor)
        return

    ring_id = 0 if group is None else group.id
    nranks = _get_global_group().nranks if group is None else group.nranks

    if _non_static_mode():
        out = _legacy_C_ops.c_allgather(tensor, 'use_calc_stream',
                                        use_calc_stream, 'ring_id', ring_id,
                                        'nranks', nranks)
    else:
        op_type = 'c_allgather'
        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=tensor.dtype)
        if not isinstance(tensor_list, list):
            raise ValueError("The type of 'tensor_list' for all_gather "
                             "should be list.")
        for elem in tensor_list:
            check_variable_and_dtype(elem, 'tensor_list', [
                'float16', 'float32', 'float64', 'int32', 'int64', 'bool',
                'int8', 'uint8', 'complex64', 'complex128'
            ], 'all_gather')
        check_variable_and_dtype(tensor, 'tensor', [
            'float16', 'float32', 'float64', 'int32', 'int64', 'bool', 'int8',
            'uint8', 'complex64', 'complex128'
        ], 'all_gather')
        helper.append_op(type=op_type,
                         inputs={'X': [tensor]},
                         outputs={'Out': [out]},
                         attrs={
                             'ring_id': ring_id,
                             'use_calc_stream': use_calc_stream,
                             'nranks': nranks
                         })

    list_of_tensor = paddle.split(out, nranks, 0)
    if is_input_complex:
        tensor_list.extend(convert_to_complex(list_of_tensor))
    else:
        tensor_list.extend(list_of_tensor)


def _convert_object_to_tensor(obj):
    _pickler = pickle.Pickler
    f = io.BytesIO()
    _pickler(f).dump(obj)
    data = np.frombuffer(f.getvalue(), dtype=np.uint8)
    tensor = paddle.to_tensor(data)
    return tensor, tensor.numel()


def _convert_tensor_to_object(tensor, len_of_tensor):
    _unpickler = pickle.Unpickler
    return _unpickler(io.BytesIO(tensor.numpy()[:len_of_tensor])).load()


def all_gather_object(object_list, obj, group=None):
    """

    Gather picklable objects from all participators and all get the result. Similiar to all_gather(), but python object can be passed in.

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

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            object_list = []
            if dist.get_rank() == 0:
                obj = {"foo": [1, 2, 3]}
            else:
                obj = {"bar": [4, 5, 6]}
            dist.all_gather_object(object_list, obj)
            print(object_list)
            # [{'foo': [1, 2, 3]}, {'bar': [4, 5, 6]}] (2 GPUs)
    """
    assert in_dygraph_mode(
    ), "all_gather_object doesn't support static graph mode."

    tensor, len_of_tensor = _convert_object_to_tensor(obj)

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
            _convert_tensor_to_object(tensor, list_len_of_tensor[i]))


def scatter(tensor, tensor_list=None, src=0, group=None, use_calc_stream=True):
    """

    Scatter a tensor to all participators. As shown below, one process is started with a GPU and the source of the scatter
    is GPU0. Through scatter operator, the data in GPU0 will be sent to all GPUs averagely.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/scatter.png
        :width: 800
        :alt: scatter
        :align: center

    Args:
        tensor (Tensor): The output Tensor. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        tensor_list (list|tuple): A list/tuple of Tensors to scatter. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32, int64, int8, uint8 or bool. Default value is None.
        src (int): The source rank id. Default value is 0.
        group (Group): The group instance return by new_group or None for global default group.
        use_calc_stream (bool): Wether to use calculation stream (True) or communication stream (False).
            Default to True.

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
    if group is not None and not group.is_member():
        return

    if not isinstance(src, int):
        raise ValueError("src should be int.")

    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        gsrc = group.get_group_rank(src)
        rank = group.rank
        nranks = group.nranks
    else:
        ring_id = 0 if group is None else group.id
        gsrc = src if group is None else group.get_group_rank(src)
        rank = _get_global_group().rank if group is None else group.rank
        nranks = _get_global_group().nranks if group is None else group.nranks
    assert gsrc >= 0, ("src rank out of group, need global rank")

    if rank != gsrc:
        tensor_list = []
        for _ in range(nranks):
            tensor_list.append(tensor)
    temp = paddle.concat(tensor_list, axis=0)
    if in_dygraph_mode():
        task = group.process_group.scatter(temp, tensor, gsrc)
        if use_calc_stream:
            task.wait()
            return None
        else:
            return task

    if _non_static_mode():
        return _legacy_C_ops.c_scatter(temp, tensor, 'use_calc_stream',
                                       use_calc_stream, 'ring_id', ring_id,
                                       'nranks', nranks, 'root', gsrc)
    op_type = 'c_scatter'
    check_variable_and_dtype(tensor, 'tensor', [
        'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8',
        'bool'
    ], 'scatter')
    helper = LayerHelper(op_type, **locals())
    helper.append_op(type=op_type,
                     inputs={'X': [temp]},
                     outputs={'Out': [tensor]},
                     attrs={
                         'ring_id': ring_id,
                         'root': gsrc,
                         'use_calc_stream': use_calc_stream,
                         'nranks': nranks,
                     })


def alltoall(in_tensor_list, out_tensor_list, group=None, use_calc_stream=True):
    """
    Scatter tensors in in_tensor_list to all participators averagely and gather the result tensors in out_tensor_list.
    As shown below, the in_tensor_list in GPU0 includes 0_0 and 0_1, and GPU1 includes 1_0 and 1_1.
    Through alltoall operator, the 0_0 in GPU0 will be sent to GPU0 and 0_1 to GPU1, 1_0 in GPU1 sent to GPU0 and 1_1 to GPU1.
    Finally the out_tensor_list in GPU0 includes 0_0 and 1_0, and GPU1 includes 0_1 and 1_1.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/alltoall.png
        :width: 800
        :alt: alltoall
        :align: center

    Args:
        in_tensor_list (list): A list of input Tensors. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        out_tensor_list (list): A list of output Tensors. The data type of its elements should be the same as the
            data type of the input Tensors.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
        use_calc_stream (bool, optional): Whether to use calculation stream (True) or communication stream. Default: True.

    Returns:
        None.

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
            dist.alltoall([data1, data2], out_tensor_list)
            print(out_tensor_list)
            # [[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]] (2 GPUs, out for rank 0)
            # [[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]] (2 GPUs, out for rank 1)
    """
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        backend = _group_map_backend[group]
        assert backend != 'gloo', ("backend gloo is not supported yet")
    else:
        ring_id = 0 if group is None else group.id

    temp = paddle.concat(in_tensor_list, axis=0)
    nranks = len(in_tensor_list)
    if in_dygraph_mode():
        if len(out_tensor_list) == 0:
            tensor_shape = list(in_tensor_list[0].shape)
            tensor_shape[0] *= nranks
            out = paddle.empty(tensor_shape, in_tensor_list[0].dtype)
        else:
            out = paddle.concat(out_tensor_list, axis=0)
        task = group.process_group.alltoall(temp, out)
        task.wait()
        out_tensor_list.clear()
        out_tensor_list.extend(paddle.split(out, nranks, 0))
        return

    if _non_static_mode():
        out = _legacy_C_ops.alltoall(temp, 'use_calc_stream', use_calc_stream,
                                     'ring_id', ring_id)
    else:
        op_type = 'alltoall'
        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(
            dtype=in_tensor_list[0].dtype)

        if not isinstance(in_tensor_list, list):
            raise ValueError("The type of 'in_tensor_list' for all_to_all "
                             "should be list.")
        for elem in in_tensor_list:
            check_variable_and_dtype(
                elem, 'in_tensor_list',
                ['float16', 'float32', 'float64', 'int32', 'int64'],
                'all_to_all')
        if not isinstance(out_tensor_list, list):
            raise ValueError("The type of 'out_tensor_list' for all_to_all "
                             "should be list.")
        if len(out_tensor_list) != 0:
            raise ValueError("The 'out_tensor_list' for all_to_all "
                             "must be an empty list.")
        helper.append_op(type=op_type,
                         inputs={'X': [temp]},
                         outputs={'Out': [out]},
                         attrs={
                             'ring_id': ring_id,
                             'use_calc_stream': use_calc_stream,
                         })
    out_tensor_list.extend(paddle.split(out, nranks, 0))


def alltoall_single(in_tensor,
                    out_tensor,
                    in_split_sizes=None,
                    out_split_sizes=None,
                    group=None,
                    use_calc_stream=True):
    """
    Scatter a single input tensor to all participators and gather the received tensors in out_tensor.

    .. note::
        ``alltoall_single`` is only supported in eager mode.

    Args:
        in_tensor (Tensor): Input tensor. The data type should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        out_tensor (Tensor): Output Tensor. The data type should be the same as the data type of the input Tensor.
        in_split_sizes (list[int], optional): Split sizes of ``in_tensor`` for dim[0]. If not given, dim[0] of ``in_tensor``
            must be divisible by group size and ``in_tensor`` will be scattered averagely to all participators. Default: None.
        out_split_sizes (list[int], optional): Split sizes of ``out_tensor`` for dim[0]. If not given, dim[0] of ``out_tensor``
            must be divisible by group size and ``out_tensor`` will be gathered averagely from all participators. Default: None.
        group (Group, optional): The group instance return by ``new_group`` or None for global default group. Default: None.
        use_calc_stream (bool, optional): Whether to use calculation stream (True) or communication stream. Default: True.

    Returns:
        None, if ``use_calc_stream`` is set to ``True``; ``Task`` of ``group``, if ``use_calc_stream`` is set to ``False``.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            rank = dist.get_rank()
            size = dist.get_world_size()

            # case 1 (2 GPUs)
            data = paddle.arange(2, dtype='int64') + rank * 2
            # data for rank 0: [0, 1]
            # data for rank 1: [2, 3]
            output = paddle.empty([2], dtype='int64')
            dist.alltoall_single(data, output)
            print(output)
            # output for rank 0: [0, 2]
            # output for rank 1: [1, 3]

            # case 2 (2 GPUs)
            in_split_sizes = [i + 1 for i in range(size)]
            # in_split_sizes for rank 0: [1, 2]
            # in_split_sizes for rank 1: [1, 2]
            out_split_sizes = [rank + 1 for i in range(size)]
            # out_split_sizes for rank 0: [1, 1]
            # out_split_sizes for rank 1: [2, 2]
            data = paddle.ones([sum(in_split_sizes), size], dtype='float32') * rank
            # data for rank 0: [[0., 0.], [0., 0.], [0., 0.]]
            # data for rank 1: [[1., 1.], [1., 1.], [1., 1.]]
            output = paddle.empty([(rank + 1) * size, size], dtype='float32')
            group = dist.new_group([0, 1])
            task = dist.alltoall_single(data,
                                        output,
                                        in_split_sizes,
                                        out_split_sizes,
                                        use_calc_stream=False,
                                        group=group)
            task.wait()
            print(output)
            # output for rank 0: [[0., 0.], [1., 1.]]
            # output for rank 1: [[0., 0.], [0., 0.], [1., 1.], [1., 1.]]

    """
    if group is not None and not group.is_member():
        return

    assert in_dygraph_mode(), "Only suppport alltoall_single in eager mode."
    # _check_single_tensor

    group = _get_default_group() if group is None else group
    backend = _group_map_backend[group]
    assert backend != 'gloo', ("backend gloo is not supported yet")

    in_split_sizes = [] if in_split_sizes is None else in_split_sizes
    out_split_sizes = [] if out_split_sizes is None else out_split_sizes

    task = group.process_group.alltoall_single(in_tensor, out_tensor,
                                               in_split_sizes, out_split_sizes)
    if use_calc_stream:
        task.wait()
        return
    else:
        return task


def _get_group_rank(global_rank, group=None):
    return global_rank if group is None else group.get_group_rank(global_rank)


def send(tensor, dst=0, group=None, use_calc_stream=True):
    """
    Send a tensor to the receiver.

    Args:
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        dst (int): The destination rank id.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
        use_calc_stream (bool, optional): Whether to use calculate stream or communication stream. Default: True.

    Returns:
        None.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            if dist.get_rank() == 0:
                data = paddle.to_tensor([7, 8, 9])
                dist.send(data, dst=1)
            else:
                data = paddle.to_tensor([1, 2, 3])
                dist.recv(data, src=0)
            print(data)
            # [7, 8, 9] (2 GPUs)
    """
    if group is not None and not group.is_member():
        return
    dst = _get_group_rank(dst, group)
    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        backend = _group_map_backend[group]
        assert backend != 'gloo', ("backend gloo is not supported yet")
        task = group.process_group.send(tensor, dst)
        if use_calc_stream:
            task.wait()
            return None
        else:
            return task

    ring_id = 0 if group is None else group.id

    if _non_static_mode():
        return _legacy_C_ops.send_v2(tensor, 'use_calc_stream', use_calc_stream,
                                     'ring_id', ring_id, 'peer', dst)
    op_type = 'send_v2'
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'send')

    helper = LayerHelper(op_type, **locals())
    helper.append_op(type=op_type,
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
            should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        src (int): The source rank id.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
        use_calc_stream (bool, optional): Whether to use calculate stream or communication stream. Default: True.

    Returns:
        None.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            if dist.get_rank() == 0:
                data = paddle.to_tensor([7, 8, 9])
                dist.send(data, dst=1)
            else:
                data = paddle.to_tensor([1, 2, 3])
                dist.recv(data, src=0)
            print(data)
            # [7, 8, 9] (2 GPUs)
    """
    if group is not None and not group.is_member():
        return

    src = _get_group_rank(src, group)
    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        backend = _group_map_backend[group]
        assert backend != 'gloo', ("backend gloo is not supported yet")
        task = group.process_group.recv(tensor, src)
        if use_calc_stream:
            task.wait()
            return None
        else:
            return task

    ring_id = 0 if group is None else group.id

    if _non_static_mode():
        return _legacy_C_ops.recv_v2(tensor, 'use_calc_stream', use_calc_stream,
                                     'ring_id', ring_id, 'peer', src, 'dtype',
                                     tensor.dtype, 'out_shape', tensor.shape)
    op_type = 'recv_v2'
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'recv')
    helper = LayerHelper(op_type, **locals())
    helper.append_op(type=op_type,
                     outputs={'Out': [tensor]},
                     attrs={
                         'ring_id': ring_id,
                         'peer': src,
                         'out_shape': tensor.shape,
                         'dtype': tensor.dtype,
                         'use_calc_stream': use_calc_stream,
                     })


def _check_single_tensor(tensor, tensor_name):
    if not isinstance(tensor, (core.eager.Tensor, paddle.Tensor)):
        raise RuntimeError("Invalid function argument. Expected parameter {}"
                           "to be of type paddle.Tensor, but it's {}".format(
                               tensor_name, type(tensor)))


def _check_tensor_list(tensor_list, tensor_name):
    if not isinstance(tensor_list, list) or \
        not all(isinstance(t, (core.eager.Tensor, paddle.Tensor)) for t in tensor_list):
        raise RuntimeError("Invalid function argument. Expected parameter {}"
                           "to be of type paddle.Tensor".format(tensor_name))


def isend(tensor, dst, group=None):
    """
    Sends a tensor asynchronously

    Args:
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        dst (int): The destination rank.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.

    Returns:
        A distributed task object.

    Warning:
        This API only supports the dygraph mode.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            if dist.get_rank() == 0:
                data = paddle.to_tensor([7, 8, 9])
                task = dist.isend(data, dst=1)
            else:
                data = paddle.to_tensor([1, 2, 3])
                task = dist.irecv(data, src=0)
            task.wait()
            print(data)
            # [7, 8, 9] (2 GPUs)

    """
    _check_single_tensor(tensor, "tensor")
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        backend = _group_map_backend[group]
        assert backend != 'gloo', ("backend gloo is not supported yet")
        group_dst_rank = group.get_group_rank(dst)
        assert group_dst_rank >= 0, ("dst rank out of group, need global rank")
        return group.process_group.send(tensor, group_dst_rank)
    else:
        raise RuntimeError("Only support eager dygraph mode.")


def irecv(tensor, src=None, group=None):
    """
    Receive a tensor to the sender.

    Args:
        tensor (Tensor): The Tensor to receive. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        src (int): The source rank id.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.

    Returns:
        A distributed task object.

    Warning:
        This API only supports the dygraph mode.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            if dist.get_rank() == 0:
                data = paddle.to_tensor([7, 8, 9])
                task = dist.isend(data, dst=1)
            else:
                data = paddle.to_tensor([1, 2, 3])
                task = dist.irecv(data, src=0)
            task.wait()
            print(data)
            # [7, 8, 9] (2 GPUs)
    """
    _check_single_tensor(tensor, "tensor")
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        backend = _group_map_backend[group]
        assert backend != 'gloo', ("backend gloo is not supported yet")
        group_src_rank = group.get_group_rank(src)
        assert group_src_rank >= 0, ("src rank out of group, need global rank")
        return group.process_group.recv(tensor, group_src_rank)
    else:
        raise RuntimeError("Only support eager dygraph mode.")


class P2POp(object):
    """
    A class that makes point-to-point operations for "batch_isend_irecv".

    This class creates the type of P2P operation, communication buffer, peer rank,
    Group. Instances of this class will be passed to
    ``paddle.distributed.batch_isend_irecv`` for point-to-point communication.

    Args:
        op (callable): A function to send data to or receive data from a peer process.
            The type of ``op`` is either ``paddle.distributed.isend`` or ``paddle.distributed.irecv``.
        tensor (Tensor): Tensor to send or receive.
        peer (int): The destination or source rank.
        group (Group, optional): The group instance return by new_group or None for global
            default group. Default: None.

    """

    def __init__(self, op, tensor, peer, group=None):
        if op not in [isend, irecv]:
            raise RuntimeError("Invalid ``op`` function. Expected ``op`` "
                               "to be of type ``paddle.distributed.isend`` or "
                               "``paddle.distributed.irecv``.")
        _check_single_tensor(tensor, "tensor")

        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = _get_default_group() if group is None else group


@contextlib.contextmanager
def _with_batch_p2p_guard(backend):
    if backend == "nccl":
        core.ProcessGroupNCCL.group_start()
    try:
        yield
    finally:
        if backend == "nccl":
            core.ProcessGroupNCCL.group_end()


def _check_p2p_op_list(p2p_op_list):
    """
    Helper to check that the ``p2p_op_list`` is a list of P2POp instances and
    all ops use the same backend.
    """
    if not isinstance(p2p_op_list, list) or not all(
            isinstance(p2p_op, P2POp) for p2p_op in p2p_op_list):
        raise RuntimeError("Invalid ``p2p_op_list``. Each op is expected to "
                           "to be of type ``paddle.distributed.P2POp``.")

    backend = _group_map_backend[p2p_op_list[0].group]
    if not all(backend == _group_map_backend[p2p_op.group]
               for p2p_op in p2p_op_list):
        raise RuntimeError("All groups need to use the same backend.")


def batch_isend_irecv(p2p_op_list):
    """
    Send or Receive a batch of tensors asynchronously and return a list of requests.

    Process each of the point-to-point operations in ``p2p_op_list`` and return the
    corresponding tasks. NCCL are currently supported.

    Args:
        p2p_op_list: A list of point-to-point operations(type of each operator is
            ``paddle.distributed.P2POp``). The order of the isend/irecv in the list
            matters and it needs to match with corresponding isend/irecv on the
            remote end.

    Returns:
        A list of distributed tasks returned by calling the corresponding
        op in the op_list.

    Warning:
        This API only supports the dygraph mode.

    Examples:
        .. code-block:: python

            # required: distributed

            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            send_t = paddle.arange(2) + rank
            # paddle.tensor([0, 1])  # Rank-0
            # paddle.tensor([1, 2])  # Rank-1

            recv_t = paddle.empty(shape=[2], dtype=send_t.dtype)

            send_op = dist.P2POp(dist.isend, send_t, (rank + 1) % world_size)
            recv_op = dist.P2POp(dist.irecv, recv_t, (rank - 1 + world_size) % world_size)

            tasks = dist.batch_isend_irecv([send_op, recv_op])

            for task in tasks:
                task.wait()

            print(recv_t)
            # paddle.tensor([1, 2])     # Rank-0
            # paddle.tensor([0, 1])     # Rank-1
    """
    _check_p2p_op_list(p2p_op_list)
    group = p2p_op_list[0].group
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        backend = _group_map_backend[group]
        tasks = []
        with _with_batch_p2p_guard(backend):
            for p2p_op in p2p_op_list:
                op = p2p_op.op
                tensor = p2p_op.tensor
                peer = p2p_op.peer
                comm_group = p2p_op.group
                task = op(tensor, peer, comm_group)
                if task is not None:
                    tasks.append(task)
        return tasks
    else:
        raise RuntimeError("Don't support static graph mode currently.")


def reduce_scatter(tensor,
                   tensor_list,
                   op=ReduceOp.SUM,
                   group=None,
                   use_calc_stream=True):
    """
    Reduces, then scatters a list of tensors to all processes in a group

    Args:
        tensor (Tensor): Output tensor. Its data type should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        tensor_list (list[Tensor]): List of tensors to reduce and scatter. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD): Optional. The operation used. Default: ReduceOp.SUM.
        group (Group, optional): The group instance return by new_group or None for global
            default group. Default: None.
        use_calc_stream (bool, optional): Whether this op should be an async op.

    Returns:
        Async task handle, if use_calc_stream is set to False.
        None, if use_calc_stream or if not part of the group.

    Warning:
        This API only supports the dygraph mode.


    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            if dist.get_rank() == 0:
                data1 = paddle.to_tensor([0, 1])
                data2 = paddle.to_tensor([2, 3])
            else:
                data1 = paddle.to_tensor([4, 5])
                data2 = paddle.to_tensor([6, 7])
            dist.reduce_scatter(data1, [data1, data2])
            print(data1)
            # [4, 6] (2 GPUs, out for rank 0)
            # [8, 10] (2 GPUs, out for rank 1)

    """
    _check_single_tensor(tensor, "tensor")
    _check_tensor_list(tensor_list, "tensor_list")

    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        op_type = _get_reduce_op(op, "reduce_scatter")
        group = _get_default_group() if group is None else group
        backend = _group_map_backend[group]
        assert backend != 'gloo', ("backend gloo is not supported yet")

        temp = paddle.concat(tensor_list, axis=0)
        task = group.process_group._reduce_scatter_base(tensor, temp, op_type)
        if use_calc_stream:
            task.wait()
            return None
        else:
            return task
    else:
        raise RuntimeError("Don't support static graph mode currently.")


def _reduce_scatter_base(output,
                         input,
                         op=ReduceOp.SUM,
                         group=None,
                         use_calc_stream=True):
    """
    Reduces, then scatters a flattened tensor to all processes in a group.

    Args:
        output (Tensor): Output tensor. Its data type should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        input (Tensor): Input tensor that is of size output tensor size times world size. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8 or bool.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD): Optional. The operation used. Default: ReduceOp.SUM.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        use_calc_stream (bool, optional): Wether to use calculation stream (True) or communication stream (False).
            Default to True.
    Returns:
        Async task handle, if use_calc_stream is set to False.
        None, if use_calc_stream or if not part of the group.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            rank = dist.get_rank()
            data = paddle.arange(4) + rank
            # [0, 1, 2, 3] (2 GPUs, for rank 0)
            # [1, 2, 3, 4] (2 GPUs, for rank 1)
            output = paddle.empty(shape=[2], dtype=data.dtype)
            dist.collective._reduce_scatter_base(output, data)
            print(output)
            # [1, 3] (2 GPUs, out for rank 0)
            # [5, 7] (2 GPUs, out for rank 1)

    """
    _check_single_tensor(output, "output")
    _check_single_tensor(input, "input")

    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        op_type = _get_reduce_op(op, "_reduce_scatter_base")
        group = _get_default_group() if group is None else group
        task = group.process_group._reduce_scatter_base(output, input, op_type)
        if use_calc_stream:
            task.wait()
            return None
        else:
            return task
    else:
        raise RuntimeError("Don't support static graph mode currently.")
