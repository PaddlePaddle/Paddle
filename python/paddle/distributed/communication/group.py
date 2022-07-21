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

import os
import paddle
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle import _C_ops
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.framework import _non_static_mode
from paddle.fluid.layers.tensor import fill_constant

__all__ = [
        "Group",
        "new_group",
        "get_group",
        "is_initialized",
        "destroy_process_group",
        "_get_global_env",
        "_set_default_backend",
        "_set_default_store",
        "_get_group_map",
        "_get_global_group",
        "_get_group_map_by_name",
        "_get_default_group",
        "_set_group_map",
        "_set_group_map_by_name",
        "_set_group_map_backend",
        "_new_ring_id",
        "_new_process_group_impl",
        "_set_custom_gid",
        "_global_env",
        "_group_map",
        "_group_map_by_name",
        "_group_map_backend",
        "_default_group_name",
        "_valid_backend_list",
        "_default_store",
        "_default_backend",
        "_custom_gid",
        ]

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

# group map by name : the map of all groups from their names
# Dict[name, Group]
_group_map_by_name = {}

# backend map by group : the map of all backend from their groups
# Dict[group, backend]
_group_map_backend = {}

# Name of the default group for init_parallel_env
_default_group_name = "_default_pg"

_valid_backend_list = ['nccl', 'gloo', 'hccl', 'heter']
_default_store = None  # the default tcp store
_default_backend = None


def _set_default_backend(backend):
    global _default_backend
    _default_backend = backend


def _set_default_store(store):
    global _default_store
    _default_store = store


def _get_group_map():
    global _group_map
    if not _group_map:
        genv = _get_global_env()
        _group_map[0] = Group(genv.rank,
                              genv.world_size,
                              ranks=list(range(genv.world_size)))
    return _group_map


def _get_global_group():
    return _get_group_map()[0]


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


# _custom_gid provides a way for users to
# set the group id, which is usually useful
# to be compatible with the static mode.
_custom_gid = None


def _set_custom_gid(gid):
    global _custom_gid
    _custom_gid = gid


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

            paddle.distributed.init_parallel_env()
            group = paddle.distributed.new_group([0, 1])

            paddle.distributed.destroy_process_group(group)
            print(paddle.distributed.is_initialized())
            # True
            paddle.distributed.destroy_process_group()
            print(paddle.distributed.is_initialized())
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
