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

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import paddle
import paddle.distributed as dist
from paddle import framework

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle.base.core import ProcessGroup


class Group:
    """
    The abstract representation of group.
    """

    def __init__(
        self,
        rank_in_group: int,
        id: int,
        ranks: list[int],
        pg: ProcessGroup | None = None,
        name: str | None = None,
    ) -> None:
        self._rank_in_group = rank_in_group
        self._world_size = len(ranks) if rank_in_group >= 0 else -1
        self._id = id
        self._ranks = ranks
        self._pg = pg
        self._name = name

    @property
    def rank(self) -> int:
        return self._rank_in_group

    @property
    def ranks(self) -> list[int]:
        return self._ranks

    @property
    def nranks(self) -> int:
        return len(self._ranks)

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def process_group(self) -> ProcessGroup:
        return self._pg

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def backend(self) -> str:
        return self._pg.name()

    @property
    def id(self) -> int:
        return self._id

    def is_member(self) -> bool:
        if self.rank < 0:
            return False
        if self.nranks < 2:
            return False
        return True

    def get_group_rank(self, rank: int) -> int | Literal[-1]:
        if self.is_member():
            return self.ranks.index(rank)
        else:
            return -1

    def get_global_rank(self, rank: int) -> int | Literal[-1]:
        """
        Get the global rank of a process within a group.

        Args:
            rank (int): The local rank within the group.

        Returns:
            If the current process is a member of the group, returns the corresponding global rank;
            otherwise returns -1.

        """
        if self.is_member():
            return self.ranks[rank]
        else:
            return -1

    def __repr__(self) -> str:
        debug_str = (
            f"rank: {self.rank}, nranks: {self.nranks}, id: {self.id}, ranks: "
        )
        debug_str += ", ".join(map(str, self.ranks))
        debug_str += "; name: "
        debug_str += self.name if self.name else "None"
        return debug_str


class _GroupManager:
    global_group_id = 0
    group_map_by_id = {}


class _DistGroupMeta(type):
    """Metaclass exposing :attr:`group.WORLD` as a dynamic class property."""

    @property
    def WORLD(cls) -> Group | None:
        try:
            return _get_global_group()
        except RuntimeError:
            return None

    @WORLD.setter
    def WORLD(cls, value: Group | None) -> None:
        # Validate before mutating any registry so a rejected assignment
        # leaves the existing default group intact.
        if value is not None:
            if not isinstance(value, Group):
                raise TypeError(
                    "group.WORLD must be a Group instance or None, got "
                    f"{type(value).__name__}"
                )
            if value.id != _GroupManager.global_group_id:
                raise ValueError(
                    f"group.WORLD expects a Group with id="
                    f"{_GroupManager.global_group_id}, got id={value.id}"
                )

        # Lazy import: ``collective`` imports from this module at its top.
        from paddle.distributed import collective as _coll

        prev = _GroupManager.group_map_by_id.pop(
            _GroupManager.global_group_id, None
        )
        _coll._group_map.pop(_coll._global_env_gid, None)
        _coll._group_map_by_name.pop(_coll._default_group_name, None)
        if prev is not None:
            _coll._group_map_backend.pop(prev, None)

        if value is None:
            return

        _GroupManager.group_map_by_id[_GroupManager.global_group_id] = value
        _coll._group_map[_coll._global_env_gid] = value
        _coll._group_map_by_name[_coll._default_group_name] = value
        if value._pg is not None:
            # ``ProcessGroup.name()`` returns the C++ backend name in upper
            # case (e.g. ``NCCL``); the registry is keyed by the lower-case
            # Python form used in ``_valid_backend_list``.
            _coll._group_map_backend[value] = value._pg.name().lower()


class _DistGroupNamespace(metaclass=_DistGroupMeta):
    """Namespace exposing :attr:`WORLD`, re-exported as
    :data:`paddle.distributed.group`.
    """


def _get_global_group():
    if _GroupManager.global_group_id not in _GroupManager.group_map_by_id:
        raise RuntimeError("The global group is not initialized.")
    return _GroupManager.group_map_by_id[_GroupManager.global_group_id]


def _add_new_group(group):
    if group.id in _GroupManager.group_map_by_id:
        raise RuntimeError(f"The group with id {group.id} already exist.")
    _GroupManager.group_map_by_id[group.id] = group


def _is_global_group(group):
    return group.id == _GroupManager.global_group_id


def _warn_cur_rank_not_in_group(group):
    global_rank = dist.get_rank()
    if group and not group.is_member():
        return True
    return False


def _get_or_throw_group_rank(global_rank, group):
    group_rank = group.get_group_rank(global_rank)
    assert group_rank >= 0, (
        f"The input rank {global_rank} can not be found inside the group {group.name}"
    )
    return group_rank


def is_initialized() -> bool:
    """

    Check whether the distributed environment has been initialized

    Returns:
        `True` if distributed environment has been initialized, otherwise `False`.

    Warning:
        This API only supports the dygraph mode.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle

            >>> print(paddle.distributed.is_initialized())
            False

            >>> paddle.distributed.init_parallel_env()
            >>> print(paddle.distributed.is_initialized())
            True

    """
    return _GroupManager.global_group_id in _GroupManager.group_map_by_id


def destroy_process_group(group: Group | None = None) -> None:
    """
    Destroy a given group for communication

    Args:
        group (Group, optional): The group to be destroyed. All of process groups, including
                                        the default group, will be destroyed and the distributed
                                        environment will be deinitialized.

    Returns : None

    Warning:
        This API only supports the dygraph mode.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> group = dist.new_group([0, 1])

            >>> dist.destroy_process_group(group)
            >>> print(dist.is_initialized())
            True
            >>> dist.destroy_process_group()
            >>> print(dist.is_initialized())
            False

    """
    group = _get_global_group() if group is None else group
    assert group.id in _GroupManager.group_map_by_id, (
        f"Destroy group with id {group.id} is invalid."
    )
    if _is_global_group(group):
        _GroupManager.group_map_by_id.clear()
        # The default group is also registered in the collective-layer
        # registries by ``init_parallel_env``; clear those slots too so a
        # follow-up ``init_process_group`` re-creates the default group
        # rather than hitting ``init_parallel_env``'s early-return path.
        from paddle.distributed import collective as _coll

        _coll._group_map.pop(_coll._global_env_gid, None)
        _coll._group_map_by_name.pop(_coll._default_group_name, None)
        _coll._group_map_backend.pop(group, None)
    else:
        del _GroupManager.group_map_by_id[group.id]


def get_group(id: int = 0) -> Group:
    """

    Get group instance by group id.

    Args:
        id (int): the group id. Default value is 0.

    Returns:
        Group: the group instance.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> gid = paddle.distributed.new_group([2, 4, 6])
            >>> paddle.distributed.get_group(gid.id)

    """

    if id in _GroupManager.group_map_by_id:
        return _GroupManager.group_map_by_id[id]
    warnings.warn(f"Group {id} is not initialized.")
    return None


def _sync_calc_stream(tensor):
    if framework.in_dynamic_mode():
        return paddle._C_ops.sync_calc_stream(tensor)
    else:
        op_type = 'c_sync_calc_stream'
        helper = framework.LayerHelper(op_type, **locals())
        helper.append_op(
            type=op_type,
            inputs={'X': [tensor]},
            outputs={'Out': [tensor]},
        )


def _sync_comm_stream(tensor, ring_id=0):
    if framework.in_dynamic_mode():
        return paddle._C_ops.sync_comm_stream([tensor], ring_id)
    else:
        op_type = 'c_sync_comm_stream'
        helper = framework.LayerHelper(op_type, **locals())
        helper.append_op(
            type=op_type,
            inputs={'X': [tensor]},
            outputs={'Out': [tensor]},
            attrs={'ring_id': ring_id},
        )


def wait(
    tensor: Tensor, group: Group | None = None, use_calc_stream: bool = True
) -> None:
    """

    wait to sync stream for group.

    Args:
        tensor (Tensor): The Tensor used before sync.
        group (Group): The Group instance to perform sync.
        use_calc_stream (bool): Whether to use calculation stream (True) or communication stream (False).
            Default to True.

    Returns:
        None.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle

            >>> paddle.distributed.init_parallel_env()
            >>> tindata = paddle.randn(shape=[2, 3])
            >>> paddle.distributed.all_reduce(tindata, sync_op=True)
            >>> paddle.distributed.wait(tindata)

    """
    if group is not None and not group.is_member():
        return

    if use_calc_stream:
        _sync_calc_stream(tensor)
    else:
        ring_id = 0 if group is None else group.id
        _sync_comm_stream(tensor, ring_id)


def barrier(group: Group | None = None) -> None:
    """

    Barrier among all participators in the group.

    Args:
        group (Group): The group instance return by new_group or None for global default group.

    Returns:
        None.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> from paddle.distributed import init_parallel_env

            >>> paddle.set_device(f'gpu:{paddle.distributed.ParallelEnv().dev_id}')
            >>> init_parallel_env()
            >>> paddle.distributed.barrier()
    """
    if group is not None and not group.is_member():
        return

    if framework.in_dynamic_mode():
        group = _get_global_group() if group is None else group
        place = framework._current_expected_place()
        if isinstance(place, framework.CPUPlace):
            task = group.process_group.barrier()
        else:
            device_id = place.get_device_id()
            task = group.process_group.barrier(device_id)
        task.wait()
        return

    ring_id = 0 if group is None else group.id

    barrier_tensor = paddle.full([1], 1, dtype="int32")
    if framework.in_dynamic_mode():
        # barrier is not available in xpu for now
        if not paddle.framework.core.is_compiled_with_xpu():
            return paddle._legacy_C_ops.barrier(
                barrier_tensor, barrier_tensor, 'ring_id', ring_id
            )
    else:
        op_type = 'barrier'
        if not isinstance(ring_id, int):
            raise ValueError("The type of 'group' for barrier must be int.")
        helper = framework.LayerHelper(op_type, **locals())
        helper.append_op(
            type=op_type,
            inputs={'X': [barrier_tensor]},
            outputs={'Out': [barrier_tensor]},
            attrs={'ring_id': ring_id},
        )


def get_backend(group: Group | None = None) -> str:
    """
    Get the backend of given group.

    Args:
        group (Group): The group to work on. Use the global group as default.

    Returns:
        Returns the name of the given group backend.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle

            >>> paddle.distributed.init_parallel_env()
            >>> paddle.distributed.get_backend()
            NCCL
    """
    if _warn_cur_rank_not_in_group(group):
        raise RuntimeError("Invalid group specified")

    group = _get_global_group() if group is None else group
    return group.backend
