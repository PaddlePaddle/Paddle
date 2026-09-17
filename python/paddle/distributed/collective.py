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
from __future__ import annotations

import datetime
import hashlib
import math
import weakref
from typing import (
    TYPE_CHECKING,
    Literal,
    TypeAlias,
)

import paddle

# (TODO: GhostScreaming) It will be removed later.
from paddle.base import core
from paddle.framework import in_dynamic_mode

from .communication.group import Group, _add_new_group, is_initialized
from .fleet.layers.mpu.mp_ops import (  # noqa: F401
    _c_concat,
    _c_identity,
    _c_lookup_table,
    _c_softmax_with_cross_entropy,
    _c_softmax_with_multi_label_cross_entropy,
    _c_split,
    _Linear,
    _linear,
    _mp_allreduce,
    _parallel_embedding,
    _parallel_linear,
    _set_var_distributed,
    split,
)

if TYPE_CHECKING:
    _BackendList: TypeAlias = Literal["gloo", "nccl", "xccl", "bkcl", "flagcx"]

    from paddle._typing import DTypeLike, ShapeLike
    from paddle.base.libpaddle import NCCLConfig

__all__ = []

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

_valid_backend_list = ['nccl', 'gloo', 'heter', 'xccl', 'bkcl', 'flagcx']
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
        _group_map[_global_env_gid] = Group(
            genv.rank, 0, list(range(genv.world_size))
        )
    return _group_map


def _get_global_group():
    return _get_group_map()[_global_env_gid]


def _get_group_map_by_name():
    global _group_map_by_name
    return _group_map_by_name


def _get_default_group():
    global _group_map_by_name
    assert is_initialized(), (
        "Call paddle.distributed.init_parallel_env first "
        "to initialize the distributed environment."
    )
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
    if in_dynamic_mode():
        global _start_ring_id
        _start_ring_id += 1
        return _start_ring_id + max(_get_global_env().nrings, 9)
    else:
        return len(_get_group_map()) + max(_get_global_env().nrings, 9)


def _new_process_group_impl(
    backend,
    store,
    rank,
    world_size,
    group_name,
    pg_options,
    group_id=0,
    nccl_comm_init_option=0,
    nccl_config=None,
):
    pg = None
    genv = _get_global_env()
    assert backend in _valid_backend_list, f"Unsupported backend: {backend}."
    if backend == "gloo":
        pg = core.ProcessGroupGloo.create(store, rank, world_size, group_id)
    elif backend == "nccl":
        pg = core.ProcessGroupNCCL.create(
            store,
            rank,
            world_size,
            group_id,
            genv.pg_timeout,
            nccl_comm_init_option,
            nccl_config,
        )
    elif backend == "xccl":
        pg = core.ProcessGroupCustom.create(
            store, genv.device_type, rank, world_size, group_id
        )
    elif backend == "bkcl":
        pg = core.ProcessGroupBKCL.create(store, rank, world_size, group_id)
    elif backend == "flagcx":
        pg = core.ProcessGroupFlagcx.create(
            store,
            rank,
            world_size,
            group_id,
            genv.pg_timeout,
            nccl_comm_init_option,
        )
    return pg


# _custom_gid provides a way for users to
# set the group id, which is usually useful
# to be compatible with the static graph mode.
_custom_gid = None


def _set_custom_gid(gid):
    global _custom_gid
    _custom_gid = gid


def new_group(
    ranks: list[int] | None = None,
    backend: Literal['nccl'] | None = None,
    timeout: datetime.timedelta = _default_timeout,
    nccl_comm_init_option: int = 0,
    nccl_config: NCCLConfig | None = None,
) -> Group:
    """

    Creates a new distributed communication group.

    Args:
        ranks (list): The global ranks of group members.
        backend (str): The backend used to create group, only nccl is supported now.
        timeout (datetime.timedelta, optional): The waiting timeout for store relevant options, default is 30 minutes.

    Returns:
        Group: The group instance.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle

            >>> paddle.distributed.init_parallel_env()
            >>> tindata = paddle.randn(shape=[2, 3])
            >>> gp = paddle.distributed.new_group([2, 4, 6])
            >>> paddle.distributed.all_reduce(tindata, group=gp, sync_op=False)

    """
    global _custom_gid
    global _group_map
    if in_dynamic_mode():
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
                "equal to that of the default global group."
            )
        size = len(ranks)
        ranks = sorted(ranks)
        if size > 1 and global_rank in ranks:
            rank = 0 if backend == 'heter' else ranks.index(global_rank)
            pg = _new_process_group_impl(
                backend,
                _default_store,
                rank,
                size,
                group_name,
                pg_options=None,
                group_id=gid,
                nccl_comm_init_option=nccl_comm_init_option,
                nccl_config=nccl_config,
            )
        else:
            rank = -1
            pg = None
        group = Group(rank, gid, ranks, pg=pg, name=group_name)
        _group_map_by_name[group_name] = group
        _group_map[gid] = group
        _group_map_backend[group] = backend
        # TODO: The method below is a new method for group management, will replace the previous
        # three in the future.
        _add_new_group(group)
        return group

    if not backend:
        backend = 'nccl'
    assert backend == 'nccl', "backend other than nccl is not supported yet"

    genv = _get_global_env()
    global_rank = genv.rank

    ring_id = _new_ring_id()

    if global_rank not in ranks:
        gp = Group(-1, ring_id, ranks)
        _group_map[ring_id] = gp
    else:
        ranks = sorted(ranks)
        group_rank = ranks.index(global_rank)
        group_size = len(ranks)
        gp = Group(group_rank, ring_id, ranks)
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
                core.NCCLParallelContext(strategy, place).init_with_ring_id(
                    ring_id
                )
            elif core.is_compiled_with_xpu():
                place = core.XPUPlace(genv.device_id)
                core.BKCLParallelContext(strategy, place).init_with_ring_id(
                    ring_id
                )
            else:
                raise AssertionError("no cuda device found")
        else:
            return gp

    # TODO(shenliang03): This is a temporary solution to solve the problem of
    # hang caused by cross-creation of new_group
    tmp = (
        paddle.to_tensor([1], dtype="int32")
        if in_dynamic_mode()
        else paddle.full([0], 1, dtype="int32")
    )
    paddle.distributed.all_reduce(tmp, sync_op=True)
    paddle.distributed.wait(tmp)
    return gp


def is_available() -> bool:
    """
    Check whether the distributed package is available.

    Returns:
        Returns True if the distributed package is available, otherwise False.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> print(paddle.distributed.is_available())

    """
    return core.is_compiled_with_dist()


def _init_parallel_env(backend: _BackendList) -> None:
    store = core.create_or_get_global_tcp_store()
    global_env = _get_global_env()
    rank = global_env.rank
    world_size = global_env.world_size
    dev_id = global_env.device_id

    if backend == "gloo":
        core.CommContextManager.create_gloo_comm_context(
            store, "0", rank, world_size
        )
    elif backend == "nccl":
        endpoints_str = ""
        for endpoint in global_env.trainer_endpoints:
            endpoints_str += endpoint
        endpoints_str += "ring_id:{}".format("0")
        endpoints_str_hash = hashlib.md5(
            endpoints_str.encode(encoding='UTF-8')
        ).hexdigest()
        core.CommContextManager.set_device_id(dev_id)
        core.CommContextManager.create_nccl_comm_context(
            store, "0", rank, world_size, endpoints_str_hash
        )
    elif backend == "xccl":
        dev_type = global_env.device_type
        paddle.device.set_device(f"{dev_type}:{dev_id}")
        core.CommContextManager.create_xccl_comm_context(
            store, "0", rank, world_size, dev_type
        )
    elif backend == "bkcl":
        endpoints_str = ""
        for endpoint in global_env.trainer_endpoints:
            endpoints_str += endpoint
        endpoints_str += "ring_id:{}".format("0")
        endpoints_str_hash = hashlib.md5(
            endpoints_str.encode(encoding='UTF-8')
        ).hexdigest()
        core.CommContextManager.set_device_id(dev_id)
        core.CommContextManager.create_bkcl_comm_context(
            store, "0", rank, world_size, endpoints_str_hash
        )


_shutdown_group_map_by_name = {}


def _get_shutdown_group_map_by_name():
    global _shutdown_group_map_by_name
    return _shutdown_group_map_by_name


def _update_shutdown_group_map_by_name(pg_name, group):
    global _shutdown_group_map_by_name
    _shutdown_group_map_by_name[pg_name] = group


def _delete_shutdown_group_map_by_name(pg_name):
    global _shutdown_group_map_by_name
    del _shutdown_group_map_by_name[pg_name]


def _clear_shutdown_group_map_by_name():
    global _shutdown_group_map_by_name
    _shutdown_group_map_by_name.clear()


def shutdown_process_group(group: Group | None = None) -> None:
    shutdown_groups = _get_shutdown_group_map_by_name()

    if group is None:
        global _default_group_name
        for pg_name, pg in _get_group_map_by_name().items():
            if (
                pg.process_group is not None
                and pg_name not in shutdown_groups
                and pg_name != _default_group_name
            ):
                pg.process_group.shutdown()
                _ZeroSMPool.discard(pg)
                _update_shutdown_group_map_by_name(pg_name, pg)
    else:
        if (
            group.process_group is not None
            and group.name not in shutdown_groups
        ):
            group.process_group.shutdown()
            _ZeroSMPool.discard(group)
            _update_shutdown_group_map_by_name(group.name, group)


def restart_process_group(group: Group | None = None) -> None:
    shutdown_groups = _get_shutdown_group_map_by_name()

    if group is None:
        for pg in shutdown_groups.values():
            pg.process_group.restart()
            _ZeroSMPool.discard(pg)
        _clear_shutdown_group_map_by_name()
    else:
        if group.process_group is not None and group.name in shutdown_groups:
            group.process_group.restart()
            _ZeroSMPool.discard(group)
            _delete_shutdown_group_map_by_name(group.name)


# Buffer address/size alignment required by ncclCommWindowRegister.
_NCCL_WINDOW_ALIGNMENT = 4096
# NCCL_WIN_COLL_SYMMETRIC, the window flag used by the zero-SM collectives.
_NCCL_WIN_COLL_SYMMETRIC = 0x01


def _nccl_symmetric_empty(shape: ShapeLike, dtype: DTypeLike) -> paddle.Tensor:
    """Allocate an uninitialized tensor that can be registered with NCCL.

    The buffer comes from ``ncclMemAlloc``: its address is aligned to
    ``_NCCL_WINDOW_ALIGNMENT`` and its allocated size padded up to it, as
    ``_register_comm_buffer`` requires, so any shape stays registrable. Tensors
    from ``paddle.empty`` come from Paddle's allocator and generally cannot be
    registered. Registration is collective, so all ranks sharing the buffer must
    allocate the same shape in the same order.
    """
    return core.nccl_mem_alloc(shape, dtype)


def _register_comm_buffer(
    tensor: paddle.Tensor,
    group: Group | None = None,
    win_flags: int = _NCCL_WIN_COLL_SYMMETRIC,
) -> int:
    """Register ``tensor`` as a NCCL symmetric memory window of ``group``.

    Registration is what makes the zero-SM paths usable: on a communicator
    created with ``cta_policy=2`` (``NCCL_CTA_POLICY_ZERO``) the collective runs
    on the Copy Engines and the RMA CPU proxy, consuming no SM. It is a
    collective requirement, not a local one: every rank of ``group`` must
    register buffers of the same size in the same order; a single collective call
    must have either all or none of its buffers registered; the address and byte
    size must be aligned to ``_NCCL_WINDOW_ALIGNMENT``, so allocate with
    ``_nccl_symmetric_empty``; and ``tensor`` must own its whole allocation,
    views are rejected.

    Repeated calls are cheap, the handle is cached per communicator. Returns 0
    when the loaded NCCL has no window API, in which case collectives silently
    keep using the SM-based path, and for a single-rank group, which has no
    communicator to register against.
    """
    if group is None:
        group = _get_global_group()
    if group.process_group is None:
        return 0
    return group.process_group.register_comm_buffer(tensor, win_flags)


def _deregister_comm_buffer(
    tensor: paddle.Tensor, group: Group | None = None
) -> None:
    """Release the window registered for ``tensor``. No-op if not registered.

    Windows are released when the communicator is destroyed, so this is only
    needed to free a buffer earlier than that.
    """
    if group is None:
        group = _get_global_group()
    if group.process_group is None:
        return
    group.process_group.deregister_comm_buffer(tensor)


class _ZeroSMPool:
    """A caching allocator whose blocks are registered NCCL symmetric windows.

    Window registration is a setup-time, collective, rank-symmetric operation, so
    it cannot be applied per call to activations from Paddle's own allocator:
    their addresses change every step. The pool instead owns a small set of
    long-lived registered blocks and hands out slices of them. Blocks are
    bucketed to powers of two so that a varying shape (MoE token counts, for
    instance) reuses one block instead of registering a new window every step.

    Which block an allocation gets is decided by a counter, not by a free list: a
    bucket owns a ring of :attr:`_RING_SIZE` blocks and the n-th allocation takes
    slot ``n % _RING_SIZE``, registering it if it is not there yet. Registration
    therefore depends on the sequence of allocations and nothing else, which is
    what the caller can keep rank-symmetric. A free list cannot be used for this:
    a block would return to it when the tensor slicing it is collected, and that
    is a local garbage-collection event, so the ranks would disagree on how many
    windows to register and hang inside NCCL.

    What the caller owes the pool is therefore the allocation sequence: all ranks
    have to reach the same allocations, for the same byte sizes, in the same
    order, and to agree on whether an input needs staging. A block is reissued
    once ``_RING_SIZE`` further allocations of its bucket have happened, and a
    handed-out tensor that is still referenced by then is refused rather than
    silently overwritten.
    """

    _instances: dict[str, _ZeroSMPool] = {}
    # Blocks a bucket keeps in rotation. It bounds the windows a bucket
    # registers, and it is how long a handed-out tensor stays valid: the ring
    # comes back around after this many allocations of the same bucket. A
    # collective usually needs two buffers of the same size at once, so leave
    # room for a few steps of those.
    _RING_SIZE = 8

    def __init__(self, group: Group) -> None:
        self._group = group
        self._rings: dict[tuple, list[paddle.Tensor]] = {}
        self._handed: dict[tuple, list[weakref.ref | None]] = {}
        self._served: dict[tuple, int] = {}
        self._spans: list[tuple[int, int]] = []

    @classmethod
    def instance(cls, group: Group) -> _ZeroSMPool:
        pool = cls._instances.get(group.name)
        if pool is None:
            pool = cls._instances[group.name] = cls(group)
        return pool

    @classmethod
    def discard(cls, group: Group) -> None:
        """Forget the pool of ``group``, whose windows are no longer registered.

        Destroying a communicator invalidates every window registered against
        it, and a restart builds a new one. The blocks this pool caches would
        otherwise still look registered to :meth:`owns`, so drop the pool: the
        next allocation registers fresh blocks against the new communicator, and
        a tensor handed out before the restart is refused rather than silently
        taking the SM-based path.
        """
        cls._instances.pop(group.name, None)

    @staticmethod
    def _bucket(numel: int, element_size: int) -> int:
        """Index of the power-of-two byte size that fits ``numel`` elements."""
        return max(0, (numel * element_size - 1).bit_length())

    @staticmethod
    def _capacity(bucket: int, element_size: int) -> int:
        """Elements a block of the ``bucket``-th power-of-two byte size holds."""
        return max((1 << bucket) // element_size, 1)

    def _register(self, dtype: DTypeLike, bucket: int) -> paddle.Tensor:
        """Allocate and register one block of ``bucket``.

        Bucketing in bytes is what keeps a varying shape on one window: it is
        also the unit a window is made of, so ranks that disagreed on the dtype
        would register different sizes for the same element count.
        """
        capacity = self._capacity(bucket, core.size_of_dtype(dtype))
        block = _nccl_symmetric_empty([capacity], dtype)
        if _register_comm_buffer(block, group=self._group) == 0:
            raise RuntimeError(
                "NCCL window registration is unavailable, so zero-SM "
                "collectives cannot be used. They need the loaded NCCL to be "
                "2.30.7 or newer."
            )
        start = block.data_ptr()
        self._spans.append((start, start + capacity * block.element_size()))
        return block

    def _take(
        self,
        shape: ShapeLike,
        dtype: DTypeLike,
        capacity: ShapeLike | None = None,
    ) -> paddle.Tensor:
        numel = math.prod(shape)
        room = numel if capacity is None else math.prod(capacity)
        if room < numel:
            raise ValueError(
                f"a zero-SM buffer of {numel} elements does not fit the "
                f"capacity of {room} it was asked to come from"
            )
        bucket = self._bucket(room, core.size_of_dtype(dtype))
        key = (dtype, bucket)
        ring = self._rings.setdefault(key, [])
        handed = self._handed.setdefault(key, [])
        served = self._served.get(key, 0)
        self._served[key] = served + 1
        slot = served % self._RING_SIZE
        if slot == len(ring):
            ring.append(self._register(dtype, bucket))
            handed.append(None)
        previous = handed[slot]
        if previous is not None and previous() is not None:
            raise RuntimeError(
                f"a tensor of {1 << bucket} bytes handed out by the zero-SM "
                f"pool is still referenced after {self._RING_SIZE} further "
                f"allocations of that size, so its block cannot be reissued. "
                f"Release it earlier, or raise _ZeroSMPool._RING_SIZE."
            )
        out = ring[slot][:numel].reshape(shape)
        # The ring owns the block, so this only records what was handed out:
        # reissuing the slot has to be able to tell whether it is still in use.
        handed[slot] = weakref.ref(out)
        return out

    def empty(
        self,
        shape: ShapeLike,
        dtype: DTypeLike,
        capacity: ShapeLike | None = None,
    ) -> paddle.Tensor:
        """Return an uninitialized registered tensor of ``shape``.

        All ranks of the group must reach this with the same byte size, in the
        same order: a registration one rank does not make hangs the others. When
        the local shape is not that size, pass the size the whole group agreed on
        as ``capacity`` and get a tensor of the local ``shape`` carved out of it,
        rather than slicing the result afterwards: the pool tracks what it hands
        out, and a slice it never saw would not stop the block being reissued.
        """
        return self._take(shape, dtype, capacity)

    def owns(self, tensor: paddle.Tensor) -> bool:
        """Whether ``tensor``'s memory sits in one of the registered windows."""
        ptr = tensor.data_ptr()
        return any(lo <= ptr < hi for lo, hi in self._spans)

    def handed_out(self, tensor: paddle.Tensor) -> bool:
        """Whether ``tensor`` is one the pool handed out and still tracks.

        Stricter than :meth:`owns`, which only answers whether the memory is
        registered: a tensor sliced out of a pooled one passes that while being
        invisible to the ring, so the block underneath it could be reissued
        while it is still in use.
        """
        return any(
            ref is not None and ref() is tensor
            for handed in self._handed.values()
            for ref in handed
        )

    def stage(self, tensor: paddle.Tensor) -> paddle.Tensor:
        """Return a registered tensor holding ``tensor``'s data.

        A tensor the pool handed out and still tracks is returned as is, so a
        caller that allocates its input from this pool pays no copy. Being merely
        inside a registered window is not enough: a slice the pool never handed
        out keeps no slot alive, so passing it through would let the ring reissue
        the block underneath while the collective is still reading it. Such a
        tensor is copied into a tracked block instead. All ranks have to agree on
        whether the copy is needed: one rank staging while another does not makes
        the registration counts diverge.
        """
        if self.handed_out(tensor):
            return tensor
        staged = self._take(tensor.shape, tensor.dtype)
        staged.copy_(tensor, False)
        return staged
