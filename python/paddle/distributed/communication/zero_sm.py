#   Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Collectives that consume no SM, mirroring ``paddle.distributed.stream``.

NCCL's zero-SM paths (``NCCL_CTA_POLICY_ZERO``) carry intra-node traffic on the
Copy Engines and inter-node traffic through the RMA CPU proxy, so a collective
stops competing for SMs with the compute it overlaps with. They only apply to
buffers registered as symmetric windows, and NCCL requires a single call to have
either all or none of its buffers registered.

Registration is a setup-time, collective, rank-symmetric operation, so it cannot
be applied per call to activations coming out of Paddle's allocator. This module
therefore pairs an allocator with the collectives: allocate the output with
:func:`empty` instead of ``paddle.empty``, and any input not already backed by
the pool is staged into it (a device-to-device copy, which also runs on a Copy
Engine). The group must have been created with ``cta_policy=2`` and the loaded
NCCL must be 2.30.7 or newer; see :func:`new_group`.

The pool decides what to register locally, without asking the group, so the
caller owes it a rank-symmetric call sequence: every rank has to reach the same
:func:`empty` calls with the same shapes in the same order, and to pass inputs
that all need staging or all do not. Size the buffers from a value the whole
group agrees on rather than from a local split. Breaking that makes one rank
register a window the others do not, which hangs the group.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from . import stream

if TYPE_CHECKING:
    from collections.abc import Callable

    from paddle import Tensor
    from paddle._typing import DTypeLike, ShapeLike
    from paddle.base.core import task
    from paddle.distributed.collective import _ZeroSMPool
    from paddle.distributed.communication.group import Group

__all__ = []


class _StagedTask:
    """Wraps a communication task to keep staged buffers alive until it lands.

    The pool recycles a block once the tensors referencing it are collected. For
    a collective running on the comm stream that is too early: a later staging
    copy on the compute stream could overwrite the block while NCCL is still
    reading it, and nothing orders the two streams.
    """

    def __init__(
        self,
        task: task | None,
        keepalive: tuple[Tensor, ...],
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        self._task = task
        self._keepalive: tuple[Tensor, ...] | None = keepalive
        self._on_complete = on_complete

    def _settle(self) -> None:
        """Run the pending write-back, once, and release the staged buffers."""
        if self._on_complete is not None:
            self._on_complete()
            self._on_complete = None
        self._keepalive = None

    def wait(self) -> None:
        if self._task is not None:
            self._task.wait()
        self._settle()

    def is_completed(self) -> bool:
        """Whether the output is readable, write-back included.

        The collective landing is not the same event as the caller's list being
        filled, so settling here is what makes a caller that polls this and then
        reads the output see the gathered values rather than stale ones.
        """
        if self._task is not None and not self._task.is_completed():
            return False
        self._settle()
        return True

    def synchronize(self) -> None:
        self.wait()

    def __getattr__(self, name: str):
        return getattr(self._task, name)


def _keep(_owner: Tensor) -> None:
    """Sole purpose is to hold a reference through :func:`weakref.finalize`."""


def _pin(view: Tensor, owner: Tensor) -> Tensor:
    """Tie ``owner``'s lifetime to ``view``'s and return ``view``.

    The pool recycles a block once the tensor handed out for it is collected, so
    views carved out of that tensor must keep it alive.
    """
    weakref.finalize(view, _keep, owner)
    return view


def _pool(group: Group | None) -> tuple[_ZeroSMPool, Group]:
    from paddle.distributed.collective import _get_global_group, _ZeroSMPool

    if group is None:
        group = _get_global_group()
    return _ZeroSMPool.instance(group), group


def empty(
    shape: ShapeLike, dtype: DTypeLike, group: Group | None = None
) -> Tensor:
    """Allocate an uninitialized tensor usable by the collectives below.

    Drop-in replacement for ``paddle.empty`` at a zero-SM call site: the buffer
    comes from a registered symmetric window, so it needs no staging copy. All
    ranks of ``group`` must allocate the same shapes in the same order.
    """
    pool, _ = _pool(group)
    return pool.empty(shape, dtype)


def all_gather(
    tensor_or_tensor_list: Tensor | list[Tensor],
    tensor: Tensor,
    group: Group | None = None,
    sync_op: bool = True,
    use_calc_stream: bool = False,
) -> task | _StagedTask | None:
    """Zero-SM all-gather with the signature of ``stream.all_gather``.

    ``tensor_or_tensor_list`` is either a single tensor from :func:`empty` or a
    tensor list; ``tensor`` may come from anywhere and is staged when needed, as
    long as every rank needs the same. The list form gathers into one pooled
    buffer, as the native implementation does anyway, and accepts a 0-D input,
    answering it with one scalar per rank. An empty list is filled with views
    over that buffer and costs no copy; a list that already holds tensors is
    written back element by element once the collective has landed.
    """
    pool, group = _pool(group)
    out = tensor_or_tensor_list

    if isinstance(out, (list, tuple)):
        if len(out) not in (0, group.nranks):
            # Checked before the allocation below so that a rejected call
            # leaves the pool, and therefore the group, untouched.
            raise ValueError(
                f"zero_sm.all_gather got an output list of {len(out)} tensors "
                f"for a group of {group.nranks} ranks"
            )
        shape = list(tensor.shape)
        scalar = not shape
        if scalar:
            # ``stream.all_gather`` answers a 0-D input with one scalar per rank
            # (``paddle.empty_like`` of a 0-D tensor), so the list form has to
            # accept one. NCCL gathers rows, so the collective runs over a row
            # per rank and the views are reshaped back to scalars. Reshaping the
            # input is a view, which keeps a pooled scalar copy-free.
            tensor = tensor.reshape([1])
            shape = [1]
        chunk = shape[0]
        buffer = pool.empty([chunk * group.nranks, *shape[1:]], tensor.dtype)
        views = [
            buffer[i * chunk : (i + 1) * chunk] for i in range(group.nranks)
        ]
        if scalar:
            views = [view.reshape([]) for view in views]
        if len(out) == 0:
            # The list aliases the buffer, so there is nothing to write back.
            task = _all_gather_into(
                pool, group, buffer, tensor, sync_op, use_calc_stream
            )
            out.extend(_pin(view, buffer) for view in views)
            return task

        def write_back():
            for dst, src in zip(out, views):
                dst.copy_(src, False)

        task = _all_gather_into(
            pool, group, buffer, tensor, sync_op, use_calc_stream
        )
        if use_calc_stream or sync_op:
            # Either the collective is ordered behind the write-back on the same
            # stream, or stream.all_gather has already waited for it. Both mean
            # the caller expects the list to be filled on return, so writing
            # back lazily would break the synchronous contract.
            write_back()
            return task
        task._on_complete = write_back
        task._keepalive = (*task._keepalive, buffer)
        return task

    if not pool.owns(out):
        raise ValueError(
            "the output of zero_sm.all_gather must come from zero_sm.empty(); "
            "NCCL requires all buffers of a call to be registered"
        )
    return _all_gather_into(pool, group, out, tensor, sync_op, use_calc_stream)


def _all_gather_into(
    pool: _ZeroSMPool,
    group: Group,
    out: Tensor,
    tensor: Tensor,
    sync_op: bool,
    use_calc_stream: bool,
) -> task | _StagedTask | None:
    staged = pool.stage(tensor)
    task = stream.all_gather(
        out,
        staged,
        group=group,
        sync_op=sync_op,
        use_calc_stream=use_calc_stream,
    )
    if use_calc_stream:
        # Same stream as the staging copy, so ordering already protects it.
        return task
    return _StagedTask(task, (staged, out))


def alltoall_single(
    out_tensor: Tensor,
    in_tensor: Tensor,
    out_split_sizes: list[int] | None = None,
    in_split_sizes: list[int] | None = None,
    group: Group | None = None,
    sync_op: bool = True,
    use_calc_stream: bool = False,
) -> task | _StagedTask:
    """Zero-SM all-to-all with the signature of ``stream.alltoall_single``.

    Uneven splits are supported, but :func:`empty` must still be reached in the
    same order with the same shapes on every rank, so size the buffers from a
    value all ranks agree on rather than from the local split.
    """
    pool, group = _pool(group)
    if not pool.owns(out_tensor):
        raise ValueError(
            "the output of zero_sm.alltoall_single must come from "
            "zero_sm.empty(); NCCL requires all buffers of a call to be "
            "registered"
        )

    staged = pool.stage(in_tensor)
    task = stream.alltoall_single(
        out_tensor,
        staged,
        out_split_sizes,
        in_split_sizes,
        group=group,
        sync_op=sync_op,
        use_calc_stream=use_calc_stream,
    )
    if use_calc_stream:
        return task
    return _StagedTask(task, (staged, out_tensor))
