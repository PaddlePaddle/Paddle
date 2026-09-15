# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import gc

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.base import core
from paddle.distributed.collective import (
    _deregister_comm_buffer,
    _nccl_symmetric_empty,
    _register_comm_buffer,
    _ZeroSMPool,
)
from paddle.distributed.communication import zero_sm
from paddle.distributed.fleet.base.topology import create_nccl_config

_ALIGNMENT = 4096
_CHUNK = 2
_HIDDEN = 4
_DTYPE = paddle.float32


def _rejects(fn, *args, **kwargs):
    """Assert that a call is refused with ValueError."""
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("the call should have raised ValueError")


class ZeroSMTestCase:
    """Exercises symmetric memory window registration and the buffer pool.

    Runs on a group created with ``cta_policy=2``, the configuration the zero-SM
    communication paths need. Every rank asserts on its own, so a failure
    anywhere fails the launcher.
    """

    def __init__(self):
        dist.init_parallel_env()
        self._rank = dist.get_rank()
        self._world = dist.get_world_size()
        config = create_nccl_config({"commName": "zero_sm_ut", "cta_policy": 2})
        self._group = dist.new_group(
            list(range(self._world)), nccl_config=config
        )
        self._expected = [float(i + 1) for i in range(self._world)]

    def _full(self, rows):
        """Input filled with this rank's id. all-to-all sends one chunk to
        every rank, so it asks for as many chunks as there are ranks."""
        return paddle.full([rows, _HIDDEN], float(self._rank + 1), dtype=_DTYPE)

    def _out(self, pooled=True):
        shape = [_CHUNK * self._world, _HIDDEN]
        if pooled:
            return zero_sm.empty(shape, _DTYPE, group=self._group)
        return paddle.empty(shape, dtype=_DTYPE)

    def _gathered(self, out):
        paddle.device.synchronize()
        return [float(out[i * _CHUNK, 0]) for i in range(self._world)]

    def test_registration(self):
        # ncclMemAlloc aligns the address; the size is padded in C++ so that a
        # shape which is not a multiple of the alignment stays registrable.
        buffer = _nccl_symmetric_empty([_CHUNK * _HIDDEN + 1], _DTYPE)
        assert buffer.shape == [_CHUNK * _HIDDEN + 1], buffer.shape
        assert buffer.dtype == _DTYPE, buffer.dtype
        assert buffer.data_ptr() % _ALIGNMENT == 0, buffer.data_ptr()

        handle = _register_comm_buffer(buffer, group=self._group)
        assert handle != 0, "window registration returned a null handle"
        # Registration is cached, so a per-step call costs nothing.
        assert handle == _register_comm_buffer(buffer, group=self._group)
        # A view does not own its allocation and must be refused.
        _rejects(_register_comm_buffer, buffer[1:], group=self._group)
        _deregister_comm_buffer(buffer, group=self._group)
        # Deregistering an unknown buffer is a no-op rather than an error.
        _deregister_comm_buffer(buffer, group=self._group)
        _rejects(_nccl_symmetric_empty, [0], _DTYPE)
        # A shape whose byte size overflows size_t must be refused rather than
        # wrapping around into a small allocation.
        _rejects(_nccl_symmetric_empty, [2**62, 2**4], _DTYPE)
        # phi::SizeOf() is 0 for UNDEFINED, which would divide by zero in the
        # overflow checks and kill the process with SIGFPE.
        _rejects(_nccl_symmetric_empty, [1], core.DataType.UNDEFINED)

        # Leaving ``group`` unset falls back to the global group.
        spare = _nccl_symmetric_empty([_HIDDEN], _DTYPE)
        assert _register_comm_buffer(spare) != 0
        _deregister_comm_buffer(spare)
        # Paddle builds no real communicator for a single-rank group, so there
        # is nothing to register against and both calls degrade to no-ops.
        solo = dist.new_group([0])
        if solo.nranks == 1:
            assert _register_comm_buffer(spare, group=solo) == 0
            _deregister_comm_buffer(spare, group=solo)

    def test_pool_stages_foreign_input(self):
        pool = _ZeroSMPool.instance(self._group)
        pooled = pool.empty([120], _DTYPE)
        assert pool.owns(pooled)
        assert pool.stage(pooled) is pooled, "a pooled input was copied"

        foreign = paddle.arange(120, dtype=_DTYPE)
        staged = pool.stage(foreign)
        assert staged is not foreign and pool.owns(staged)
        np.testing.assert_allclose(staged.numpy(), foreign.numpy())

    def test_collective_over_registered_buffers(self):
        # Registration must not change what a collective computes: run one over
        # pooled buffers on the cta_policy=2 group and check the result.
        pool = _ZeroSMPool.instance(self._group)
        source = pool.empty([_CHUNK, _HIDDEN], _DTYPE)
        source[:] = float(self._rank + 1)
        out = pool.empty([_CHUNK * self._world, _HIDDEN], _DTYPE)
        dist.stream.all_gather(
            out, source, group=self._group, use_calc_stream=True
        )
        paddle.device.synchronize()
        gathered = [float(out[i * _CHUNK, 0]) for i in range(self._world)]
        assert gathered == self._expected, gathered

    def test_ring_reuse_ignores_local_state(self):
        # A shape that changes between steps (MoE token counts, for instance)
        # must not register a new window every time. Sizes landing in the same
        # power-of-two bucket share one ring, and the ring comes back around
        # after _RING_SIZE allocations, so the number of windows a bucket
        # registers depends on the allocation sequence and nothing else.
        pool = _ZeroSMPool.instance(self._group)
        ring = _ZeroSMPool._RING_SIZE
        # Row counts no other case touches, all in one bucket, so the ring
        # starts out empty and stays a single one.
        rows = tuple(600 + 40 * step for step in range(ring))
        element_size = core.size_of_dtype(_DTYPE)
        buckets = {
            _ZeroSMPool._bucket(taken * _HIDDEN, element_size) for taken in rows
        }
        assert len(buckets) == 1, buckets
        windows = len(pool._spans)

        first = pool.empty([rows[0], _HIDDEN], _DTYPE)
        assert pool.owns(first)
        address = first.data_ptr()
        del first
        for taken in rows[1:]:
            spare = pool.empty([taken, _HIDDEN], _DTYPE)
            del spare
        assert len(pool._spans) == windows + ring, len(pool._spans)

        # One turn later the first block comes back, and no rank had to consult
        # its garbage collector to know that.
        again = pool.empty([rows[0], _HIDDEN], _DTYPE)
        assert again.data_ptr() == address, "the ring did not come back around"
        assert len(pool._spans) == windows + ring, len(pool._spans)
        del again

    def test_registration_survives_one_sided_release(self):
        # Only rank 0 drops its reference and collects before the next
        # allocation. Registration must not depend on that: a free list would
        # let rank 0 reuse its block while rank 1 registers a new window, and
        # the next collective would hang.
        pool = _ZeroSMPool.instance(self._group)
        held = pool.empty([5000, _HIDDEN], _DTYPE)
        assert pool.owns(held)
        if self._rank == 0:
            del held
            gc.collect()
        spare = pool.empty([5100, _HIDDEN], _DTYPE)
        assert pool.owns(spare)

        counts = []
        dist.all_gather(
            counts,
            paddle.to_tensor([len(pool._spans)], dtype="int64"),
            group=self._group,
        )
        windows = [int(count[0]) for count in counts]
        assert len(set(windows)) == 1, windows

        # The group is still usable, so the registrations stayed in step.
        source = pool.empty([_CHUNK, _HIDDEN], _DTYPE)
        source[:] = float(self._rank + 1)
        out = pool.empty([_CHUNK * self._world, _HIDDEN], _DTYPE)
        dist.stream.all_gather(
            out, source, group=self._group, use_calc_stream=True
        )
        paddle.device.synchronize()
        gathered = [float(out[i * _CHUNK, 0]) for i in range(self._world)]
        assert gathered == self._expected, gathered

    def test_all_gather_into_tensor(self):
        for sync_op, use_calc_stream in (
            (True, True),
            (True, False),
            (False, False),
        ):
            out = self._out()
            task = zero_sm.all_gather(
                out,
                self._full(_CHUNK),
                group=self._group,
                sync_op=sync_op,
                use_calc_stream=use_calc_stream,
            )
            if use_calc_stream:
                # Ordered behind the staging copy on the same stream.
                assert task is None or not isinstance(task, zero_sm._StagedTask)
            elif not sync_op:
                assert isinstance(task, zero_sm._StagedTask), type(task)
                assert isinstance(task.is_completed(), bool)
                # Unknown attributes are forwarded to the wrapped task.
                try:
                    forwarded = task.no_such_attribute
                    raise AssertionError(f"expected no {forwarded}")
                except AttributeError:
                    pass
                task.synchronize()
            assert self._gathered(out) == self._expected

        # An input already backed by the pool skips the staging copy.
        pooled = zero_sm.empty([_CHUNK, _HIDDEN], _DTYPE, group=self._group)
        pooled[:] = float(self._rank + 1)
        out = self._out()
        zero_sm.all_gather(out, pooled, group=self._group, use_calc_stream=True)
        assert _ZeroSMPool.instance(self._group).owns(pooled)
        assert self._gathered(out) == self._expected

        # The default group owns a pool of its own. It is not a cta_policy=2
        # group, so only the plumbing is under test here.
        out = zero_sm.empty([_CHUNK * self._world, _HIDDEN], _DTYPE)
        zero_sm.all_gather(out, self._full(_CHUNK), use_calc_stream=True)
        assert self._gathered(out) == self._expected

    def test_all_gather_into_list(self):
        out = []
        zero_sm.all_gather(
            out, self._full(_CHUNK), group=self._group, use_calc_stream=True
        )
        paddle.device.synchronize()
        assert len(out) == self._world, len(out)
        assert [float(item[0, 0]) for item in out] == self._expected

        # The list holds views of a pooled block. Dropping every other
        # reference and allocating again must not hand that block out twice.
        gc.collect()
        filler = self._out()
        filler[:] = -1.0
        paddle.device.synchronize()
        assert [float(item[0, 0]) for item in out] == self._expected

        for sync_op, use_calc_stream in (
            (True, True),
            (True, False),
            (False, False),
        ):
            out = [
                paddle.zeros([_CHUNK, _HIDDEN], dtype=_DTYPE)
                for _ in range(self._world)
            ]
            task = zero_sm.all_gather(
                out,
                self._full(_CHUNK),
                group=self._group,
                sync_op=sync_op,
                use_calc_stream=use_calc_stream,
            )
            if not sync_op and not use_calc_stream:
                # Only the fully asynchronous form defers the write-back; a
                # sync_op call must have filled the list before returning.
                task.wait()
            paddle.device.synchronize()
            values = [float(item[0, 0]) for item in out]
            assert values == self._expected, (
                sync_op,
                use_calc_stream,
                values,
            )

    def test_rejects_unregistered_output(self):
        # NCCL requires all buffers of a call to be registered or none of them,
        # so an output that does not come from the pool cannot be used.
        for collective in (zero_sm.all_gather, zero_sm.alltoall_single):
            _rejects(
                collective,
                self._out(pooled=False),
                self._full(_CHUNK * self._world),
                group=self._group,
                use_calc_stream=True,
            )
        # A mis-sized output list is refused as well.
        _rejects(
            zero_sm.all_gather,
            [paddle.zeros([_CHUNK, _HIDDEN], dtype=_DTYPE)],
            self._full(_CHUNK),
            group=self._group,
            use_calc_stream=True,
        )

    def test_alltoall_single(self):
        # With explicit splits both buffers have to be pooled, so the input is
        # allocated here rather than handed over unregistered: the pool would
        # otherwise size a window from the local row count.
        splits = [_CHUNK] * self._world
        for sync_op, use_calc_stream in ((True, True), (False, False)):
            out = self._out()
            source = zero_sm.empty(
                [_CHUNK * self._world, _HIDDEN], _DTYPE, group=self._group
            )
            source[:] = float(self._rank + 1)
            task = zero_sm.alltoall_single(
                out,
                source,
                splits,
                splits,
                group=self._group,
                sync_op=sync_op,
                use_calc_stream=use_calc_stream,
            )
            if not use_calc_stream:
                assert isinstance(task, zero_sm._StagedTask), type(task)
                task.wait()
            assert self._gathered(out) == self._expected

        # Without explicit splits the input is spread evenly, and a legal call
        # gives every rank the same row count, so staging it is safe.
        out = self._out()
        zero_sm.alltoall_single(
            out,
            self._full(_CHUNK * self._world),
            group=self._group,
            use_calc_stream=True,
        )
        assert self._gathered(out) == self._expected

        # An unregistered input cannot be staged behind explicit splits.
        _rejects(
            zero_sm.alltoall_single,
            self._out(),
            self._full(_CHUNK * self._world),
            splits,
            splits,
            group=self._group,
            use_calc_stream=True,
        )

        # Buffers that are not registered windows cannot take the symmetric
        # ncclAlltoAll path; the Send/Recv implementation must still be correct.
        out = self._out(pooled=False)
        dist.alltoall_single(
            out, self._full(_CHUNK * self._world), group=self._group
        )
        assert self._gathered(out) == self._expected

    def test_alltoall_single_uneven_pooled(self):
        # Uneven splits through zero_sm.alltoall_single itself: rank r sends one
        # row to each peer plus r extra rows to itself, which is legal and gives
        # every rank a different row count. Both buffers are asked for with a
        # capacity the whole group agrees on, so the pool registers the same
        # window everywhere while handing out the local rows, and the shapes
        # stay a private matter.
        splits = [1] * self._world
        splits[self._rank] += self._rank
        rows = sum(splits)
        capacity = [2 * self._world, _HIDDEN]

        in_tensor = zero_sm.empty(
            [rows, _HIDDEN], _DTYPE, group=self._group, capacity=capacity
        )
        out_tensor = zero_sm.empty(
            [rows, _HIDDEN], _DTYPE, group=self._group, capacity=capacity
        )
        in_tensor[:] = float(self._rank + 1)
        out_tensor[:] = -1.0

        zero_sm.alltoall_single(
            out_tensor,
            in_tensor,
            splits,
            splits,
            group=self._group,
            use_calc_stream=True,
        )
        paddle.device.synchronize()

        # Rank j receives splits[i] rows of value i + 1 from rank i.
        offset = 0
        for peer in range(self._world):
            received = float(out_tensor[offset, 0])
            assert received == float(peer + 1), (self._rank, peer, received)
            offset += splits[peer]

    def test_rejects_untracked_slice(self):
        # The pool tracks the tensors it hands out, not slices taken from them
        # afterwards: such a slice keeps pointing into a registered window while
        # letting the ring believe the block is idle, so a caller that dropped
        # the parent and kept the slice would have it overwritten once the ring
        # comes back around. The collectives refuse it and point at the capacity
        # argument, which carves the local shape out without a stray slice.
        pool = _ZeroSMPool.instance(self._group)
        buffer = zero_sm.empty(
            [2 * self._world, _HIDDEN], _DTYPE, group=self._group
        )
        sliced = buffer[: self._world]
        del buffer
        gc.collect()

        # Registered memory, but nothing the pool would hold a block for.
        assert pool.owns(sliced)
        assert not pool.handed_out(sliced)

        splits = [1] * self._world
        _rejects(
            zero_sm.alltoall_single,
            sliced,
            sliced,
            splits,
            splits,
            group=self._group,
            use_calc_stream=True,
        )
        _rejects(
            zero_sm.all_gather,
            sliced,
            self._full(1),
            group=self._group,
            use_calc_stream=True,
        )

        # The capacity argument is the supported way to get there, and what it
        # returns is tracked.
        local = zero_sm.empty(
            [self._world, _HIDDEN],
            _DTYPE,
            group=self._group,
            capacity=[2 * self._world, _HIDDEN],
        )
        assert pool.handed_out(local)
        assert local.shape == [self._world, _HIDDEN], local.shape

    def test_stages_untracked_slice_input(self):
        # Same hazard on the input side, where there is no need to refuse: an
        # input that the pool does not track is copied into a block it does, so
        # the collective never reads through a slice the ring believes is idle.
        pool = _ZeroSMPool.instance(self._group)
        buffer = zero_sm.empty([2 * _CHUNK, _HIDDEN], _DTYPE, group=self._group)
        buffer[:] = float(self._rank + 1)
        sliced = buffer[:_CHUNK]
        del buffer
        gc.collect()
        assert pool.owns(sliced) and not pool.handed_out(sliced)

        # Being inside a registered window is not enough to skip the copy.
        staged = pool.stage(sliced)
        assert staged is not sliced, "an untracked slice was passed through"
        assert pool.handed_out(staged)
        paddle.device.synchronize()
        np.testing.assert_allclose(staged.numpy(), sliced.numpy())

        # End to end, over both collectives that stage their input.
        out = self._out()
        zero_sm.all_gather(out, sliced, group=self._group, use_calc_stream=True)
        paddle.device.synchronize()
        gathered = [float(out[i * _CHUNK, 0]) for i in range(self._world)]
        assert gathered == self._expected, gathered

        rows = _CHUNK * self._world
        out_tensor = zero_sm.empty([rows, _HIDDEN], _DTYPE, group=self._group)
        even = zero_sm.empty([rows, _HIDDEN], _DTYPE, group=self._group)
        even[:] = float(self._rank + 1)
        untracked = even[:]
        del even
        gc.collect()
        assert not pool.handed_out(untracked)
        zero_sm.alltoall_single(
            out_tensor,
            untracked,
            group=self._group,
            use_calc_stream=True,
        )
        paddle.device.synchronize()
        received = [
            float(out_tensor[i * _CHUNK, 0]) for i in range(self._world)
        ]
        assert received == self._expected, received

    def test_all_gather_list_settles_on_is_completed(self):
        # A caller is entitled to poll is_completed() and read the output as
        # soon as it returns True, so completion has to cover the write-back
        # into the list, not just the collective landing.
        out = [
            paddle.zeros([_CHUNK, _HIDDEN], dtype=_DTYPE)
            for _ in range(self._world)
        ]
        task = zero_sm.all_gather(
            out, self._full(_CHUNK), group=self._group, sync_op=False
        )
        assert isinstance(task, zero_sm._StagedTask), type(task)
        for _ in range(1000000):
            if task.is_completed():
                break
        else:
            raise AssertionError("the task never reported completion")

        # Deliberately no wait() here: is_completed() has to have settled it.
        paddle.device.synchronize()
        values = [float(item[0, 0]) for item in out]
        assert values == self._expected, values

    def test_all_gather_scalar_list(self):
        # A 0-D tensor is a legal input to the list form: stream.all_gather
        # answers it with one scalar per rank, so this must too rather than
        # indexing an empty shape.
        scalar = paddle.to_tensor(float(self._rank + 1), dtype=_DTYPE)
        assert scalar.shape == [], scalar.shape

        out = []
        zero_sm.all_gather(out, scalar, group=self._group, use_calc_stream=True)
        paddle.device.synchronize()
        assert len(out) == self._world, len(out)
        assert [item.shape for item in out] == [[]] * self._world
        assert [float(item) for item in out] == self._expected

        # A list that already holds scalars is written back element by element.
        out = [paddle.zeros([], dtype=_DTYPE) for _ in range(self._world)]
        zero_sm.all_gather(out, scalar, group=self._group, use_calc_stream=True)
        paddle.device.synchronize()
        assert [float(item) for item in out] == self._expected

    def test_alltoall_single_uneven_registered(self):
        # A legal all-to-all whose splits are uniform on rank 0 and uneven on
        # every other rank: rank r sends one row to each peer plus r extra rows
        # to itself. Picking the symmetric ncclAlltoAll from the local splits
        # would make rank 0 take it alone while the others run Send/Recv, and
        # hang the group, so an explicitly split call must never take it.
        rows = self._world + self._rank
        splits = [1] * self._world
        splits[self._rank] += self._rank

        # Registration has to stay rank-symmetric, so every rank registers the
        # same byte size and slices its own rows out of that window.
        capacity = 2 * self._world * _HIDDEN
        send = _nccl_symmetric_empty([capacity], _DTYPE)
        recv = _nccl_symmetric_empty([capacity], _DTYPE)
        assert _register_comm_buffer(send, group=self._group) != 0
        assert _register_comm_buffer(recv, group=self._group) != 0
        in_tensor = send[: rows * _HIDDEN].reshape([rows, _HIDDEN])
        out_tensor = recv[: rows * _HIDDEN].reshape([rows, _HIDDEN])
        in_tensor[:] = float(self._rank + 1)
        out_tensor[:] = -1.0

        dist.alltoall_single(
            out_tensor, in_tensor, splits, splits, group=self._group
        )
        paddle.device.synchronize()

        # Rank j receives splits[i] rows of value i + 1 from rank i.
        offset = 0
        for peer in range(self._world):
            received = float(out_tensor[offset, 0])
            assert received == float(peer + 1), (self._rank, peer, received)
            offset += splits[peer]

        _deregister_comm_buffer(send, group=self._group)
        _deregister_comm_buffer(recv, group=self._group)

    def run_test_case(self):
        for name in sorted(n for n in dir(self) if n.startswith("test_")):
            getattr(self, name)()


if __name__ == "__main__":
    ZeroSMTestCase().run_test_case()
