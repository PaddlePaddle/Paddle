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

    def test_pool_recycles_blocks(self):
        pool = _ZeroSMPool.instance(self._group)
        first = pool.empty([100], _DTYPE)
        address = first.data_ptr()
        assert pool.owns(first)
        del first
        gc.collect()

        # Capacities are bucketed to powers of two, so a different but
        # same-bucket shape reuses the block instead of registering a new one.
        second = pool.empty([120], _DTYPE)
        assert second.data_ptr() == address, "the pool did not reuse the block"
        assert pool.stage(second) is second, "a pooled input was copied"

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

    def test_bucketed_dynamic_shapes(self):
        # A shape that changes between steps (MoE token counts, for instance)
        # must not register a new window every time: two sizes landing in the
        # same power-of-two bucket share one block. Every rank asks for the same
        # shapes in the same order, which is what the pool requires from its
        # caller now that it decides locally.
        pool = _ZeroSMPool.instance(self._group)
        first = pool.empty([48, _HIDDEN], _DTYPE)
        assert pool.owns(first)
        address = first.data_ptr()
        windows = len(pool._spans)
        del first
        gc.collect()

        spare = pool.empty([60, _HIDDEN], _DTYPE)
        assert spare.data_ptr() == address, "the pool did not reuse the block"
        assert len(pool._spans) == windows, "a redundant window was registered"

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
        # The split sizes have to be identical on every rank: the pool
        # registers its blocks collectively, so all ranks must ask for the
        # same shapes in the same order.
        splits = [_CHUNK] * self._world
        for sync_op, use_calc_stream in ((True, True), (False, False)):
            out = self._out()
            task = zero_sm.alltoall_single(
                out,
                self._full(_CHUNK * self._world),
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

        # Without explicit splits the input is spread evenly.
        out = self._out()
        zero_sm.alltoall_single(
            out,
            self._full(_CHUNK * self._world),
            group=self._group,
            use_calc_stream=True,
        )
        assert self._gathered(out) == self._expected

        # Buffers that are not registered windows cannot take the symmetric
        # ncclAlltoAll path; the Send/Recv implementation must still be correct.
        out = self._out(pooled=False)
        dist.alltoall_single(
            out, self._full(_CHUNK * self._world), group=self._group
        )
        assert self._gathered(out) == self._expected

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
