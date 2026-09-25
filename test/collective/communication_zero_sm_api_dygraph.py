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

    def test_asymmetric_dynamic_shapes(self):
        # A dynamic shape (MoE token counts, for instance) puts the ranks in
        # different buckets, and registration is collective: the pool has to
        # negotiate one byte size for the whole group instead of following the
        # local request. Doing it locally deadlocks right here.
        pool = _ZeroSMPool.instance(self._group)
        local = pool.empty([(self._rank + 1) * 48, _HIDDEN], _DTYPE)
        assert pool.owns(local)

        # Whatever each rank asked for, the block it registered has the same
        # byte size everywhere.
        span = pool._spans[-1]
        sizes = []
        dist.all_gather(
            sizes,
            paddle.to_tensor([span[1] - span[0]], dtype="int64"),
            group=self._group,
        )
        nbytes = [int(size[0]) for size in sizes]
        assert len(set(nbytes)) == 1, nbytes

        # Now make the free lists disagree on purpose: only rank 0 hands its
        # block back before the next allocation. Deciding on the local cache
        # state would make the ranks register a different number of windows.
        if self._rank == 0:
            del local
            gc.collect()
        spare = pool.empty([(self._rank + 1) * 48, _HIDDEN], _DTYPE)
        assert pool.owns(spare)

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

    def run_test_case(self):
        for name in sorted(n for n in dir(self) if n.startswith("test_")):
            getattr(self, name)()


if __name__ == "__main__":
    ZeroSMTestCase().run_test_case()
