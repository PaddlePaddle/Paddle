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


class ZeroSMTestCase:
    """Exercises the registered-buffer pool and the zero-SM collectives.

    Runs on a group created with ``cta_policy=2``, the configuration the
    zero-SM communication paths need. Every rank asserts on its own, so a
    failure anywhere fails the launcher.
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

    def _input(self):
        return paddle.full(
            [_CHUNK, _HIDDEN], float(self._rank + 1), dtype=_DTYPE
        )

    def _out_buffer(self):
        return zero_sm.empty(
            [_CHUNK * self._world, _HIDDEN], _DTYPE, group=self._group
        )

    def _a2a_input(self):
        # all-to-all sends one chunk to every rank, so the input holds as many
        # chunks as there are ranks.
        return paddle.full(
            [_CHUNK * self._world, _HIDDEN], float(self._rank + 1), dtype=_DTYPE
        )

    def _gathered(self, out):
        paddle.device.synchronize()
        return [float(out[i * _CHUNK, 0]) for i in range(self._world)]

    def test_low_level_registration(self):
        buffer = _nccl_symmetric_empty([_CHUNK * _HIDDEN + 1], _DTYPE)
        assert buffer.shape == [_CHUNK * _HIDDEN + 1], buffer.shape
        assert buffer.dtype == _DTYPE, buffer.dtype
        # ncclMemAlloc aligns the address; the size is padded in C++ so that a
        # shape which is not a multiple of the alignment stays registrable.
        assert buffer.data_ptr() % _ALIGNMENT == 0, buffer.data_ptr()

        handle = _register_comm_buffer(buffer, group=self._group)
        assert handle != 0, "window registration returned a null handle"
        # Registration is cached, so a per-step call costs nothing.
        assert handle == _register_comm_buffer(buffer, group=self._group)

        # A view does not own its allocation and must be refused.
        try:
            _register_comm_buffer(buffer[1:], group=self._group)
            raise AssertionError("registering a view should have raised")
        except ValueError:
            pass

        _deregister_comm_buffer(buffer, group=self._group)
        # Deregistering an unknown buffer is a no-op rather than an error.
        _deregister_comm_buffer(buffer, group=self._group)

        try:
            _nccl_symmetric_empty([0], _DTYPE)
            raise AssertionError("an empty shape should have raised")
        except ValueError:
            pass

    def test_pool_recycles_blocks(self):
        pool = _ZeroSMPool.instance(self._group)
        first = zero_sm.empty([100], _DTYPE, group=self._group)
        address = first.data_ptr()
        assert pool.owns(first)
        del first
        gc.collect()

        # Capacities are bucketed to powers of two, so a different but
        # same-bucket shape reuses the block instead of registering a new one.
        second = zero_sm.empty([120], _DTYPE, group=self._group)
        assert second.data_ptr() == address, "the pool did not reuse the block"
        assert pool.stage(second) is second, "a pooled input was copied"

        foreign = paddle.arange(120, dtype=_DTYPE)
        staged = pool.stage(foreign)
        assert staged is not foreign and pool.owns(staged)
        np.testing.assert_allclose(staged.numpy(), foreign.numpy())

    def test_all_gather_into_tensor(self):
        source = self._input()
        for sync_op, use_calc_stream in (
            (True, True),
            (True, False),
            (False, False),
        ):
            out = self._out_buffer()
            task = zero_sm.all_gather(
                out,
                source,
                group=self._group,
                sync_op=sync_op,
                use_calc_stream=use_calc_stream,
            )
            if use_calc_stream:
                # Ordered behind the staging copy on the same stream.
                assert task is None or not isinstance(task, zero_sm._StagedTask)
            else:
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
        out = self._out_buffer()
        zero_sm.all_gather(out, pooled, group=self._group, use_calc_stream=True)
        assert _ZeroSMPool.instance(self._group).owns(pooled)
        assert self._gathered(out) == self._expected

    def test_all_gather_into_empty_list(self):
        out = []
        zero_sm.all_gather(
            out, self._input(), group=self._group, use_calc_stream=True
        )
        paddle.device.synchronize()
        assert len(out) == self._world, len(out)
        values = [float(item[0, 0]) for item in out]
        assert values == self._expected, values

        # The list holds views of a pooled block. Dropping every other
        # reference and allocating again must not hand that block out twice.
        gc.collect()
        filler = zero_sm.empty(
            [_CHUNK * self._world, _HIDDEN], _DTYPE, group=self._group
        )
        filler[:] = -1.0
        paddle.device.synchronize()
        assert [float(item[0, 0]) for item in out] == self._expected

    def test_all_gather_into_prefilled_list(self):
        for use_calc_stream in (True, False):
            out = [
                paddle.zeros([_CHUNK, _HIDDEN], dtype=_DTYPE)
                for _ in range(self._world)
            ]
            task = zero_sm.all_gather(
                out,
                self._input(),
                group=self._group,
                sync_op=use_calc_stream,
                use_calc_stream=use_calc_stream,
            )
            if not use_calc_stream:
                task.wait()
            paddle.device.synchronize()
            values = [float(item[0, 0]) for item in out]
            assert values == self._expected, (use_calc_stream, values)

    def test_all_gather_rejects_bad_output(self):
        # NCCL requires all buffers of a call to be registered or none of them,
        # so an output that does not come from the pool cannot be used.
        try:
            zero_sm.all_gather(
                paddle.empty([_CHUNK * self._world, _HIDDEN], dtype=_DTYPE),
                self._input(),
                group=self._group,
                use_calc_stream=True,
            )
            raise AssertionError("an unregistered output should have raised")
        except ValueError:
            pass

        try:
            zero_sm.all_gather(
                [paddle.zeros([_CHUNK, _HIDDEN], dtype=_DTYPE)],
                self._input(),
                group=self._group,
                use_calc_stream=True,
            )
            raise AssertionError("a mis-sized output list should have raised")
        except ValueError:
            pass

    def test_alltoall_single(self):
        # The split sizes have to be identical on every rank: the pool
        # registers its blocks collectively, so all ranks must ask for the
        # same shapes in the same order.
        splits = [_CHUNK] * self._world
        for sync_op, use_calc_stream in ((True, True), (False, False)):
            source = self._a2a_input()
            out = self._out_buffer()
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

        # Without explicit splits the input is spread evenly.
        out = self._out_buffer()
        zero_sm.alltoall_single(
            out, self._a2a_input(), group=self._group, use_calc_stream=True
        )
        assert self._gathered(out) == self._expected

    def test_alltoall_single_rejects_bad_output(self):
        try:
            zero_sm.alltoall_single(
                paddle.empty([_CHUNK * self._world, _HIDDEN], dtype=_DTYPE),
                self._a2a_input(),
                group=self._group,
                use_calc_stream=True,
            )
            raise AssertionError("an unregistered output should have raised")
        except ValueError:
            pass

    def test_alltoall_single_falls_back_off_the_pool(self):
        # Buffers that are not registered windows cannot take the symmetric
        # ncclAlltoAll path; the Send/Recv implementation must still be correct.
        source = paddle.full(
            [_CHUNK * self._world, _HIDDEN], float(self._rank + 1), dtype=_DTYPE
        )
        out = paddle.empty([_CHUNK * self._world, _HIDDEN], dtype=_DTYPE)
        dist.alltoall_single(out, source, group=self._group)
        assert self._gathered(out) == self._expected

    def test_default_group(self):
        # Leaving ``group`` unset falls back to the global group, which owns a
        # pool of its own. It is not a cta_policy=2 group, so the collective
        # keeps its usual kernels; only the plumbing is under test here.
        out = zero_sm.empty([_CHUNK * self._world, _HIDDEN], _DTYPE)
        zero_sm.all_gather(out, self._input(), use_calc_stream=True)
        assert self._gathered(out) == self._expected

        buffer = _nccl_symmetric_empty([_HIDDEN], _DTYPE)
        assert _register_comm_buffer(buffer) != 0
        _deregister_comm_buffer(buffer)

    def test_single_rank_group_is_a_noop(self):
        # Paddle builds no real communicator for a single-rank group, so there
        # is nothing to register against and both calls degrade to no-ops.
        solo = dist.new_group([0])
        if solo.nranks != 1:
            return
        buffer = _nccl_symmetric_empty([_HIDDEN], _DTYPE)
        assert _register_comm_buffer(buffer, group=solo) == 0
        _deregister_comm_buffer(buffer, group=solo)

    def run_test_case(self):
        self.test_low_level_registration()
        self.test_pool_recycles_blocks()
        self.test_all_gather_into_tensor()
        self.test_all_gather_into_empty_list()
        self.test_all_gather_into_prefilled_list()
        self.test_all_gather_rejects_bad_output()
        self.test_alltoall_single()
        self.test_alltoall_single_rejects_bad_output()
        self.test_alltoall_single_falls_back_off_the_pool()
        self.test_default_group()
        self.test_single_rank_group_is_a_noop()


if __name__ == "__main__":
    ZeroSMTestCase().run_test_case()
