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

"""Single-process unit tests for ``paddle.distributed.fsdp.fully_shard_fusion``.

The three behaviours covered here are pure/host-side logic that needs no
distributed launch:

* ``_find_exposed_param_names`` -- which of a layer's own parameters leak into
  its outputs, including the bounded-depth walk.
* the fused-kernel fp32 gradient hand-off inside the grad hook.
* the tuple branch of the forward post hook that splices exposed parameters
  back untouched while routing the rest through the backward all_gather hook.
* ``unshard_params`` -- the synchronous gather / reshard context manager and
  the double-buffer bookkeeping it restores on exit.
"""

import types
import unittest

import numpy as np

import paddle
from paddle.distributed.fsdp import fully_shard_fusion as fsf


class _NoopCommManager:
    """comm_manager stand-in: the hooks only need these to be callable.

    ``buffer_manager.tie_param_name`` is read when registering the tie-weight
    pre hook; a value no real param name matches keeps that path dormant.
    """

    def __init__(self):
        self.buffer_manager = types.SimpleNamespace(tie_param_name=None)

    def _maybe_zero_grads(self):
        pass

    def _ensure_grads_writable(self, param):
        pass

    def shard_params(self, params, is_backward=False):
        pass

    def reduce_scatter_grads(self, param):
        pass


class _FakeParam:
    """Minimal fused parameter: a lazily sized ``main_grad`` plus a slot that
    captures the hook ``_register_grad_hook_once`` installs."""

    def __init__(self):
        self.main_grad = None
        self.registered_hook = None

    def get_main_grad(self, shape):
        if self.main_grad is None:
            self.main_grad = paddle.zeros(shape, dtype="float32")
        return self.main_grad

    def _register_grad_hook(self, hook):
        self.registered_hook = hook


class _NamedParam:
    def __init__(self, name):
        self.name = name


class _FakeLayer:
    def __init__(self, params):
        self._params = list(params)
        self.post_hook = None
        self.pre_hook = None

    def parameters(self, include_sublayers=True):
        return self._params

    def register_forward_post_hook(self, hook):
        self.post_hook = hook

    def register_forward_pre_hook(self, hook):
        self.pre_hook = hook


class TestFindExposedParamNames(unittest.TestCase):
    def _tensor(self):
        return paddle.to_tensor([1.0, 2.0])

    def test_empty_own_names_short_circuits(self):
        t = self._tensor()
        self.assertEqual(fsf._find_exposed_param_names(t, set()), set())

    def test_bare_tensor_hit_and_miss(self):
        t = self._tensor()
        self.assertEqual(fsf._find_exposed_param_names(t, {t.name}), {t.name})
        self.assertEqual(
            fsf._find_exposed_param_names(t, {"not-a-name"}), set()
        )

    def test_tuple_returns_only_exposed_members(self):
        hidden, weight, bias = self._tensor(), self._tensor(), self._tensor()
        found = fsf._find_exposed_param_names(
            (hidden, weight, bias), {weight.name, bias.name}
        )
        self.assertEqual(found, {weight.name, bias.name})

    def test_dict_values_are_scanned(self):
        hidden, weight = self._tensor(), self._tensor()
        found = fsf._find_exposed_param_names(
            {"h": hidden, "w": weight}, {weight.name}
        )
        self.assertEqual(found, {weight.name})

    def test_non_container_returns_empty(self):
        self.assertEqual(fsf._find_exposed_param_names(123, {"x"}), set())

    def test_depth_bound_stops_deep_nesting(self):
        # A tensor two container levels deep is still found (the tensor check
        # precedes the guard); one level deeper trips ``_depth >= 2``.
        t = self._tensor()
        self.assertEqual(
            fsf._find_exposed_param_names(((t,),), {t.name}), {t.name}
        )
        self.assertEqual(
            fsf._find_exposed_param_names((((t,),),), {t.name}), set()
        )


class TestFusedKernelGradHook(unittest.TestCase):
    """Covers the fp32 hand-off in the grad hook from ``_register_grad_hook_once``."""

    def _make_hooked_param(self):
        fsdp = object.__new__(fsf.FullyShardFusion)
        fsdp.comm_manager = _NoopCommManager()
        param = _FakeParam()
        fsdp._register_grad_hook_once(param)
        self.assertIsNotNone(param.registered_hook)
        return param

    def test_single_fp32_grad_wins_over_autograd_grad(self):
        param = self._make_hooked_param()
        param._fused_kernel_fp32_grad = paddle.to_tensor([1.0, 2.0, 3.0])
        # The autograd-delivered (bf16-rounded) grad must be ignored and freed.
        incoming = paddle.to_tensor([9.0, 9.0, 9.0])
        param.registered_hook(incoming)
        np.testing.assert_array_equal(param.main_grad.numpy(), [1.0, 2.0, 3.0])
        self.assertIsNone(param._fused_kernel_fp32_grad)
        self.assertFalse(incoming._is_initialized())

    def test_fp32_grad_list_added_in_order(self):
        param = self._make_hooked_param()
        param._fused_kernel_fp32_grad = [
            paddle.to_tensor([1.0, 1.0]),
            paddle.to_tensor([10.0, 20.0]),
        ]
        param.registered_hook(None)
        np.testing.assert_array_equal(param.main_grad.numpy(), [11.0, 21.0])
        self.assertIsNone(param._fused_kernel_fp32_grad)

    def test_without_fp32_grad_falls_back_to_autograd_grad(self):
        param = self._make_hooked_param()
        incoming = paddle.to_tensor([2.0, 4.0])
        param.registered_hook(incoming)
        np.testing.assert_array_equal(param.main_grad.numpy(), [2.0, 4.0])
        self.assertFalse(incoming._is_initialized())


class TestForwardPostHookTupleBranch(unittest.TestCase):
    """Covers the tuple branch of the forward post hook built by
    ``_register_fusion_layer_hooks``."""

    def _capture_post_hook(self, param_names):
        fsdp = object.__new__(fsf.FullyShardFusion)
        fsdp.comm_manager = _NoopCommManager()
        layer = _FakeLayer([_NamedParam(n) for n in param_names])
        fsdp._register_fusion_layer_hooks(layer, recursive=False)
        self.assertIsNotNone(layer.post_hook)
        return layer

    def test_exposed_param_spliced_back_untouched(self):
        hidden = paddle.to_tensor([1.0, 2.0])
        weight = paddle.to_tensor([3.0, 4.0])  # exposed LM-head weight
        layer = self._capture_post_hook([weight.name])
        result = layer.post_hook(layer, (), (hidden, weight))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        # Exposed param returned by identity; it must be the very same object.
        self.assertIs(result[1], weight)
        # The activation is routed through the (identity) backward hook.
        np.testing.assert_array_equal(result[0].numpy(), hidden.numpy())

    def test_all_exposed_returns_outputs_unchanged(self):
        weight = paddle.to_tensor([3.0, 4.0])
        layer = self._capture_post_hook([weight.name])
        outputs = (weight,)
        result = layer.post_hook(layer, (), outputs)
        self.assertIs(result, outputs)

    def test_no_exposed_params_routes_all_through_hook(self):
        a = paddle.to_tensor([1.0])
        b = paddle.to_tensor([2.0])
        layer = self._capture_post_hook([])  # nothing exposed
        result = layer.post_hook(layer, (), (a, b))
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0].numpy(), a.numpy())
        np.testing.assert_array_equal(result[1].numpy(), b.numpy())


class _FakeParamsBuffer:
    def __init__(self, status, is_sharded=True):
        self.status = status
        self.is_sharded = is_sharded
        self.clear_calls = 0

    def clear_tmp_buffer(self):
        self.clear_calls += 1


class _RecordingCommManager:
    """comm_manager stand-in that mimics the double-buffer accounting of a
    real gather (FREED -> USING, count += 1) and records how it was called."""

    def __init__(
        self, buffer_manager, buffer_cnt_in_using=0, fail_gather=False
    ):
        self.buffer_manager = buffer_manager
        self.enable_overlap = True
        self.prefetch_by_size = False
        self.buffer_cnt_in_using = buffer_cnt_in_using
        self.fail_gather = fail_gather
        self.gathered = None
        self.sharded = None
        self.overlap_during_gather = None
        self.grad_enabled_during_gather = None

    def all_gather_params(self, params):
        self.overlap_during_gather = self.enable_overlap
        self.grad_enabled_during_gather = paddle.is_grad_enabled()
        if self.fail_gather:
            raise RuntimeError("gather failed")
        self.gathered = list(params)
        bm = self.buffer_manager
        for gid in {bm.param_to_buffer_id[p.name] for p in params}:
            buf = bm.buffer_groups[gid].params_buffer
            if buf.status == fsf.BufferState.FREED:
                buf.status = fsf.BufferState.USING
                self.buffer_cnt_in_using += 1

    def shard_params(self, params, is_backward=False):
        # The real shard_params only drops the param views; buffer state is
        # left for unshard_params to restore.
        self.sharded = list(params)

    _free_params_buffer = fsf.FSDPCommManager._free_params_buffer


class TestUnshardParams(unittest.TestCase):
    """Covers ``FullyShardFusion.unshard_params``."""

    def _make_fsdp(self, statuses, param_to_gid, **cm_kwargs):
        groups = [
            types.SimpleNamespace(params_buffer=_FakeParamsBuffer(*s))
            for s in statuses
        ]
        bm = types.SimpleNamespace(
            param_to_buffer_id=dict(param_to_gid), buffer_groups=groups
        )
        fsdp = object.__new__(fsf.FullyShardFusion)
        fsdp.buffer_manager = bm
        fsdp.comm_manager = _RecordingCommManager(bm, **cm_kwargs)
        return fsdp, groups

    def test_freed_buffer_gathered_then_restored(self):
        fsdp, groups = self._make_fsdp(
            [(fsf.BufferState.FREED,)], {"w0": 0, "w1": 0}
        )
        cm = fsdp.comm_manager
        params = [_NamedParam("w0"), _NamedParam("w1")]
        with fsdp.unshard_params(params):
            self.assertEqual(cm.gathered, params)
            self.assertIsNone(cm.sharded)
            self.assertEqual(
                groups[0].params_buffer.status, fsf.BufferState.USING
            )
            self.assertEqual(cm.buffer_cnt_in_using, 1)
            # Overlap is restored and grad re-enabled once the gather returns.
            self.assertTrue(cm.enable_overlap)
            self.assertTrue(paddle.is_grad_enabled())
        # Gather is synchronous and runs without autograd.
        self.assertFalse(cm.overlap_during_gather)
        self.assertFalse(cm.grad_enabled_during_gather)
        self.assertEqual(cm.sharded, params)
        buf = groups[0].params_buffer
        self.assertEqual(buf.status, fsf.BufferState.FREED)
        self.assertEqual(buf.clear_calls, 1)
        self.assertEqual(cm.buffer_cnt_in_using, 0)

    def test_already_gathered_buffer_left_untouched(self):
        # A buffer that was live before entry belongs to the caller's
        # schedule: no clear, no state change, no count change.
        fsdp, groups = self._make_fsdp(
            [(fsf.BufferState.READY,)], {"w": 0}, buffer_cnt_in_using=1
        )
        cm = fsdp.comm_manager
        params = [_NamedParam("w")]
        with fsdp.unshard_params(params):
            pass
        buf = groups[0].params_buffer
        self.assertEqual(cm.sharded, params)
        self.assertEqual(buf.status, fsf.BufferState.READY)
        self.assertEqual(buf.clear_calls, 0)
        self.assertEqual(cm.buffer_cnt_in_using, 1)

    def test_unmanaged_params_are_ignored(self):
        fsdp, groups = self._make_fsdp([(fsf.BufferState.FREED,)], {"w": 0})
        cm = fsdp.comm_manager
        managed = _NamedParam("w")
        with fsdp.unshard_params([managed, _NamedParam("not-fsdp")]):
            pass
        self.assertEqual(cm.gathered, [managed])
        self.assertEqual(cm.sharded, [managed])

    def test_mixed_groups_only_freed_ones_restored(self):
        fsdp, groups = self._make_fsdp(
            [(fsf.BufferState.FREED,), (fsf.BufferState.USING,)],
            {"a": 0, "b": 1},
            buffer_cnt_in_using=1,
        )
        cm = fsdp.comm_manager
        with fsdp.unshard_params([_NamedParam("a"), _NamedParam("b")]):
            self.assertEqual(cm.buffer_cnt_in_using, 2)
        self.assertEqual(groups[0].params_buffer.status, fsf.BufferState.FREED)
        self.assertEqual(groups[0].params_buffer.clear_calls, 1)
        self.assertEqual(groups[1].params_buffer.status, fsf.BufferState.USING)
        self.assertEqual(groups[1].params_buffer.clear_calls, 0)
        self.assertEqual(cm.buffer_cnt_in_using, 1)

    def test_unsharded_buffer_does_not_decrement_count(self):
        fsdp, groups = self._make_fsdp(
            [(fsf.BufferState.FREED, False)], {"w": 0}
        )
        cm = fsdp.comm_manager
        cm.buffer_cnt_in_using = 3
        with fsdp.unshard_params([_NamedParam("w")]):
            cm.buffer_cnt_in_using = 3
        self.assertEqual(groups[0].params_buffer.status, fsf.BufferState.FREED)
        self.assertEqual(groups[0].params_buffer.clear_calls, 1)
        self.assertEqual(cm.buffer_cnt_in_using, 3)

    def test_count_never_goes_negative(self):
        fsdp, groups = self._make_fsdp([(fsf.BufferState.FREED,)], {"w": 0})
        cm = fsdp.comm_manager
        with fsdp.unshard_params([_NamedParam("w")]):
            cm.buffer_cnt_in_using = 0
        self.assertEqual(cm.buffer_cnt_in_using, 0)
        self.assertEqual(groups[0].params_buffer.status, fsf.BufferState.FREED)

    def test_body_exception_still_reshards(self):
        fsdp, groups = self._make_fsdp([(fsf.BufferState.FREED,)], {"w": 0})
        cm = fsdp.comm_manager
        params = [_NamedParam("w")]
        with self.assertRaises(ValueError), fsdp.unshard_params(params):
            raise ValueError("boom")
        self.assertEqual(cm.sharded, params)
        self.assertEqual(groups[0].params_buffer.status, fsf.BufferState.FREED)
        self.assertEqual(groups[0].params_buffer.clear_calls, 1)
        self.assertEqual(cm.buffer_cnt_in_using, 0)

    def test_gather_exception_restores_overlap(self):
        fsdp, groups = self._make_fsdp(
            [(fsf.BufferState.FREED,)], {"w": 0}, fail_gather=True
        )
        cm = fsdp.comm_manager
        params = [_NamedParam("w")]
        with self.assertRaises(RuntimeError), fsdp.unshard_params(params):
            self.fail("body must not run when the gather fails")
        self.assertTrue(cm.enable_overlap)
        self.assertTrue(paddle.is_grad_enabled())
        self.assertIsNone(cm.sharded)

    def test_empty_params_is_noop_pair(self):
        fsdp, groups = self._make_fsdp([(fsf.BufferState.FREED,)], {"w": 0})
        cm = fsdp.comm_manager
        with fsdp.unshard_params([]):
            pass
        self.assertEqual(cm.gathered, [])
        self.assertEqual(cm.sharded, [])
        self.assertEqual(groups[0].params_buffer.clear_calls, 0)

    def test_buffer_freed_inside_block_not_freed_twice(self):
        # A buffer already freed inside the block (e.g. by a reshard) must
        # not be freed again on exit.
        fsdp, groups = self._make_fsdp([(fsf.BufferState.FREED,)], {"w": 0})
        cm = fsdp.comm_manager
        with fsdp.unshard_params([_NamedParam("w")]):
            cm._free_params_buffer(groups[0].params_buffer)
        self.assertEqual(cm.buffer_cnt_in_using, 0)
        self.assertEqual(groups[0].params_buffer.status, fsf.BufferState.FREED)
        self.assertEqual(groups[0].params_buffer.clear_calls, 1)


def _make_comm_manager(layout, **kwargs):
    # layout: (fsdp_unit_id, is_expert, numel, is_sharded, trainable) per bucket.
    groups = [
        types.SimpleNamespace(
            fsdp_unit_id=unit,
            is_expert_param=expert,
            params_buffer=types.SimpleNamespace(
                is_sharded=sharded,
                total_buffer_size=numel,
                trainable=trainable,
                status=fsf.BufferState.FREED,
                clear_tmp_buffer=lambda: None,
            ),
        )
        for unit, expert, numel, sharded, trainable in layout
    ]
    bm = types.SimpleNamespace(buffer_groups=groups)
    return fsf.FSDPCommManager(bm, **kwargs)


# Units 0..3, each a dense + an expert bucket of 10 + 30 elements.
_LAYOUT = [
    (u, e, n, True, True)
    for u in range(4)
    for e, n in ((False, 10), (True, 30))
]


def _fake_grads_buffer(numel, waited, i):
    return types.SimpleNamespace(
        comm_task=types.SimpleNamespace(wait=lambda: waited.append(i)),
        total_buffer_size=numel,
        is_sharded=False,
        accumulate_reduced_grad=lambda: None,
    )


class TestPrefetchByLayer(unittest.TestCase):
    """Covers the default per-unit prefetch of FSDPCommManager."""

    def test_default_is_by_layer(self):
        cm = _make_comm_manager(_LAYOUT)
        self.assertFalse(cm.prefetch_by_size)
        self.assertEqual(cm.prefetch_units, 1)
        # 2 sharded groups per unit.
        self.assertEqual(cm.gather_buffer_limit, 4)
        self.assertEqual(cm.grad_queue_limit, 2)

    def test_prefetch_units_ahead(self):
        cm = _make_comm_manager(_LAYOUT)
        self.assertEqual(cm._next_buffer_id([0, 1], False), [2, 3])
        self.assertEqual(cm._next_buffer_id([6, 7], True), [5, 4])
        cm = _make_comm_manager(_LAYOUT, prefetch_units=2)
        self.assertEqual(cm._next_buffer_id([0, 1], False), [2, 3, 4, 5])

    def test_zero_disables_prefetch(self):
        cm = _make_comm_manager(_LAYOUT, prefetch_units=0)
        self.assertEqual(cm._next_buffer_id([0, 1], False), [])

    def test_free_counts_buffers(self):
        cm = _make_comm_manager(_LAYOUT)
        buf = cm.buffer_manager.buffer_groups[0].params_buffer
        cm._count_gathered(buf)
        self.assertEqual(cm.buffer_cnt_in_using, 1)
        self.assertEqual(cm.gathered_numel, 0)
        cm._free_params_buffer(buf)
        cm._free_params_buffer(buf)
        self.assertEqual(cm.buffer_cnt_in_using, 0)

    def test_grad_queue_bounded_by_count(self):
        cm = _make_comm_manager(_LAYOUT)
        waited = []
        for i in range(4):
            cm.grad_reduce_queue.append(_fake_grads_buffer(100, waited, i))
            cm._grad_queue_numel += 100
        cm._wait_for_grad_comm()
        self.assertEqual(waited, [0, 1])
        cm._wait_for_grad_comm(queue_limit=0)
        self.assertEqual(waited, [0, 1, 2, 3])
        self.assertEqual(cm._grad_queue_numel, 0)


class TestPrefetchBySize(unittest.TestCase):
    """Covers the by-size (element count) prefetch of FSDPCommManager."""

    def test_sizes_derived_from_unit_size(self):
        cm = _make_comm_manager(_LAYOUT, comm_unit_size=80)
        self.assertTrue(cm.prefetch_by_size)
        self.assertEqual(cm.grad_queue_capacity, 80)
        self.assertEqual(cm.gather_prefetch_size, 40)

    def test_forward_prefetches_next_unit(self):
        cm = _make_comm_manager(_LAYOUT, comm_unit_size=80)
        self.assertEqual(cm._extend_by_prefetch_size([0, 1], False), [2, 3])
        self.assertEqual(cm._extend_by_prefetch_size([6, 7], False), [])

    def test_backward_walks_in_reverse(self):
        cm = _make_comm_manager(_LAYOUT, comm_unit_size=80)
        self.assertEqual(cm._extend_by_prefetch_size([6, 7], True), [5, 4])

    def test_at_least_one_bucket_group_when_positive(self):
        cm = _make_comm_manager(_LAYOUT, comm_unit_size=2)
        self.assertEqual(cm._extend_by_prefetch_size([0], False), [1])

    def test_zero_disables_prefetch(self):
        cm = _make_comm_manager(_LAYOUT, comm_unit_size=0)
        self.assertEqual(cm._extend_by_prefetch_size([0, 1], False), [])

    def test_prefetch_units_ignored(self):
        cm = _make_comm_manager(_LAYOUT, prefetch_units=3, comm_unit_size=80)
        self.assertEqual(cm._extend_by_prefetch_size([0, 1], False), [2, 3])

    def test_backward_skips_frozen_buckets(self):
        layout = list(_LAYOUT)
        layout[4] = (2, False, 10, True, False)
        cm = _make_comm_manager(layout, comm_unit_size=80)
        self.assertEqual(cm._extend_by_prefetch_size([6, 7], True), [5, 3])

    def test_skips_unsharded_buckets(self):
        layout = list(_LAYOUT)
        layout[2] = (1, False, 1000, False, True)
        cm = _make_comm_manager(layout, comm_unit_size=80)
        self.assertEqual(cm._extend_by_prefetch_size([0, 1], False), [3, 4])
        _, _, limit = cm._prefetch_plan([2, 3], False)
        self.assertEqual(limit, 80)

    def test_plan_keep_and_limit(self):
        cm = _make_comm_manager(_LAYOUT, comm_unit_size=20)
        targets, keep, limit = cm._prefetch_plan([0, 1], False)
        self.assertEqual(targets, (2,))
        self.assertEqual(keep, frozenset({0, 1, 2}))
        # Window (50) exceeds one communication unit (20).
        self.assertEqual(limit, 50)
        self.assertIs(cm._prefetch_plan([0, 1], False)[1], keep)

    def test_free_counts_numel(self):
        cm = _make_comm_manager(_LAYOUT, comm_unit_size=80)
        buf = cm.buffer_manager.buffer_groups[1].params_buffer
        cm._count_gathered(buf)
        self.assertEqual(cm.gathered_numel, 30)
        self.assertEqual(cm.buffer_cnt_in_using, 0)
        cm._free_params_buffer(buf)
        self.assertEqual(cm.gathered_numel, 0)

    def test_grad_queue_bounded_by_numel(self):
        cm = _make_comm_manager(_LAYOUT, comm_unit_size=25)
        waited = []
        for i in range(4):
            cm.grad_reduce_queue.append(_fake_grads_buffer(10, waited, i))
            cm._grad_queue_numel += 10
        cm._wait_for_grad_comm()
        self.assertEqual(waited, [0, 1])
        self.assertEqual(cm._grad_queue_numel, 20)
        cm._wait_for_grad_comm(queue_limit=0)
        self.assertEqual(waited, [0, 1, 2, 3])
        self.assertEqual(cm._grad_queue_numel, 0)

    def test_grad_queue_reserves_next_numel(self):
        cm = _make_comm_manager(_LAYOUT, comm_unit_size=25)
        waited = []
        for i in range(2):
            cm.grad_reduce_queue.append(_fake_grads_buffer(10, waited, i))
            cm._grad_queue_numel += 10
        # 20 + 10 > 25: retire one so the next buffer fits.
        cm._wait_for_grad_comm(next_numel=10)
        self.assertEqual(waited, [0])
        self.assertEqual(cm._grad_queue_numel, 10)
        # A buffer larger than the capacity drains the queue and stops.
        cm._wait_for_grad_comm(next_numel=100)
        self.assertEqual(waited, [0, 1])
        self.assertEqual(cm._grad_queue_numel, 0)
        cm._wait_for_grad_comm(next_numel=100)
        self.assertEqual(waited, [0, 1])

    def test_release_ready_buffers_oldest_first(self):
        cm = _make_comm_manager(_LAYOUT, comm_unit_size=80)
        bufs = [g.params_buffer for g in cm.buffer_manager.buffer_groups]
        # gid 0, 1, 3 READY (70), gid 2 SYNCING (10), gid 4 USING (10).
        statuses = {
            0: fsf.BufferState.READY,
            1: fsf.BufferState.READY,
            2: fsf.BufferState.SYNCING,
            3: fsf.BufferState.READY,
            4: fsf.BufferState.USING,
        }
        for gid, status in statuses.items():
            bufs[gid].status = status
            cm.gathered_numel += bufs[gid].total_buffer_size
        self.assertEqual(cm.gathered_numel, 90)

        # Fits already: nothing freed.
        cm._release_ready_buffers(10, 100)
        self.assertEqual(cm.gathered_numel, 90)

        # gid 0 is kept, so gid 1 (30) goes first and is enough.
        cm._release_ready_buffers(10, 70, keep={0})
        self.assertEqual(bufs[1].status, fsf.BufferState.FREED)
        self.assertEqual(bufs[0].status, fsf.BufferState.READY)
        self.assertEqual(bufs[3].status, fsf.BufferState.READY)
        self.assertEqual(cm.gathered_numel, 60)

        # Only READY buffers are freed, even if the limit stays exceeded.
        cm._release_ready_buffers(10, 0)
        self.assertEqual(bufs[0].status, fsf.BufferState.FREED)
        self.assertEqual(bufs[3].status, fsf.BufferState.FREED)
        self.assertEqual(bufs[2].status, fsf.BufferState.SYNCING)
        self.assertEqual(bufs[4].status, fsf.BufferState.USING)
        self.assertEqual(cm.gathered_numel, 20)


class _FakeTask:
    def __init__(self, log, gid):
        self.log = log
        self.gid = gid

    def wait(self):
        self.log.append(self.gid)


class _GatherParamsBuffer:
    """Sharded params buffer whose "gathered" storage is filled with its
    gid, so a bound param shows which buffer it was viewed from."""

    def __init__(self, gid, params, is_sharded, issued, waited):
        self.gid = gid
        self.is_sharded = is_sharded
        self.trainable = True
        self.status = (
            fsf.BufferState.FREED if is_sharded else fsf.BufferState.USING
        )
        self.comm_task = None
        self.param_offsets = {}
        offset = 0
        for param in params:
            self.param_offsets[param.name] = offset
            offset += param._numel()
        self.total_buffer_size = offset
        self.tmp_data_buffer = None
        self._issued = issued
        self._waited = waited

    def get_tmp_buffer(self):
        if self.tmp_data_buffer is None:
            self.tmp_data_buffer = paddle.full(
                [self.total_buffer_size], float(self.gid), dtype="float32"
            )
        return self.tmp_data_buffer

    def do_all_gather(self):
        self.get_tmp_buffer()
        self._issued.append(self.gid)
        return _FakeTask(self._waited, self.gid)

    def clear_tmp_buffer(self):
        self.tmp_data_buffer = None


class TestPrefetchBySizeSequence(unittest.TestCase):
    """Drives FSDPCommManager.all_gather_params / shard_params through whole
    forward + backward passes in prefetch-by-size mode."""

    def _build(self, comm_unit_size, layout=_LAYOUT):
        self.issued, self.waited = [], []
        groups, param_to_gid = [], {}
        for gid, (unit, expert, numel, sharded, _) in enumerate(layout):
            params = [
                paddle.create_parameter([numel // 2], "float32"),
                paddle.create_parameter([numel - numel // 2], "float32"),
            ]
            for param in params:
                param_to_gid[param.name] = gid
            groups.append(
                types.SimpleNamespace(
                    fsdp_unit_id=unit,
                    is_expert_param=expert,
                    params=params,
                    grads_buffer=None,
                    params_buffer=_GatherParamsBuffer(
                        gid, params, sharded, self.issued, self.waited
                    ),
                )
            )
        bm = types.SimpleNamespace(
            buffer_groups=groups, param_to_buffer_id=param_to_gid
        )
        self.cm = fsf.FSDPCommManager(bm, comm_unit_size=comm_unit_size)
        self.units = {}
        for gid, g in enumerate(groups):
            self.units.setdefault(g.fsdp_unit_id, []).append(gid)
        return groups

    def _unit_params(self, unit):
        groups = self.cm.buffer_manager.buffer_groups
        return [p for gid in self.units[unit] for p in groups[gid].params]

    def _check_accounting(self):
        live = sum(
            g.params_buffer.total_buffer_size
            for g in self.cm.buffer_manager.buffer_groups
            if g.params_buffer.is_sharded
            and g.params_buffer.status != fsf.BufferState.FREED
        )
        self.assertEqual(self.cm.gathered_numel, live)

    def _gather(self, unit, is_backward):
        params = self._unit_params(unit)
        self.cm.all_gather_params(params, is_backward=is_backward)
        _, _, limit = self.cm._prefetch_plan(self.units[unit], is_backward)
        self.assertLessEqual(self.cm.gathered_numel, limit)
        self._check_accounting()
        groups = self.cm.buffer_manager.buffer_groups
        for gid in self.units[unit]:
            buf = groups[gid].params_buffer
            if buf.is_sharded:
                self.assertEqual(buf.status, fsf.BufferState.USING)
            # Each param is a view of its own gathered buffer.
            for param in groups[gid].params:
                np.testing.assert_array_equal(
                    param.numpy(), np.full(param.shape, gid, "float32")
                )
        return params

    def _run_step(self):
        groups = self.cm.buffer_manager.buffer_groups
        order = sorted(self.units)
        statuses_before = {}
        for unit in order:
            statuses_before[("fwd", unit)] = [
                groups[gid].params_buffer.status for gid in self.units[unit]
            ]
            params = self._gather(unit, is_backward=False)
            self.cm.shard_params(params)
            self._check_accounting()
        for unit in reversed(order):
            statuses_before[("bwd", unit)] = [
                groups[gid].params_buffer.status for gid in self.units[unit]
            ]
            params = self._gather(unit, is_backward=True)
            # Sharded units after this one were released on entering it.
            for later in order:
                if later <= unit:
                    continue
                for gid in self.units[later]:
                    buf = groups[gid].params_buffer
                    if not buf.is_sharded:
                        continue
                    self.assertEqual(buf.status, fsf.BufferState.FREED)
            self.cm.shard_params(params, is_backward=True)
            self._check_accounting()
        self.cm.reset_params_buffer_status()
        self.assertEqual(self.cm.gathered_numel, 0)
        for g in groups:
            if g.params_buffer.is_sharded:
                self.assertEqual(g.params_buffer.status, fsf.BufferState.FREED)
        # Every issued all_gather was waited on exactly once.
        self.assertEqual(sorted(self.issued), sorted(self.waited))
        return statuses_before

    def test_window_fits_limit(self):
        # One unit = 40 elements; 80 holds the current and the next unit.
        self._build(comm_unit_size=80)
        for _ in range(2):
            self.issued.clear()
            self.waited.clear()
            before = self._run_step()
            self.assertLessEqual(self.cm.gathered_numel, 80)
            # Forward units after the first were prefetched by the previous
            # call (SYNCING); the last unit is reused at backward start.
            for unit in (1, 2, 3):
                self.assertEqual(
                    before[("fwd", unit)], [fsf.BufferState.SYNCING] * 2
                )
            self.assertEqual(before[("bwd", 3)], [fsf.BufferState.READY] * 2)
            self.assertEqual(before[("bwd", 2)], [fsf.BufferState.READY] * 2)
            self.assertEqual(before[("bwd", 1)], [fsf.BufferState.SYNCING] * 2)
            # 8 forward gathers; backward regathers units 1 and 0 only.
            self.assertEqual(self.issued, [0, 1, 2, 3, 4, 5, 6, 7, 3, 2, 1, 0])

    def test_small_limit(self):
        # Window larger than comm_unit_size: bounded by the plan's limit.
        self._build(comm_unit_size=20)
        for _ in range(2):
            self.issued.clear()
            self.waited.clear()
            self._run_step()

    def test_zero_disables_prefetch(self):
        self._build(comm_unit_size=0)
        before = self._run_step()
        for unit in (1, 2, 3):
            self.assertEqual(before[("fwd", unit)], [fsf.BufferState.FREED] * 2)
        # Gathered on demand, waited right away.
        self.assertEqual(self.issued, self.waited)

    def test_large_limit(self):
        # Everything prefetched; no READY buffer is ever freed in forward.
        self._build(comm_unit_size=1000)
        before = self._run_step()
        self.assertEqual(self.issued[:8], list(range(8)))
        for unit in (1, 2, 3):
            self.assertEqual(
                before[("fwd", unit)], [fsf.BufferState.SYNCING] * 2
            )

    def test_unsharded_buckets_not_counted(self):
        layout = list(_LAYOUT)
        layout[2] = (1, False, 10, False, True)
        self._build(comm_unit_size=80, layout=layout)
        self._run_step()
        self.assertNotIn(2, self.issued)


if __name__ == "__main__":
    unittest.main()
