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
        cm = _make_comm_manager(_LAYOUT, prefetch_sizes=80)
        self.assertTrue(cm.prefetch_by_size)
        self.assertEqual(cm.grad_queue_capacity, 80)
        self.assertEqual(cm.gather_prefetch_size, 40)

    def test_forward_prefetches_next_unit(self):
        cm = _make_comm_manager(_LAYOUT, prefetch_sizes=80)
        self.assertEqual(cm._extend_by_prefetch_size([0, 1], False), [2, 3])
        self.assertEqual(cm._extend_by_prefetch_size([6, 7], False), [])

    def test_backward_walks_in_reverse(self):
        cm = _make_comm_manager(_LAYOUT, prefetch_sizes=80)
        self.assertEqual(cm._extend_by_prefetch_size([6, 7], True), [5, 4])

    def test_at_least_one_bucket_group_when_positive(self):
        cm = _make_comm_manager(_LAYOUT, prefetch_sizes=2)
        self.assertEqual(cm._extend_by_prefetch_size([0], False), [1])

    def test_zero_disables_prefetch(self):
        cm = _make_comm_manager(_LAYOUT, prefetch_sizes=0)
        self.assertEqual(cm._extend_by_prefetch_size([0, 1], False), [])

    def test_prefetch_units_ignored(self):
        cm = _make_comm_manager(_LAYOUT, prefetch_units=3, prefetch_sizes=80)
        self.assertEqual(cm._extend_by_prefetch_size([0, 1], False), [2, 3])

    def test_backward_skips_frozen_buckets(self):
        layout = list(_LAYOUT)
        layout[4] = (2, False, 10, True, False)
        cm = _make_comm_manager(layout, prefetch_sizes=80)
        self.assertEqual(cm._extend_by_prefetch_size([6, 7], True), [5, 3])

    def test_skips_unsharded_buckets(self):
        layout = list(_LAYOUT)
        layout[2] = (1, False, 1000, False, True)
        cm = _make_comm_manager(layout, prefetch_sizes=80)
        self.assertEqual(cm._extend_by_prefetch_size([0, 1], False), [3, 4])
        _, _, limit = cm._prefetch_plan([2, 3], False)
        self.assertEqual(limit, 80)

    def test_plan_keep_and_limit(self):
        cm = _make_comm_manager(_LAYOUT, prefetch_sizes=20)
        targets, keep, limit = cm._prefetch_plan([0, 1], False)
        self.assertEqual(targets, (2,))
        self.assertEqual(keep, frozenset({0, 1, 2}))
        # Window (50) exceeds one communication unit (20).
        self.assertEqual(limit, 50)
        self.assertIs(cm._prefetch_plan([0, 1], False)[1], keep)

    def test_free_counts_numel(self):
        cm = _make_comm_manager(_LAYOUT, prefetch_sizes=80)
        buf = cm.buffer_manager.buffer_groups[1].params_buffer
        cm._count_gathered(buf)
        self.assertEqual(cm.gathered_numel, 30)
        self.assertEqual(cm.buffer_cnt_in_using, 0)
        cm._free_params_buffer(buf)
        self.assertEqual(cm.gathered_numel, 0)

    def test_grad_queue_bounded_by_numel(self):
        cm = _make_comm_manager(_LAYOUT, prefetch_sizes=25)
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


if __name__ == "__main__":
    unittest.main()
