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

"""Single-process unit tests for ``paddle.optimizer.mixed_inner_optimizer``.

Covers the inner-optimizer registry, the ``_GatheredMasterWeights`` write-through
view and ``MixedInnerOptimizer`` (construction / routing / state-dict aggregation
and routing / fused-storage build / state-layout aggregation). All single
process, real paddle dygraph -- the sharding shell that wraps this optimizer is
covered by the collective tests under test/collective/fleet/.
"""

import itertools
import unittest

import paddle
import paddle.optimizer.mixed_inner_optimizer as mixed_mod
from paddle.optimizer.inner_optimizer import InnerOptimizer
from paddle.optimizer.mixed_inner_optimizer import (
    INNER_OPTIMIZER_REGISTRY,
    MixedInnerOptimizer,
    _GatheredMasterWeights,
    get_inner_optimizer,
    register_inner_optimizer,
)

_UID = itertools.count()


def _named_param(name, shape=(4, 4), dtype="float32", value=1.0):
    """Create a parameter with an EXACT name (caller guarantees uniqueness)."""
    return paddle.create_parameter(
        shape=list(shape),
        dtype=dtype,
        name=name,
        default_initializer=paddle.nn.initializer.Constant(value),
    )


def _param(name_hint, shape=(4, 4), dtype="float32", value=1.0):
    # dygraph forbids duplicate parameter names within a process, so make every
    # created parameter's name unique regardless of how many times a test reuses
    # the same hint.
    return _named_param(f"{name_hint}_{next(_UID)}", shape, dtype, value)


class _SGDInner(InnerOptimizer):
    """moment1-only inner optimizer (non-default state layout)."""

    def _ensure_accumulators(self, param):
        if (
            "moment1" in self._accumulators
            and param.name in self._accumulators["moment1"]
        ):
            return
        self._ensure_master_weight(param)
        self._add_accumulator(
            "moment1",
            param,
            dtype=paddle.float32,
            fill_value=0.0,
            shape=param.shape,
        )

    def _update_parameters(self, params_grads, lr):
        with paddle.no_grad():
            for param, grad in params_grads:
                m = self._get_accumulator("moment1", param)
                paddle.assign(0.9 * m + grad.astype(m.dtype), m)
                param.subtract_(
                    (self._effective_lr(param, lr) * m).astype(param.dtype)
                )

    def optimizer_state_layout(self):
        return {
            "vector": ["moment1_0"],
            "scalar": [],
            "master": "fp32_master_0",
        }


class _AdamInner(InnerOptimizer):
    """moment1/moment2 + beta pow (inherits the default AdamW layout)."""

    def _ensure_accumulators(self, param):
        if (
            "moment1" in self._accumulators
            and param.name in self._accumulators["moment1"]
        ):
            return
        self._ensure_master_weight(param)
        for a in ("moment1", "moment2"):
            self._add_accumulator(
                a,
                param,
                dtype=paddle.float32,
                fill_value=0.0,
                shape=param.shape,
            )
        for a, v in (("beta1_pow_acc", 0.9), ("beta2_pow_acc", 0.999)):
            self._add_accumulator(
                a, param, dtype=paddle.float32, fill_value=v, shape=[1]
            )

    def _update_parameters(self, params_grads, lr):
        with paddle.no_grad():
            for param, grad in params_grads:
                m = self._get_accumulator("moment1", param)
                paddle.assign(0.9 * m + grad.astype(m.dtype), m)
                param.subtract_(
                    (self._effective_lr(param, lr) * m).astype(param.dtype)
                )


class _ConflictMaster(_SGDInner):
    def optimizer_state_layout(self):
        return {
            "vector": ["moment1_0"],
            "scalar": [],
            "master": "other_master_0",
        }


class _GradClipHookSub(_SGDInner):
    """Exposes ``_set_grad_clip_none`` so the chain uses that hook."""

    def _set_grad_clip_none(self):
        self._grad_clip = None


def _make_mixed(**kwargs):
    """Two SGD groups (paramsA=2, paramsB=1) wrapped in a MixedInnerOptimizer."""
    pa = [_param("mixA0"), _param("mixA1")]
    pb = [_param("mixB0")]
    subA = _SGDInner(parameters=pa, learning_rate=0.1)
    subB = _SGDInner(parameters=pb, learning_rate=0.1)
    mixed = MixedInnerOptimizer([subA, subB], learning_rate=0.1, **kwargs)
    return mixed, subA, subB, pa, pb


class TestRegistry(unittest.TestCase):
    def tearDown(self):
        for n in ("t_reg", "t_reg2", "t_dup"):
            INNER_OPTIMIZER_REGISTRY.pop(n, None)

    def test_register_and_get(self):
        register_inner_optimizer("t_reg", _SGDInner)
        self.assertIs(get_inner_optimizer("t_reg"), _SGDInner)

    def test_register_same_is_noop(self):
        register_inner_optimizer("t_reg2", _SGDInner)
        register_inner_optimizer("t_reg2", _SGDInner)  # no raise
        self.assertIs(get_inner_optimizer("t_reg2"), _SGDInner)

    def test_register_shadow_raises(self):
        register_inner_optimizer("t_dup", _SGDInner)
        with self.assertRaises(ValueError):
            register_inner_optimizer("t_dup", _AdamInner)

    def test_register_bad_name(self):
        with self.assertRaises(ValueError):
            register_inner_optimizer("", _SGDInner)
        with self.assertRaises(ValueError):
            register_inner_optimizer(123, _SGDInner)

    def test_register_non_inner_optimizer(self):
        with self.assertRaises(TypeError):
            register_inner_optimizer("t_bad", dict)

    def test_get_unknown(self):
        with self.assertRaises(KeyError):
            get_inner_optimizer("does_not_exist_xyz")


class TestGatheredMasterWeights(unittest.TestCase):
    def _seed(self):
        mixed, subA, subB, pa, pb = _make_mixed()
        subA._master_weights[pa[0].name] = paddle.ones([2, 2])
        subB._master_weights[pb[0].name] = paddle.ones([2, 2])
        return mixed, subA, subB, pa, pb

    def test_getitem_len_iter_contains(self):
        mixed, subA, subB, pa, pb = self._seed()
        view = mixed._master_weights
        self.assertEqual(len(view), 2)
        self.assertEqual(set(view), {pa[0].name, pb[0].name})
        self.assertIn(pa[0].name, view)
        self.assertNotIn("nope", view)
        self.assertIs(view[pa[0].name], subA._master_weights[pa[0].name])
        with self.assertRaises(KeyError):
            _ = view["nope"]

    def test_setitem_write_through(self):
        mixed, subA, subB, pa, pb = _make_mixed()
        mw = paddle.ones([3, 3])
        mixed._master_weights[pa[0].name] = mw  # routes to owner subA
        self.assertIs(subA._master_weights[pa[0].name], mw)
        self.assertIs(mixed._master_weights[pa[0].name], mw)

    def test_delitem(self):
        mixed, subA, subB, pa, pb = self._seed()
        del mixed._master_weights[pa[0].name]
        self.assertNotIn(pa[0].name, subA._master_weights)
        with self.assertRaises(KeyError):
            del mixed._master_weights["nope"]

    def test_iter_duplicate_raises(self):
        mixed, subA, subB, pa, pb = _make_mixed()
        # same key present on two subs -> iteration must flag the ambiguity
        subA._master_weights["dup"] = paddle.ones([1])
        subB._master_weights["dup"] = paddle.ones([1])
        with self.assertRaises(ValueError):
            list(mixed._master_weights)

    def test_setter_ignores_dict_assignment(self):
        mixed, *_ = _make_mixed()
        mixed._master_weights = {"x": paddle.ones([1])}  # ignored by design
        self.assertIsInstance(mixed._master_weights, _GatheredMasterWeights)
        self.assertNotIn("x", mixed._master_weights)


class TestMixedInit(unittest.TestCase):
    def test_empty_subs(self):
        with self.assertRaises(ValueError):
            MixedInnerOptimizer([], learning_rate=0.1)

    def test_sub_without_apply_optimize(self):
        class _NotAnOpt:
            _parameter_list = []

        with self.assertRaises(TypeError):
            MixedInnerOptimizer([_NotAnOpt()], learning_rate=0.1)

    def test_duplicate_param_across_subs(self):
        p = _param("dupparam")
        subA = _SGDInner(parameters=[p], learning_rate=0.1)
        subB = _SGDInner(parameters=[p], learning_rate=0.1)
        with self.assertRaises(ValueError):
            MixedInnerOptimizer([subA, subB], learning_rate=0.1)

    def test_union_param_list_and_grad_clip_nulled(self):
        clip = paddle.nn.ClipGradByGlobalNorm(1.0)
        pa = [_param("uA0"), _param("uA1")]
        pb = [_param("uB0")]
        # subA clears its clip via the _set_grad_clip_none hook, subB directly.
        subA = _GradClipHookSub(
            parameters=pa, learning_rate=0.1, grad_clip=clip
        )
        subB = _SGDInner(parameters=pb, learning_rate=0.1, grad_clip=clip)
        mixed = MixedInnerOptimizer(
            [subA, subB], learning_rate=0.1, grad_clip=clip
        )
        self.assertEqual(len(mixed._parameter_list), 3)
        # chain owns clipping; subs must not clip again
        self.assertIsNone(subA._grad_clip)
        self.assertIsNone(subB._grad_clip)
        self.assertIs(mixed._grad_clip, clip)

    def test_merged_param_info_map(self):
        pa = [_param("miA0")]
        pb = [_param("miB0")]
        subA = _SGDInner(
            parameters=pa,
            learning_rate=0.1,
            param_info_map={pa[0].name: object()},
        )
        subB = _SGDInner(
            parameters=pb,
            learning_rate=0.1,
            param_info_map={pb[0].name: object()},
        )
        mixed = MixedInnerOptimizer(
            [subA, subB], learning_rate=0.1, param_info_map={"extra": object()}
        )
        self.assertEqual(
            set(mixed._param_info_map), {pa[0].name, pb[0].name, "extra"}
        )


class TestMixedMasterWeightRouting(unittest.TestCase):
    def test_set_master_weight_routes(self):
        mixed, subA, subB, pa, pb = _make_mixed()
        mw = paddle.ones([2, 2])
        mixed._set_master_weight(pa[0].name, mw)
        self.assertIs(subA._master_weights[pa[0].name], mw)

    def test_set_master_weight_unrouted(self):
        mixed, *_ = _make_mixed()
        with self.assertRaises(KeyError):
            mixed._set_master_weight("nope", paddle.ones([1]))

    def test_create_master_weight_routes(self):
        pa = [_param("cmA0", dtype="bfloat16")]
        pb = [_param("cmB0", dtype="bfloat16")]
        subA = _SGDInner(parameters=pa, learning_rate=0.1, multi_precision=True)
        subB = _SGDInner(parameters=pb, learning_rate=0.1, multi_precision=True)
        mixed = MixedInnerOptimizer(
            [subA, subB], learning_rate=0.1, multi_precision=True
        )
        mixed._create_master_weight(pa[0])
        self.assertIn(pa[0].name, subA._master_weights)

    def test_create_master_weight_unrouted(self):
        mixed, *_ = _make_mixed()
        with self.assertRaises(KeyError):
            mixed._create_master_weight(_param("orphan"))


class TestMixedApplyOptimize(unittest.TestCase):
    def test_routing_updates_each_sub(self):
        mixed, subA, subB, pa, pb = _make_mixed()
        grads = [(p, paddle.ones_like(p)) for p in pa + pb]
        mixed._apply_optimize(None, None, grads)
        self.assertIn(pa[0].name, subA._accumulators["moment1"])
        self.assertIn(pb[0].name, subB._accumulators["moment1"])

    def test_grad_none_skipped(self):
        mixed, subA, subB, pa, pb = _make_mixed()
        mixed._apply_optimize(
            None, None, [(pa[0], None), (pb[0], paddle.ones_like(pb[0]))]
        )
        self.assertNotIn("moment1", subA._accumulators)
        self.assertIn(pb[0].name, subB._accumulators["moment1"])

    def test_unrouted_param_raises(self):
        mixed, *_ = _make_mixed()
        orphan = _param("orphan2")
        with self.assertRaises(KeyError):
            mixed._apply_optimize(
                None, None, [(orphan, paddle.ones_like(orphan))]
            )

    def test_grad_clip_once(self):
        clip = paddle.nn.ClipGradByGlobalNorm(1e-6)
        pa = [_param("gcA0")]
        pb = [_param("gcB0")]
        subA = _SGDInner(parameters=pa, learning_rate=0.1)
        subB = _SGDInner(parameters=pb, learning_rate=0.1)
        mixed = MixedInnerOptimizer(
            [subA, subB], learning_rate=0.1, grad_clip=clip
        )
        grads = [(p, paddle.ones_like(p)) for p in pa + pb]
        mixed._apply_optimize(None, None, grads)
        self.assertIn(pa[0].name, subA._accumulators["moment1"])

    def test_bypass(self):
        mixed, subA, subB, pa, pb = _make_mixed()
        saved = mixed_mod.g_shard_bypass_dygraph_optimizer
        mixed_mod.g_shard_bypass_dygraph_optimizer = 1
        try:
            mixed._apply_optimize(
                None, None, [(pa[0], paddle.ones_like(pa[0]))]
            )
        finally:
            mixed_mod.g_shard_bypass_dygraph_optimizer = saved
        self.assertEqual(subA._accumulators, {})

    def test_static_rejected(self):
        mixed, *_ = _make_mixed()
        paddle.enable_static()
        try:
            with self.assertRaises(NotImplementedError):
                mixed._apply_optimize(None, None, [])
        finally:
            paddle.disable_static()


class TestMixedCreateAccumulatorsAndMerge(unittest.TestCase):
    def test_create_accumulators_routes(self):
        mixed, subA, subB, pa, pb = _make_mixed()
        mixed._create_accumulators(None, pa + pb)
        self.assertIn(pa[0].name, subA._accumulators["moment1"])
        self.assertIn(pb[0].name, subB._accumulators["moment1"])

    def test_merged_accumulators_fresh(self):
        mixed, subA, subB, pa, pb = _make_mixed()
        mixed._apply_optimize(
            None, None, [(p, paddle.ones_like(p)) for p in pa + pb]
        )
        merged = mixed.merged_accumulators()
        self.assertIn("moment1", merged)
        self.assertEqual(set(merged["moment1"]), {p.name for p in pa + pb})
        self.assertIsNot(mixed.merged_accumulators(), merged)


class TestMixedStateLayout(unittest.TestCase):
    def test_layout_union(self):
        pa = [_param("laA0")]
        pb = [_param("laB0")]
        subA = _SGDInner(parameters=pa, learning_rate=0.1)  # moment1 only
        subB = _AdamInner(parameters=pb, learning_rate=0.1)  # default layout
        mixed = MixedInnerOptimizer([subA, subB], learning_rate=0.1)
        layout = mixed.optimizer_state_layout()
        self.assertEqual(layout["vector"], ["moment1_0", "moment2_0"])
        self.assertEqual(
            layout["scalar"], ["beta1_pow_acc_0", "beta2_pow_acc_0"]
        )
        self.assertEqual(layout["master"], "fp32_master_0")

    def test_layout_master_conflict(self):
        pa = [_param("lcA0")]
        pb = [_param("lcB0")]
        subA = _SGDInner(parameters=pa, learning_rate=0.1)
        subB = _ConflictMaster(parameters=pb, learning_rate=0.1)
        mixed = MixedInnerOptimizer([subA, subB], learning_rate=0.1)
        with self.assertRaises(ValueError):
            mixed.optimizer_state_layout()


class TestMixedStateDict(unittest.TestCase):
    def test_roundtrip_with_scheduler_and_master(self):
        sched = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[100], values=[0.1, 0.01]
        )
        pa = [_param("rtA0", dtype="bfloat16")]
        pb = [_param("rtB0", dtype="bfloat16")]
        subA = _SGDInner(
            parameters=pa, learning_rate=sched, multi_precision=True
        )
        subB = _SGDInner(
            parameters=pb, learning_rate=sched, multi_precision=True
        )
        mixed = MixedInnerOptimizer(
            [subA, subB], learning_rate=sched, multi_precision=True
        )
        grads = [(p, paddle.ones_like(p)) for p in pa + pb]
        mixed._apply_optimize(None, None, grads)
        sd0 = mixed.state_dict()
        # one chain-level LR_Scheduler; bf16 params got fp32 masters
        self.assertIn("LR_Scheduler", sd0)
        self.assertEqual(set(sd0["master_weights"]), {pa[0].name, pb[0].name})
        acc_keys = [
            k for k in sd0 if k not in ("LR_Scheduler", "master_weights")
        ]
        self.assertTrue(any(pa[0].name in k for k in acc_keys))
        self.assertTrue(any(pb[0].name in k for k in acc_keys))
        restore = {}
        for k, v in sd0.items():
            if k == "master_weights":
                restore[k] = {pk: pv.clone() for pk, pv in v.items()}
            elif k == "LR_Scheduler":
                restore[k] = v
            else:
                restore[k] = v.clone()
        mixed._apply_optimize(None, None, grads)  # perturb
        # restore: chain scheduler restore + master routing + sub LR reinject
        mixed.set_state_dict(restore)
        self.assertAlmostEqual(
            float(
                subA._accumulators["moment1"][pa[0].name]
                .astype("float32")
                .mean()
            ),
            1.0,
            places=2,
        )

    def test_state_dict_master_dedup(self):
        mixed, subA, subB, pa, pb = _make_mixed()
        subA._master_weights[pa[0].name] = paddle.ones([2, 2])
        subB._master_weights[pb[0].name] = paddle.ones([2, 2])
        sd = mixed.state_dict()
        self.assertEqual(set(sd["master_weights"]), {pa[0].name, pb[0].name})

    def test_set_state_dict_longest_prefix_routing(self):
        # "<base>" is a prefix of "<base>_extra": the longest owning name must
        # win so each accumulator lands on its true owner.
        base = f"linroute{next(_UID)}"
        pa = [_named_param(base)]
        pb = [_named_param(base + "_extra")]
        subA = _SGDInner(parameters=pa, learning_rate=0.1)
        subB = _SGDInner(parameters=pb, learning_rate=0.1)
        mixed = MixedInnerOptimizer([subA, subB], learning_rate=0.1)
        mixed._apply_optimize(
            None,
            None,
            [
                (pa[0], paddle.ones_like(pa[0])),
                (pb[0], 3.0 * paddle.ones_like(pb[0])),
            ],
        )
        sd0 = mixed.state_dict()
        restore = {
            k: (v.clone() if hasattr(v, "clone") else v) for k, v in sd0.items()
        }
        mixed._apply_optimize(
            None,
            None,
            [
                (pa[0], paddle.ones_like(pa[0])),
                (pb[0], 3.0 * paddle.ones_like(pb[0])),
            ],
        )
        mixed.set_state_dict(restore)
        self.assertAlmostEqual(
            float(subA._accumulators["moment1"][base].mean()), 1.0, places=5
        )
        self.assertAlmostEqual(
            float(subB._accumulators["moment1"][base + "_extra"].mean()),
            3.0,
            places=5,
        )


class TestMixedFusionStorage(unittest.TestCase):
    def test_no_fusion_by_default(self):
        mixed, *_ = _make_mixed()
        mixed._maybe_refuse()  # not enabled -> no-op
        self.assertIsNone(mixed.fusion_storage)

    def test_use_fusion_storage_builds_single_buffer(self):
        mixed, subA, subB, pa, pb = _make_mixed()
        mixed.use_fusion_storage()
        mixed._apply_optimize(
            None, None, [(p, paddle.ones_like(p)) for p in pa + pb]
        )
        self.assertIsNotNone(mixed.fusion_storage)
        v = mixed._fuse_buffer_version
        mixed._maybe_refuse()  # tensors still shared -> no rebuild
        self.assertEqual(mixed._fuse_buffer_version, v)

    def test_maybe_refuse_static_noop(self):
        mixed, *_ = _make_mixed()
        mixed._use_fusion_storage = True
        paddle.enable_static()
        try:
            mixed._maybe_refuse()  # not dygraph -> returns without building
        finally:
            paddle.disable_static()
        self.assertIsNone(mixed.fusion_storage)

    def test_fusion_rebuild_on_realloc(self):
        mixed, subA, subB, pa, pb = _make_mixed()
        mixed.use_fusion_storage()
        mixed._apply_optimize(
            None, None, [(p, paddle.ones_like(p)) for p in pa + pb]
        )
        v = mixed._fuse_buffer_version
        # Reallocate an accumulator so it no longer aliases the fused buffer;
        # _maybe_refuse must notice and rebuild (version bumps).
        name = pa[0].name
        subA._accumulators["moment1"][name] = paddle.ones_like(
            subA._accumulators["moment1"][name]
        )
        mixed._maybe_refuse()
        self.assertGreater(mixed._fuse_buffer_version, v)


if __name__ == "__main__":
    unittest.main()
