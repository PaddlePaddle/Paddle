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
view and ``MixedInnerOptimizer`` (construction / routing / ``keep_whole``
delegation / state-dict aggregation and routing / fused-storage build /
state-layout aggregation). All single process, real paddle dygraph -- the
sharding shell that wraps this optimizer is covered by
test/collective/fleet/hybrid_parallel_sharding_mixed_model.py.
"""

import itertools
import unittest

import paddle
import paddle.optimizer.mixed_inner_optimizer as mixed_mod
from paddle.base import core
from paddle.distributed.fsdp.fully_shard import _mark_params_for_optimizer
from paddle.optimizer.inner_optimizer import (
    KEEP_WHOLE_UNDECLARED,
    InnerOptimizer,
)
from paddle.optimizer.mixed_inner_optimizer import (
    INNER_OPTIMIZER_REGISTRY,
    MixedInnerOptimizer,
    _GatheredMasterWeights,
    get_inner_optimizer,
    register_inner_optimizer,
)

_UID = itertools.count()

# FusionStorage only knows the memory alignment of gpu / npu / xpu, so the two
# cases that really build the fused buffer need an accelerator; the rest of the
# file is device agnostic.
_skip_without_accelerator = unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "FusionStorage requires an accelerator device",
)


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

    # Element-wise update -> shardable (V2).
    keep_whole = False

    def __init__(self, parameters, **kwargs):
        # weight_decay has no default in the base class; these tests do not
        # exercise decay, so pick 0 unless a test overrides it.
        kwargs.setdefault("weight_decay", 0.0)
        super().__init__(parameters=parameters, **kwargs)

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
    """moment1/moment2 + beta pow (an AdamW-shaped state layout)."""

    # Declared the other way round from _SGDInner so a mixed optimizer built
    # from the two really has to answer keep_whole per parameter.
    keep_whole = True

    def __init__(self, parameters, **kwargs):
        kwargs.setdefault("weight_decay", 0.0)
        super().__init__(parameters=parameters, **kwargs)

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

    def optimizer_state_layout(self):
        return {
            "vector": ["moment1_0", "moment2_0"],
            "scalar": ["beta1_pow_acc_0", "beta2_pow_acc_0"],
            "master": "fp32_master_0",
        }


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


class _EitherInner(_SGDInner):
    """Element-wise math declared as "either mode works" (config picks)."""

    keep_whole = None


class _UndeclaredInner(_SGDInner):
    """Forgot to declare ``keep_whole`` -> rejected at construction.

    ``None`` is a valid declaration ("either mode works"), so the undeclared
    case is spelled with the sentinel.
    """

    keep_whole = KEEP_WHOLE_UNDECLARED


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

    def test_sub_not_inner_optimizer(self):
        # Rejected even though it does expose a callable _apply_optimize:
        # everything downstream needs the full InnerOptimizer contract
        # (_param_info_map, optimizer_state_layout(), _master_weights, ...),
        # and a duck-typed sub would only surface much later, inside the
        # sharding shell, as a misleading "has no param_info".
        class _NotAnOpt:
            _parameter_list = []

            def _apply_optimize(self, loss, startup_program, params_grads):
                pass

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

    def test_sub_without_declared_keep_whole(self):
        # Fail fast when the sub is built, long before the sharding shell's
        # classification loop would ask it.
        with self.assertRaisesRegex(TypeError, "keep_whole"):
            _UndeclaredInner(parameters=[_param("kwU0")], learning_rate=0.1)

    def test_keep_whole_routed_per_param(self):
        pa = [_param("kwA0")]
        pb = [_param("kwB0")]
        subA = _SGDInner(parameters=pa, learning_rate=0.1)  # V2
        subB = _AdamInner(parameters=pb, learning_rate=0.1)  # V1
        mixed = MixedInnerOptimizer([subA, subB], learning_rate=0.1)
        # The chain is heterogeneous: it declares ``None`` only to satisfy the
        # base class and answers per parameter instead.
        self.assertIsNone(MixedInnerOptimizer.keep_whole)
        self.assertIs(mixed.keep_whole_of(pa[0].name), False)
        self.assertIs(mixed.keep_whole_of(pb[0].name), True)
        with self.assertRaises(KeyError):
            mixed.keep_whole_of("nope")

    def test_keep_whole_narrowed_per_group(self):
        # A sub that supports both modes may be pinned by the config; the
        # chain must report the narrowed value, not the class default.
        pa = [_param("kwN0")]
        pb = [_param("kwN1")]
        subA = _EitherInner(parameters=pa, learning_rate=0.1)
        subB = _EitherInner(parameters=pb, learning_rate=0.1)
        self.assertIs(subA.set_keep_whole(True), True)
        mixed = MixedInnerOptimizer([subA, subB], learning_rate=0.1)
        self.assertIs(mixed.keep_whole_of(pa[0].name), True)
        self.assertIs(mixed.keep_whole_of(pb[0].name), False)  # default V2
        # Narrowing must be done on the sub that owns the group: the chain
        # itself has no single keep_whole and says so instead of silently
        # writing an attribute nobody reads.
        with self.assertRaisesRegex(TypeError, "no single keep_whole"):
            mixed.set_keep_whole(True)

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

    def test_set_master_weight_without_store(self):
        mixed, subA, _subB, pa, _pb = _make_mixed()
        subA._master_weights = None
        with self.assertRaises(AttributeError):
            mixed._set_master_weight(pa[0].name, paddle.ones([1]))

    def test_create_master_weight_not_implemented(self):
        mixed, subA, _subB, pa, _pb = _make_mixed()
        subA._create_master_weight = None
        with self.assertRaises(AttributeError):
            mixed._create_master_weight(pa[0])


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

    def test_ensure_accumulators_routes(self):
        # The aggregate owns no state: the per-parameter hook hands the
        # parameter to the sub-optimizer that owns it.
        mixed, subA, subB, pa, pb = _make_mixed()
        mixed._ensure_accumulators(pb[0])
        self.assertIn(pb[0].name, subB._accumulators["moment1"])
        self.assertNotIn("moment1", subA._accumulators)

    def test_ensure_accumulators_unrouted_raises(self):
        mixed, _subA, _subB, _pa, _pb = _make_mixed()
        with self.assertRaises(KeyError):
            mixed._ensure_accumulators(_param("eaX0"))

    def test_update_parameters_is_never_reached(self):
        # _apply_optimize clips once and then dispatches, so the aggregate's
        # own update math is unreachable; getting there means the dispatch
        # was bypassed.
        mixed, _subA, _subB, _pa, _pb = _make_mixed()
        with self.assertRaises(AssertionError):
            mixed._update_parameters([], 0.1)


class TestMixedStateLayout(unittest.TestCase):
    def test_layout_union(self):
        pa = [_param("laA0")]
        pb = [_param("laB0")]
        subA = _SGDInner(parameters=pa, learning_rate=0.1)  # moment1 only
        subB = _AdamInner(parameters=pb, learning_rate=0.1)  # AdamW-shaped
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

    def test_state_dict_master_weight_not_owned(self):
        # A master weight for a parameter the sub does not own is ignored.
        mixed, subA, _subB, _pa, _pb = _make_mixed()
        subA._master_weights["not_owned_by_any_sub"] = paddle.ones([1])
        self.assertNotIn("master_weights", mixed.state_dict())

    def test_state_dict_master_weight_in_two_subs(self):
        # The same parameter owned (and mastered) by two subs is ambiguous.
        mixed, subA, subB, pa, _pb = _make_mixed()
        subB._parameter_list.append(pa[0])
        subA._master_weights[pa[0].name] = paddle.ones([2, 2])
        subB._master_weights[pa[0].name] = paddle.ones([2, 2])
        with self.assertRaisesRegex(ValueError, "more than one"):
            mixed.state_dict()

    def test_state_dict_keeps_routed_fused_buffer_master(self):
        # FSDP steps fused buffers (fuse_params_*) that no sub lists as a
        # parameter; their masters are kept once the buffer is routed.
        mixed, subA, subB, _pa, _pb = _make_mixed()
        fused_a, fused_b = (f"fuse_params_{next(_UID)}" for _ in range(2))
        mixed._route[fused_a] = subA
        mixed._route[fused_b] = subB
        subA._master_weights[fused_a] = paddle.ones([4])
        subB._master_weights[fused_b] = paddle.full([4], 2.0)
        # Routed to subA: subB's copy is not its own and is dropped.
        subB._master_weights[fused_a] = paddle.zeros([4])
        masters = mixed.state_dict()["master_weights"]
        self.assertEqual(set(masters), {fused_a, fused_b})
        self.assertEqual(float(masters[fused_a].mean()), 1.0)
        self.assertEqual(float(masters[fused_b].mean()), 2.0)

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

    @_skip_without_accelerator
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

    @_skip_without_accelerator
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


def _layer_of(params):
    layer = paddle.nn.Layer()
    for i, p in enumerate(params):
        layer.add_parameter(f"p{i}", p)
    return layer


class TestMarkParamsForFSDP(unittest.TestCase):
    """fully_shard(optimizer=mixed) marks each param by its owning sub."""

    def _mixed(self, top_decay_fun=False):
        """subA exempts pa[1]; with ``top_decay_fun`` the top level exempts
        pb[0] (and claims pa[1] decays, which subA's callback overrides)."""
        pa = [_param("markA0"), _param("markA1")]
        pb = [_param("markB0"), _param("markB1")]
        exempt_a, exempt_b = pa[1].name, pb[0].name
        subA = _SGDInner(
            parameters=pa,
            learning_rate=0.1,
            weight_decay=0.1,
            apply_decay_param_fun=lambda name: name != exempt_a,
        )
        subB = _AdamInner(parameters=pb, learning_rate=0.1)
        mixed = MixedInnerOptimizer(
            [subA, subB],
            learning_rate=0.1,
            apply_decay_param_fun=(lambda name: name != exempt_b)
            if top_decay_fun
            else None,
        )
        return mixed, pa, pb

    def test_routing_and_decay(self):
        mixed, pa, pb = self._mixed(top_decay_fun=True)
        _mark_params_for_optimizer(_layer_of(pa + pb), mixed)
        self.assertEqual(
            [p.fsdp_optimizer_group for p in pa + pb], [0, 0, 1, 1]
        )
        self.assertEqual(
            [p.use_muon for p in pa + pb], [False, False, True, True]
        )
        self.assertEqual(
            [p.no_weight_decay for p in pa + pb], [False, True, True, False]
        )
        self.assertFalse(any(p.use_hyperball_sharded for p in pa + pb))

    def test_no_decay_fun_anywhere(self):
        mixed, pa, pb = self._mixed()
        _mark_params_for_optimizer(_layer_of(pa + pb), mixed)
        # subB and the top level have no callback: subB's params decay.
        self.assertEqual(
            [p.no_weight_decay for p in pa + pb], [False, True, False, False]
        )

    def test_frozen_unrouted_param(self):
        mixed, pa, pb = self._mixed()
        frozen = _param("markFrozen")
        frozen.stop_gradient = True
        _mark_params_for_optimizer(_layer_of([*pa, *pb, frozen]), mixed)
        self.assertIsNone(frozen.fsdp_optimizer_group)
        self.assertFalse(frozen.use_muon)

    def test_trainable_unrouted_param_raises(self):
        mixed, pa, pb = self._mixed()
        stray = _param("markStray")
        with self.assertRaisesRegex(ValueError, "misses trainable params"):
            _mark_params_for_optimizer(_layer_of([*pa, *pb, stray]), mixed)


if __name__ == "__main__":
    unittest.main()
