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

"""Single-process unit tests for ``paddle.optimizer.inner_optimizer``.

Covers the generic ``ParamInfo`` metadata carrier and the ``InnerOptimizer``
base class (argument validation, the class-level ``keep_whole`` declaration,
the accumulator / master-weight helpers, the ``_apply_optimize`` template and
``step``). The distributed sharding behaviour that consumes
``keep_whole_of()`` is covered by
test/collective/fleet/hybrid_parallel_sharding_mixed_model.py.
"""

import unittest

import paddle
import paddle.optimizer.inner_optimizer as inner_mod
from paddle.optimizer.inner_optimizer import InnerOptimizer, ParamInfo


class _SGDInner(InnerOptimizer):
    """Minimal concrete InnerOptimizer: momentum SGD with one buffer."""

    _moment_acc_str = "moment1"
    # Element-wise update -> the shell may shard it (V2).
    keep_whole = False

    def __init__(self, parameters, momentum=0.9, **kwargs):
        # weight_decay has no default in the base class; these tests do not
        # exercise decay, so pick 0 unless a test overrides it.
        kwargs.setdefault("weight_decay", 0.0)
        super().__init__(parameters=parameters, **kwargs)
        self._momentum = momentum

    def _ensure_accumulators(self, param):
        if (
            self._moment_acc_str in self._accumulators
            and param.name in self._accumulators[self._moment_acc_str]
        ):
            return
        self._ensure_master_weight(param)
        self._add_accumulator(
            self._moment_acc_str,
            param,
            dtype=paddle.float32,
            fill_value=0.0,
            shape=param.shape,
        )

    def _update_parameters(self, params_grads, lr):
        with paddle.no_grad():
            for param, grad in params_grads:
                m = self._get_accumulator(self._moment_acc_str, param)
                paddle.assign(self._momentum * m + grad.astype(m.dtype), m)
                eff = self._effective_lr(param, lr)
                master = self._master_weights.get(param.name)
                if master is not None:
                    master.subtract_(eff * m)
                    paddle.assign(master.astype(param.dtype), param)
                else:
                    param.subtract_((eff * m).astype(param.dtype))

    def optimizer_state_layout(self):
        # Matches _ensure_accumulators above: one per-element buffer, no scalar.
        return {
            "vector": [f"{self._moment_acc_str}_0"],
            "scalar": [],
            "master": "fp32_master_0",
        }


class _BareInner(InnerOptimizer):
    """Re-exposes the base hooks so their raises stay reachable.

    The three hooks are ``@abstractmethod``; delegating to the base
    implementation is a concrete override (so the class is instantiable) that
    still raises when called.
    """

    keep_whole = False

    def _ensure_accumulators(self, param):
        return InnerOptimizer._ensure_accumulators(self, param)

    def _update_parameters(self, params_grads, lr):
        return InnerOptimizer._update_parameters(self, params_grads, lr)

    def optimizer_state_layout(self):
        return InnerOptimizer.optimizer_state_layout(self)


class _NoLayoutInner(InnerOptimizer):
    """Implements the two update hooks but forgets the state layout.

    ``optimizer_state_layout`` stays abstract -> ABCMeta refuses to
    instantiate this class, which is the whole point: a missing declaration
    must fail at construction, not at the first checkpoint save.
    """

    keep_whole = False
    _moment_acc_str = "moment1"

    _ensure_accumulators = _SGDInner._ensure_accumulators
    _update_parameters = _SGDInner._update_parameters


class _NoUpdateInner(_SGDInner):
    """Forgets ``_update_parameters`` -> also not instantiable."""

    _update_parameters = InnerOptimizer._update_parameters


class _EitherInner(_SGDInner):
    """Element-wise math that works on a shard AND on a whole tensor."""

    keep_whole = None


class _WholeOnlyInner(_SGDInner):
    """Whole-matrix math (e.g. Newton-Schulz): only V1 is correct."""

    keep_whole = True


class _UndeclaredInner(_SGDInner):
    """Forgot to declare ``keep_whole`` -> construction raises.

    ``None`` is a *valid* declaration ("either mode works"), so the undeclared
    case has to be spelled with the sentinel the base class defaults to.
    """

    keep_whole = inner_mod.KEEP_WHOLE_UNDECLARED


class _BadDeclInner(_SGDInner):
    """Declared something that is neither a bool nor ``None``."""

    keep_whole = "whole"


def _param(shape=(4, 4), dtype="float32", value=1.0):
    return paddle.create_parameter(
        shape=list(shape),
        dtype=dtype,
        default_initializer=paddle.nn.initializer.Constant(value),
    )


class TestParamInfo(unittest.TestCase):
    def test_carries_no_field(self):
        # V1/V2 is a property of the optimizer class, not of a parameter, so
        # ParamInfo has no field of its own ...
        self.assertFalse(getattr(ParamInfo, "__dataclass_fields__", {}))
        # ... but it is a plain dataclass, so optimizer-specific metadata can
        # be attached by subclassing or by setting attributes.
        pi = ParamInfo()
        pi.custom = 7
        self.assertEqual(pi.custom, 7)


class TestKeepWhole(unittest.TestCase):
    def test_declared_value_is_returned(self):
        self.assertIs(_SGDInner.declared_keep_whole(), False)
        self.assertIs(_WholeOnlyInner.declared_keep_whole(), True)
        self.assertIsNone(_EitherInner.declared_keep_whole())
        opt = _SGDInner(parameters=[_param()])
        # Same answer for every parameter the optimizer owns.
        self.assertIs(opt.keep_whole_of(opt._parameter_list[0].name), False)
        self.assertIs(opt.keep_whole_of("no-such-param"), False)

    def test_undeclared_raises(self):
        with self.assertRaisesRegex(TypeError, "keep_whole"):
            _UndeclaredInner.declared_keep_whole()
        # The resolution happens in __init__, so forgetting to declare fails
        # at construction rather than much later during sharding.
        with self.assertRaisesRegex(TypeError, "keep_whole"):
            _UndeclaredInner(parameters=[_param()])

    def test_bad_declaration_raises(self):
        with self.assertRaisesRegex(TypeError, "keep_whole"):
            _BadDeclInner.declared_keep_whole()
        with self.assertRaisesRegex(TypeError, "keep_whole"):
            _BadDeclInner(parameters=[_param()])

    def test_either_defaults_to_v2(self):
        # ``None`` means "the config picks"; with no request the default is V2,
        # which spreads each parameter across the sharding ranks.
        self.assertIs(_EitherInner.resolve_keep_whole(None), False)
        opt = _EitherInner(parameters=[_param()])
        self.assertIs(opt.keep_whole_of(opt._parameter_list[0].name), False)

    def test_either_accepts_both_requests(self):
        self.assertIs(_EitherInner.resolve_keep_whole(True), True)
        self.assertIs(_EitherInner.resolve_keep_whole(False), False)
        opt = _EitherInner(parameters=[_param()])
        self.assertIs(opt.set_keep_whole(True), True)
        self.assertIs(opt.keep_whole_of("whatever"), True)
        self.assertIs(opt.set_keep_whole(False), False)
        self.assertIs(opt.keep_whole_of("whatever"), False)
        # Back to "no preference" -> the class default again.
        self.assertIs(opt.set_keep_whole(None), False)

    def test_request_must_match_a_hard_declaration(self):
        # A class that only supports one mode may be asked for that mode (or
        # for nothing), but asking for the other one is a configuration error.
        self.assertIs(_WholeOnlyInner.resolve_keep_whole(None), True)
        self.assertIs(_WholeOnlyInner.resolve_keep_whole(True), True)
        with self.assertRaisesRegex(ValueError, "only supports keep_whole"):
            _WholeOnlyInner.resolve_keep_whole(False)
        with self.assertRaisesRegex(ValueError, "only supports keep_whole"):
            _SGDInner.resolve_keep_whole(True)
        opt = _WholeOnlyInner(parameters=[_param()])
        self.assertIs(opt.keep_whole_of("x"), True)
        with self.assertRaisesRegex(ValueError, "only supports keep_whole"):
            opt.set_keep_whole(False)
        # The failed request must not have changed the instance.
        self.assertIs(opt.keep_whole_of("x"), True)

    def test_non_bool_request_rejected(self):
        for bad in ("true", 1, 0.0):
            with self.assertRaises(TypeError):
                _EitherInner.resolve_keep_whole(bad)


class TestInnerOptimizerInit(unittest.TestCase):
    def test_parameters_none(self):
        with self.assertRaises(ValueError):
            _SGDInner(parameters=None)

    def test_parameters_not_list(self):
        with self.assertRaises(TypeError):
            _SGDInner(parameters=_param())

    def test_parameters_param_groups_rejected(self):
        with self.assertRaises(TypeError):
            _SGDInner(parameters=[{"params": [_param()]}])

    def test_bad_grad_clip(self):
        with self.assertRaises(TypeError):
            _SGDInner(parameters=[_param()], grad_clip=object())

    def test_weight_decay_is_required(self):
        # No safe default: 0.0 silently disables regularisation and a nonzero
        # value silently changes the objective.
        with self.assertRaisesRegex(ValueError, "weight_decay"):
            _SGDInner(parameters=[_param()], weight_decay=None)

    def test_defaults(self):
        opt = _SGDInner(parameters=[_param()])
        self.assertEqual(opt._master_weights, {})
        self.assertEqual(opt._param_info_map, {})
        self.assertIsNone(opt._lr_ratio)


class TestInnerOptimizerUpdate(unittest.TestCase):
    def test_step_updates_and_creates_accumulator(self):
        p = _param()
        before = p.numpy().copy()
        opt = _SGDInner(parameters=[p], learning_rate=0.1)
        p.clear_gradient()
        (p * p).sum().backward()
        opt.step()
        self.assertIn("moment1", opt._accumulators)
        self.assertIn(p.name, opt._accumulators["moment1"])
        self.assertFalse((p.numpy() == before).all())

    def test_grad_none_is_skipped(self):
        p = _param()
        opt = _SGDInner(parameters=[p], learning_rate=0.1)
        # grad None -> no accumulator, no update
        opt._apply_optimize(None, None, [(p, None)])
        self.assertEqual(opt._accumulators, {})

    def test_effective_lr_with_ratio(self):
        p = _param()
        opt = _SGDInner(parameters=[p], lr_ratio=lambda _p: 0.5)
        self.assertAlmostEqual(opt._effective_lr(p, 0.2), 0.1)
        opt2 = _SGDInner(parameters=[p])
        self.assertAlmostEqual(opt2._effective_lr(p, 0.2), 0.2)

    def test_lr_scheduler_resolved(self):
        p = _param()
        sched = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[10], values=[0.1, 0.01]
        )
        opt = _SGDInner(parameters=[p], learning_rate=sched)
        p.clear_gradient()
        (p * p).sum().backward()
        opt.step()  # exercises the LRScheduler() resolution branch
        self.assertIn(p.name, opt._accumulators["moment1"])

    def test_master_weight_created_for_bf16(self):
        p = _param(dtype="bfloat16")
        opt = _SGDInner(parameters=[p], multi_precision=True, learning_rate=0.1)
        p.clear_gradient()
        (p.astype("float32") * p.astype("float32")).sum().backward()
        opt.step()
        self.assertIn(p.name, opt._master_weights)
        self.assertEqual(opt._master_weights[p.name].dtype, paddle.float32)

    def test_grad_clip_applied(self):
        p = _param()
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1e-6)
        opt = _SGDInner(parameters=[p], learning_rate=0.1, grad_clip=clip)
        p.clear_gradient()
        (p * p).sum().backward()
        opt.step()  # goes through the grad_clip branch of _apply_optimize
        self.assertIn(p.name, opt._accumulators["moment1"])

    def test_shard_bypass(self):
        p = _param()
        before = p.numpy().copy()
        opt = _SGDInner(parameters=[p], learning_rate=0.1)
        saved = inner_mod.g_shard_bypass_dygraph_optimizer
        inner_mod.g_shard_bypass_dygraph_optimizer = 1
        try:
            opt._apply_optimize(None, None, [(p, paddle.ones_like(p))])
        finally:
            inner_mod.g_shard_bypass_dygraph_optimizer = saved
        # bypass -> nothing happened
        self.assertEqual(opt._accumulators, {})
        self.assertTrue((p.numpy() == before).all())


class TestInnerOptimizerMisc(unittest.TestCase):
    def test_declared_state_layout(self):
        opt = _SGDInner(parameters=[_param()])
        self.assertEqual(
            opt.optimizer_state_layout(),
            {
                "vector": ["moment1_0"],
                "scalar": [],
                "master": "fp32_master_0",
            },
        )

    def test_required_hooks_are_abstract(self):
        # All three are part of the minimal path of a custom optimizer: a
        # subclass that misses any one of them is not instantiable, so the
        # mistake surfaces at construction instead of at the first save.
        self.assertEqual(
            InnerOptimizer.__abstractmethods__,
            frozenset(
                {
                    "_ensure_accumulators",
                    "_update_parameters",
                    "optimizer_state_layout",
                }
            ),
        )
        for cls in (InnerOptimizer, _NoLayoutInner, _NoUpdateInner):
            with self.assertRaisesRegex(TypeError, "abstract"):
                cls(parameters=[_param()], weight_decay=0.0)

    def test_create_accumulators_targets(self):
        p = _param()
        opt = _SGDInner(parameters=[p])
        opt._create_accumulators(
            None, [p]
        )  # flex-load target creation, no step
        self.assertIn(p.name, opt._accumulators["moment1"])

    def test_abstract_hooks_raise(self):
        opt = _BareInner(parameters=[_param()], weight_decay=0.0)
        with self.assertRaises(NotImplementedError):
            opt._ensure_accumulators(opt._parameter_list[0])
        with self.assertRaises(NotImplementedError):
            opt._update_parameters([], 0.1)
        with self.assertRaises(NotImplementedError):
            opt.optimizer_state_layout()

    def test_static_mode_rejected(self):
        p = _param()
        opt = _SGDInner(parameters=[p])
        paddle.enable_static()
        try:
            with self.assertRaises(NotImplementedError):
                opt._create_accumulators(None, [p])
            with self.assertRaises(NotImplementedError):
                opt._apply_optimize(None, None, [])
        finally:
            paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
