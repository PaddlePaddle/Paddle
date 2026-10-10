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
base class (argument validation, the accumulator / master-weight helpers, the
``_apply_optimize`` template and ``step``). The distributed sharding behaviour
that consumes ``ParamInfo.keep_whole`` is covered by the collective tests under
test/collective/fleet/.
"""

import unittest

import paddle
import paddle.optimizer.inner_optimizer as inner_mod
from paddle.optimizer.inner_optimizer import InnerOptimizer, ParamInfo


class _SGDInner(InnerOptimizer):
    """Minimal concrete InnerOptimizer: momentum SGD with one buffer."""

    _moment_acc_str = "moment1"

    def __init__(self, parameters, momentum=0.9, **kwargs):
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


class _BareInner(InnerOptimizer):
    """Does not override the abstract hooks -> exercises their raises."""


def _param(shape=(4, 4), dtype="float32", value=1.0):
    return paddle.create_parameter(
        shape=list(shape),
        dtype=dtype,
        default_initializer=paddle.nn.initializer.Constant(value),
    )


class TestParamInfo(unittest.TestCase):
    def test_default_and_explicit(self):
        self.assertIsNone(ParamInfo().keep_whole)
        self.assertTrue(ParamInfo(keep_whole=True).keep_whole)
        # A plain dataclass -> callers may attach optimizer-specific fields.
        pi = ParamInfo()
        pi.custom = 7
        self.assertEqual(pi.custom, 7)


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
    def test_default_state_layout(self):
        opt = _SGDInner(parameters=[_param()])
        self.assertEqual(
            opt.optimizer_state_layout(),
            {
                "vector": ["moment1_0", "moment2_0"],
                "scalar": ["beta1_pow_acc_0", "beta2_pow_acc_0"],
                "master": "fp32_master_0",
            },
        )

    def test_create_accumulators_targets(self):
        p = _param()
        opt = _SGDInner(parameters=[p])
        opt._create_accumulators(
            None, [p]
        )  # flex-load target creation, no step
        self.assertIn(p.name, opt._accumulators["moment1"])

    def test_abstract_hooks_raise(self):
        opt = _BareInner(parameters=[_param()])
        with self.assertRaises(NotImplementedError):
            opt._ensure_accumulators(opt._parameter_list[0])
        with self.assertRaises(NotImplementedError):
            opt._update_parameters([], 0.1)

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
