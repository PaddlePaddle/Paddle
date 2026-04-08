# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle
from paddle.amp import AmpScaler, GradScaler


class TestGradScalerParamAlias(unittest.TestCase):
    """Test ParamAliasDecorator: PyTorch aliases, Paddle names, mixed,
    positional args, single alias, conflict detection, and non-aliased params."""

    def _assert_scaler(self, scaler, **expected):
        """Helper to check multiple scaler attributes at once."""
        attr_map = {
            'enable': '_enable',
            'init_loss_scaling': '_init_loss_scaling',
            'incr_ratio': '_incr_ratio',
            'decr_ratio': '_decr_ratio',
            'incr_every_n_steps': '_incr_every_n_steps',
            'decr_every_n_nan_or_inf': '_decr_every_n_nan_or_inf',
            'use_dynamic_loss_scaling': '_use_dynamic_loss_scaling',
        }
        for key, val in expected.items():
            self.assertEqual(getattr(scaler, attr_map[key]), val, msg=key)

    def test_default_values(self):
        self._assert_scaler(
            GradScaler(),
            enable=True,
            init_loss_scaling=2.0**16,
            incr_ratio=2.0,
            decr_ratio=0.5,
            incr_every_n_steps=2000,
            decr_every_n_nan_or_inf=1,
            use_dynamic_loss_scaling=True,
        )

    def test_pytorch_style_kwargs(self):
        self._assert_scaler(
            GradScaler(
                enabled=True,
                init_scale=1024.0,
                growth_factor=3.0,
                backoff_factor=0.25,
                growth_interval=500,
            ),
            enable=True,
            init_loss_scaling=1024.0,
            incr_ratio=3.0,
            decr_ratio=0.25,
            incr_every_n_steps=500,
        )

    def test_mixed_kwargs(self):
        self._assert_scaler(
            GradScaler(
                enable=True,
                init_loss_scaling=1024.0,
                growth_factor=3.0,
                decr_ratio=0.25,
                growth_interval=500,
            ),
            init_loss_scaling=1024.0,
            incr_ratio=3.0,
            decr_ratio=0.25,
            incr_every_n_steps=500,
        )

    def test_single_alias_each(self):
        self.assertFalse(GradScaler(enabled=False)._enable)
        self.assertEqual(GradScaler(init_scale=512.0)._init_loss_scaling, 512.0)
        self.assertEqual(GradScaler(growth_factor=5.0)._incr_ratio, 5.0)
        self.assertEqual(GradScaler(backoff_factor=0.1)._decr_ratio, 0.1)
        self.assertEqual(
            GradScaler(growth_interval=100)._incr_every_n_steps, 100
        )

    def test_positional_args(self):
        self._assert_scaler(
            GradScaler(True, 1024.0, 3.0, 0.25, 500, 2, True),
            enable=True,
            init_loss_scaling=1024.0,
            incr_ratio=3.0,
            decr_ratio=0.25,
            incr_every_n_steps=500,
            decr_every_n_nan_or_inf=2,
        )

    def test_positional_with_alias_kwarg(self):
        self._assert_scaler(
            GradScaler(True, 1024.0, growth_factor=5.0),
            enable=True,
            init_loss_scaling=1024.0,
            incr_ratio=5.0,
        )

    def test_non_aliased_with_aliases(self):
        self._assert_scaler(
            GradScaler(
                enabled=True,
                init_scale=2048.0,
                decr_every_n_nan_or_inf=3,
                use_dynamic_loss_scaling=False,
            ),
            enable=True,
            init_loss_scaling=2048.0,
            decr_every_n_nan_or_inf=3,
            use_dynamic_loss_scaling=False,
        )

    def test_pytorch_vs_paddle_equivalence(self):
        pt = GradScaler(
            enabled=True,
            init_scale=1024.0,
            growth_factor=3.0,
            backoff_factor=0.25,
            growth_interval=500,
        )
        pd = GradScaler(
            enable=True,
            init_loss_scaling=1024.0,
            incr_ratio=3.0,
            decr_ratio=0.25,
            incr_every_n_steps=500,
        )
        for attr in (
            '_enable',
            '_init_loss_scaling',
            '_incr_ratio',
            '_decr_ratio',
            '_incr_every_n_steps',
        ):
            self.assertEqual(getattr(pt, attr), getattr(pd, attr), msg=attr)

    def test_conflict_raises_error(self):
        conflicts = [
            {"enable": True, "enabled": False},
            {"init_loss_scaling": 1024.0, "init_scale": 2048.0},
            {"incr_ratio": 2.0, "growth_factor": 3.0},
            {"decr_ratio": 0.5, "backoff_factor": 0.25},
            {"incr_every_n_steps": 1000, "growth_interval": 2000},
        ]
        for kwargs in conflicts:
            with self.assertRaises(ValueError, msg=str(kwargs)):
                GradScaler(**kwargs)

    def test_torch_positional_no_device(self):
        # PyTorch older API: init_scale first (float -> detected as torch positional)
        self._assert_scaler(
            GradScaler(1024.0, 3.0, 0.25, 500, True),
            enable=True,
            init_loss_scaling=1024.0,
            incr_ratio=3.0,
            decr_ratio=0.25,
            incr_every_n_steps=500,
        )

    def test_torch_positional_no_device_disabled(self):
        # PyTorch: GradScaler(init_scale, ..., enabled=False)
        scaler = GradScaler(1024.0, 2.0, 0.5, 2000, False)
        self.assertFalse(scaler._enable)
        self.assertEqual(scaler._init_loss_scaling, 1.0)

    def test_torch_positional_with_device(self):
        # PyTorch newer API: device string first
        self._assert_scaler(
            GradScaler('cuda', 1024.0, 3.0, 0.25, 500, True),
            enable=True,
            init_loss_scaling=1024.0,
            incr_ratio=3.0,
            decr_ratio=0.25,
            incr_every_n_steps=500,
        )

    def test_torch_device_kwarg_dropped(self):
        # device kwarg is silently ignored (no Paddle equivalent)
        self._assert_scaler(
            GradScaler(device='cuda', init_scale=2048.0, growth_factor=3.0),
            init_loss_scaling=2048.0,
            incr_ratio=3.0,
        )

    def test_torch_device_string_with_kwargs(self):
        # device as positional string combined with torch keyword aliases
        self._assert_scaler(
            GradScaler('cuda', init_scale=512.0, enabled=True),
            enable=True,
            init_loss_scaling=512.0,
        )

    def test_torch_positional_partial(self):
        # Only init_scale positionally, rest default
        self._assert_scaler(
            GradScaler(4096.0),
            init_loss_scaling=4096.0,
            incr_ratio=2.0,
            decr_ratio=0.5,
            incr_every_n_steps=2000,
        )

    def test_torch_device_string_only(self):
        # GradScaler('cuda') — device only, all remaining params default
        self._assert_scaler(
            GradScaler('cuda'),
            enable=True,
            init_loss_scaling=2.0**16,
            incr_ratio=2.0,
            decr_ratio=0.5,
            incr_every_n_steps=2000,
        )

    def test_torch_device_string_partial_positional(self):
        # GradScaler('cuda', init_scale) — device + only first positional param
        self._assert_scaler(
            GradScaler('cuda', 1024.0),
            enable=True,
            init_loss_scaling=1024.0,
            incr_ratio=2.0,
            decr_ratio=0.5,
            incr_every_n_steps=2000,
        )

    def test_torch_positional_with_kwarg_disabled(self):
        # GradScaler(init_scale, enabled=False) — positional float + torch kwarg
        scaler = GradScaler(1024.0, enabled=False)
        self.assertFalse(scaler._enable)
        self.assertEqual(scaler._init_loss_scaling, 1.0)

    def test_disabled_scaler(self):
        scaler = GradScaler(enabled=False)
        self.assertFalse(scaler._enable)
        self.assertEqual(scaler._init_loss_scaling, 1.0)


class TestGradScalerPytorchCompatMethods(unittest.TestCase):
    """Test PyTorch-compatible getter/setter methods."""

    def test_is_enabled(self):
        self.assertTrue(GradScaler(enabled=True).is_enabled())
        self.assertFalse(GradScaler(enabled=False).is_enabled())

    def test_get_scale(self):
        self.assertEqual(GradScaler(init_scale=2048.0).get_scale(), 2048.0)
        self.assertEqual(GradScaler(enabled=False).get_scale(), 0.0)

    def test_growth_factor_get_set(self):
        s = GradScaler(growth_factor=3.0)
        self.assertEqual(s.get_growth_factor(), 3.0)
        s.set_growth_factor(5.0)
        self.assertEqual(s.get_growth_factor(), 5.0)
        self.assertEqual(s.get_incr_ratio(), 5.0)

    def test_backoff_factor_get_set(self):
        s = GradScaler(backoff_factor=0.25)
        self.assertEqual(s.get_backoff_factor(), 0.25)
        s.set_backoff_factor(0.1)
        self.assertEqual(s.get_backoff_factor(), 0.1)
        self.assertEqual(s.get_decr_ratio(), 0.1)

    def test_growth_interval_get_set(self):
        s = GradScaler(growth_interval=500)
        self.assertEqual(s.get_growth_interval(), 500)
        s.set_growth_interval(100)
        self.assertEqual(s.get_growth_interval(), 100)
        self.assertEqual(s.get_incr_every_n_steps(), 100)


class TestGradScalerCallPathsAndInheritance(unittest.TestCase):
    """Test all public call paths and AmpScaler vs GradScaler defaults."""

    def test_all_paths_same_class(self):
        self.assertIs(paddle.device.amp.GradScaler, GradScaler)
        self.assertIs(paddle.cuda.amp.GradScaler, GradScaler)

    def test_gradscaler_is_subclass_of_ampscaler(self):
        self.assertTrue(issubclass(GradScaler, AmpScaler))

    def test_alias_via_device_and_cuda(self):
        s1 = paddle.device.amp.GradScaler(init_scale=512.0, growth_factor=4.0)
        self.assertEqual(s1._init_loss_scaling, 512.0)
        self.assertEqual(s1._incr_ratio, 4.0)

        s2 = paddle.cuda.amp.GradScaler(backoff_factor=0.3, growth_interval=800)
        self.assertEqual(s2._decr_ratio, 0.3)
        self.assertEqual(s2._incr_every_n_steps, 800)

    def test_defaults_differ_from_ampscaler(self):
        a, g = AmpScaler(), GradScaler()
        # Intentionally different: GradScaler aligns with PyTorch
        self.assertEqual(a._init_loss_scaling, 2.0**15)
        self.assertEqual(g._init_loss_scaling, 2.0**16)
        self.assertEqual(a._incr_every_n_steps, 1000)
        self.assertEqual(g._incr_every_n_steps, 2000)
        # Shared defaults stay the same
        self.assertEqual(a._incr_ratio, g._incr_ratio)
        self.assertEqual(a._decr_ratio, g._decr_ratio)
        self.assertEqual(a._decr_every_n_nan_or_inf, g._decr_every_n_nan_or_inf)


if __name__ == "__main__":
    unittest.main()
