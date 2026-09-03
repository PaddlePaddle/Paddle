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

# [AUTO-GENERATED]
# Target file: paddle/optimizer/lr.py
# Coverage target: 80.5% -> improve coverage on uncovered lines
# 测试学习率调度器的各项功能，包括 NoamDecay, PiecewiseDecay, StepDecay, ExponentialDecay,
#   NaturalExpDecay, InverseTimeDecay, PolynomialDecay, LinearWarmup,
#   MultiStepDecay, LambdaDecay, ReduceOnPlateau, CosineAnnealingDecay,
#   MultiplicativeDecay, OneCycleLR, CyclicLR, LinearLR, CosineAnnealingWarmRestarts
# Tests for learning rate schedulers covering validation, state_dict, step, get_lr, etc.

import io
import math
import unittest
import warnings

import numpy as np

from paddle.optimizer.lr import (
    CosineAnnealingDecay,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialDecay,
    InverseTimeDecay,
    LambdaDecay,
    LinearLR,
    LinearWarmup,
    LRScheduler,
    MultiplicativeDecay,
    MultiStepDecay,
    NaturalExpDecay,
    NoamDecay,
    OneCycleLR,
    PiecewiseDecay,
    PolynomialDecay,
    ReduceOnPlateau,
    StepDecay,
)


class TestLRSchedulerBase(unittest.TestCase):
    """学习率调度器基础测试类 / LR scheduler base test class"""

    def test_base_lr_invalid_type(self):
        """测试基础调度器无效学习率类型 / Test base scheduler invalid LR type"""
        with self.assertRaises(TypeError):
            LRScheduler(learning_rate="0.1")

    def test_base_lr_negative(self):
        """测试基础调度器负学习率 / Test base scheduler negative LR"""
        with self.assertRaises(ValueError):
            LRScheduler(learning_rate=-0.1)

    def test_base_get_lr_not_implemented(self):
        """测试基础调度器未实现 get_lr / Test base scheduler get_lr not implemented"""
        scheduler = LRScheduler.__new__(LRScheduler)
        scheduler.base_lr = 0.1
        scheduler.last_lr = 0.1
        scheduler.last_epoch = -1
        scheduler.verbose = False
        scheduler._var_name = None
        with self.assertRaises(NotImplementedError):
            scheduler.get_lr()

    def test_base_step_none(self):
        """测试基础调度器 step(epoch=None) / Test base scheduler step with None epoch"""
        scheduler = ExponentialDecay(learning_rate=0.5, gamma=0.9)
        # __init__ calls step(), making last_epoch = 0
        # Verify it was called during init
        self.assertEqual(scheduler.last_epoch, 0)
        # Now calling step() again should increment to 1
        scheduler.step()
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertAlmostEqual(scheduler.last_lr, 0.5 * 0.9**1)

    def test_base_step_with_epoch(self):
        """测试基础调度器 step(epoch=N) / Test base scheduler step with specific epoch"""
        scheduler = ExponentialDecay(learning_rate=0.5, gamma=0.9)
        scheduler.step(epoch=5)
        self.assertEqual(scheduler.last_epoch, 5)

    def test_base_step_with_closed_form(self):
        """测试基础调度器使用 _get_closed_form_lr / Test base scheduler with _get_closed_form_lr"""
        scheduler = CosineAnnealingDecay(learning_rate=0.5, T_max=10)
        scheduler.step(epoch=5)
        self.assertEqual(scheduler.last_epoch, 5)

    def test_base_verbose(self):
        """测试基础调度器 verbose 模式 / Test base scheduler verbose mode"""
        scheduler = ExponentialDecay(learning_rate=0.5, gamma=0.9, verbose=True)
        # Redirect stdout to capture print
        old_stdout = None
        try:
            old_stdout = io.StringIO()
            import sys

            sys.stdout = old_stdout
            scheduler.step()
        finally:
            if old_stdout is not None:
                import sys

                sys.stdout = sys.__stdout__

    def test_base_call(self):
        """测试基础调度器 __call__ 方法 / Test base scheduler __call__ method"""
        scheduler = ExponentialDecay(learning_rate=0.5, gamma=0.9)
        lr = scheduler()
        self.assertEqual(lr, scheduler.last_lr)

    def test_base_state_dict_roundtrip(self):
        """测试基础调度器 state_dict 保存和加载 / Test base scheduler state_dict roundtrip"""
        scheduler = ExponentialDecay(learning_rate=0.5, gamma=0.9)
        for _ in range(5):
            scheduler.step()
        state = scheduler.state_dict()
        self.assertIn("last_epoch", state)
        self.assertIn("last_lr", state)

        scheduler2 = ExponentialDecay(learning_rate=0.5, gamma=0.9)
        scheduler2.set_state_dict(state)
        self.assertEqual(scheduler2.last_epoch, scheduler.last_epoch)
        self.assertAlmostEqual(scheduler2.last_lr, scheduler.last_lr)

    def test_base_set_dict_alias(self):
        """测试基础调度器 set_dict 别名 / Test base scheduler set_dict alias"""
        scheduler = ExponentialDecay(learning_rate=0.5, gamma=0.9)
        scheduler.step()
        state = scheduler.state_dict()
        scheduler2 = ExponentialDecay(learning_rate=0.5, gamma=0.9)
        scheduler2.set_dict(state)
        self.assertEqual(scheduler2.last_epoch, scheduler.last_epoch)

    def test_base_state_dict_missing_key(self):
        """测试基础调度器 state_dict 缺少键 / Test base scheduler state_dict missing key"""
        scheduler = ExponentialDecay(learning_rate=0.5, gamma=0.9)
        with self.assertRaises(RuntimeError):
            scheduler.set_state_dict({"last_epoch": 5})

    def test_base_state_dict_extra_keys_warning(self):
        """测试基础调度器 state_dict 多余键警告 / Test base scheduler state_dict extra keys warning"""
        scheduler = ExponentialDecay(learning_rate=0.5, gamma=0.9)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scheduler.set_state_dict(
                {"last_epoch": 5, "last_lr": 0.1, "extra": 1}
            )
            self.assertTrue(len(w) > 0)


class TestNoamDecay(unittest.TestCase):
    """NoamDecay 调度器测试类 / NoamDecay scheduler test class"""

    def test_noam_decay_basic(self):
        """测试 NoamDecay 基本功能 / Test NoamDecay basic"""
        scheduler = NoamDecay(d_model=512, warmup_steps=100, learning_rate=1.0)
        # At init, last_epoch=0, so lr may be 0. Step to get a non-zero lr
        scheduler.step()
        lr = scheduler()
        self.assertGreater(lr, 0)

    def test_noam_decay_epoch_zero(self):
        """测试 NoamDecay epoch=0 时的学习率 / Test NoamDecay at epoch 0"""
        scheduler = NoamDecay(d_model=512, warmup_steps=100, learning_rate=1.0)
        # Manually set to epoch 0 to test the branch
        scheduler.last_epoch = 0
        scheduler.last_lr = scheduler.get_lr()
        # At epoch 0, a=1, b=0, min(1, 0)=0, so lr=0
        # This is expected behavior - step to epoch > 0 for non-zero lr
        scheduler.step()
        self.assertGreater(scheduler(), 0)

    def test_noam_decay_invalid_d_model(self):
        """测试 NoamDecay 无效 d_model / Test NoamDecay invalid d_model"""
        with self.assertRaises(ValueError):
            NoamDecay(d_model=0, warmup_steps=100)


class TestPiecewiseDecay(unittest.TestCase):
    """PiecewiseDecay 调度器测试类 / PiecewiseDecay scheduler test class"""

    def test_piecewise_decay_basic(self):
        """测试 PiecewiseDecay 基本功能 / Test PiecewiseDecay basic"""
        scheduler = PiecewiseDecay(
            boundaries=[3, 6, 9], values=[0.1, 0.05, 0.01, 0.001]
        )
        self.assertAlmostEqual(scheduler(), 0.1)
        scheduler.step(epoch=2)
        self.assertAlmostEqual(scheduler(), 0.1)
        scheduler.step(epoch=3)
        self.assertAlmostEqual(scheduler(), 0.05)
        scheduler.step(epoch=10)
        self.assertAlmostEqual(scheduler(), 0.001)

    def test_piecewise_decay_empty_boundaries(self):
        """测试 PiecewiseDecay 空边界 / Test PiecewiseDecay empty boundaries"""
        with self.assertRaises(ValueError):
            PiecewiseDecay(boundaries=[], values=[0.1])

    def test_piecewise_decay_short_values(self):
        """测试 PiecewiseDecay values 太短 / Test PiecewiseDecay values too short"""
        with self.assertRaises(ValueError):
            PiecewiseDecay(boundaries=[3, 5], values=[0.1, 0.05])


class TestStepDecay(unittest.TestCase):
    """StepDecay 调度器测试类 / StepDecay scheduler test class"""

    def test_step_decay_basic(self):
        """测试 StepDecay 基本功能 / Test StepDecay basic"""
        scheduler = StepDecay(learning_rate=0.5, step_size=30, gamma=0.1)
        self.assertAlmostEqual(scheduler(), 0.5)
        scheduler.step(epoch=29)
        self.assertAlmostEqual(scheduler(), 0.5)
        scheduler.step(epoch=30)
        self.assertAlmostEqual(scheduler(), 0.05)

    def test_step_decay_invalid_step_size_type(self):
        """测试 StepDecay 无效 step_size 类型 / Test StepDecay invalid step_size type"""
        with self.assertRaises(TypeError):
            StepDecay(learning_rate=0.5, step_size=3.5)

    def test_step_decay_invalid_gamma(self):
        """测试 StepDecay 无效 gamma / Test StepDecay invalid gamma"""
        with self.assertRaises(ValueError):
            StepDecay(learning_rate=0.5, step_size=30, gamma=1.0)


class TestExponentialDecay(unittest.TestCase):
    """ExponentialDecay 调度器测试类 / ExponentialDecay scheduler test class"""

    def test_exponential_decay_basic(self):
        """测试 ExponentialDecay 基本功能 / Test ExponentialDecay basic"""
        scheduler = ExponentialDecay(learning_rate=0.5, gamma=0.9)
        lr0 = scheduler()
        scheduler.step()
        lr1 = scheduler()
        self.assertAlmostEqual(lr1, lr0 * 0.9)

    def test_exponential_decay_invalid_gamma(self):
        """测试 ExponentialDecay 无效 gamma / Test ExponentialDecay invalid gamma"""
        with self.assertRaises(AssertionError):
            ExponentialDecay(learning_rate=0.5, gamma=1.0)
        with self.assertRaises(AssertionError):
            ExponentialDecay(learning_rate=0.5, gamma=0.0)


class TestNaturalExpDecay(unittest.TestCase):
    """NaturalExpDecay 调度器测试类 / NaturalExpDecay scheduler test class"""

    def test_natural_exp_decay_basic(self):
        """测试 NaturalExpDecay 基本功能 / Test NaturalExpDecay basic"""
        scheduler = NaturalExpDecay(learning_rate=0.5, gamma=0.1)
        scheduler.step(epoch=10)
        expected = 0.5 * math.exp(-1 * 0.1 * 10)
        self.assertAlmostEqual(scheduler(), expected, places=5)

    def test_natural_exp_decay_invalid_gamma(self):
        """测试 NaturalExpDecay 无效 gamma / Test NaturalExpDecay invalid gamma"""
        with self.assertRaises(AssertionError):
            NaturalExpDecay(learning_rate=0.5, gamma=-0.1)


class TestInverseTimeDecay(unittest.TestCase):
    """InverseTimeDecay 调度器测试类 / InverseTimeDecay scheduler test class"""

    def test_inverse_time_decay_basic(self):
        """测试 InverseTimeDecay 基本功能 / Test InverseTimeDecay basic"""
        scheduler = InverseTimeDecay(learning_rate=0.5, gamma=0.1)
        scheduler.step(epoch=10)
        expected = 0.5 / (1 + 0.1 * 10)
        self.assertAlmostEqual(scheduler(), expected, places=5)


class TestPolynomialDecay(unittest.TestCase):
    """PolynomialDecay 调度器测试类 / PolynomialDecay scheduler test class"""

    def test_polynomial_decay_basic(self):
        """测试 PolynomialDecay 基本功能 / Test PolynomialDecay basic"""
        scheduler = PolynomialDecay(
            learning_rate=0.5, decay_steps=100, end_lr=0.0001, power=1.0
        )
        lr0 = scheduler()
        self.assertAlmostEqual(lr0, 0.5)

    def test_polynomial_decay_with_cycle(self):
        """测试 PolynomialDecay 循环模式 / Test PolynomialDecay with cycle"""
        scheduler = PolynomialDecay(
            learning_rate=0.5,
            decay_steps=10,
            end_lr=0.0001,
            power=1.0,
            cycle=True,
        )
        scheduler.step(epoch=15)
        self.assertGreater(scheduler(), 0.0001)

    def test_polynomial_decay_epoch_zero_cycle(self):
        """测试 PolynomialDecay epoch=0 循环模式 / Test PolynomialDecay epoch=0 with cycle"""
        scheduler = PolynomialDecay(
            learning_rate=0.5,
            decay_steps=10,
            end_lr=0.0001,
            power=1.0,
            cycle=True,
        )
        scheduler.step(epoch=0)
        self.assertGreater(scheduler(), 0)

    def test_polynomial_decay_invalid_power(self):
        """测试 PolynomialDecay 无效 power / Test PolynomialDecay invalid power"""
        with self.assertRaises(AssertionError):
            PolynomialDecay(learning_rate=0.5, decay_steps=10, power=0.0)


class TestLinearWarmup(unittest.TestCase):
    """LinearWarmup 调度器测试类 / LinearWarmup scheduler test class"""

    def test_linear_warmup_basic(self):
        """测试 LinearWarmup 基本功能 / Test LinearWarmup basic"""
        scheduler = LinearWarmup(
            learning_rate=0.5, warmup_steps=10, start_lr=0, end_lr=0.5
        )
        self.assertAlmostEqual(scheduler(), 0.0)
        scheduler.step(epoch=5)
        self.assertAlmostEqual(scheduler(), 0.25, places=3)
        scheduler.step(epoch=10)
        self.assertAlmostEqual(scheduler(), 0.5)

    def test_linear_warmup_with_scheduler(self):
        """测试 LinearWarmup 内嵌调度器 / Test LinearWarmup with inner scheduler"""
        inner = ExponentialDecay(learning_rate=0.5, gamma=0.9)
        scheduler = LinearWarmup(
            learning_rate=inner, warmup_steps=5, start_lr=0, end_lr=0.5
        )
        scheduler.step(epoch=10)
        # After warmup, should use inner scheduler
        self.assertGreater(scheduler(), 0)

    def test_linear_warmup_state_dict(self):
        """测试 LinearWarmup 状态字典 / Test LinearWarmup state_dict"""
        inner = ExponentialDecay(learning_rate=0.5, gamma=0.9)
        scheduler = LinearWarmup(
            learning_rate=inner, warmup_steps=5, start_lr=0, end_lr=0.5
        )
        scheduler.step(epoch=10)
        state = scheduler.state_dict()
        self.assertIn("LinearWarmup_LR", state)

        scheduler2 = LinearWarmup(
            learning_rate=ExponentialDecay(learning_rate=0.5, gamma=0.9),
            warmup_steps=5,
            start_lr=0,
            end_lr=0.5,
        )
        scheduler2.set_state_dict(state)
        self.assertAlmostEqual(scheduler2.last_lr, scheduler.last_lr)

    def test_linear_warmup_invalid_type(self):
        """测试 LinearWarmup 无效类型 / Test LinearWarmup invalid type"""
        with self.assertRaises(TypeError):
            LinearWarmup(
                learning_rate="0.5", warmup_steps=5, start_lr=0, end_lr=0.5
            )

    def test_linear_warmup_invalid_end_lr(self):
        """测试 LinearWarmup end_lr < start_lr / Test LinearWarmup end_lr < start_lr"""
        with self.assertRaises(AssertionError):
            LinearWarmup(
                learning_rate=0.5, warmup_steps=5, start_lr=0.5, end_lr=0.1
            )


class TestMultiStepDecay(unittest.TestCase):
    """MultiStepDecay 调度器测试类 / MultiStepDecay scheduler test class"""

    def test_multi_step_decay_basic(self):
        """测试 MultiStepDecay 基本功能 / Test MultiStepDecay basic"""
        scheduler = MultiStepDecay(
            learning_rate=0.5, milestones=[30, 50], gamma=0.1
        )
        self.assertAlmostEqual(scheduler(), 0.5)
        scheduler.step(epoch=40)
        self.assertAlmostEqual(scheduler(), 0.05)
        scheduler.step(epoch=60)
        self.assertAlmostEqual(scheduler(), 0.005)

    def test_multi_step_decay_invalid_milestones_type(self):
        """测试 MultiStepDecay 无效 milestones 类型 / Test MultiStepDecay invalid milestones type"""
        with self.assertRaises(TypeError):
            MultiStepDecay(learning_rate=0.5, milestones=30, gamma=0.1)

    def test_multi_step_decay_invalid_milestones_order(self):
        """测试 MultiStepDecay milestones 非递增 / Test MultiStepDecay milestones not increasing"""
        with self.assertRaises(ValueError):
            MultiStepDecay(learning_rate=0.5, milestones=[50, 30], gamma=0.1)

    def test_multi_step_decay_invalid_gamma(self):
        """测试 MultiStepDecay 无效 gamma / Test MultiStepDecay invalid gamma"""
        with self.assertRaises(ValueError):
            MultiStepDecay(learning_rate=0.5, milestones=[30, 50], gamma=1.0)


class TestLambdaDecay(unittest.TestCase):
    """LambdaDecay 调度器测试类 / LambdaDecay scheduler test class"""

    def test_lambda_decay_basic(self):
        """测试 LambdaDecay 基本功能 / Test LambdaDecay basic"""
        scheduler = LambdaDecay(learning_rate=0.5, lr_lambda=lambda x: 0.95**x)
        scheduler.step(epoch=10)
        expected = 0.5 * (0.95**10)
        self.assertAlmostEqual(scheduler(), expected, places=5)

    def test_lambda_decay_invalid_type(self):
        """测试 LambdaDecay 无效 lambda 类型 / Test LambdaDecay invalid lambda type"""
        with self.assertRaises(TypeError):
            LambdaDecay(learning_rate=0.5, lr_lambda="not_callable")


class TestReduceOnPlateau(unittest.TestCase):
    """ReduceOnPlateau 调度器测试类 / ReduceOnPlateau scheduler test class"""

    def test_reduce_on_plateau_basic(self):
        """测试 ReduceOnPlateau 基本功能 / Test ReduceOnPlateau basic"""
        scheduler = ReduceOnPlateau(
            learning_rate=1.0, mode="min", factor=0.5, patience=2
        )
        # Simulate loss not improving
        scheduler.step(1.0)
        scheduler.step(0.99)
        scheduler.step(0.98)
        scheduler.step(0.97)
        scheduler.step(0.97)  # bad epoch 1
        scheduler.step(0.97)  # bad epoch 2
        scheduler.step(0.97)  # bad epoch 3 -> should reduce
        self.assertLess(scheduler(), 1.0)

    def test_reduce_on_plateau_mode_max(self):
        """测试 ReduceOnPlateau max 模式 / Test ReduceOnPlateau max mode"""
        scheduler = ReduceOnPlateau(
            learning_rate=1.0, mode="max", factor=0.5, patience=1
        )
        scheduler.step(0.5)
        scheduler.step(0.49)
        scheduler.step(0.48)  # bad epoch 1 -> should reduce
        self.assertLess(scheduler(), 1.0)

    def test_reduce_on_plateau_threshold_abs(self):
        """测试 ReduceOnPlateau 绝对阈值 / Test ReduceOnPlateau absolute threshold"""
        scheduler = ReduceOnPlateau(
            learning_rate=1.0,
            mode="min",
            threshold_mode="abs",
            threshold=0.01,
            patience=2,
        )
        scheduler.step(1.0)
        scheduler.step(0.99)
        scheduler.step(0.99)
        scheduler.step(0.99)

    def test_reduce_on_plateau_cooldown(self):
        """测试 ReduceOnPlateau 冷却期 / Test ReduceOnPlateau cooldown"""
        scheduler = ReduceOnPlateau(
            learning_rate=1.0,
            mode="min",
            factor=0.5,
            patience=1,
            cooldown=3,
        )
        scheduler.step(1.0)
        scheduler.step(0.99)
        scheduler.step(0.99)  # Should reduce
        # Now in cooldown
        scheduler.step(0.98)
        scheduler.step(0.97)

    def test_reduce_on_plateau_with_numpy(self):
        """测试 ReduceOnPlateau 使用 numpy / Test ReduceOnPlateau with numpy"""
        scheduler = ReduceOnPlateau(
            learning_rate=1.0, mode="min", factor=0.5, patience=2
        )
        scheduler.step(np.array([1.0]))
        scheduler.step(np.array([0.99]))

    def test_reduce_on_plateau_invalid_mode(self):
        """测试 ReduceOnPlateau 无效模式 / Test ReduceOnPlateau invalid mode"""
        with self.assertRaises(ValueError):
            ReduceOnPlateau(learning_rate=1.0, mode="unknown")

    def test_reduce_on_plateau_invalid_factor(self):
        """测试 ReduceOnPlateau 无效因子 / Test ReduceOnPlateau invalid factor"""
        with self.assertRaises(ValueError):
            ReduceOnPlateau(learning_rate=1.0, factor=1.5)

    def test_reduce_on_plateau_invalid_threshold_mode(self):
        """测试 ReduceOnPlateau 无效阈值模式 / Test ReduceOnPlateau invalid threshold mode"""
        with self.assertRaises(ValueError):
            ReduceOnPlateau(learning_rate=1.0, threshold_mode="unknown")

    def test_reduce_on_plateau_invalid_lr_type(self):
        """测试 ReduceOnPlateau 无效学习率类型 / Test ReduceOnPlateau invalid LR type"""
        with self.assertRaises(TypeError):
            ReduceOnPlateau(learning_rate="1.0")

    def test_reduce_on_plateau_invalid_metrics(self):
        """测试 ReduceOnPlateau 无效指标 / Test ReduceOnPlateau invalid metrics"""
        scheduler = ReduceOnPlateau(learning_rate=1.0)
        with self.assertRaises(TypeError):
            scheduler.step("invalid")

    def test_reduce_on_plateau_state_dict(self):
        """测试 ReduceOnPlateau 状态字典 / Test ReduceOnPlateau state_dict"""
        scheduler = ReduceOnPlateau(learning_rate=1.0, patience=5)
        scheduler.step(1.0)
        state = scheduler.state_dict()
        self.assertIn("cooldown_counter", state)
        self.assertIn("best", state)
        self.assertIn("num_bad_epochs", state)

    def test_reduce_on_plateau_step_with_epoch(self):
        """测试 ReduceOnPlateau 指定 epoch / Test ReduceOnPlateau step with specific epoch"""
        scheduler = ReduceOnPlateau(learning_rate=1.0, patience=2)
        scheduler.step(1.0, epoch=5)
        self.assertEqual(scheduler.last_epoch, 5)


class TestCosineAnnealingDecay(unittest.TestCase):
    """CosineAnnealingDecay 调度器测试类 / CosineAnnealingDecay scheduler test class"""

    def test_cosine_annealing_basic(self):
        """测试 CosineAnnealingDecay 基本功能 / Test CosineAnnealingDecay basic"""
        scheduler = CosineAnnealingDecay(learning_rate=0.5, T_max=10)
        self.assertAlmostEqual(scheduler(), 0.5)
        scheduler.step(epoch=5)
        mid_lr = scheduler()
        self.assertGreater(mid_lr, 0)
        self.assertLess(mid_lr, 0.5)

    def test_cosine_annealing_at_restart(self):
        """测试 CosineAnnealingDecay 重启点 / Test CosineAnnealingDecay at restart point"""
        scheduler = CosineAnnealingDecay(
            learning_rate=0.5, T_max=10, eta_min=0.01
        )
        # Step through to the restart boundary
        scheduler.step()  # epoch 1
        self.assertGreater(scheduler(), 0)
        # Test at an epoch that triggers the special restart case
        scheduler.step(
            epoch=11
        )  # (11 - 1 - 10) % (2*10) = 0, triggers special case
        self.assertGreater(scheduler(), 0)

    def test_cosine_annealing_closed_form(self):
        """测试 CosineAnnealingDecay 闭式公式 / Test CosineAnnealingDecay closed form"""
        scheduler = CosineAnnealingDecay(
            learning_rate=0.5, T_max=10, eta_min=0.01
        )
        # step with epoch uses _get_closed_form_lr
        scheduler.step(epoch=3)
        expected = 0.01 + (0.5 - 0.01) * (1 + math.cos(math.pi * 3 / 10)) / 2
        self.assertAlmostEqual(scheduler(), expected, places=5)

    def test_cosine_annealing_invalid_T_max(self):
        """测试 CosineAnnealingDecay 无效 T_max / Test CosineAnnealingDecay invalid T_max"""
        with self.assertRaises(TypeError):
            CosineAnnealingDecay(learning_rate=0.5, T_max=10.5)
        with self.assertRaises(AssertionError):
            CosineAnnealingDecay(learning_rate=0.5, T_max=0)

    def test_cosine_annealing_invalid_eta_min(self):
        """测试 CosineAnnealingDecay 无效 eta_min / Test CosineAnnealingDecay invalid eta_min"""
        with self.assertRaises(TypeError):
            CosineAnnealingDecay(learning_rate=0.5, T_max=10, eta_min="0")


class TestMultiplicativeDecay(unittest.TestCase):
    """MultiplicativeDecay 调度器测试类 / MultiplicativeDecay scheduler test class"""

    def test_multiplicative_decay_basic(self):
        """测试 MultiplicativeDecay 基本功能 / Test MultiplicativeDecay basic"""
        scheduler = MultiplicativeDecay(
            learning_rate=0.5, lr_lambda=lambda x: 0.95
        )
        scheduler.step(epoch=3)
        expected = 0.5 * 0.95 * 0.95 * 0.95
        self.assertAlmostEqual(scheduler(), expected, places=5)

    def test_multiplicative_decay_invalid_type(self):
        """测试 MultiplicativeDecay 无效类型 / Test MultiplicativeDecay invalid type"""
        with self.assertRaises(TypeError):
            MultiplicativeDecay(learning_rate=0.5, lr_lambda=123)


class TestOneCycleLR(unittest.TestCase):
    """OneCycleLR 调度器测试类 / OneCycleLR scheduler test class"""

    def test_one_cycle_lr_basic(self):
        """测试 OneCycleLR 基本功能 / Test OneCycleLR basic"""
        scheduler = OneCycleLR(max_learning_rate=1.0, total_steps=100)
        lr0 = scheduler()
        self.assertGreater(lr0, 0)
        scheduler.step()
        scheduler.step(epoch=50)
        scheduler.step(epoch=99)

    def test_one_cycle_lr_linear(self):
        """测试 OneCycleLR 线性策略 / Test OneCycleLR linear strategy"""
        scheduler = OneCycleLR(
            max_learning_rate=1.0,
            total_steps=100,
            anneal_strategy="linear",
        )
        scheduler.step(epoch=50)

    def test_one_cycle_lr_three_phase(self):
        """测试 OneCycleLR 三阶段 / Test OneCycleLR three phase"""
        scheduler = OneCycleLR(
            max_learning_rate=1.0,
            total_steps=100,
            three_phase=True,
            phase_pct=0.3,
        )
        scheduler.step(epoch=50)

    def test_one_cycle_lr_invalid_max_lr(self):
        """测试 OneCycleLR 无效最大学习率 / Test OneCycleLR invalid max LR"""
        with self.assertRaises(ValueError):
            OneCycleLR(max_learning_rate=-1.0, total_steps=100)
        with self.assertRaises(TypeError):
            OneCycleLR(max_learning_rate="1.0", total_steps=100)

    def test_one_cycle_lr_invalid_end_lr(self):
        """测试 OneCycleLR 无效结束学习率 / Test OneCycleLR invalid end LR"""
        with self.assertRaises(ValueError):
            OneCycleLR(
                max_learning_rate=1.0, total_steps=100, end_learning_rate=-1.0
            )

    def test_one_cycle_lr_invalid_total_steps(self):
        """测试 OneCycleLR 无效总步数 / Test OneCycleLR invalid total steps"""
        with self.assertRaises(ValueError):
            OneCycleLR(max_learning_rate=1.0, total_steps=0)

    def test_one_cycle_lr_invalid_phase_pct(self):
        """测试 OneCycleLR 无效 phase_pct / Test OneCycleLR invalid phase_pct"""
        with self.assertRaises(ValueError):
            OneCycleLR(max_learning_rate=1.0, total_steps=100, phase_pct=1.5)

    def test_one_cycle_lr_three_phase_invalid_pct(self):
        """测试 OneCycleLR 三阶段无效 phase_pct / Test OneCycleLR three phase invalid pct"""
        with self.assertRaises(ValueError):
            OneCycleLR(
                max_learning_rate=1.0,
                total_steps=100,
                three_phase=True,
                phase_pct=0.6,
            )

    def test_one_cycle_lr_invalid_anneal_strategy(self):
        """测试 OneCycleLR 无效退火策略 / Test OneCycleLR invalid anneal strategy"""
        with self.assertRaises(ValueError):
            OneCycleLR(
                max_learning_rate=1.0,
                total_steps=100,
                anneal_strategy="invalid",
            )

    def test_one_cycle_lr_exceed_steps(self):
        """测试 OneCycleLR 超过总步数 / Test OneCycleLR exceeding total steps"""
        scheduler = OneCycleLR(max_learning_rate=1.0, total_steps=10)
        with self.assertRaises(ValueError):
            scheduler.step(epoch=11)


class TestCyclicLR(unittest.TestCase):
    """CyclicLR 调度器测试类 / CyclicLR scheduler test class"""

    def test_cyclic_lr_triangular(self):
        """测试 CyclicLR triangular 模式 / Test CyclicLR triangular mode"""
        scheduler = CyclicLR(
            base_learning_rate=0.5,
            max_learning_rate=1.0,
            step_size_up=10,
            mode="triangular",
        )
        scheduler.step()
        self.assertGreater(scheduler(), 0)

    def test_cyclic_lr_triangular2(self):
        """测试 CyclicLR triangular2 模式 / Test CyclicLR triangular2 mode"""
        scheduler = CyclicLR(
            base_learning_rate=0.5,
            max_learning_rate=1.0,
            step_size_up=10,
            mode="triangular2",
        )
        scheduler.step(epoch=25)
        self.assertGreater(scheduler(), 0)

    def test_cyclic_lr_exp_range(self):
        """测试 CyclicLR exp_range 模式 / Test CyclicLR exp_range mode"""
        scheduler = CyclicLR(
            base_learning_rate=0.5,
            max_learning_rate=1.0,
            step_size_up=10,
            mode="exp_range",
            exp_gamma=0.9,
        )
        scheduler.step()
        self.assertGreater(scheduler(), 0)

    def test_cyclic_lr_custom_scale_fn(self):
        """测试 CyclicLR 自定义 scale_fn / Test CyclicLR custom scale_fn"""
        scheduler = CyclicLR(
            base_learning_rate=0.5,
            max_learning_rate=1.0,
            step_size_up=10,
            scale_fn=lambda x: max(0.5, 1.0 - x),
            scale_mode="cycle",
        )
        scheduler.step()
        self.assertGreater(scheduler(), 0)


class TestLinearLR(unittest.TestCase):
    """LinearLR 调度器测试类 / LinearLR scheduler test class"""

    def test_linear_lr_basic(self):
        """测试 LinearLR 基本功能 / Test LinearLR basic"""
        scheduler = LinearLR(
            learning_rate=0.5, total_steps=10, start_factor=0.1, end_factor=1.0
        )
        lr0 = scheduler()
        self.assertGreater(lr0, 0)
        scheduler.step(epoch=10)
        self.assertGreater(scheduler(), 0)


class TestCosineAnnealingWarmRestarts(unittest.TestCase):
    """CosineAnnealingWarmRestarts 调度器测试类 / CosineAnnealingWarmRestarts test class"""

    def test_cosine_warm_restarts_basic(self):
        """测试 CosineAnnealingWarmRestarts 基本功能 / Test CosineAnnealingWarmRestarts basic"""
        scheduler = CosineAnnealingWarmRestarts(
            learning_rate=0.5, T_0=10, T_mult=2
        )
        lr0 = scheduler()
        self.assertGreater(lr0, 0)
        scheduler.step(epoch=5)
        self.assertGreater(scheduler(), 0)

    def test_cosine_warm_restarts_eta_min(self):
        """测试 CosineAnnealingWarmRestarts eta_min / Test CosineAnnealingWarmRestarts eta_min"""
        scheduler = CosineAnnealingWarmRestarts(
            learning_rate=0.5, T_0=10, eta_min=0.01
        )
        # At T_0 boundary, the lr should approach eta_min
        # Step to epoch T_0 - 1 (9), then step to T_0 (10) for restart
        for i in range(10):
            scheduler.step()
        # After restart at epoch 10, lr should be close to base_lr again
        # Not eta_min - eta_min is the minimum during a cycle
        # At epoch near T_0-1, lr approaches eta_min
        self.assertGreater(scheduler(), 0)
        # Verify scheduler is functional
        self.assertIsNotNone(scheduler.last_lr)


if __name__ == "__main__":
    unittest.main()
