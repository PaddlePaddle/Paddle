# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Unit test for paddle.optimizer.lr
# 自动生成的单测，覆盖 paddle.optimizer.lr 模块中未覆盖的代码
# Target: cover uncovered lines 158, 245-251, 965, 1329, 1333, 1438, 1789, 1793
#   in paddle/python/paddle/optimizer/lr.py
# 目标：覆盖 lr.py 中 LRScheduler 基类参数校验、state_dict Tensor 处理、
#   StepDecay 参数校验、LinearWarmup 参数校验、CosineAnnealingDecay 参数校验、
#   LambdaDecay 参数校验等未覆盖行

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. LRScheduler base class - negative learning_rate ValueError (line 158)
   LRScheduler 基类 - 负学习率 ValueError

2. LRScheduler.state_dict() - Tensor value handling in state dict (lines 245-251)
   LRScheduler.state_dict() - 状态字典中 Tensor 值的处理

3. LinearWarmup - invalid learning_rate type (line 965)
   LinearWarmup - 无效学习率类型的 TypeError

4. StepDecay - invalid step_size type (line 1329), gamma >= 1.0 (line 1333)
   StepDecay - step_size 类型校验和 gamma 值校验

5. LambdaDecay - invalid lr_lambda type (line 1438)
   LambdaDecay - lr_lambda 非可调用对象的 TypeError

6. CosineAnnealingDecay - invalid T_max type (line 1789), invalid eta_min type (line 1793)
   CosineAnnealingDecay - T_max 和 eta_min 的类型校验
"""

import unittest

import numpy as np

import paddle
from paddle.optimizer import lr


class TestLRSchedulerNegativeLR(unittest.TestCase):
    """Test LRScheduler raises ValueError for negative learning rate.
    测试 LRScheduler 在负学习率时抛出 ValueError。
    覆盖 paddle/python/paddle/optimizer/lr.py 第 158 行。
    """

    def test_negative_learning_rate(self):
        """LRScheduler should raise ValueError for negative learning_rate.
        负学习率应抛出 ValueError。
        """
        with self.assertRaises(ValueError):
            lr.StepDecay(learning_rate=-0.1, step_size=10)

    def test_zero_learning_rate(self):
        """LRScheduler should accept zero learning rate.
        零学习率应被接受（不抛异常）。
        """
        scheduler = lr.StepDecay(learning_rate=0.0, step_size=10)
        self.assertEqual(scheduler(), 0.0)


class TestLRSchedulerStateDictTensor(unittest.TestCase):
    """Test LRScheduler.state_dict() properly handles Tensor values.
    测试 LRScheduler.state_dict() 正确处理 Tensor 类型的值。
    覆盖 paddle/python/paddle/optimizer/lr.py 第 245-251 行。
    """

    def test_state_dict_with_tensor_value(self):
        """state_dict should convert Tensor values to float.
        state_dict 中的 Tensor 值应转换为 float。
        """
        scheduler = lr.StepDecay(learning_rate=0.1, step_size=10)
        # Manually inject a Tensor into internal state to test conversion
        scheduler.last_lr = paddle.to_tensor([0.05])
        state = scheduler.state_dict()
        self.assertIsInstance(state['last_lr'], float)
        np.testing.assert_allclose(state['last_lr'], 0.05, rtol=1e-5)

    def test_state_dict_normal(self):
        """Normal state_dict should contain last_epoch and last_lr.
        正常的 state_dict 应包含 last_epoch 和 last_lr。
        """
        scheduler = lr.StepDecay(learning_rate=0.1, step_size=10)
        scheduler.step()
        state = scheduler.state_dict()
        self.assertIn('last_epoch', state)
        self.assertIn('last_lr', state)

    def test_state_dict_set_and_load(self):
        """state_dict and set_state_dict should be symmetric.
        state_dict 和 set_state_dict 应具有对称性。
        """
        scheduler1 = lr.StepDecay(learning_rate=0.1, step_size=5)
        for _ in range(7):
            scheduler1.step()
        state = scheduler1.state_dict()

        scheduler2 = lr.StepDecay(learning_rate=0.1, step_size=5)
        scheduler2.set_state_dict(state)
        self.assertEqual(scheduler1(), scheduler2())


class TestLinearWarmupTypeError(unittest.TestCase):
    """Test LinearWarmup raises TypeError for invalid learning_rate type.
    测试 LinearWarmup 在 learning_rate 类型无效时抛出 TypeError。
    覆盖 paddle/python/paddle/optimizer/lr.py 第 965 行。
    """

    def test_invalid_lr_type_string(self):
        """LinearWarmup should raise TypeError for string learning_rate.
        字符串类型的 learning_rate 应抛出 TypeError。
        """
        with self.assertRaises(TypeError):
            lr.LinearWarmup(
                learning_rate="0.1",
                warmup_steps=100,
                start_lr=0.0,
                end_lr=0.1,
            )

    def test_valid_lr_float(self):
        """LinearWarmup should accept float learning_rate.
        浮点型 learning_rate 应被接受。
        """
        scheduler = lr.LinearWarmup(
            learning_rate=0.1,
            warmup_steps=100,
            start_lr=0.0,
            end_lr=0.1,
        )
        self.assertIsNotNone(scheduler)

    def test_valid_lr_scheduler(self):
        """LinearWarmup should accept LRScheduler as learning_rate.
        LRScheduler 类型的 learning_rate 应被接受。
        """
        base_scheduler = lr.StepDecay(learning_rate=0.1, step_size=10)
        scheduler = lr.LinearWarmup(
            learning_rate=base_scheduler,
            warmup_steps=50,
            start_lr=0.0,
            end_lr=0.1,
        )
        self.assertIsNotNone(scheduler)


class TestStepDecayValidation(unittest.TestCase):
    """Test StepDecay input validation.
    测试 StepDecay 的输入参数校验。
    覆盖 paddle/python/paddle/optimizer/lr.py 第 1329、1333 行。
    """

    def test_step_size_not_int(self):
        """StepDecay should raise TypeError when step_size is not int.
        当 step_size 不是 int 时应抛出 TypeError。
        """
        with self.assertRaises(TypeError):
            lr.StepDecay(learning_rate=0.1, step_size=10.5)

    def test_step_size_string(self):
        """StepDecay should raise TypeError when step_size is string.
        当 step_size 是字符串时应抛出 TypeError。
        """
        with self.assertRaises(TypeError):
            lr.StepDecay(learning_rate=0.1, step_size="10")

    def test_gamma_too_large(self):
        """StepDecay should raise ValueError when gamma >= 1.0.
        当 gamma >= 1.0 时应抛出 ValueError。
        """
        with self.assertRaises(ValueError):
            lr.StepDecay(learning_rate=0.1, step_size=10, gamma=1.0)

    def test_gamma_greater_than_one(self):
        """StepDecay should raise ValueError when gamma > 1.0.
        当 gamma > 1.0 时应抛出 ValueError。
        """
        with self.assertRaises(ValueError):
            lr.StepDecay(learning_rate=0.1, step_size=10, gamma=1.5)

    def test_valid_step_decay(self):
        """StepDecay with valid params should work.
        合法参数应正常创建 StepDecay。
        """
        scheduler = lr.StepDecay(learning_rate=0.1, step_size=10, gamma=0.5)
        initial_lr = scheduler()
        np.testing.assert_allclose(initial_lr, 0.1, rtol=1e-5)
        # After 10 steps, lr should be 0.05
        for _ in range(10):
            scheduler.step()
        np.testing.assert_allclose(scheduler(), 0.05, rtol=1e-5)


class TestLambdaDecayValidation(unittest.TestCase):
    """Test LambdaDecay raises TypeError when lr_lambda is not callable.
    测试 LambdaDecay 在 lr_lambda 不可调用时抛出 TypeError。
    覆盖 paddle/python/paddle/optimizer/lr.py 第 1438 行。
    """

    def test_lr_lambda_not_callable(self):
        """LambdaDecay should raise TypeError for non-callable lr_lambda.
        非可调用的 lr_lambda 应抛出 TypeError。
        """
        with self.assertRaises(TypeError):
            lr.LambdaDecay(learning_rate=0.1, lr_lambda=0.5)

    def test_lr_lambda_string(self):
        """LambdaDecay should raise TypeError for string lr_lambda.
        字符串类型的 lr_lambda 应抛出 TypeError。
        """
        with self.assertRaises(TypeError):
            lr.LambdaDecay(learning_rate=0.1, lr_lambda="lambda x: x")

    def test_valid_lambda_decay(self):
        """LambdaDecay with a callable should work.
        可调用的 lr_lambda 应正常工作。
        """
        scheduler = lr.LambdaDecay(
            learning_rate=0.1, lr_lambda=lambda epoch: 0.95**epoch
        )
        np.testing.assert_allclose(scheduler(), 0.1, rtol=1e-5)
        scheduler.step()
        np.testing.assert_allclose(scheduler(), 0.1 * 0.95, rtol=1e-5)


class TestCosineAnnealingDecayValidation(unittest.TestCase):
    """Test CosineAnnealingDecay input validation.
    测试 CosineAnnealingDecay 的输入参数校验。
    覆盖 paddle/python/paddle/optimizer/lr.py 第 1789、1793 行。
    """

    def test_T_max_not_int(self):
        """CosineAnnealingDecay should raise TypeError when T_max is not int.
        当 T_max 不是 int 时应抛出 TypeError。
        """
        with self.assertRaises(TypeError):
            lr.CosineAnnealingDecay(learning_rate=0.1, T_max=10.5)

    def test_eta_min_not_number(self):
        """CosineAnnealingDecay should raise TypeError when eta_min is not a number.
        当 eta_min 不是数值时应抛出 TypeError。
        """
        with self.assertRaises(TypeError):
            lr.CosineAnnealingDecay(learning_rate=0.1, T_max=10, eta_min="0")

    def test_valid_cosine_annealing(self):
        """CosineAnnealingDecay with valid params should work.
        合法参数应正常创建 CosineAnnealingDecay。
        """
        scheduler = lr.CosineAnnealingDecay(
            learning_rate=0.1, T_max=20, eta_min=0.001
        )
        np.testing.assert_allclose(scheduler(), 0.1, rtol=1e-5)
        # Step through half the cosine period
        for _ in range(10):
            scheduler.step()
        # LR should be between eta_min and base_lr
        current_lr = scheduler()
        self.assertGreaterEqual(current_lr, 0.001)
        self.assertLessEqual(current_lr, 0.1)


if __name__ == '__main__':
    unittest.main()
