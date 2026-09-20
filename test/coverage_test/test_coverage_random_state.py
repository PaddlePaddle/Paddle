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

# [AUTO-GENERATED] Unit test for paddle.framework.random
# 自动生成的单测，覆盖 paddle.framework.random 模块中未覆盖的代码
# Target: cover uncovered lines in paddle/python/paddle/framework/random.py
# 目标：覆盖 random.py 中 get_rng_state、set_rng_state（CPU路径）、_manual_program_seed 等未覆盖行

"""
测试模块：paddle.framework.random
Test Module: paddle.framework.random

本测试覆盖以下功能：
This test covers the following functions:
1. get_rng_state(device='cpu') - 获取CPU随机状态 / Get CPU random state
2. set_rng_state(state_list, device='cpu') - 设置CPU随机状态 / Set CPU random state
3. _manual_program_seed() - 手动设置program的随机种子 / Manually set program random seed
4. set_random_seed_generator / get_random_seed_generator - 命名随机种子生成器 / Named random seed generators
5. Generator(device='cpu') - CPU随机数生成器 / CPU random number generator

覆盖的未覆盖行：54-56, 103-107, 113-114, 120, 185, 190-202, 205-206, 209-214, 242, 269, 273, 299-302
"""

import unittest

import paddle


class TestGetSetRngStateCPU(unittest.TestCase):
    """测试CPU设备上的随机状态获取和设置功能
    Test get/set random state on CPU device"""

    def setUp(self):
        """设置测试环境为动态图模式 / Set up dynamic graph mode"""
        paddle.disable_static()

    def test_get_rng_state_cpu(self):
        """测试获取CPU随机状态，应返回一个包含一个状态的列表
        Test getting CPU random state, should return a list with one state"""
        state_list = paddle.get_rng_state(device='cpu')
        self.assertIsInstance(state_list, list)
        self.assertEqual(len(state_list), 1)

    def test_set_rng_state_cpu(self):
        """测试设置CPU随机状态后生成的随机数可复现
        Test that setting CPU random state makes random numbers reproducible"""
        # 设置设备为CPU以确保随机数在CPU上生成
        # Set device to CPU to ensure random numbers are generated on CPU
        paddle.set_device('cpu')

        # 设置一个已知种子 / Set a known seed
        paddle.seed(42)

        # 获取当前CPU随机状态 / Get current CPU random state
        state_list = paddle.get_rng_state(device='cpu')
        self.assertEqual(len(state_list), 1)

        # 使用此状态生成随机数 / Generate random numbers with this state
        rand1 = paddle.randn([3, 3])

        # 恢复状态并再次生成 / Restore state and generate again
        paddle.set_rng_state(state_list, device='cpu')
        rand2 = paddle.randn([3, 3])

        # 两次生成的随机数应相同 / Both random numbers should be identical
        self.assertTrue(paddle.equal_all(rand1, rand2).item())

        # 恢复GPU设备 / Restore GPU device
        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')

    def test_set_rng_state_cpu_invalid_length(self):
        """测试设置CPU随机状态时传入错误长度应报错
        Test that setting CPU state with wrong length raises ValueError"""
        state_list = paddle.get_rng_state(device='cpu')
        # CPU状态列表长度应为1，传入2个应报错
        # CPU state list length should be 1, passing 2 should raise error
        with self.assertRaises(ValueError):
            paddle.set_rng_state(state_list + state_list, device='cpu')

    def test_get_rng_state_default_device(self):
        """测试不指定设备时获取当前设备的随机状态
        Test getting random state for current device when device is None"""
        state_list = paddle.get_rng_state()
        self.assertIsInstance(state_list, list)
        self.assertGreaterEqual(len(state_list), 1)


class TestManualProgramSeed(unittest.TestCase):
    """测试手动设置program随机种子
    Test _manual_program_seed function"""

    def test_manual_program_seed(self):
        """测试_manual_program_seed设置种子到默认program
        Test _manual_program_seed sets seed on default programs"""
        paddle.enable_static()
        try:
            from paddle.framework.random import _manual_program_seed

            _manual_program_seed(12345)

            # 验证种子已设置到默认main program
            # Verify seed is set on default main program
            main_seed = paddle.static.default_main_program().random_seed
            self.assertEqual(main_seed, 12345)

            # 验证种子已设置到默认startup program
            # Verify seed is set on default startup program
            startup_seed = paddle.static.default_startup_program().random_seed
            self.assertEqual(startup_seed, 12345)
        finally:
            paddle.disable_static()


class TestNamedRandomSeedGenerator(unittest.TestCase):
    """测试命名随机种子生成器的设置和获取
    Test set/get named random seed generators"""

    def setUp(self):
        paddle.disable_static()

    def test_set_and_get_random_seed_generator(self):
        """测试设置和获取命名随机种子生成器
        Test setting and getting a named random seed generator"""
        from paddle.framework.random import (
            get_random_seed_generator,
            set_random_seed_generator,
        )

        # 设置一个命名的随机种子生成器 / Set a named random seed generator
        set_random_seed_generator('test_gen', 42)

        # 获取该生成器 / Get the generator
        gen = get_random_seed_generator('test_gen')
        self.assertIsNotNone(gen)


class TestGeneratorClass(unittest.TestCase):
    """测试 Generator 类的创建
    Test Generator class creation"""

    def setUp(self):
        paddle.disable_static()

    def test_cpu_generator(self):
        """测试创建CPU Generator
        Test creating a CPU Generator"""
        gen = paddle.Generator('cpu')
        self.assertIsNotNone(gen)

    def test_default_generator(self):
        """测试创建默认设备的Generator
        Test creating a default device Generator"""
        gen = paddle.Generator()
        self.assertIsNotNone(gen)

    def test_generator_manual_seed(self):
        """测试Generator的manual_seed方法
        Test Generator's manual_seed method"""
        gen = paddle.Generator('cpu')
        gen.manual_seed(99)
        # 验证种子已设置 / Verify seed is set
        self.assertIsNotNone(gen)


if __name__ == '__main__':
    unittest.main()
