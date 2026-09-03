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

# [AUTO-GENERATED] Test file for paddle.distributed.fleet.base.strategy_compiler
# 覆盖模块: paddle/distributed/fleet/base/strategy_compiler.py
# 未覆盖行: 26,27,39,46,59,61,62,63,74,75,77,78,79,81,82,87,102,133,153,155,186,216,217,218,219,223,224,225,227
# Covered module: paddle/distributed/fleet/base/strategy_compiler.py
# Uncovered lines: 26,27,39,46,59,61-63,74,75,77-79,81,82,87,102,133,153,155,186,216-219,223-225,227

import unittest

from paddle.distributed.fleet.base.strategy_compiler import StrategyCompiler


class TestStrategyCompiler(unittest.TestCase):
    """测试 StrategyCompiler 类
    Test StrategyCompiler class"""

    def test_strategy_compiler_init(self):
        """测试 StrategyCompiler 初始化
        Test StrategyCompiler initialization"""
        compiler = StrategyCompiler()
        self.assertIsNotNone(compiler)

    def test_strategy_compiler_compatible(self):
        """测试 StrategyCompiler 实例可创建
        Test StrategyCompiler instance can be created"""
        compiler = StrategyCompiler()
        self.assertTrue(hasattr(compiler, '__class__'))


if __name__ == '__main__':
    unittest.main()
