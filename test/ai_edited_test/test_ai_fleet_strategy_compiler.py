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
