# [AUTO-GENERATED] Test file for paddle.distributed.fleet.utils.mix_precision_utils
# 覆盖模块: paddle/distributed/fleet/utils/mix_precision_utils.py
# Uncovered lines: 84,92,119,122,129,133,138,148-157,162,166,171,174,175,178,192-195,202-205,212-215,218-222

import unittest

import paddle
from paddle.distributed.fleet.utils.mix_precision_utils import (
    MixPrecisionOptimizer,
)


class TestMixPrecisionOptimizer(unittest.TestCase):
    """测试 MixPrecisionOptimizer 类
    Test MixPrecisionOptimizer class"""

    def test_mix_precision_optimizer_import(self):
        """测试 MixPrecisionOptimizer 可导入
        Test MixPrecisionOptimizer can be imported"""
        self.assertIsNotNone(MixPrecisionOptimizer)


if __name__ == '__main__':
    unittest.main()
