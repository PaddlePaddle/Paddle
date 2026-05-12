# [AUTO-GENERATED] Test file for paddle.distributed.fleet.meta_parallel modules
# 覆盖模块: paddle/distributed/fleet/meta_parallel/tensor_parallel.py, paddle/distributed/fleet/meta_parallel/segment_parallel.py
# 未覆盖行: tensor_parallel: 38,39,42,43,48,49,52,53; segment_parallel: 35,36,39,40
# Covered module: paddle/distributed/fleet/meta_parallel/tensor_parallel.py, segment_parallel.py
# Uncovered lines: tensor_parallel: 38,39,42,43,48,49,52,53; segment_parallel: 35,36,39,40

import unittest

from paddle.distributed.fleet.meta_parallel.tensor_parallel import (
    TensorParallel,
)
from paddle.distributed.fleet.meta_parallel.segment_parallel import (
    SegmentParallel,
)


class TestTensorParallel(unittest.TestCase):
    """测试 TensorParallel 类
    Test TensorParallel class"""

    def test_tensor_parallel_import(self):
        """测试 TensorParallel 可以被导入
        Test TensorParallel can be imported"""
        self.assertIsNotNone(TensorParallel)


class TestSegmentParallel(unittest.TestCase):
    """测试 SegmentParallel 类
    Test SegmentParallel class"""

    def test_segment_parallel_import(self):
        """测试 SegmentParallel 可以被导入
        Test SegmentParallel can be imported"""
        self.assertIsNotNone(SegmentParallel)


if __name__ == '__main__':
    unittest.main()
