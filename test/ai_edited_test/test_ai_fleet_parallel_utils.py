# [AUTO-GENERATED] Test file for paddle.distributed.fleet.utils.sequence_parallel_utils and tensor_parallel_utils
# 覆盖模块: paddle/distributed/fleet/utils/sequence_parallel_utils.py, paddle/distributed/fleet/utils/tensor_parallel_utils.py
# Uncovered lines: sequence_parallel_utils: 46-166; tensor_parallel_utils: 61-189

import unittest

from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    GatherOp,
    ScatterOp,
    AllGatherOp,
    ReduceScatterOp,
)
from paddle.distributed.fleet.utils.tensor_parallel_utils import (
    insert_synchronization,
    insert_sync_op,
)


class TestGatherOp(unittest.TestCase):
    """测试 GatherOp 类
    Test GatherOp class"""

    def test_gather_op_import(self):
        """测试 GatherOp 可以被导入
        Test GatherOp can be imported"""
        self.assertTrue(GatherOp is not None)


class TestScatterOp(unittest.TestCase):
    """测试 ScatterOp 类
    Test ScatterOp class"""

    def test_scatter_op_import(self):
        """测试 ScatterOp 可以被导入
        Test ScatterOp can be imported"""
        self.assertTrue(ScatterOp is not None)


class TestAllGatherOp(unittest.TestCase):
    """测试 AllGatherOp 类
    Test AllGatherOp class"""

    def test_all_gather_op_import(self):
        """测试 AllGatherOp 可以被导入
        Test AllGatherOp can be imported"""
        self.assertTrue(AllGatherOp is not None)


class TestReduceScatterOp(unittest.TestCase):
    """测试 ReduceScatterOp 类
    Test ReduceScatterOp class"""

    def test_reduce_scatter_op_import(self):
        """测试 ReduceScatterOp 可以被导入
        Test ReduceScatterOp can be imported"""
        self.assertTrue(ReduceScatterOp is not None)


class TestTensorParallelUtils(unittest.TestCase):
    """测试张量并行工具函数
    Test tensor parallel utility functions"""

    def test_insert_synchronization_import(self):
        """测试 insert_synchronization 可以被导入
        Test insert_synchronization can be imported"""
        self.assertTrue(callable(insert_synchronization))

    def test_insert_sync_op_import(self):
        """测试 insert_sync_op 可以被导入
        Test insert_sync_op can be imported"""
        self.assertTrue(callable(insert_sync_op))


if __name__ == '__main__':
    unittest.main()
