# [AUTO-GENERATED] Test file for paddle.distributed.fleet.base.topology
# 覆盖模块: paddle/distributed/fleet/base/topology.py
# Uncovered lines: 61,116,442,588-622,644,693,696,722,723,725,773,794,795,797-799

import unittest

from paddle.distributed.fleet.base.topology import CommunicateTopology


class TestCommunicateTopology(unittest.TestCase):
    """测试 CommunicateTopology 类
    Test CommunicateTopology class"""

    def test_communicate_topology_default_init(self):
        """测试 CommunicateTopology 默认初始化
        Test CommunicateTopology default initialization"""
        topo = CommunicateTopology()
        self.assertIsNotNone(topo)

    def test_communicate_topology_custom_init(self):
        """测试 CommunicateTopology 自定义初始化
        Test CommunicateTopology custom initialization"""
        topo = CommunicateTopology(
            hybrid_group_names=["data", "model"],
            dims=[2, 4],
        )
        self.assertIsNotNone(topo)

    def test_communicate_topology_world_size(self):
        """测试 CommunicateTopology world_size
        Test CommunicateTopology world_size"""
        topo = CommunicateTopology(
            hybrid_group_names=["data", "model"],
            dims=[2, 4],
        )
        self.assertEqual(topo.world_size(), 8)


if __name__ == '__main__':
    unittest.main()
