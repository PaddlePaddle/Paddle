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

# [AUTO-GENERATED] Test file for paddle.distributed.communication.group
# 覆盖模块: paddle/distributed/communication/group.py
# 未覆盖行: 85,92,109,128,134,251,252,253,254,258,259,261,262,263,264,265,271,272,274,275,276,277,278,312,313,315,316,318,319,351,354,363,364,365,370,401,403
# Covered module: paddle/distributed/communication/group.py
# Uncovered lines: 85,92,109,128,134,251,252,253,254,258,259,261,262,263,264,265,271,272,274,275,276,277,278,312,313,315,316,318,319

import unittest

from paddle.distributed.communication.group import (
    Group,
    _add_new_group,
    _get_global_group,
    _GroupManager,
    _is_global_group,
    is_initialized,
)


class TestGroup(unittest.TestCase):
    """测试 Group 类
    Test Group class"""

    def test_group_init(self):
        """测试 Group 初始化
        Test Group initialization"""
        group = Group(rank_in_group=0, id=1, ranks=[0, 1, 2])
        self.assertEqual(group.rank, 0)
        self.assertEqual(group.id, 1)
        self.assertEqual(group.ranks, [0, 1, 2])
        self.assertEqual(group.nranks, 3)
        self.assertEqual(group.world_size, 3)

    def test_group_is_member(self):
        """测试 Group.is_member 方法
        Test Group.is_member method"""
        group = Group(rank_in_group=0, id=1, ranks=[0, 1, 2])
        self.assertTrue(group.is_member())

    def test_group_is_not_member_negative_rank(self):
        """测试负 rank 的 Group 不是成员
        Test Group with negative rank is not a member"""
        group = Group(rank_in_group=-1, id=1, ranks=[0, 1, 2])
        self.assertFalse(group.is_member())

    def test_group_is_not_member_single_rank(self):
        """测试只有1个 rank 的 Group 不是成员
        Test Group with single rank is not a member"""
        group = Group(rank_in_group=0, id=1, ranks=[0])
        self.assertFalse(group.is_member())

    def test_group_get_group_rank(self):
        """测试 Group.get_group_rank 方法
        Test Group.get_group_rank method"""
        group = Group(rank_in_group=0, id=1, ranks=[0, 1, 2])
        self.assertEqual(group.get_group_rank(1), 1)
        self.assertEqual(group.get_group_rank(0), 0)

    def test_group_get_group_rank_not_member(self):
        """测试非成员的 Group.get_group_rank 返回 -1
        Test Group.get_group_rank returns -1 for non-member"""
        group = Group(rank_in_group=-1, id=1, ranks=[0, 1, 2])
        self.assertEqual(group.get_group_rank(0), -1)

    def test_group_get_global_rank(self):
        """测试 Group.get_global_rank 方法
        Test Group.get_global_rank method"""
        group = Group(rank_in_group=0, id=1, ranks=[10, 20, 30])
        self.assertEqual(group.get_global_rank(0), 10)
        self.assertEqual(group.get_global_rank(1), 20)

    def test_group_get_global_rank_not_member(self):
        """测试非成员的 Group.get_global_rank 返回 -1
        Test Group.get_global_rank returns -1 for non-member"""
        group = Group(rank_in_group=-1, id=1, ranks=[0, 1, 2])
        self.assertEqual(group.get_global_rank(0), -1)

    def test_group_repr(self):
        """测试 Group.__repr__ 方法
        Test Group.__repr__ method"""
        group = Group(rank_in_group=0, id=1, ranks=[0, 1], name="test_group")
        repr_str = repr(group)
        self.assertIn("rank: 0", repr_str)
        self.assertIn("nranks: 2", repr_str)
        self.assertIn("test_group", repr_str)

    def test_group_repr_no_name(self):
        """测试没有名称的 Group.__repr__ 方法
        Test Group.__repr__ method without name"""
        group = Group(rank_in_group=0, id=1, ranks=[0, 1])
        repr_str = repr(group)
        self.assertIn("None", repr_str)

    def test_group_name_property(self):
        """测试 Group.name 属性
        Test Group.name property"""
        group = Group(rank_in_group=0, id=1, ranks=[0, 1], name="my_group")
        self.assertEqual(group.name, "my_group")

    def test_group_name_none(self):
        """测试 Group.name 为 None
        Test Group.name is None"""
        group = Group(rank_in_group=0, id=1, ranks=[0, 1])
        self.assertIsNone(group.name)


class TestGroupManager(unittest.TestCase):
    """测试 _GroupManager 类
    Test _GroupManager class"""

    def setUp(self):
        self._orig_map = _GroupManager.group_map_by_id.copy()
        self._orig_id = _GroupManager.global_group_id
        _GroupManager.group_map_by_id.clear()

    def tearDown(self):
        _GroupManager.group_map_by_id.clear()
        _GroupManager.group_map_by_id.update(self._orig_map)
        _GroupManager.global_group_id = self._orig_id

    def test_add_new_group(self):
        """测试 _add_new_group 函数
        Test _add_new_group function"""
        group = Group(rank_in_group=0, id=1, ranks=[0, 1])
        _add_new_group(group)
        self.assertIn(1, _GroupManager.group_map_by_id)

    def test_add_duplicate_group(self):
        """测试添加重复 id 的 group 报错
        Test adding duplicate group id raises error"""
        group = Group(rank_in_group=0, id=1, ranks=[0, 1])
        _add_new_group(group)
        with self.assertRaises(RuntimeError):
            _add_new_group(group)

    def test_get_global_group_not_initialized(self):
        """测试未初始化时获取全局 group 报错
        Test getting global group when not initialized raises error"""
        with self.assertRaises(RuntimeError):
            _get_global_group()

    def test_is_initialized_false(self):
        """测试未初始化时 is_initialized 返回 False
        Test is_initialized returns False when not initialized"""
        self.assertFalse(is_initialized())

    def test_is_global_group(self):
        """测试 _is_global_group 函数
        Test _is_global_group function"""
        _GroupManager.global_group_id = 0
        group = Group(rank_in_group=0, id=0, ranks=[0, 1])
        _add_new_group(group)
        self.assertTrue(_is_global_group(group))

    def test_is_not_global_group(self):
        """测试非全局 group 的 _is_global_group 返回 False
        Test _is_global_group returns False for non-global group"""
        _GroupManager.global_group_id = 0
        group = Group(rank_in_group=0, id=1, ranks=[0, 1])
        _add_new_group(group)
        self.assertFalse(_is_global_group(group))


if __name__ == '__main__':
    unittest.main()
