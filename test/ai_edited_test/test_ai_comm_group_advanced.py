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
# 覆盖模块: paddle/distributed/communication/group.py (82.0%)
# 目标覆盖: destroy_process_group, get_group, _sync_calc_stream,
#            _sync_comm_stream, wait, barrier, get_backend, _warn_cur_rank_not_in_group,
#            _get_or_throw_group_rank, Group.nranks with rank<0
# Covered module: paddle/distributed/communication/group.py
# Target coverage: destroy_process_group, get_group, _sync_calc_stream,
#                  _sync_comm_stream, wait, barrier, get_backend, _warn_cur_rank_not_in_group,
#                  _get_or_throw_group_rank, Group.nranks with rank<0

import unittest
from unittest.mock import MagicMock, patch

from paddle.distributed.communication.group import (
    Group,
    _add_new_group,
    _get_or_throw_group_rank,
    _GroupManager,
    _sync_calc_stream,
    _sync_comm_stream,
    _warn_cur_rank_not_in_group,
    barrier,
    destroy_process_group,
    get_backend,
    get_group,
    is_initialized,
    wait,
)


class TestGroupAdditional(unittest.TestCase):
    """Group 类补充测试 / Additional tests for Group class"""

    def test_group_world_size_negative_rank(self):
        """测试负 rank 时 world_size 为 -1
        Test world_size is -1 when rank is negative"""
        group = Group(rank_in_group=-1, id=1, ranks=[0, 1, 2])
        self.assertEqual(group.world_size, -1)

    def test_group_nranks_negative_rank(self):
        """测试负 rank 时 nranks 仍返回 ranks 列表长度
        Test nranks returns len(ranks) even when rank is negative"""
        group = Group(rank_in_group=-1, id=1, ranks=[0, 1, 2])
        self.assertEqual(group.nranks, 3)

    def test_group_id_property(self):
        """测试 Group.id 属性
        Test Group.id property"""
        group = Group(rank_in_group=0, id=42, ranks=[0, 1])
        self.assertEqual(group.id, 42)


class TestGroupManagerAdvanced(unittest.TestCase):
    """_GroupManager 高级测试 / Advanced tests for _GroupManager"""

    def setUp(self):
        self._orig_map = _GroupManager.group_map_by_id.copy()
        self._orig_id = _GroupManager.global_group_id
        _GroupManager.group_map_by_id.clear()
        _GroupManager.global_group_id = 0

    def tearDown(self):
        _GroupManager.group_map_by_id.clear()
        _GroupManager.group_map_by_id.update(self._orig_map)
        _GroupManager.global_group_id = self._orig_id

    def test_destroy_process_group_global_group(self):
        """测试销毁全局 group 时清空所有 group
        Test destroying global group clears all groups"""
        global_group = Group(rank_in_group=0, id=0, ranks=[0, 1])
        _add_new_group(global_group)
        local_group = Group(rank_in_group=0, id=1, ranks=[0, 1])
        _add_new_group(local_group)
        self.assertTrue(is_initialized())
        destroy_process_group(global_group)
        self.assertFalse(is_initialized())

    def test_destroy_process_group_non_global_group(self):
        """测试销毁非全局 group 时只删除该 group
        Test destroying non-global group only removes that group"""
        global_group = Group(rank_in_group=0, id=0, ranks=[0, 1])
        _add_new_group(global_group)
        local_group = Group(rank_in_group=0, id=1, ranks=[0, 1])
        _add_new_group(local_group)
        destroy_process_group(local_group)
        self.assertTrue(is_initialized())
        self.assertNotIn(1, _GroupManager.group_map_by_id)
        self.assertIn(0, _GroupManager.group_map_by_id)

    def test_destroy_process_group_default_none_uses_global(self):
        """测试 destroy_process_group(group=None) 使用全局 group
        Test destroy_process_group(None) destroys global group"""
        global_group = Group(rank_in_group=0, id=0, ranks=[0, 1])
        _add_new_group(global_group)
        destroy_process_group()
        self.assertFalse(is_initialized())

    def test_get_group_existing(self):
        """测试获取已存在的 group
        Test getting an existing group"""
        g = Group(rank_in_group=0, id=5, ranks=[0, 1, 2])
        _add_new_group(g)
        result = get_group(5)
        self.assertEqual(result.id, 5)

    def test_get_group_not_existing(self):
        """测试获取不存在的 group 返回 None 并发出警告
        Test getting non-existing group returns None with warning"""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_group(999)
            self.assertIsNone(result)
            self.assertTrue(len(w) > 0)
            self.assertIn("999", str(w[0].message))

    def test_get_group_default_id_zero(self):
        """测试 get_group 默认使用 id=0
        Test get_group uses default id=0"""
        result = get_group()
        # With no groups registered, returns None with warning
        self.assertIsNone(result)

    def test_is_initialized_true(self):
        """测试初始化后 is_initialized 返回 True
        Test is_initialized returns True after initialization"""
        g = Group(rank_in_group=0, id=0, ranks=[0, 1])
        _add_new_group(g)
        self.assertTrue(is_initialized())


class TestWarnCurRankNotInGroup(unittest.TestCase):
    """_warn_cur_rank_not_in_group 函数测试
    Test _warn_cur_rank_not_in_group function"""

    @patch(
        'paddle.distributed.communication.group.dist.get_rank', return_value=5
    )
    def test_warn_not_in_group_true(self, mock_rank):
        """测试当前 rank 不在 group 中返回 True
        Test returns True when current rank is not in group"""
        group = Group(rank_in_group=-1, id=1, ranks=[0, 1, 2])
        self.assertTrue(_warn_cur_rank_not_in_group(group))

    @patch(
        'paddle.distributed.communication.group.dist.get_rank', return_value=1
    )
    def test_warn_in_group_false(self, mock_rank):
        """测试当前 rank 在 group 中返回 False
        Test returns False when current rank is in group"""
        group = Group(rank_in_group=1, id=1, ranks=[0, 1, 2])
        self.assertFalse(_warn_cur_rank_not_in_group(group))

    def test_warn_none_group(self):
        """测试 group 为 None 时返回 False
        Test returns False when group is None"""
        self.assertFalse(_warn_cur_rank_not_in_group(None))


class TestGetOrThrowGroupRank(unittest.TestCase):
    """_get_or_throw_group_rank 函数测试
    Test _get_or_throw_group_rank function"""

    def test_get_or_throw_group_rank_valid(self):
        """测试有效 rank 返回正确的 group rank
        Test valid rank returns correct group rank"""
        group = Group(rank_in_group=0, id=1, ranks=[10, 20, 30])
        self.assertEqual(_get_or_throw_group_rank(20, group), 1)

    def test_get_or_throw_group_rank_invalid(self):
        """测试无效 rank 抛出异常（rank 不在 group 中）
        Test invalid rank raises exception (rank not in group)"""
        group = Group(rank_in_group=0, id=1, ranks=[10, 20, 30])
        with self.assertRaises((AssertionError, ValueError)):
            _get_or_throw_group_rank(99, group)


class TestSyncCalcStream(unittest.TestCase):
    """_sync_calc_stream 函数测试
    Test _sync_calc_stream function"""

    @patch(
        'paddle.distributed.communication.group.framework.in_dynamic_mode',
        return_value=True,
    )
    @patch(
        'paddle.distributed.communication.group.paddle._C_ops.sync_calc_stream'
    )
    def test_sync_calc_stream_dygraph(self, mock_sync, mock_dygraph):
        """测试动态图模式下调用 sync_calc_stream
        Test sync_calc_stream in dynamic mode"""
        tensor = MagicMock()
        _sync_calc_stream(tensor)
        mock_sync.assert_called_once_with(tensor)

    @patch(
        'paddle.distributed.communication.group.framework.in_dynamic_mode',
        return_value=False,
    )
    @patch('paddle.distributed.communication.group.framework.LayerHelper')
    def test_sync_calc_stream_static(self, mock_helper_cls, mock_dygraph):
        """测试静态图模式下使用 LayerHelper 添加 op
        Test sync_calc_stream in static mode uses LayerHelper"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper
        tensor = MagicMock()
        _sync_calc_stream(tensor)
        mock_helper.append_op.assert_called_once()
        call_kwargs = mock_helper.append_op.call_args[1]
        self.assertEqual(call_kwargs['type'], 'c_sync_calc_stream')
        self.assertIn('X', call_kwargs['inputs'])
        self.assertIn('Out', call_kwargs['outputs'])


class TestSyncCommStream(unittest.TestCase):
    """_sync_comm_stream 函数测试
    Test _sync_comm_stream function"""

    @patch(
        'paddle.distributed.communication.group.framework.in_dynamic_mode',
        return_value=True,
    )
    @patch(
        'paddle.distributed.communication.group.paddle._C_ops.sync_comm_stream'
    )
    def test_sync_comm_stream_dygraph(self, mock_sync, mock_dygraph):
        """测试动态图模式下调用 sync_comm_stream
        Test sync_comm_stream in dynamic mode"""
        tensor = MagicMock()
        _sync_comm_stream(tensor, ring_id=3)
        mock_sync.assert_called_once_with([tensor], 3)

    @patch(
        'paddle.distributed.communication.group.framework.in_dynamic_mode',
        return_value=False,
    )
    @patch('paddle.distributed.communication.group.framework.LayerHelper')
    def test_sync_comm_stream_static(self, mock_helper_cls, mock_dygraph):
        """测试静态图模式下使用 LayerHelper 添加 op
        Test sync_comm_stream in static mode uses LayerHelper"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper
        tensor = MagicMock()
        _sync_comm_stream(tensor, ring_id=5)
        mock_helper.append_op.assert_called_once()
        call_kwargs = mock_helper.append_op.call_args[1]
        self.assertEqual(call_kwargs['type'], 'c_sync_comm_stream')
        self.assertEqual(call_kwargs['attrs']['ring_id'], 5)


class TestWait(unittest.TestCase):
    """wait 函数测试 / Test wait function"""

    def test_wait_not_member_returns_early(self):
        """测试非成员 group 时 wait 直接返回
        Test wait returns early when group is not member"""
        group = Group(rank_in_group=-1, id=1, ranks=[0, 1])
        tensor = MagicMock()
        # Should not raise
        wait(tensor, group=group)

    @patch('paddle.distributed.communication.group._sync_calc_stream')
    def test_wait_calc_stream(self, mock_sync):
        """测试使用计算流时调用 _sync_calc_stream
        Test wait with calc stream calls _sync_calc_stream"""
        tensor = MagicMock()
        wait(tensor, group=None, use_calc_stream=True)
        mock_sync.assert_called_once_with(tensor)

    @patch('paddle.distributed.communication.group._sync_comm_stream')
    def test_wait_comm_stream_with_group(self, mock_sync):
        """测试使用通信流且有 group 时传递正确的 ring_id
        Test wait with comm stream and group passes correct ring_id"""
        group = MagicMock()
        group.is_member.return_value = True
        group.id = 42
        tensor = MagicMock()
        wait(tensor, group=group, use_calc_stream=False)
        mock_sync.assert_called_once_with(tensor, 42)

    @patch('paddle.distributed.communication.group._sync_comm_stream')
    def test_wait_comm_stream_no_group(self, mock_sync):
        """测试使用通信流且无 group 时 ring_id 为 0
        Test wait with comm stream and no group uses ring_id=0"""
        tensor = MagicMock()
        wait(tensor, group=None, use_calc_stream=False)
        mock_sync.assert_called_once_with(tensor, 0)


class TestBarrier(unittest.TestCase):
    """barrier 函数测试 / Test barrier function"""

    def test_barrier_not_member_returns_early(self):
        """测试非成员 group 时 barrier 直接返回
        Test barrier returns early for non-member group"""
        group = Group(rank_in_group=-1, id=1, ranks=[0, 1])
        barrier(group=group)

    @patch(
        'paddle.distributed.communication.group.framework.in_dynamic_mode',
        return_value=True,
    )
    @patch(
        'paddle.distributed.communication.group.framework._current_expected_place'
    )
    def test_barrier_cpu_place(self, mock_place, mock_dygraph):
        """测试 CPU 位置下的 barrier
        Test barrier on CPU place"""
        from paddle.framework import CPUPlace

        mock_cpu_instance = CPUPlace()
        mock_place.return_value = mock_cpu_instance

        mock_pg = MagicMock()
        mock_task = MagicMock()
        mock_pg.barrier.return_value = mock_task

        group = MagicMock()
        group.is_member.return_value = True
        group.process_group = mock_pg

        with patch(
            'paddle.distributed.communication.group._get_global_group',
            return_value=group,
        ):
            barrier()

        mock_pg.barrier.assert_called_once()
        mock_task.wait.assert_called_once()

    @patch(
        'paddle.distributed.communication.group.framework.in_dynamic_mode',
        return_value=True,
    )
    @patch(
        'paddle.distributed.communication.group.framework._current_expected_place'
    )
    def test_barrier_gpu_place(self, mock_place, mock_dygraph):
        """测试 GPU 位置下的 barrier（传递 device_id）
        Test barrier on GPU place passes device_id"""
        mock_gpu_place = MagicMock()
        mock_gpu_place.get_device_id.return_value = 3
        mock_place.return_value = mock_gpu_place
        # Not a CPUPlace
        mock_place.return_value.__class__ = type('GPUPlace', (), {})

        mock_pg = MagicMock()
        mock_task = MagicMock()
        mock_pg.barrier.return_value = mock_task

        group = MagicMock()
        group.is_member.return_value = True
        group.process_group = mock_pg

        with patch(
            'paddle.distributed.communication.group._get_global_group',
            return_value=group,
        ):
            barrier()

        mock_pg.barrier.assert_called_once_with(3)
        mock_task.wait.assert_called_once()

    @patch(
        'paddle.distributed.communication.group.framework.in_dynamic_mode',
        return_value=False,
    )
    @patch('paddle.distributed.communication.group.framework.LayerHelper')
    @patch('paddle.distributed.communication.group.paddle.full')
    def test_barrier_static_mode(
        self, mock_full, mock_helper_cls, mock_dygraph
    ):
        """测试静态图模式下的 barrier
        Test barrier in static mode"""
        mock_barrier_tensor = MagicMock()
        mock_full.return_value = mock_barrier_tensor
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper

        barrier()
        mock_helper.append_op.assert_called_once()
        call_kwargs = mock_helper.append_op.call_args[1]
        self.assertEqual(call_kwargs['type'], 'barrier')
        self.assertEqual(call_kwargs['attrs']['ring_id'], 0)

    @patch(
        'paddle.distributed.communication.group.framework.in_dynamic_mode',
        return_value=False,
    )
    @patch('paddle.distributed.communication.group.framework.LayerHelper')
    @patch('paddle.distributed.communication.group.paddle.full')
    def test_barrier_static_mode_non_int_ring_id(
        self, mock_full, mock_helper_cls, mock_dygraph
    ):
        """测试静态图模式下非整数 ring_id 抛出 ValueError
        Test barrier in static mode with non-int ring_id raises ValueError"""
        mock_full.return_value = MagicMock()
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper

        group = MagicMock()
        group.is_member.return_value = True
        group.id = "not_an_int"
        with self.assertRaises(ValueError):
            barrier(group=group)


class TestGetBackend(unittest.TestCase):
    """get_backend 函数测试 / Test get_backend function"""

    @patch(
        'paddle.distributed.communication.group._warn_cur_rank_not_in_group',
        return_value=True,
    )
    def test_get_backend_invalid_group(self, mock_warn):
        """测试无效 group 时 get_backend 抛出 RuntimeError
        Test get_backend raises RuntimeError for invalid group"""
        with self.assertRaises(RuntimeError):
            get_backend(group=MagicMock())

    @patch(
        'paddle.distributed.communication.group._warn_cur_rank_not_in_group',
        return_value=False,
    )
    @patch('paddle.distributed.communication.group._get_global_group')
    def test_get_backend_none_group(self, mock_get_global, mock_warn):
        """测试 group=None 时使用全局 group
        Test get_backend with None group uses global group"""
        mock_group = MagicMock()
        mock_group.backend = "NCCL"
        mock_get_global.return_value = mock_group
        result = get_backend(group=None)
        self.assertEqual(result, "NCCL")

    @patch(
        'paddle.distributed.communication.group._warn_cur_rank_not_in_group',
        return_value=False,
    )
    def test_get_backend_with_group(self, mock_warn):
        """测试指定 group 时返回对应后端
        Test get_backend with specified group returns its backend"""
        group = MagicMock()
        group.backend = "GLOO"
        result = get_backend(group=group)
        self.assertEqual(result, "GLOO")


if __name__ == '__main__':
    unittest.main()
