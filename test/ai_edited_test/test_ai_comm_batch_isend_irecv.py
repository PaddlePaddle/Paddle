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

# [AUTO-GENERATED] Test file for paddle.distributed.communication.batch_isend_irecv
# 覆盖模块: paddle/distributed/communication/batch_isend_irecv.py (90.7%)
# 目标覆盖: P2POp, _coalescing_manager, _check_p2p_op_list, batch_isend_irecv
# Covered module: paddle/distributed/communication/batch_isend_irecv.py
# Target coverage: P2POp, _coalescing_manager, _check_p2p_op_list, batch_isend_irecv

import importlib
import unittest
from unittest.mock import MagicMock, patch

batch_mod = importlib.import_module(
    'paddle.distributed.communication.batch_isend_irecv'
)

import paddle.distributed as dist


class TestP2POp(unittest.TestCase):
    """P2POp 类测试 / Test P2POp class"""

    @patch.object(batch_mod, '_get_global_group')
    def test_p2p_op_init_isend_no_group(self, mock_get_global):
        """测试使用 isend 且无 group 创建 P2POp
        Test creating P2POp with isend and no group"""
        mock_group = MagicMock()
        mock_get_global.return_value = mock_group
        tensor = MagicMock()
        op = batch_mod.P2POp(dist.isend, tensor, peer=1)
        self.assertEqual(op.op, dist.isend)
        self.assertEqual(op.tensor, tensor)
        self.assertEqual(op.peer, 1)
        self.assertEqual(op.group, mock_group)

    @patch.object(batch_mod, '_get_global_group')
    def test_p2p_op_init_irecv_no_group(self, mock_get_global):
        """测试使用 irecv 且无 group 创建 P2POp
        Test creating P2POp with irecv and no group"""
        mock_group = MagicMock()
        mock_get_global.return_value = mock_group
        tensor = MagicMock()
        op = batch_mod.P2POp(dist.irecv, tensor, peer=0)
        self.assertEqual(op.op, dist.irecv)
        self.assertEqual(op.peer, 0)

    def test_p2p_op_init_with_group(self):
        """测试指定 group 创建 P2POp
        Test creating P2POp with specified group"""
        group = MagicMock()
        tensor = MagicMock()
        op = batch_mod.P2POp(dist.isend, tensor, peer=1, group=group)
        self.assertEqual(op.group, group)

    def test_p2p_op_init_invalid_op(self):
        """测试无效操作函数抛出 RuntimeError
        Test P2POp with invalid op raises RuntimeError"""
        tensor = MagicMock()
        with self.assertRaises(RuntimeError):
            batch_mod.P2POp(lambda t, p, g: None, tensor, peer=1)


class TestCheckP2POpList(unittest.TestCase):
    """_check_p2p_op_list 函数测试
    Test _check_p2p_op_list function"""

    def test_check_valid_list(self):
        """测试有效的 P2POp 列表不抛出异常
        Test valid P2POp list does not raise"""
        group = MagicMock()
        group.backend = "NCCL"
        tensor = MagicMock()
        op1 = batch_mod.P2POp(dist.isend, tensor, peer=1, group=group)
        op2 = batch_mod.P2POp(dist.irecv, tensor, peer=0, group=group)
        # Should not raise
        batch_mod._check_p2p_op_list([op1, op2])

    def test_check_not_list(self):
        """测试非列表输入抛出 RuntimeError
        Test non-list input raises RuntimeError"""
        with self.assertRaises(RuntimeError):
            batch_mod._check_p2p_op_list("not a list")

    def test_check_non_p2p_op_elements(self):
        """测试列表中包含非 P2POp 元素抛出 RuntimeError
        Test list with non-P2POp elements raises RuntimeError"""
        with self.assertRaises(RuntimeError):
            batch_mod._check_p2p_op_list(["not a p2p op"])

    def test_check_empty_list(self):
        """测试空列表抛出异常（RuntimeError 或 IndexError）
        Test empty list raises RuntimeError or IndexError"""
        with self.assertRaises((RuntimeError, IndexError)):
            batch_mod._check_p2p_op_list([])

    def test_check_mixed_backend(self):
        """测试混合后端抛出 RuntimeError
        Test mixed backends raises RuntimeError"""
        group1 = MagicMock()
        group1.backend = "NCCL"
        group2 = MagicMock()
        group2.backend = "GLOO"
        tensor = MagicMock()
        op1 = batch_mod.P2POp(dist.isend, tensor, peer=1, group=group1)
        op2 = batch_mod.P2POp(dist.irecv, tensor, peer=0, group=group2)
        with self.assertRaises(RuntimeError):
            batch_mod._check_p2p_op_list([op1, op2])


class TestCoalescingManager(unittest.TestCase):
    """_coalescing_manager 上下文管理器测试
    Test _coalescing_manager context manager"""

    @patch.object(batch_mod, '_get_global_group')
    def test_coalescing_manager_no_tasks(self, mock_get_global):
        """测试无任务时的合并管理器
        Test coalescing manager with no tasks"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_get_global.return_value = mock_group

        with batch_mod._coalescing_manager(mock_group):
            pass

        mock_pg._start_coalescing.assert_called_once()
        mock_pg._end_coalescing.assert_called_once()

    @patch.object(batch_mod, '_get_global_group')
    def test_coalescing_manager_none_group(self, mock_get_global):
        """测试 group=None 时使用全局 group
        Test coalescing manager with None group uses global group"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_get_global.return_value = mock_group

        with batch_mod._coalescing_manager(None):
            pass

        mock_get_global.assert_called_once()
        mock_pg._start_coalescing.assert_called_once()

    @patch.object(batch_mod, '_get_global_group')
    def test_coalescing_manager_with_tasks(self, mock_get_global):
        """测试有任务时的合并管理器
        Test coalescing manager with tasks"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_get_global.return_value = mock_group
        tasks = [MagicMock(), MagicMock()]

        with batch_mod._coalescing_manager(mock_group, tasks):
            pass

        mock_pg._end_coalescing.assert_called_once_with(tasks)

    @patch.object(batch_mod, '_get_global_group')
    def test_coalescing_manager_empty_tasks_list(self, mock_get_global):
        """测试空任务列表时的合并管理器
        Test coalescing manager with empty tasks list"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_get_global.return_value = mock_group
        tasks = []

        with batch_mod._coalescing_manager(mock_group, tasks):
            pass

        mock_pg._end_coalescing.assert_called_once()


class TestBatchIsendIrecv(unittest.TestCase):
    """batch_isend_irecv 函数测试 / Test batch_isend_irecv function"""

    @patch.object(batch_mod, '_warn_cur_rank_not_in_group', return_value=True)
    def test_batch_isend_irecv_not_in_group(self, mock_warn):
        """测试当前 rank 不在 group 中时直接返回
        Test batch_isend_irecv returns early when rank not in group"""
        group = MagicMock()
        group.backend = "NCCL"
        tensor = MagicMock()
        op = batch_mod.P2POp(dist.isend, tensor, peer=1, group=group)
        result = batch_mod.batch_isend_irecv([op])
        self.assertIsNone(result)

    @patch.object(batch_mod.framework, 'in_dynamic_mode', return_value=False)
    @patch.object(batch_mod, '_warn_cur_rank_not_in_group', return_value=False)
    def test_batch_isend_irecv_static_mode(self, mock_warn, mock_dygraph):
        """测试静态图模式下抛出 RuntimeError
        Test batch_isend_irecv in static mode raises RuntimeError"""
        group = MagicMock()
        group.backend = "NCCL"
        tensor = MagicMock()
        op = batch_mod.P2POp(dist.isend, tensor, peer=1, group=group)
        with self.assertRaises(RuntimeError):
            batch_mod.batch_isend_irecv([op])

    @patch.object(batch_mod, '_get_global_group')
    @patch.object(batch_mod, '_warn_cur_rank_not_in_group', return_value=False)
    @patch.object(batch_mod, '_coalescing_manager')
    @patch.object(batch_mod.framework, 'in_dynamic_mode', return_value=True)
    def test_batch_isend_irecv_dygraph(
        self, mock_dygraph, mock_coalescing, mock_warn, mock_get_global
    ):
        """测试动态图模式下的正常执行
        Test batch_isend_irecv in dynamic mode"""
        from contextlib import nullcontext

        mock_coalescing.return_value = nullcontext()

        mock_group = MagicMock()
        mock_group.backend = "NCCL"
        mock_get_global.return_value = mock_group

        tensor1 = MagicMock()
        tensor2 = MagicMock()
        op1 = batch_mod.P2POp(dist.isend, tensor1, peer=1, group=mock_group)
        op2 = batch_mod.P2POp(dist.irecv, tensor2, peer=0, group=mock_group)

        mock_task1 = MagicMock()
        mock_task2 = MagicMock()
        op1.op = MagicMock(return_value=mock_task1)
        op2.op = MagicMock(return_value=mock_task2)

        result = batch_mod.batch_isend_irecv([op1, op2])
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    @patch.object(batch_mod, '_coalescing_manager')
    @patch.object(batch_mod.framework, 'in_dynamic_mode', return_value=True)
    def test_batch_isend_irecv_with_none_task(
        self, mock_dygraph, mock_coalescing
    ):
        """测试操作返回 None 时不添加到任务列表
        Test batch_isend_irecv skips None tasks"""
        from contextlib import nullcontext

        mock_coalescing.return_value = nullcontext()

        group = MagicMock()
        group.backend = "NCCL"
        tensor = MagicMock()
        op = batch_mod.P2POp(dist.isend, tensor, peer=1, group=group)

        # Make the op return None
        op.op = MagicMock(return_value=None)

        result = batch_mod.batch_isend_irecv([op])
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
