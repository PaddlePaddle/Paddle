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

# [AUTO-GENERATED] Test file for paddle.distributed.communication.stream.scatter
# 覆盖模块: paddle/distributed/communication/stream/scatter.py (65.1%)
# 目标覆盖: _scatter_tensor_in_dygraph, _scatter_in_dygraph,
#            _scatter_in_static_mode, scatter (所有分支)
# Covered module: paddle/distributed/communication/stream/scatter.py
# Target coverage: _scatter_tensor_in_dygraph, _scatter_in_dygraph,
#                  _scatter_in_static_mode, scatter (all branches)

import importlib
import unittest
from unittest.mock import MagicMock, patch

sc_mod = importlib.import_module(
    'paddle.distributed.communication.stream.scatter'
)


class TestScatterTensorInDygraph(unittest.TestCase):
    """_scatter_tensor_in_dygraph 函数测试
    Test _scatter_tensor_in_dygraph function"""

    def test_with_calc_stream(self):
        """测试使用 calc stream 调用
        Test with use_calc_stream=True"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_task = MagicMock()
        mock_pg.scatter_tensor_on_calc_stream.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        result = sc_mod._scatter_tensor_in_dygraph(
            out_tensor, in_tensor, 0, mock_group, True, True
        )
        self.assertEqual(result, mock_task)
        mock_pg.scatter_tensor_on_calc_stream.assert_called_once_with(
            out_tensor, in_tensor, 0
        )

    def test_sync_op_waits(self):
        """测试同步操作时等待任务完成
        Test sync_op waits for task completion"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_task = MagicMock()
        mock_pg.scatter_tensor.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        result = sc_mod._scatter_tensor_in_dygraph(
            out_tensor, in_tensor, 0, mock_group, True, False
        )
        mock_pg.scatter_tensor.assert_called_once_with(
            out_tensor, in_tensor, 0, True
        )
        mock_task.wait.assert_called_once()

    def test_async_no_wait(self):
        """测试异步操作不等待
        Test async op does not wait"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_task = MagicMock()
        mock_pg.scatter_tensor.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        result = sc_mod._scatter_tensor_in_dygraph(
            out_tensor, in_tensor, 0, mock_group, False, False
        )
        mock_task.wait.assert_not_called()


class TestScatterInDygraph(unittest.TestCase):
    """_scatter_in_dygraph 函数测试
    Test _scatter_in_dygraph function"""

    def test_src_rank_empty_list_raises(self):
        """测试 src rank 的空 tensor_list 抛出 RuntimeError
        Test empty tensor_list on src rank raises RuntimeError"""
        mock_group = MagicMock()
        mock_group.rank = 0
        out_tensor = MagicMock()
        with self.assertRaises(RuntimeError):
            sc_mod._scatter_in_dygraph(
                out_tensor, [], 0, mock_group, True, False
            )

    def test_non_src_rank_tensor_list_replaced(self):
        """测试非 src rank 的 tensor_list 被替换
        Test non-src rank tensor_list is replaced"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_group.rank = 1
        mock_group.nranks = 3
        mock_task = MagicMock()
        mock_pg.scatter.return_value = mock_task

        out_tensor = MagicMock()
        result = sc_mod._scatter_in_dygraph(
            out_tensor, [MagicMock()], 0, mock_group, True, False
        )
        # The tensor_list should have been replaced with [out_tensor] * nranks
        call_args = mock_pg.scatter.call_args
        actual_list = call_args[0][1]
        self.assertEqual(len(actual_list), 3)

    def test_with_calc_stream(self):
        """测试使用 calc stream
        Test with use_calc_stream=True"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_group.rank = 1
        mock_group.nranks = 2
        mock_task = MagicMock()
        mock_pg.scatter_on_calc_stream.return_value = mock_task

        out_tensor = MagicMock()
        result = sc_mod._scatter_in_dygraph(
            out_tensor, [MagicMock()], 0, mock_group, True, True
        )
        self.assertEqual(result, mock_task)
        mock_pg.scatter_on_calc_stream.assert_called_once()

    def test_sync_op_waits(self):
        """测试同步操作时等待
        Test sync_op waits for task"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_group.rank = 0
        mock_group.nranks = 2
        mock_task = MagicMock()
        mock_pg.scatter.return_value = mock_task

        t1 = MagicMock()
        out_tensor = MagicMock()
        result = sc_mod._scatter_in_dygraph(
            out_tensor, [t1], 0, mock_group, True, False
        )
        mock_task.wait.assert_called_once()


class TestScatterInStaticMode(unittest.TestCase):
    """_scatter_in_static_mode 函数测试
    Test _scatter_in_static_mode function"""

    @patch.object(sc_mod.paddle, 'stack')
    @patch.object(sc_mod.paddle, 'concat', return_value=MagicMock())
    @patch.object(sc_mod.dist, 'get_world_size', return_value=2)
    @patch.object(sc_mod.dist, 'get_rank', return_value=0)
    @patch.object(sc_mod.framework, 'LayerHelper')
    def test_static_mode_with_list_0d(
        self,
        mock_helper_cls,
        mock_rank,
        mock_world_size,
        mock_concat,
        mock_stack,
    ):
        """测试静态图模式 0-D tensor 列表使用 stack
        Test static mode with 0-D tensor list uses stack"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper

        t1 = MagicMock()
        t1.shape = []  # 0-D
        t2 = MagicMock()
        t2.shape = []
        out_tensor = MagicMock()

        sc_mod._scatter_in_static_mode(
            out_tensor, [t1, t2], 0, None, True, False
        )
        mock_stack.assert_called_once()

    @patch.object(sc_mod.paddle, 'concat')
    @patch.object(sc_mod.dist, 'get_world_size', return_value=2)
    @patch.object(sc_mod.dist, 'get_rank', return_value=0)
    @patch.object(sc_mod.framework, 'LayerHelper')
    def test_static_mode_with_list_nd(
        self, mock_helper_cls, mock_rank, mock_world_size, mock_concat
    ):
        """测试静态图模式 N-D tensor 列表使用 concat
        Test static mode with N-D tensor list uses concat"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper
        mock_concat.return_value = MagicMock()

        t1 = MagicMock()
        t1.shape = [2, 3]
        t2 = MagicMock()
        t2.shape = [2, 3]
        out_tensor = MagicMock()

        sc_mod._scatter_in_static_mode(
            out_tensor, [t1, t2], 0, None, True, False
        )
        mock_concat.assert_called_once()

    @patch.object(sc_mod.paddle, 'concat')
    @patch.object(sc_mod.dist, 'get_world_size', return_value=2)
    @patch.object(sc_mod.dist, 'get_rank', return_value=0)
    @patch.object(sc_mod.framework, 'LayerHelper')
    def test_static_mode_with_single_tensor(
        self, mock_helper_cls, mock_rank, mock_world_size, mock_concat
    ):
        """测试静态图模式单个 tensor 直接使用
        Test static mode with single tensor uses it directly"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper

        in_tensor = MagicMock()
        out_tensor = MagicMock()

        sc_mod._scatter_in_static_mode(
            out_tensor, in_tensor, 0, None, True, False
        )
        mock_concat.assert_not_called()
        mock_helper.append_op.assert_called_once()
        call_kwargs = mock_helper.append_op.call_args[1]
        self.assertEqual(call_kwargs['type'], 'c_scatter')

    @patch.object(sc_mod.paddle, 'concat')
    @patch.object(sc_mod.dist, 'get_world_size', return_value=2)
    @patch.object(sc_mod.dist, 'get_rank', return_value=0)
    @patch.object(sc_mod.framework, 'LayerHelper')
    def test_static_mode_src_empty_list_raises(
        self, mock_helper_cls, mock_rank, mock_world_size, mock_concat
    ):
        """测试静态图模式 src rank 空列表抛出 RuntimeError
        Test static mode src rank empty list raises RuntimeError"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper

        out_tensor = MagicMock()
        with self.assertRaises(RuntimeError):
            sc_mod._scatter_in_static_mode(out_tensor, [], 0, None, True, False)

    @patch.object(sc_mod, 'paddle')
    @patch.object(sc_mod.dist, 'get_world_size', return_value=2)
    @patch.object(sc_mod.dist, 'get_rank', return_value=1)
    @patch.object(sc_mod.framework, 'LayerHelper')
    def test_static_mode_non_src_empty_list_replaced(
        self, mock_helper_cls, mock_rank, mock_world_size, mock_paddle
    ):
        """测试静态图模式非 src rank 空列表被替换
        Test static mode non-src rank empty list is replaced"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper
        mock_paddle.concat = MagicMock(return_value=MagicMock())
        mock_paddle.stack = MagicMock(return_value=MagicMock())

        out_tensor = MagicMock()
        out_tensor.shape = [2, 3]
        sc_mod._scatter_in_static_mode(out_tensor, [], 0, None, True, False)
        # Non-src rank replaces empty list with [out_tensor] * nranks
        # Then uses stack (0-D) or concat (N-D) depending on shape
        self.assertTrue(mock_paddle.concat.called or mock_paddle.stack.called)

    @patch.object(sc_mod.paddle, 'concat')
    @patch.object(sc_mod.dist, 'get_world_size', return_value=2)
    @patch.object(sc_mod.dist, 'get_rank', return_value=0)
    @patch.object(sc_mod.framework, 'LayerHelper')
    def test_static_mode_with_group_uses_group_id(
        self, mock_helper_cls, mock_rank, mock_world_size, mock_concat
    ):
        """测试静态图模式指定 group 使用 group.id
        Test static mode with group uses group.id"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper

        group = MagicMock()
        group.id = 7
        in_tensor = MagicMock()
        out_tensor = MagicMock()

        sc_mod._scatter_in_static_mode(
            out_tensor, in_tensor, 0, group, True, False
        )
        call_kwargs = mock_helper.append_op.call_args[1]
        self.assertEqual(call_kwargs['attrs']['ring_id'], 7)


class TestScatter(unittest.TestCase):
    """scatter 主函数测试 / Test scatter main function"""

    @patch.object(sc_mod, '_warn_cur_rank_not_in_group', return_value=True)
    def test_not_in_group_returns_none(self, mock_warn):
        """测试不在 group 中时返回 None
        Test returns None when not in group"""
        result = sc_mod.scatter(MagicMock(), None)
        self.assertIsNone(result)

    def test_async_with_calc_stream_raises(self):
        """测试异步 + calc stream 抛出 RuntimeError
        Test async + calc stream raises RuntimeError"""
        tensor = MagicMock()
        with self.assertRaises(RuntimeError):
            sc_mod.scatter(tensor, None, sync_op=False, use_calc_stream=True)

    @patch.object(sc_mod, '_warn_cur_rank_not_in_group', return_value=False)
    @patch.object(sc_mod.framework, 'in_dynamic_mode', return_value=True)
    @patch.object(sc_mod.paddle, 'is_tensor', return_value=True)
    @patch.object(sc_mod, '_get_or_throw_group_rank', return_value=0)
    @patch.object(sc_mod, '_scatter_tensor_in_dygraph')
    @patch.object(sc_mod, '_get_global_group')
    @patch.object(sc_mod.dist, 'get_rank', return_value=0)
    def test_dygraph_tensor_input(
        self,
        mock_dist_rank,
        mock_get_global,
        mock_scatter_tensor,
        mock_get_rank,
        mock_is_tensor,
        mock_dygraph,
        mock_warn,
    ):
        """测试动态图模式 tensor 输入
        Test dygraph mode with tensor input"""
        mock_group = MagicMock()
        mock_get_global.return_value = mock_group
        mock_task = MagicMock()
        mock_scatter_tensor.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        result = sc_mod.scatter(out_tensor, in_tensor, src=0)
        self.assertEqual(result, mock_task)
        mock_scatter_tensor.assert_called_once()

    @patch.object(sc_mod, '_warn_cur_rank_not_in_group', return_value=False)
    @patch.object(sc_mod.framework, 'in_dynamic_mode', return_value=True)
    @patch.object(sc_mod.paddle, 'is_tensor', return_value=False)
    @patch.object(sc_mod, '_get_or_throw_group_rank', return_value=0)
    @patch.object(sc_mod, '_scatter_in_dygraph')
    @patch.object(sc_mod, '_get_global_group')
    @patch.object(sc_mod.dist, 'get_rank', return_value=0)
    def test_dygraph_list_input(
        self,
        mock_dist_rank,
        mock_get_global,
        mock_scatter,
        mock_get_rank,
        mock_is_tensor,
        mock_dygraph,
        mock_warn,
    ):
        """测试动态图模式列表输入
        Test dygraph mode with list input"""
        mock_group = MagicMock()
        mock_get_global.return_value = mock_group
        mock_task = MagicMock()
        mock_scatter.return_value = mock_task

        out_tensor = MagicMock()
        in_list = [MagicMock()]
        result = sc_mod.scatter(out_tensor, in_list, src=0)
        self.assertEqual(result, mock_task)
        mock_scatter.assert_called_once()

    @patch.object(sc_mod, '_warn_cur_rank_not_in_group', return_value=False)
    @patch.object(sc_mod.framework, 'in_dynamic_mode', return_value=False)
    @patch.object(sc_mod, '_scatter_in_static_mode')
    def test_static_mode(self, mock_static, mock_dygraph, mock_warn):
        """测试静态图模式
        Test static mode"""
        mock_static.return_value = None
        out_tensor = MagicMock()
        in_list = MagicMock()
        result = sc_mod.scatter(out_tensor, in_list)
        mock_static.assert_called_once()

    @patch.object(sc_mod, '_warn_cur_rank_not_in_group', return_value=False)
    @patch.object(sc_mod.framework, 'in_dynamic_mode', return_value=False)
    def test_static_mode_with_group_raises(self, mock_dygraph, mock_warn):
        """测试静态图模式带 group 抛出 AssertionError
        Test static mode with group raises AssertionError"""
        group = MagicMock()
        out_tensor = MagicMock()
        with self.assertRaises(AssertionError):
            sc_mod.scatter(out_tensor, None, group=group)


if __name__ == '__main__':
    unittest.main()
