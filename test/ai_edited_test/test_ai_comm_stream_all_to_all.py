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

# [AUTO-GENERATED] Test file for paddle.distributed.communication.stream.all_to_all
# 覆盖模块: paddle/distributed/communication/stream/all_to_all.py (84.8%)
# 目标覆盖: _all_to_all_tensor_in_dygraph, _all_to_all_in_dygraph,
#            _all_to_all_in_static_mode, alltoall, _alltoall_single_in_dygraph,
#            alltoall_single (所有分支)
# Covered module: paddle/distributed/communication/stream/all_to_all.py
# Target coverage: _all_to_all_tensor_in_dygraph, _all_to_all_in_dygraph,
#                  _all_to_all_in_static_mode, alltoall, _alltoall_single_in_dygraph,
#                  alltoall_single (all branches)

import importlib
import unittest
from unittest.mock import MagicMock, patch

a2a_mod = importlib.import_module(
    'paddle.distributed.communication.stream.all_to_all'
)


class TestAllToAllTensorInDygraph(unittest.TestCase):
    """_all_to_all_tensor_in_dygraph 函数测试
    Test _all_to_all_tensor_in_dygraph function"""

    def test_with_calc_stream(self):
        """测试使用 calc stream 调用
        Test with use_calc_stream=True"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_task = MagicMock()
        mock_pg.all_to_all_tensor_on_calc_stream.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        result = a2a_mod._all_to_all_tensor_in_dygraph(
            out_tensor, in_tensor, mock_group, True, True
        )
        self.assertEqual(result, mock_task)
        mock_pg.all_to_all_tensor_on_calc_stream.assert_called_once_with(
            out_tensor, in_tensor
        )

    def test_sync_op_waits(self):
        """测试同步操作等待任务
        Test sync_op waits for task"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_task = MagicMock()
        mock_pg.all_to_all_tensor.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        result = a2a_mod._all_to_all_tensor_in_dygraph(
            out_tensor, in_tensor, mock_group, True, False
        )
        mock_task.wait.assert_called_once()

    def test_async_no_wait(self):
        """测试异步操作不等待
        Test async op does not wait"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_task = MagicMock()
        mock_pg.all_to_all_tensor.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        a2a_mod._all_to_all_tensor_in_dygraph(
            out_tensor, in_tensor, mock_group, False, False
        )
        mock_task.wait.assert_not_called()


class TestAllToAllInDygraph(unittest.TestCase):
    """_all_to_all_in_dygraph 函数测试
    Test _all_to_all_in_dygraph function"""

    def test_empty_in_list_raises(self):
        """测试空输入列表抛出 RuntimeError
        Test empty input list raises RuntimeError"""
        mock_group = MagicMock()
        out_list = [MagicMock()]
        with self.assertRaises(RuntimeError):
            a2a_mod._all_to_all_in_dygraph(
                out_list, [], mock_group, True, False
            )

    @patch.object(a2a_mod.paddle, 'empty_like')
    def test_empty_out_list_filled(self, mock_empty_like):
        """测试空输出列表自动填充
        Test empty output list is auto-filled"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_task = MagicMock()
        mock_pg.all_to_all.return_value = mock_task

        in1 = MagicMock()
        in2 = MagicMock()
        mock_empty_like.return_value = MagicMock()

        out_list = []
        result = a2a_mod._all_to_all_in_dygraph(
            out_list, [in1, in2], mock_group, True, False
        )
        self.assertEqual(mock_empty_like.call_count, 2)

    def test_with_calc_stream(self):
        """测试使用 calc stream
        Test with use_calc_stream=True"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_task = MagicMock()
        mock_pg.all_to_all_on_calc_stream.return_value = mock_task

        out_list = [MagicMock(), MagicMock()]
        in_list = [MagicMock(), MagicMock()]
        result = a2a_mod._all_to_all_in_dygraph(
            out_list, in_list, mock_group, True, True
        )
        self.assertEqual(result, mock_task)
        mock_pg.all_to_all_on_calc_stream.assert_called_once_with(
            out_list, in_list
        )


class TestAllToAllInStaticMode(unittest.TestCase):
    """_all_to_all_in_static_mode 函数测试
    Test _all_to_all_in_static_mode function"""

    @patch.object(
        a2a_mod.paddle, 'unstack', return_value=[MagicMock(), MagicMock()]
    )
    @patch.object(a2a_mod.paddle, 'split')
    @patch.object(a2a_mod.paddle, 'stack', return_value=MagicMock())
    @patch.object(a2a_mod.paddle, 'concat', return_value=MagicMock())
    @patch.object(a2a_mod.dist, 'get_world_size', return_value=2)
    @patch.object(a2a_mod.framework, 'LayerHelper')
    def test_static_mode_list_0d_uses_stack_and_unstack(
        self,
        mock_helper_cls,
        mock_world_size,
        mock_concat,
        mock_stack,
        mock_split,
        mock_unstack,
    ):
        """测试静态图模式 0-D tensor 列表使用 stack
        Test static mode with 0-D tensors uses stack for input"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper
        mock_out = MagicMock()
        mock_helper.create_variable_for_type_inference.return_value = mock_out
        mock_op = MagicMock()
        mock_helper.append_op.return_value = mock_op

        in1 = MagicMock()
        in1.shape = []
        in2 = MagicMock()
        in2.shape = []

        out_list = []
        a2a_mod._all_to_all_in_static_mode(
            out_list, [in1, in2], None, True, False
        )
        mock_stack.assert_called_once()
        mock_unstack.assert_called_once()

    @patch.object(
        a2a_mod.paddle, 'split', return_value=[MagicMock(), MagicMock()]
    )
    @patch.object(a2a_mod.paddle, 'concat', return_value=MagicMock())
    @patch.object(a2a_mod.dist, 'get_world_size', return_value=2)
    @patch.object(a2a_mod.framework, 'LayerHelper')
    def test_static_mode_list_nd_uses_concat_and_split(
        self, mock_helper_cls, mock_world_size, mock_concat, mock_split
    ):
        """测试静态图模式 N-D tensor 列表使用 concat
        Test static mode with N-D tensors uses concat for input"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper
        mock_out = MagicMock()
        mock_helper.create_variable_for_type_inference.return_value = mock_out
        mock_op = MagicMock()
        mock_helper.append_op.return_value = mock_op

        in1 = MagicMock()
        in1.shape = [2, 3]
        in2 = MagicMock()
        in2.shape = [2, 3]

        out_list = []
        a2a_mod._all_to_all_in_static_mode(
            out_list, [in1, in2], None, True, False
        )
        mock_concat.assert_called_once()

    @patch.object(a2a_mod.paddle, 'split')
    @patch.object(a2a_mod.paddle, 'concat')
    @patch.object(a2a_mod.dist, 'get_world_size', return_value=2)
    @patch.object(a2a_mod.framework, 'LayerHelper')
    def test_static_mode_with_single_tensor_io(
        self, mock_helper_cls, mock_world_size, mock_concat, mock_split
    ):
        """测试静态图模式单个 tensor 输入输出
        Test static mode with single tensor input/output"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper

        in_tensor = MagicMock()
        out_tensor = MagicMock()

        result = a2a_mod._all_to_all_in_static_mode(
            out_tensor, in_tensor, None, True, False
        )
        mock_helper.append_op.assert_called_once()

    @patch.object(a2a_mod.paddle, 'split')
    @patch.object(a2a_mod.paddle, 'concat')
    @patch.object(a2a_mod.dist, 'get_world_size', return_value=2)
    @patch.object(a2a_mod.framework, 'LayerHelper')
    def test_static_mode_empty_in_list_raises(
        self, mock_helper_cls, mock_world_size, mock_concat, mock_split
    ):
        """测试静态图模式空输入列表抛出 RuntimeError
        Test static mode empty input list raises RuntimeError"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper

        out_list = []
        with self.assertRaises(RuntimeError):
            a2a_mod._all_to_all_in_static_mode(out_list, [], None, True, False)

    @patch.object(a2a_mod.paddle, 'split')
    @patch.object(a2a_mod.paddle, 'concat')
    @patch.object(a2a_mod.dist, 'get_world_size', return_value=2)
    @patch.object(a2a_mod.framework, 'LayerHelper')
    def test_static_mode_non_empty_out_list_raises(
        self, mock_helper_cls, mock_world_size, mock_concat, mock_split
    ):
        """测试静态图模式非空输出列表抛出 ValueError
        Test static mode non-empty out list raises ValueError"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper

        out_list = [MagicMock()]
        in_list = [MagicMock()]
        with self.assertRaises(ValueError):
            a2a_mod._all_to_all_in_static_mode(
                out_list, in_list, None, True, False
            )

    @patch.object(a2a_mod.dist, 'wait')
    @patch.object(
        a2a_mod.paddle, 'split', return_value=[MagicMock(), MagicMock()]
    )
    @patch.object(a2a_mod.paddle, 'concat', return_value=MagicMock())
    @patch.object(a2a_mod.dist, 'get_world_size', return_value=2)
    @patch.object(a2a_mod.framework, 'LayerHelper')
    def test_static_mode_async_waits_then_splits(
        self,
        mock_helper_cls,
        mock_world_size,
        mock_concat,
        mock_split,
        mock_wait,
    ):
        """测试静态图模式异步操作先等待再拆分
        Test static mode async waits then splits"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper
        mock_out = MagicMock()
        mock_helper.create_variable_for_type_inference.return_value = mock_out
        mock_op = MagicMock()
        mock_helper.append_op.return_value = mock_op

        in1 = MagicMock()
        in1.shape = [2, 3]
        in2 = MagicMock()
        in2.shape = [2, 3]

        out_list = []
        a2a_mod._all_to_all_in_static_mode(
            out_list, [in1, in2], None, False, False
        )
        mock_wait.assert_called_once()


class TestAllToAll(unittest.TestCase):
    """alltoall 主函数测试 / Test alltoall main function"""

    @patch.object(a2a_mod, '_warn_cur_rank_not_in_group', return_value=True)
    def test_not_in_group_returns_none(self, mock_warn):
        """测试不在 group 中时返回 None
        Test returns None when not in group"""
        result = a2a_mod.alltoall([], MagicMock())
        self.assertIsNone(result)

    def test_async_with_calc_stream_raises(self):
        """测试异步 + calc stream 抛出 RuntimeError
        Test async + calc stream raises RuntimeError"""
        with self.assertRaises(RuntimeError):
            a2a_mod.alltoall(
                [], MagicMock(), sync_op=False, use_calc_stream=True
            )

    def test_none_output_raises(self):
        """测试输出为 None 抛出 RuntimeError
        Test None output raises RuntimeError"""
        with self.assertRaises(RuntimeError):
            a2a_mod.alltoall(None, MagicMock())

    def test_none_input_raises(self):
        """测试输入为 None 抛出 RuntimeError
        Test None input raises RuntimeError"""
        with self.assertRaises(RuntimeError):
            a2a_mod.alltoall([], None)

    @patch.object(a2a_mod, '_all_to_all_tensor_in_dygraph')
    @patch.object(a2a_mod.framework, 'in_dynamic_mode', return_value=True)
    @patch.object(a2a_mod.paddle, 'is_tensor', return_value=True)
    @patch.object(a2a_mod, '_warn_cur_rank_not_in_group', return_value=False)
    @patch.object(a2a_mod, '_get_global_group')
    def test_dygraph_both_tensors(
        self,
        mock_get_global,
        mock_warn,
        mock_is_tensor,
        mock_dygraph,
        mock_tensor_fn,
    ):
        """测试动态图模式 tensor 输入输出
        Test dygraph mode with tensor input/output"""
        mock_group = MagicMock()
        mock_get_global.return_value = mock_group
        mock_task = MagicMock()
        mock_tensor_fn.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        result = a2a_mod.alltoall(out_tensor, in_tensor)
        self.assertEqual(result, mock_task)
        mock_tensor_fn.assert_called_once()

    @patch.object(a2a_mod, '_all_to_all_in_dygraph')
    @patch.object(a2a_mod.framework, 'in_dynamic_mode', return_value=True)
    @patch.object(a2a_mod.paddle, 'is_tensor', return_value=False)
    @patch.object(a2a_mod, '_warn_cur_rank_not_in_group', return_value=False)
    @patch.object(a2a_mod, '_get_global_group')
    def test_dygraph_both_lists(
        self,
        mock_get_global,
        mock_warn,
        mock_is_tensor,
        mock_dygraph,
        mock_list_fn,
    ):
        """测试动态图模式列表输入输出
        Test dygraph mode with list input/output"""
        mock_group = MagicMock()
        mock_get_global.return_value = mock_group
        mock_task = MagicMock()
        mock_list_fn.return_value = mock_task

        out_list = []
        in_list = [MagicMock()]
        result = a2a_mod.alltoall(out_list, in_list)
        self.assertEqual(result, mock_task)
        mock_list_fn.assert_called_once()

    @patch.object(a2a_mod.framework, 'in_dynamic_mode', return_value=True)
    @patch.object(a2a_mod.paddle, 'is_tensor')
    @patch.object(a2a_mod, '_warn_cur_rank_not_in_group', return_value=False)
    @patch.object(a2a_mod, '_get_global_group')
    def test_dygraph_mixed_types_raises(
        self, mock_get_global, mock_warn, mock_is_tensor, mock_dygraph
    ):
        """测试动态图模式混合类型输入抛出 RuntimeError
        Test dygraph mode with mixed types raises RuntimeError"""
        mock_group = MagicMock()
        mock_get_global.return_value = mock_group

        # out is list (is_tensor=False), in is tensor (is_tensor=True)
        mock_is_tensor.side_effect = [False, True]

        with self.assertRaises(RuntimeError):
            a2a_mod.alltoall([], MagicMock())

    @patch.object(a2a_mod, '_all_to_all_in_static_mode')
    @patch.object(a2a_mod.framework, 'in_dynamic_mode', return_value=False)
    @patch.object(a2a_mod, '_warn_cur_rank_not_in_group', return_value=False)
    def test_static_mode(self, mock_warn, mock_dygraph, mock_static):
        """测试静态图模式
        Test static mode"""
        mock_static.return_value = None
        out_list = []
        in_list = [MagicMock()]
        result = a2a_mod.alltoall(out_list, in_list)
        mock_static.assert_called_once()

    @patch.object(a2a_mod.framework, 'in_dynamic_mode', return_value=False)
    @patch.object(a2a_mod, '_warn_cur_rank_not_in_group', return_value=False)
    def test_static_mode_with_group_raises(self, mock_warn, mock_dygraph):
        """测试静态图模式带 group 抛出 AssertionError
        Test static mode with group raises AssertionError"""
        group = MagicMock()
        with self.assertRaises(AssertionError):
            a2a_mod.alltoall([], [MagicMock()], group=group)


class TestAllToAllSingleInDygraph(unittest.TestCase):
    """_alltoall_single_in_dygraph 函数测试
    Test _alltoall_single_in_dygraph function"""

    def test_none_split_sizes_converted_to_empty(self):
        """测试 None split sizes 转为空列表
        Test None split sizes converted to empty list"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_task = MagicMock()
        mock_pg.all_to_all_single.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        result = a2a_mod._alltoall_single_in_dygraph(
            out_tensor, in_tensor, None, None, mock_group, True, False
        )
        call_args = mock_pg.all_to_all_single.call_args
        self.assertEqual(call_args[0][2], [])
        self.assertEqual(call_args[0][3], [])

    def test_with_calc_stream(self):
        """测试使用 calc stream
        Test with use_calc_stream=True"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_task = MagicMock()
        mock_pg.all_to_all_single_on_calc_stream.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        result = a2a_mod._alltoall_single_in_dygraph(
            out_tensor, in_tensor, [1, 2], [2, 1], mock_group, True, True
        )
        self.assertEqual(result, mock_task)
        mock_pg.all_to_all_single_on_calc_stream.assert_called_once_with(
            out_tensor, in_tensor, [1, 2], [2, 1]
        )

    def test_sync_op_waits(self):
        """测试同步操作等待任务
        Test sync_op waits for task"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_task = MagicMock()
        mock_pg.all_to_all_single.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        a2a_mod._alltoall_single_in_dygraph(
            out_tensor, in_tensor, [1], [1], mock_group, True, False
        )
        mock_task.wait.assert_called_once()

    def test_async_no_wait(self):
        """测试异步操作不等待
        Test async op does not wait"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_task = MagicMock()
        mock_pg.all_to_all_single.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        a2a_mod._alltoall_single_in_dygraph(
            out_tensor, in_tensor, [1], [1], mock_group, False, False
        )
        mock_task.wait.assert_not_called()


class TestAllToAllSingle(unittest.TestCase):
    """alltoall_single 主函数测试 / Test alltoall_single main function"""

    @patch.object(a2a_mod, '_warn_cur_rank_not_in_group', return_value=True)
    def test_not_in_group_returns_none(self, mock_warn):
        """测试不在 group 中时返回 None
        Test returns None when not in group"""
        result = a2a_mod.alltoall_single(MagicMock(), MagicMock())
        self.assertIsNone(result)

    def test_async_with_calc_stream_raises(self):
        """测试异步 + calc stream 抛出 RuntimeError
        Test async + calc stream raises RuntimeError"""
        with self.assertRaises(RuntimeError):
            a2a_mod.alltoall_single(
                MagicMock(), MagicMock(), sync_op=False, use_calc_stream=True
            )

    @patch.object(a2a_mod, '_alltoall_single_in_dygraph')
    @patch.object(a2a_mod.framework, 'in_dynamic_mode', return_value=True)
    @patch.object(a2a_mod, '_warn_cur_rank_not_in_group', return_value=False)
    @patch.object(a2a_mod, '_get_global_group')
    def test_dygraph_mode(
        self, mock_get_global, mock_warn, mock_dygraph, mock_single
    ):
        """测试动态图模式正常执行
        Test dygraph mode normal execution"""
        mock_group = MagicMock()
        mock_get_global.return_value = mock_group
        mock_task = MagicMock()
        mock_single.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        result = a2a_mod.alltoall_single(out_tensor, in_tensor)
        self.assertEqual(result, mock_task)
        mock_single.assert_called_once_with(
            out_tensor, in_tensor, None, None, mock_group, True, False
        )

    @patch.object(a2a_mod, '_alltoall_single_in_dygraph')
    @patch.object(a2a_mod.framework, 'in_dynamic_mode', return_value=True)
    @patch.object(a2a_mod, '_warn_cur_rank_not_in_group', return_value=False)
    @patch.object(a2a_mod, '_get_global_group')
    def test_dygraph_with_split_sizes(
        self, mock_get_global, mock_warn, mock_dygraph, mock_single
    ):
        """测试动态图模式带 split sizes
        Test dygraph mode with split sizes"""
        mock_group = MagicMock()
        mock_get_global.return_value = mock_group
        mock_single.return_value = MagicMock()

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        a2a_mod.alltoall_single(out_tensor, in_tensor, [2, 3], [3, 2])
        call_args = mock_single.call_args[0]
        self.assertEqual(call_args[2], [2, 3])
        self.assertEqual(call_args[3], [3, 2])

    @patch.object(a2a_mod.framework, 'in_dynamic_mode', return_value=False)
    @patch.object(a2a_mod, '_warn_cur_rank_not_in_group', return_value=False)
    def test_static_mode_raises(self, mock_warn, mock_dygraph):
        """测试静态图模式抛出 RuntimeError
        Test static mode raises RuntimeError"""
        with self.assertRaises(RuntimeError):
            a2a_mod.alltoall_single(MagicMock(), MagicMock())


if __name__ == '__main__':
    unittest.main()
