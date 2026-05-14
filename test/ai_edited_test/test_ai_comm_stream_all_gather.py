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

# [AUTO-GENERATED] Test file for paddle.distributed.communication.stream.all_gather
# 覆盖模块: paddle/distributed/communication/stream/all_gather.py (61.8%)
# 目标覆盖: _all_gather_into_tensor_in_dygraph, _all_gather_in_dygraph,
#            _all_gather_in_static_mode, all_gather (所有分支)
# Covered module: paddle/distributed/communication/stream/all_gather.py
# Target coverage: _all_gather_into_tensor_in_dygraph, _all_gather_in_dygraph,
#                  _all_gather_in_static_mode, all_gather (all branches)

import importlib
import unittest
from unittest.mock import MagicMock, patch

ag_mod = importlib.import_module(
    'paddle.distributed.communication.stream.all_gather'
)


class TestAllGatherIntoTensorInDygraph(unittest.TestCase):
    """_all_gather_into_tensor_in_dygraph 函数测试
    Test _all_gather_into_tensor_in_dygraph function"""

    @patch.object(ag_mod, '_get_global_group')
    def test_with_calc_stream(self, mock_get_global):
        """测试使用 calc stream 调用
        Test with use_calc_stream=True"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_get_global.return_value = mock_group
        mock_task = MagicMock()
        mock_pg.all_gather_into_tensor_on_calc_stream.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        result = ag_mod._all_gather_into_tensor_in_dygraph(
            out_tensor, in_tensor, None, True, True
        )
        self.assertEqual(result, mock_task)
        mock_pg.all_gather_into_tensor_on_calc_stream.assert_called_once_with(
            out_tensor, in_tensor
        )

    @patch.object(ag_mod, '_get_global_group')
    def test_without_calc_stream_sync(self, mock_get_global):
        """测试不使用 calc stream 且同步操作
        Test without calc stream and sync_op=True"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_get_global.return_value = mock_group
        mock_task = MagicMock()
        mock_pg.all_gather_into_tensor.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        result = ag_mod._all_gather_into_tensor_in_dygraph(
            out_tensor, in_tensor, None, True, False
        )
        self.assertEqual(result, mock_task)
        mock_pg.all_gather_into_tensor.assert_called_once_with(
            out_tensor, in_tensor, True
        )
        mock_task.wait.assert_called_once()

    @patch.object(ag_mod, '_get_global_group')
    def test_without_calc_stream_async(self, mock_get_global):
        """测试不使用 calc stream 且异步操作
        Test without calc stream and sync_op=False"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_get_global.return_value = mock_group
        mock_task = MagicMock()
        mock_pg.all_gather_into_tensor.return_value = mock_task

        out_tensor = MagicMock()
        in_tensor = MagicMock()
        result = ag_mod._all_gather_into_tensor_in_dygraph(
            out_tensor, in_tensor, None, False, False
        )
        mock_task.wait.assert_not_called()

    @patch.object(ag_mod, '_get_global_group')
    def test_with_explicit_group(self, mock_get_global):
        """测试使用显式 group（不使用全局 group）
        Test with explicit group (not global)"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        out_tensor = MagicMock()
        in_tensor = MagicMock()

        result = ag_mod._all_gather_into_tensor_in_dygraph(
            out_tensor, in_tensor, mock_group, True, True
        )
        mock_get_global.assert_not_called()
        mock_pg.all_gather_into_tensor_on_calc_stream.assert_called_once()


class TestAllGatherInDygraph(unittest.TestCase):
    """_all_gather_in_dygraph 函数测试
    Test _all_gather_in_dygraph function"""

    @patch.object(ag_mod.paddle, 'empty_like')
    @patch.object(ag_mod, '_get_global_group')
    def test_empty_tensor_list_filled(self, mock_get_global, mock_empty_like):
        """测试空 tensor_list 时自动填充
        Test empty tensor_list is auto-filled"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_group.nranks = 3
        mock_get_global.return_value = mock_group
        mock_task = MagicMock()
        mock_pg.all_gather.return_value = mock_task

        tensor_list = []
        tensor = MagicMock()
        mock_empty_like.return_value = MagicMock()

        result = ag_mod._all_gather_in_dygraph(
            tensor_list, tensor, None, True, False
        )
        self.assertEqual(mock_empty_like.call_count, 3)
        mock_pg.all_gather.assert_called_once()

    @patch.object(ag_mod, '_get_global_group')
    def test_non_empty_tensor_list(self, mock_get_global):
        """测试非空 tensor_list 直接使用
        Test non-empty tensor_list is used directly"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_group.nranks = 2
        mock_get_global.return_value = mock_group
        mock_task = MagicMock()
        mock_pg.all_gather.return_value = mock_task

        t1 = MagicMock()
        t2 = MagicMock()
        tensor_list = [t1, t2]
        tensor = MagicMock()

        result = ag_mod._all_gather_in_dygraph(
            tensor_list, tensor, None, True, False
        )
        mock_pg.all_gather.assert_called_once_with(tensor_list, tensor, True)

    @patch.object(ag_mod, '_get_global_group')
    def test_with_calc_stream(self, mock_get_global):
        """测试使用 calc stream
        Test with use_calc_stream=True"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_group.nranks = 2
        mock_get_global.return_value = mock_group
        mock_task = MagicMock()
        mock_pg.all_gather_on_calc_stream.return_value = mock_task

        tensor_list = [MagicMock(), MagicMock()]
        tensor = MagicMock()
        result = ag_mod._all_gather_in_dygraph(
            tensor_list, tensor, None, True, True
        )
        mock_pg.all_gather_on_calc_stream.assert_called_once_with(
            tensor_list, tensor
        )

    @patch.object(ag_mod, '_get_global_group')
    def test_async_no_wait(self, mock_get_global):
        """测试异步操作不等待
        Test async op does not wait"""
        mock_pg = MagicMock()
        mock_group = MagicMock()
        mock_group.process_group = mock_pg
        mock_group.nranks = 2
        mock_get_global.return_value = mock_group
        mock_task = MagicMock()
        mock_pg.all_gather.return_value = mock_task

        tensor_list = [MagicMock(), MagicMock()]
        tensor = MagicMock()
        ag_mod._all_gather_in_dygraph(tensor_list, tensor, None, False, False)
        mock_task.wait.assert_not_called()


class TestAllGatherInStaticMode(unittest.TestCase):
    """_all_gather_in_static_mode 函数测试
    Test _all_gather_in_static_mode function"""

    @patch.object(ag_mod.paddle, 'split')
    @patch.object(ag_mod, 'paddle')
    @patch.object(ag_mod.dist, 'get_world_size', return_value=2)
    @patch.object(ag_mod.framework, 'LayerHelper')
    def test_static_mode_sync_op(
        self, mock_helper_cls, mock_world_size, mock_paddle, mock_split
    ):
        """测试静态图模式同步操作
        Test static mode with sync_op=True"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper
        mock_out = MagicMock()
        mock_helper.create_variable_for_type_inference.return_value = mock_out
        mock_op = MagicMock()
        mock_helper.append_op.return_value = mock_op

        mock_paddle.unstack = MagicMock()
        mock_paddle.split = MagicMock(return_value=[MagicMock(), MagicMock()])

        tensor_list = [MagicMock(), MagicMock()]
        tensor = MagicMock()
        tensor.dtype = 'float32'
        tensor.shape = [2, 3]

        ag_mod._all_gather_in_static_mode(tensor_list, tensor, None, True)

        mock_helper.append_op.assert_called_once()

    @patch.object(ag_mod, 'paddle')
    @patch.object(ag_mod.dist, 'get_world_size', return_value=2)
    @patch.object(ag_mod.framework, 'LayerHelper')
    def test_static_mode_0d_tensor_uses_unstack(
        self, mock_helper_cls, mock_world_size, mock_paddle
    ):
        """测试静态图模式 0-D tensor 使用 unstack
        Test static mode with 0-D tensor uses unstack"""
        mock_helper = MagicMock()
        mock_helper_cls.return_value = mock_helper
        mock_out = MagicMock()
        mock_helper.create_variable_for_type_inference.return_value = mock_out
        mock_op = MagicMock()
        mock_helper.append_op.return_value = mock_op

        mock_paddle.unstack = MagicMock(return_value=[MagicMock(), MagicMock()])

        tensor = MagicMock()
        tensor.dtype = 'float32'
        tensor.shape = []  # 0-D tensor

        ag_mod._all_gather_in_static_mode(
            [MagicMock(), MagicMock()], tensor, None, True
        )
        mock_paddle.unstack.assert_called_once()


class TestAllGather(unittest.TestCase):
    """all_gather 主函数测试 / Test all_gather main function"""

    def test_all_gather_not_member_raises(self):
        """测试非成员 group 时抛出 RuntimeError
        Test all_gather with non-member group raises RuntimeError"""
        group = MagicMock()
        group.is_member.return_value = False
        tensor = MagicMock()
        tensor_list = MagicMock()
        with self.assertRaises(RuntimeError):
            ag_mod.all_gather(tensor_list, tensor, group=group)

    def test_all_gather_async_calc_stream_raises(self):
        """测试异步操作且使用 calc stream 时抛出 RuntimeError
        Test all_gather with async + calc stream raises RuntimeError"""
        group = MagicMock()
        group.is_member.return_value = True
        tensor = MagicMock()
        tensor_list = MagicMock()
        with self.assertRaises(RuntimeError):
            ag_mod.all_gather(
                tensor_list,
                tensor,
                group=group,
                sync_op=False,
                use_calc_stream=True,
            )

    @patch.object(ag_mod, '_all_gather_into_tensor_in_dygraph')
    @patch.object(ag_mod.framework, 'in_dynamic_mode', return_value=True)
    @patch.object(ag_mod.paddle, 'is_tensor', return_value=True)
    def test_dygraph_tensor_output(
        self, mock_is_tensor, mock_dygraph, mock_into_tensor
    ):
        """测试动态图模式输出为 tensor
        Test dygraph mode with tensor output"""
        mock_task = MagicMock()
        mock_into_tensor.return_value = mock_task
        out_tensor = MagicMock()
        in_tensor = MagicMock()
        result = ag_mod.all_gather(out_tensor, in_tensor)
        self.assertEqual(result, mock_task)
        mock_into_tensor.assert_called_once_with(
            out_tensor, in_tensor, None, True, False
        )

    @patch.object(ag_mod, '_all_gather_in_dygraph')
    @patch.object(ag_mod.framework, 'in_dynamic_mode', return_value=True)
    @patch.object(ag_mod.paddle, 'is_tensor', return_value=False)
    def test_dygraph_list_output(
        self, mock_is_tensor, mock_dygraph, mock_in_dygraph
    ):
        """测试动态图模式输出为列表
        Test dygraph mode with list output"""
        mock_task = MagicMock()
        mock_in_dygraph.return_value = mock_task
        out_list = MagicMock()
        in_tensor = MagicMock()
        result = ag_mod.all_gather(out_list, in_tensor)
        self.assertEqual(result, mock_task)
        mock_in_dygraph.assert_called_once()

    @patch.object(ag_mod, '_all_gather_in_static_mode')
    @patch.object(ag_mod.framework, 'in_dynamic_mode', return_value=False)
    @patch.object(ag_mod.paddle, 'is_tensor', return_value=False)
    def test_static_mode_list(self, mock_is_tensor, mock_dygraph, mock_static):
        """测试静态图模式列表输出
        Test static mode with list output"""
        mock_static.return_value = None
        out_list = MagicMock()
        in_tensor = MagicMock()
        result = ag_mod.all_gather(out_list, in_tensor)
        mock_static.assert_called_once()

    @patch.object(ag_mod.framework, 'in_dynamic_mode', return_value=False)
    @patch.object(ag_mod.paddle, 'is_tensor', return_value=True)
    def test_static_mode_tensor_raises(self, mock_is_tensor, mock_dygraph):
        """测试静态图模式 tensor 输出抛出 RuntimeError
        Test static mode with tensor output raises RuntimeError"""
        out_tensor = MagicMock()
        in_tensor = MagicMock()
        with self.assertRaises(RuntimeError):
            ag_mod.all_gather(out_tensor, in_tensor)

    @patch.object(ag_mod.framework, 'in_dynamic_mode', return_value=False)
    @patch.object(ag_mod.paddle, 'is_tensor', return_value=False)
    def test_static_mode_with_group_raises(self, mock_is_tensor, mock_dygraph):
        """测试静态图模式带 group 抛出 AssertionError
        Test static mode with group raises AssertionError"""
        group = MagicMock()
        group.is_member.return_value = True
        out_list = MagicMock()
        in_tensor = MagicMock()
        with self.assertRaises(AssertionError):
            ag_mod.all_gather(out_list, in_tensor, group=group)


if __name__ == '__main__':
    unittest.main()
