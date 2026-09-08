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

# [AUTO-GENERATED] Test file for paddle.distributed.communication.all_reduce
# 覆盖模块: paddle/distributed/communication/all_reduce.py (72.7%)
# 目标覆盖: all_reduce 函数，包括 AVG 分支和默认分支
# Covered module: paddle/distributed/communication/all_reduce.py
# Target coverage: all_reduce function, including AVG fallback and default path

import importlib
import unittest
from unittest.mock import MagicMock, patch

all_reduce_mod = importlib.import_module(
    'paddle.distributed.communication.all_reduce'
)
stream_mod = importlib.import_module('paddle.distributed.communication.stream')

from paddle.distributed.communication.reduce import ReduceOp


class TestAllReduce(unittest.TestCase):
    """all_reduce 函数测试
    Test all_reduce function"""

    @patch.object(stream_mod, 'all_reduce')
    def test_all_reduce_default(self, mock_stream_all_reduce):
        """测试默认参数调用 all_reduce
        Test all_reduce with default parameters"""
        tensor = MagicMock()
        mock_task = MagicMock()
        mock_stream_all_reduce.return_value = mock_task

        result = all_reduce_mod.all_reduce(tensor)

        mock_stream_all_reduce.assert_called_once()
        call_kwargs = mock_stream_all_reduce.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.SUM)
        self.assertEqual(call_kwargs['sync_op'], True)
        self.assertEqual(call_kwargs['use_calc_stream'], False)
        self.assertEqual(result, mock_task)

    @patch.object(stream_mod, 'all_reduce')
    def test_all_reduce_with_max_op(self, mock_stream_all_reduce):
        """测试使用 MAX 操作调用 all_reduce
        Test all_reduce with MAX operation"""
        tensor = MagicMock()
        mock_stream_all_reduce.return_value = MagicMock()
        all_reduce_mod.all_reduce(tensor, op=ReduceOp.MAX)
        mock_stream_all_reduce.assert_called_once()
        call_kwargs = mock_stream_all_reduce.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.MAX)

    @patch.object(stream_mod, 'all_reduce')
    def test_all_reduce_with_min_op(self, mock_stream_all_reduce):
        """测试使用 MIN 操作调用 all_reduce
        Test all_reduce with MIN operation"""
        tensor = MagicMock()
        mock_stream_all_reduce.return_value = MagicMock()
        all_reduce_mod.all_reduce(tensor, op=ReduceOp.MIN)
        mock_stream_all_reduce.assert_called_once()
        call_kwargs = mock_stream_all_reduce.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.MIN)

    @patch.object(stream_mod, 'all_reduce')
    def test_all_reduce_with_prod_op(self, mock_stream_all_reduce):
        """测试使用 PROD 操作调用 all_reduce
        Test all_reduce with PROD operation"""
        tensor = MagicMock()
        mock_stream_all_reduce.return_value = MagicMock()
        all_reduce_mod.all_reduce(tensor, op=ReduceOp.PROD)
        mock_stream_all_reduce.assert_called_once()
        call_kwargs = mock_stream_all_reduce.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.PROD)

    @patch.object(stream_mod, 'all_reduce')
    def test_all_reduce_with_group(self, mock_stream_all_reduce):
        """测试指定 group 调用 all_reduce
        Test all_reduce with specified group"""
        tensor = MagicMock()
        group = MagicMock()
        group.nranks = 4
        mock_stream_all_reduce.return_value = MagicMock()
        all_reduce_mod.all_reduce(tensor, op=ReduceOp.SUM, group=group)
        mock_stream_all_reduce.assert_called_once()
        call_kwargs = mock_stream_all_reduce.call_args[1]
        self.assertEqual(call_kwargs['group'], group)

    @patch.object(stream_mod, 'all_reduce')
    def test_all_reduce_async(self, mock_stream_all_reduce):
        """测试异步模式调用 all_reduce
        Test all_reduce with async mode (sync_op=False)"""
        tensor = MagicMock()
        mock_stream_all_reduce.return_value = MagicMock()
        all_reduce_mod.all_reduce(tensor, sync_op=False)
        mock_stream_all_reduce.assert_called_once()
        call_kwargs = mock_stream_all_reduce.call_args[1]
        self.assertEqual(call_kwargs['sync_op'], False)

    @patch.object(
        all_reduce_mod.paddle.base.core, 'nccl_version', return_value=21000
    )
    @patch.object(stream_mod, 'all_reduce')
    def test_all_reduce_avg_nccl_ge_21000(
        self, mock_stream_all_reduce, mock_nccl
    ):
        """测试 AVG 操作且 nccl >= 2.10 时直接调用 stream
        Test AVG with nccl >= 2.10 calls stream directly"""
        tensor = MagicMock()
        mock_stream_all_reduce.return_value = MagicMock()
        all_reduce_mod.all_reduce(tensor, op=ReduceOp.AVG)
        mock_stream_all_reduce.assert_called_once()
        call_kwargs = mock_stream_all_reduce.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.AVG)

    @patch('paddle.distributed.collective._get_global_group')
    @patch.object(stream_mod, 'all_reduce')
    @patch.object(
        all_reduce_mod.paddle.base.core, 'nccl_version', return_value=20900
    )
    def test_all_reduce_avg_nccl_lt_21000_no_group(
        self, mock_nccl, mock_stream, mock_get_global
    ):
        """测试 AVG 操作且 nccl < 2.10 且无 group 时，先 scale 再 SUM
        Test AVG with nccl < 2.10 and no group scales tensor then uses SUM"""
        tensor = MagicMock()
        mock_task = MagicMock()
        mock_stream.return_value = mock_task

        global_group = MagicMock()
        global_group.nranks = 4
        mock_get_global.return_value = global_group

        result = all_reduce_mod.all_reduce(tensor, op=ReduceOp.AVG, group=None)
        tensor.scale_.assert_called_once_with(1.0 / 4)
        mock_stream.assert_called_once()
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.SUM)
        self.assertEqual(call_kwargs['group'], global_group)
        self.assertEqual(result, mock_task)

    @patch.object(stream_mod, 'all_reduce')
    @patch.object(
        all_reduce_mod.paddle.base.core, 'nccl_version', return_value=20900
    )
    def test_all_reduce_avg_nccl_lt_21000_with_group(
        self, mock_nccl, mock_stream
    ):
        """测试 AVG 操作且 nccl < 2.10 且有 group 时，使用指定 group
        Test AVG with nccl < 2.10 and specified group uses that group"""
        tensor = MagicMock()
        group = MagicMock()
        group.nranks = 2
        mock_task = MagicMock()
        mock_stream.return_value = mock_task

        result = all_reduce_mod.all_reduce(tensor, op=ReduceOp.AVG, group=group)
        tensor.scale_.assert_called_once_with(1.0 / 2)
        mock_stream.assert_called_once()
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.SUM)
        self.assertEqual(call_kwargs['group'], group)


if __name__ == '__main__':
    unittest.main()
