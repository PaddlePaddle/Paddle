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

# [AUTO-GENERATED] Test file for paddle.distributed.communication.reduce_scatter
# 覆盖模块: paddle/distributed/communication/reduce_scatter.py (72.2%)
# 目标覆盖: reduce_scatter, _reduce_scatter_base（所有分支）
# Covered module: paddle/distributed/communication/reduce_scatter.py
# Target coverage: reduce_scatter, _reduce_scatter_base (all branches)

import importlib
import unittest
from unittest.mock import MagicMock, patch

reduce_scatter_mod = importlib.import_module(
    'paddle.distributed.communication.reduce_scatter'
)
stream_mod = importlib.import_module('paddle.distributed.communication.stream')
stream_reduce_scatter_mod = importlib.import_module(
    'paddle.distributed.communication.stream.reduce_scatter'
)

from paddle.distributed.communication.reduce import ReduceOp


class TestReduceScatter(unittest.TestCase):
    """reduce_scatter 函数测试
    Test reduce_scatter function"""

    @patch.object(stream_mod, 'reduce_scatter')
    def test_reduce_scatter_default(self, mock_stream):
        """测试默认参数调用 reduce_scatter
        Test reduce_scatter with default parameters"""
        tensor = MagicMock()
        tensor_list = [MagicMock(), MagicMock()]
        mock_task = MagicMock()
        mock_stream.return_value = mock_task
        result = reduce_scatter_mod.reduce_scatter(tensor, tensor_list)
        mock_stream.assert_called_once()
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.SUM)
        self.assertEqual(call_kwargs['sync_op'], True)
        self.assertEqual(call_kwargs['use_calc_stream'], False)
        self.assertEqual(result, mock_task)

    @patch.object(stream_mod, 'reduce_scatter')
    def test_reduce_scatter_with_max(self, mock_stream):
        """测试 MAX 操作调用 reduce_scatter
        Test reduce_scatter with MAX operation"""
        tensor = MagicMock()
        tensor_list = [MagicMock()]
        mock_stream.return_value = MagicMock()
        reduce_scatter_mod.reduce_scatter(tensor, tensor_list, op=ReduceOp.MAX)
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.MAX)

    @patch.object(stream_mod, 'reduce_scatter')
    def test_reduce_scatter_with_min(self, mock_stream):
        """测试 MIN 操作调用 reduce_scatter
        Test reduce_scatter with MIN operation"""
        tensor = MagicMock()
        tensor_list = [MagicMock()]
        mock_stream.return_value = MagicMock()
        reduce_scatter_mod.reduce_scatter(tensor, tensor_list, op=ReduceOp.MIN)
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.MIN)

    @patch.object(stream_mod, 'reduce_scatter')
    def test_reduce_scatter_with_prod(self, mock_stream):
        """测试 PROD 操作调用 reduce_scatter
        Test reduce_scatter with PROD operation"""
        tensor = MagicMock()
        tensor_list = [MagicMock()]
        mock_stream.return_value = MagicMock()
        reduce_scatter_mod.reduce_scatter(tensor, tensor_list, op=ReduceOp.PROD)
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.PROD)

    @patch.object(stream_mod, 'reduce_scatter')
    def test_reduce_scatter_with_group(self, mock_stream):
        """测试指定 group 调用 reduce_scatter
        Test reduce_scatter with specified group"""
        tensor = MagicMock()
        tensor_list = [MagicMock()]
        group = MagicMock()
        mock_stream.return_value = MagicMock()
        reduce_scatter_mod.reduce_scatter(tensor, tensor_list, group=group)
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs['group'], group)

    @patch.object(stream_mod, 'reduce_scatter')
    def test_reduce_scatter_async(self, mock_stream):
        """测试异步模式调用 reduce_scatter
        Test reduce_scatter with async mode"""
        tensor = MagicMock()
        tensor_list = [MagicMock()]
        mock_stream.return_value = MagicMock()
        reduce_scatter_mod.reduce_scatter(tensor, tensor_list, sync_op=False)
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs['sync_op'], False)

    def test_reduce_scatter_invalid_op(self):
        """测试无效操作类型抛出 RuntimeError
        Test reduce_scatter with invalid op raises RuntimeError"""
        tensor = MagicMock()
        tensor_list = [MagicMock()]
        with self.assertRaises(RuntimeError):
            reduce_scatter_mod.reduce_scatter(tensor, tensor_list, op=999)

    @patch.object(
        reduce_scatter_mod.paddle.base.core, 'nccl_version', return_value=21000
    )
    @patch.object(stream_mod, 'reduce_scatter')
    def test_reduce_scatter_avg_nccl_ge(self, mock_stream, mock_nccl):
        """测试 AVG 操作且 nccl >= 2.10 时直接调用 stream
        Test reduce_scatter AVG with nccl >= 2.10 calls stream directly"""
        tensor = MagicMock()
        tensor_list = [MagicMock()]
        mock_stream.return_value = MagicMock()
        reduce_scatter_mod.reduce_scatter(tensor, tensor_list, op=ReduceOp.AVG)
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.AVG)

    @patch('paddle.distributed.collective._get_global_group')
    @patch.object(stream_mod, 'reduce_scatter')
    @patch.object(
        reduce_scatter_mod.paddle.base.core, 'nccl_version', return_value=20900
    )
    def test_reduce_scatter_avg_nccl_lt_no_group(
        self, mock_nccl, mock_stream, mock_get_global
    ):
        """测试 AVG 操作且 nccl < 2.10 且无 group 时，先 scale 再 SUM
        Test reduce_scatter AVG with nccl < 2.10 and no group"""
        tensor = MagicMock()
        tensor_list = [MagicMock()]
        global_group = MagicMock()
        global_group.nranks = 2
        mock_get_global.return_value = global_group
        mock_task = MagicMock()
        mock_stream.return_value = mock_task

        result = reduce_scatter_mod.reduce_scatter(
            tensor, tensor_list, op=ReduceOp.AVG
        )
        tensor.scale_.assert_called_once_with(1.0 / 2)
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.SUM)
        self.assertEqual(call_kwargs['group'], global_group)
        self.assertEqual(result, mock_task)

    @patch.object(stream_mod, 'reduce_scatter')
    @patch.object(
        reduce_scatter_mod.paddle.base.core, 'nccl_version', return_value=20900
    )
    def test_reduce_scatter_avg_nccl_lt_with_group(
        self, mock_nccl, mock_stream
    ):
        """测试 AVG 操作且 nccl < 2.10 且有 group 时，使用指定 group
        Test reduce_scatter AVG with nccl < 2.10 and specified group"""
        tensor = MagicMock()
        tensor_list = [MagicMock()]
        group = MagicMock()
        group.nranks = 3
        mock_stream.return_value = MagicMock()

        reduce_scatter_mod.reduce_scatter(
            tensor, tensor_list, op=ReduceOp.AVG, group=group
        )
        tensor.scale_.assert_called_once_with(1.0 / 3)
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.SUM)
        self.assertEqual(call_kwargs['group'], group)


class TestReduceScatterBase(unittest.TestCase):
    """_reduce_scatter_base 函数测试
    Test _reduce_scatter_base function"""

    @patch.object(reduce_scatter_mod, '_reduce_scatter_base_stream')
    def test_reduce_scatter_base_sum(self, mock_stream_base):
        """测试 SUM 操作的 _reduce_scatter_base
        Test _reduce_scatter_base with SUM"""
        output = MagicMock()
        input_tensor = MagicMock()
        mock_task = MagicMock()
        mock_stream_base.return_value = mock_task
        result = reduce_scatter_mod._reduce_scatter_base(output, input_tensor)
        mock_stream_base.assert_called_once()
        call_kwargs = mock_stream_base.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.SUM)
        self.assertEqual(call_kwargs['sync_op'], True)
        self.assertEqual(call_kwargs['use_calc_stream'], False)
        self.assertEqual(result, mock_task)

    @patch.object(reduce_scatter_mod, '_reduce_scatter_base_stream')
    def test_reduce_scatter_base_max(self, mock_stream_base):
        """测试 MAX 操作的 _reduce_scatter_base
        Test _reduce_scatter_base with MAX"""
        output = MagicMock()
        input_tensor = MagicMock()
        mock_stream_base.return_value = MagicMock()
        reduce_scatter_mod._reduce_scatter_base(
            output, input_tensor, op=ReduceOp.MAX
        )
        call_kwargs = mock_stream_base.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.MAX)

    @patch.object(reduce_scatter_mod, '_reduce_scatter_base_stream')
    def test_reduce_scatter_base_min(self, mock_stream_base):
        """测试 MIN 操作的 _reduce_scatter_base
        Test _reduce_scatter_base with MIN"""
        output = MagicMock()
        input_tensor = MagicMock()
        mock_stream_base.return_value = MagicMock()
        reduce_scatter_mod._reduce_scatter_base(
            output, input_tensor, op=ReduceOp.MIN
        )
        call_kwargs = mock_stream_base.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.MIN)

    @patch.object(reduce_scatter_mod, '_reduce_scatter_base_stream')
    def test_reduce_scatter_base_prod(self, mock_stream_base):
        """测试 PROD 操作的 _reduce_scatter_base
        Test _reduce_scatter_base with PROD"""
        output = MagicMock()
        input_tensor = MagicMock()
        mock_stream_base.return_value = MagicMock()
        reduce_scatter_mod._reduce_scatter_base(
            output, input_tensor, op=ReduceOp.PROD
        )
        call_kwargs = mock_stream_base.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.PROD)

    @patch.object(reduce_scatter_mod, '_reduce_scatter_base_stream')
    def test_reduce_scatter_base_with_group(self, mock_stream_base):
        """测试指定 group 的 _reduce_scatter_base
        Test _reduce_scatter_base with group"""
        output = MagicMock()
        input_tensor = MagicMock()
        group = MagicMock()
        mock_stream_base.return_value = MagicMock()
        reduce_scatter_mod._reduce_scatter_base(
            output, input_tensor, group=group
        )
        call_kwargs = mock_stream_base.call_args[1]
        self.assertEqual(call_kwargs['group'], group)

    def test_reduce_scatter_base_invalid_op(self):
        """测试无效操作类型抛出 RuntimeError
        Test _reduce_scatter_base with invalid op raises RuntimeError"""
        output = MagicMock()
        input_tensor = MagicMock()
        with self.assertRaises(RuntimeError):
            reduce_scatter_mod._reduce_scatter_base(
                output, input_tensor, op=999
            )

    def test_reduce_scatter_base_avg_invalid(self):
        """测试 _reduce_scatter_base 不支持 AVG 操作
        Test _reduce_scatter_base does not support AVG"""
        output = MagicMock()
        input_tensor = MagicMock()
        with self.assertRaises(RuntimeError):
            reduce_scatter_mod._reduce_scatter_base(
                output, input_tensor, op=ReduceOp.AVG
            )


if __name__ == '__main__':
    unittest.main()
