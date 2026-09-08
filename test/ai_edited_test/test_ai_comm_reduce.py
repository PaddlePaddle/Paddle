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

# [AUTO-GENERATED] Test file for paddle.distributed.communication.reduce
# 覆盖模块: paddle/distributed/communication/reduce.py (71.4%)
# 目标覆盖: ReduceOp, _get_reduce_op, _to_inplace_op, reduce, is_avg_reduce_op_supported
# Covered module: paddle/distributed/communication/reduce.py
# Target coverage: ReduceOp, _get_reduce_op, _to_inplace_op, reduce, is_avg_reduce_op_supported

import importlib
import unittest
from unittest.mock import MagicMock, patch

reduce_mod = importlib.import_module('paddle.distributed.communication.reduce')
stream_mod = importlib.import_module('paddle.distributed.communication.stream')

from paddle.distributed.communication.reduce import (
    ReduceOp,
    _get_reduce_op,
    _to_inplace_op,
    is_avg_reduce_op_supported,
)


class TestReduceOp(unittest.TestCase):
    """ReduceOp 类测试 / Test ReduceOp class"""

    def test_reduce_op_values(self):
        """测试 ReduceOp 枚举值
        Test ReduceOp enum values"""
        self.assertEqual(ReduceOp.SUM, 0)
        self.assertEqual(ReduceOp.MAX, 1)
        self.assertEqual(ReduceOp.MIN, 2)
        self.assertEqual(ReduceOp.PROD, 3)
        self.assertEqual(ReduceOp.AVG, 4)


class TestGetReduceOp(unittest.TestCase):
    """_get_reduce_op 函数测试 / Test _get_reduce_op function"""

    @patch.object(reduce_mod.framework.core, 'ReduceOp')
    def test_get_reduce_op_sum(self, mock_reduce_op_cls):
        """测试 SUM 映射 / Test SUM mapping"""
        mock_reduce_op_cls.SUM = 'sum'
        result = _get_reduce_op(ReduceOp.SUM)
        self.assertEqual(result, 'sum')

    @patch.object(reduce_mod.framework.core, 'ReduceOp')
    def test_get_reduce_op_max(self, mock_reduce_op_cls):
        """测试 MAX 映射 / Test MAX mapping"""
        mock_reduce_op_cls.MAX = 'max'
        result = _get_reduce_op(ReduceOp.MAX)
        self.assertEqual(result, 'max')

    @patch.object(reduce_mod.framework.core, 'ReduceOp')
    def test_get_reduce_op_min(self, mock_reduce_op_cls):
        """测试 MIN 映射 / Test MIN mapping"""
        mock_reduce_op_cls.MIN = 'min'
        result = _get_reduce_op(ReduceOp.MIN)
        self.assertEqual(result, 'min')

    @patch.object(reduce_mod.framework.core, 'ReduceOp')
    def test_get_reduce_op_prod(self, mock_reduce_op_cls):
        """测试 PROD 映射 / Test PROD mapping"""
        mock_reduce_op_cls.PRODUCT = 'product'
        result = _get_reduce_op(ReduceOp.PROD)
        self.assertEqual(result, 'product')

    @patch.object(reduce_mod.framework.core, 'ReduceOp')
    def test_get_reduce_op_avg(self, mock_reduce_op_cls):
        """测试 AVG 映射 / Test AVG mapping"""
        mock_reduce_op_cls.AVG = 'avg'
        result = _get_reduce_op(ReduceOp.AVG)
        self.assertEqual(result, 'avg')

    def test_get_reduce_op_invalid(self):
        """测试无效操作类型抛出 ValueError
        Test invalid op type raises ValueError"""
        with self.assertRaises(ValueError):
            _get_reduce_op(999)


class TestToInplaceOp(unittest.TestCase):
    """_to_inplace_op 函数测试 / Test _to_inplace_op function"""

    def test_to_inplace_op(self):
        """测试添加下划线后缀
        Test adding underscore suffix"""
        self.assertEqual(_to_inplace_op('scale'), 'scale_')
        self.assertEqual(_to_inplace_op('all_reduce'), 'all_reduce_')
        self.assertEqual(_to_inplace_op(''), '_')


class TestIsAvgReduceOpSupported(unittest.TestCase):
    """is_avg_reduce_op_supported 函数测试
    Test is_avg_reduce_op_supported function"""

    @patch.object(reduce_mod.paddle, 'is_compiled_with_cuda', return_value=True)
    @patch.object(
        reduce_mod.paddle.base.core, 'nccl_version', return_value=21000
    )
    def test_avg_supported_cuda_nccl_ge(self, mock_nccl, mock_cuda):
        """测试 CUDA 编译且 nccl >= 2.10 时 AVG 支持
        Test AVG supported when compiled with CUDA and nccl >= 2.10"""
        self.assertTrue(is_avg_reduce_op_supported())

    @patch.object(reduce_mod.paddle, 'is_compiled_with_cuda', return_value=True)
    @patch.object(
        reduce_mod.paddle.base.core, 'nccl_version', return_value=20900
    )
    def test_avg_not_supported_cuda_nccl_lt(self, mock_nccl, mock_cuda):
        """测试 CUDA 编译但 nccl < 2.10 时 AVG 不支持
        Test AVG not supported when compiled with CUDA but nccl < 2.10"""
        self.assertFalse(is_avg_reduce_op_supported())

    @patch.object(
        reduce_mod.paddle, 'is_compiled_with_cuda', return_value=False
    )
    def test_avg_not_supported_no_cuda(self, mock_cuda):
        """测试非 CUDA 编译时 AVG 不支持
        Test AVG not supported when not compiled with CUDA"""
        self.assertFalse(is_avg_reduce_op_supported())


class TestReduce(unittest.TestCase):
    """reduce 函数测试 / Test reduce function"""

    @patch.object(stream_mod, 'reduce')
    def test_reduce_default(self, mock_stream_reduce):
        """测试默认参数调用 reduce
        Test reduce with default parameters"""
        tensor = MagicMock()
        mock_task = MagicMock()
        mock_stream_reduce.return_value = mock_task
        result = reduce_mod.reduce(tensor, dst=0)
        mock_stream_reduce.assert_called_once()
        call_kwargs = mock_stream_reduce.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.SUM)
        self.assertEqual(call_kwargs['dst'], 0)
        self.assertEqual(call_kwargs['sync_op'], True)
        self.assertEqual(call_kwargs['use_calc_stream'], False)
        self.assertEqual(result, mock_task)

    @patch.object(stream_mod, 'reduce')
    def test_reduce_with_max_op(self, mock_stream_reduce):
        """测试 MAX 操作调用 reduce
        Test reduce with MAX operation"""
        tensor = MagicMock()
        mock_stream_reduce.return_value = MagicMock()
        reduce_mod.reduce(tensor, dst=1, op=ReduceOp.MAX)
        call_kwargs = mock_stream_reduce.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.MAX)

    @patch.object(stream_mod, 'reduce')
    def test_reduce_with_group(self, mock_stream_reduce):
        """测试指定 group 调用 reduce
        Test reduce with specified group"""
        tensor = MagicMock()
        group = MagicMock()
        mock_stream_reduce.return_value = MagicMock()
        reduce_mod.reduce(tensor, dst=0, group=group)
        call_kwargs = mock_stream_reduce.call_args[1]
        self.assertEqual(call_kwargs['group'], group)

    @patch.object(stream_mod, 'reduce')
    def test_reduce_async(self, mock_stream_reduce):
        """测试异步模式调用 reduce
        Test reduce with async mode"""
        tensor = MagicMock()
        mock_stream_reduce.return_value = MagicMock()
        reduce_mod.reduce(tensor, dst=0, sync_op=False)
        call_kwargs = mock_stream_reduce.call_args[1]
        self.assertEqual(call_kwargs['sync_op'], False)

    @patch.object(
        reduce_mod.paddle.base.core, 'nccl_version', return_value=21000
    )
    @patch.object(stream_mod, 'reduce')
    def test_reduce_avg_nccl_ge(self, mock_stream_reduce, mock_nccl):
        """测试 AVG 操作且 nccl >= 2.10 时直接调用 stream
        Test reduce AVG with nccl >= 2.10 calls stream directly"""
        tensor = MagicMock()
        mock_stream_reduce.return_value = MagicMock()
        reduce_mod.reduce(tensor, dst=0, op=ReduceOp.AVG)
        call_kwargs = mock_stream_reduce.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.AVG)

    @patch('paddle.distributed.collective._get_global_group')
    @patch.object(stream_mod, 'reduce')
    @patch.object(
        reduce_mod.paddle.base.core, 'nccl_version', return_value=20900
    )
    def test_reduce_avg_nccl_lt_no_group(
        self, mock_nccl, mock_stream, mock_get_global
    ):
        """测试 AVG 操作且 nccl < 2.10 且无 group 时，先 scale 再 SUM
        Test reduce AVG with nccl < 2.10 and no group scales tensor then uses SUM"""
        tensor = MagicMock()
        global_group = MagicMock()
        global_group.nranks = 3
        mock_get_global.return_value = global_group
        mock_task = MagicMock()
        mock_stream.return_value = mock_task

        result = reduce_mod.reduce(tensor, dst=0, op=ReduceOp.AVG)
        tensor.scale_.assert_called_once_with(1.0 / 3)
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.SUM)
        self.assertEqual(call_kwargs['group'], global_group)

    @patch.object(stream_mod, 'reduce')
    @patch.object(
        reduce_mod.paddle.base.core, 'nccl_version', return_value=20900
    )
    def test_reduce_avg_nccl_lt_with_group(self, mock_nccl, mock_stream):
        """测试 AVG 操作且 nccl < 2.10 且有 group 时，使用指定 group
        Test reduce AVG with nccl < 2.10 and specified group"""
        tensor = MagicMock()
        group = MagicMock()
        group.nranks = 5
        mock_stream.return_value = MagicMock()

        reduce_mod.reduce(tensor, dst=2, op=ReduceOp.AVG, group=group)
        tensor.scale_.assert_called_once_with(1.0 / 5)
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs['op'], ReduceOp.SUM)
        self.assertEqual(call_kwargs['group'], group)
        self.assertEqual(call_kwargs['dst'], 2)


if __name__ == '__main__':
    unittest.main()
