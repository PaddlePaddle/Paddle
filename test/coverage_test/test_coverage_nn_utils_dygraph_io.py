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

# [AUTO-GENERATED] Test file for paddle.nn.utils.dygraph_utils and paddle.io.multiprocess_utils
# 覆盖模块: paddle/nn/utils/dygraph_utils.py, paddle/io/multiprocess_utils.py
# 未覆盖行: dygraph_utils: 30,31,33; multiprocess_utils: 34,36,37,38,39,45,47,53,66,67,68,70,74,81,82,83,84,85,90,91,92,96,97,98,99,135
# Covered module: paddle/nn/utils/dygraph_utils.py, paddle/io/multiprocess_utils.py
# Uncovered lines: dygraph_utils: 30,31,33; multiprocess_utils: 34,36,37,38,39,45,47,53,66,67,68,70,74,81,82,83,84,85,90,91,92,96,97,98,99,135

import queue
import unittest

import numpy as np

import paddle
from paddle.io.multiprocess_utils import (
    MP_STATUS_CHECK_INTERVAL,
    CleanupFuncRegistrar,
    _clear_multiprocess_queue_set,
    _set_SIGCHLD_handler,
    multiprocess_queue_set,
)
from paddle.nn.utils.dygraph_utils import _append_bias_in_dygraph


class TestAppendBiasInDygraph(unittest.TestCase):
    """测试 _append_bias_in_dygraph 函数
    Test _append_bias_in_dygraph function"""

    def test_append_bias_none(self):
        """测试 bias=None 时直接返回输入
        Test _append_bias_in_dygraph returns input when bias is None"""
        input_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        result = _append_bias_in_dygraph(input_tensor, bias=None)
        np.testing.assert_allclose(result.numpy(), input_tensor.numpy())

    def test_append_bias_add(self):
        """测试 bias 不为 None 时的加法操作
        Test _append_bias_in_dygraph adds bias to input"""
        input_tensor = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        bias = paddle.to_tensor([0.5, 1.0])
        result = _append_bias_in_dygraph(input_tensor, bias)
        expected = input_tensor + bias
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6)

    def test_append_bias_1d_input(self):
        """测试1D输入的 bias append
        Test _append_bias_in_dygraph with 1D input"""
        input_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        bias = paddle.to_tensor([0.1, 0.2, 0.3])
        result = _append_bias_in_dygraph(input_tensor, bias, axis=0)
        expected = input_tensor + bias
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6)


class TestMultiprocessUtils(unittest.TestCase):
    """测试 paddle.io.multiprocess_utils 模块
    Test paddle.io.multiprocess_utils module"""

    def test_mp_status_check_interval(self):
        """测试 MP_STATUS_CHECK_INTERVAL 常量
        Test MP_STATUS_CHECK_INTERVAL constant"""
        self.assertEqual(MP_STATUS_CHECK_INTERVAL, 5.0)

    def test_clear_multiprocess_queue_set_empty(self):
        """测试清空空的队列集合
        Test clearing empty queue set"""
        original = multiprocess_queue_set.copy()
        try:
            multiprocess_queue_set.clear()
            # Should not raise
            _clear_multiprocess_queue_set()
        finally:
            multiprocess_queue_set.update(original)

    def test_clear_multiprocess_queue_set_with_data(self):
        """测试清空有数据的队列集合
        Test clearing queue set with data"""
        original = multiprocess_queue_set.copy()
        try:
            multiprocess_queue_set.clear()
            q = queue.Queue()
            q.put("test_data")
            multiprocess_queue_set.add(q)
            _clear_multiprocess_queue_set()
            # Queue should be empty after clearing
            self.assertTrue(q.empty())
        finally:
            multiprocess_queue_set.clear()
            multiprocess_queue_set.update(original)

    def test_cleanup_func_registrar_register_callable(self):
        """测试 CleanupFuncRegistrar 注册可调用对象
        Test CleanupFuncRegistrar registers callable objects"""
        call_count = 0

        def test_func():
            nonlocal call_count
            call_count += 1

        # Reset the registrar state
        CleanupFuncRegistrar._executed_func_set = set()
        CleanupFuncRegistrar._registered_func_set = set()

        CleanupFuncRegistrar.register(test_func)

        # Should be in registered set
        self.assertIn(test_func, CleanupFuncRegistrar._registered_func_set)

        # Cleanup
        CleanupFuncRegistrar._registered_func_set.discard(test_func)
        CleanupFuncRegistrar._executed_func_set.discard(test_func)

    def test_cleanup_func_registrar_non_callable(self):
        """测试 CleanupFuncRegistrar 注册不可调用对象时报错
        Test CleanupFuncRegistrar raises error for non-callable"""
        CleanupFuncRegistrar._executed_func_set = set()
        CleanupFuncRegistrar._registered_func_set = set()

        with self.assertRaises(TypeError):
            CleanupFuncRegistrar.register(123)

    def test_set_sigchld_handler(self):
        """测试 _set_SIGCHLD_handler 函数
        Test _set_SIGCHLD_handler function"""
        import paddle.io.multiprocess_utils as mp_utils

        original_state = mp_utils._SIGCHLD_handler_set
        try:
            mp_utils._SIGCHLD_handler_set = False
            _set_SIGCHLD_handler()
            self.assertTrue(mp_utils._SIGCHLD_handler_set)
            # Calling again should not change anything
            _set_SIGCHLD_handler()
            self.assertTrue(mp_utils._SIGCHLD_handler_set)
        finally:
            mp_utils._SIGCHLD_handler_set = original_state


if __name__ == '__main__':
    unittest.main()
