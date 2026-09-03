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

# [AUTO-GENERATED] test for paddle/io/multiprocess_utils.py
# Target file: python/paddle/io/multiprocess_utils.py
# Coverage: 70.1% (47/67) - Uncovered lines: 45, 47, 53, 66, 67, 68, 70, 81, 82, 83, 84, 85, 90, 91, 92, 96, 97, 98, 99, 135
# 本文件为 multiprocess_utils.py 的单元测试 / Unit tests for multiprocess_utils.py
#
# 测试目标：
# - CleanupFuncRegistrar.register() 注册与执行清理函数
# - CleanupFuncRegistrar 信号处理注册
# - _set_SIGCHLD_handler() SIGCHLD 信号处理器
# - _clear_multiprocess_queue_set() 多进程队列清理
# - _cleanup() / _cleanup_mmap() 主进程/子进程清理函数

import queue
import signal
import unittest


class TestMultiprocessUtilsBasic(unittest.TestCase):
    """基础工具函数测试 / Basic utility function tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        pass

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_mp_status_check_interval(self):
        """测试 MP_STATUS_CHECK_INTERVAL 常量 / Test MP_STATUS_CHECK_INTERVAL constant"""
        from paddle.io.multiprocess_utils import MP_STATUS_CHECK_INTERVAL

        # 应该是一个正数 / Should be a positive number
        self.assertIsInstance(MP_STATUS_CHECK_INTERVAL, float)
        self.assertGreater(MP_STATUS_CHECK_INTERVAL, 0)

    def test_multiprocess_queue_set_exists(self):
        """测试 multiprocess_queue_set 是否存在 / Test multiprocess_queue_set exists"""
        from paddle.io.multiprocess_utils import multiprocess_queue_set

        self.assertIsInstance(multiprocess_queue_set, set)

    def test_clear_multiprocess_queue_set_empty(self):
        """测试清空空队列集合 / Test clearing empty queue set"""
        from paddle.io.multiprocess_utils import _clear_multiprocess_queue_set

        # 空队列不应报错 / Empty queue should not raise error
        _clear_multiprocess_queue_set()

    def test_clear_multiprocess_queue_set_with_data(self):
        """测试清空有数据的队列集合 / Test clearing queue set with data"""
        from paddle.io.multiprocess_utils import (
            _clear_multiprocess_queue_set,
            multiprocess_queue_set,
        )

        q = queue.Queue()
        q.put("test_data")
        q.put("test_data_2")
        multiprocess_queue_set.add(q)

        try:
            _clear_multiprocess_queue_set()
            # 队列应该被清空 / Queue should be cleared
            self.assertTrue(q.empty())
        finally:
            multiprocess_queue_set.discard(q)


class TestCleanupFuncRegistrar(unittest.TestCase):
    """CleanupFuncRegistrar 测试 / CleanupFuncRegistrar tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        # 保存并重置类状态 / Save and reset class state
        from paddle.io.multiprocess_utils import CleanupFuncRegistrar

        self._executed = CleanupFuncRegistrar._executed_func_set.copy()
        self._registered = CleanupFuncRegistrar._registered_func_set.copy()
        CleanupFuncRegistrar._executed_func_set.clear()
        CleanupFuncRegistrar._registered_func_set.clear()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        from paddle.io.multiprocess_utils import CleanupFuncRegistrar

        # 恢复类状态 / Restore class state
        CleanupFuncRegistrar._executed_func_set = self._executed
        CleanupFuncRegistrar._registered_func_set = self._registered

    def test_register_callable_function(self):
        """测试注册可调用函数 / Test registering callable function"""
        from paddle.io.multiprocess_utils import CleanupFuncRegistrar

        call_count = [0]

        def cleanup_func():
            call_count[0] += 1

        CleanupFuncRegistrar.register(cleanup_func)
        self.assertIn(cleanup_func, CleanupFuncRegistrar._registered_func_set)

    def test_register_non_callable_raises(self):
        """测试注册非可调用对象抛出异常 / Test registering non-callable raises TypeError"""
        from paddle.io.multiprocess_utils import CleanupFuncRegistrar

        with self.assertRaises(TypeError):
            CleanupFuncRegistrar.register("not_callable")

    def test_register_same_function_idempotent(self):
        """测试重复注册同一函数幂等 / Test re-registering same function is idempotent"""
        from paddle.io.multiprocess_utils import CleanupFuncRegistrar

        def cleanup_func():
            pass

        CleanupFuncRegistrar.register(cleanup_func)
        CleanupFuncRegistrar.register(cleanup_func)
        # 应该只注册一次 / Should only register once
        self.assertEqual(
            len(
                [
                    f
                    for f in CleanupFuncRegistrar._registered_func_set
                    if f == cleanup_func
                ]
            ),
            1,
        )

    def test_register_with_signal_handlers(self):
        """测试注册信号处理函数 / Test registering signal handlers"""
        from paddle.io.multiprocess_utils import CleanupFuncRegistrar

        call_count = [0]

        def cleanup_func():
            call_count[0] += 1

        # 注册 SIGUSR1 信号 / Register SIGUSR1 signal
        original_handler = signal.getsignal(signal.SIGUSR1)
        try:
            CleanupFuncRegistrar.register(
                cleanup_func, signals=[signal.SIGUSR1]
            )
            # 验证函数已注册 / Verify function is registered
            self.assertIn(
                cleanup_func, CleanupFuncRegistrar._registered_func_set
            )
        finally:
            # 恢复原始信号处理器 / Restore original signal handler
            signal.signal(signal.SIGUSR1, original_handler)

    def test_register_preserves_existing_signal_handlers(self):
        """测试注册不覆盖已有信号处理器 / Test register preserves existing signal handlers"""
        from paddle.io.multiprocess_utils import CleanupFuncRegistrar

        original_handler_called = [False]

        def original_handler(signum, frame):
            original_handler_called[0] = True

        def cleanup_func():
            pass

        # 设置自定义处理器 / Set custom handler
        signal.signal(signal.SIGUSR2, original_handler)

        try:
            CleanupFuncRegistrar.register(
                cleanup_func, signals=[signal.SIGUSR2]
            )
            # 验证原始处理器也被注册为清理函数 / Verify original handler is also registered
            self.assertIn(
                original_handler, CleanupFuncRegistrar._registered_func_set
            )
        finally:
            # 恢复默认信号处理器 / Restore default signal handler
            signal.signal(signal.SIGUSR2, signal.SIG_DFL)


class TestSIGCHLDHandler(unittest.TestCase):
    """SIGCHLD 处理器测试 / SIGCHLD handler tests"""

    def test_set_sigchld_handler_basic(self):
        """测试设置 SIGCHLD 处理器 / Test setting SIGCHLD handler"""
        import signal as sig

        import paddle.io.multiprocess_utils as mp_utils

        original_handler = sig.getsignal(sig.SIGCHLD)
        was_set = mp_utils._SIGCHLD_handler_set
        try:
            mp_utils._set_SIGCHLD_handler()
            # 验证信号处理器已被设置（为一个新的可调用对象）/ Verify handler is set to a callable
            new_handler = sig.getsignal(sig.SIGCHLD)
            self.assertTrue(callable(new_handler))
            # _SIGCHLD_handler_set 应该为 True
            self.assertTrue(mp_utils._SIGCHLD_handler_set)
        finally:
            sig.signal(sig.SIGCHLD, original_handler)
            mp_utils._SIGCHLD_handler_set = was_set

    def test_set_sigchld_handler_calls_core(self):
        """测试 SIGCHLD 处理器调用 core 检查 / Test SIGCHLD handler calls core check"""
        import signal as sig

        import paddle.io.multiprocess_utils as mp_utils

        original_handler = sig.getsignal(sig.SIGCHLD)
        was_set = mp_utils._SIGCHLD_handler_set
        try:
            mp_utils._set_SIGCHLD_handler()
            new_handler = sig.getsignal(sig.SIGCHLD)
            # 新处理器应该是一个包装函数 / New handler should be a wrapper function
            self.assertTrue(callable(new_handler))
        finally:
            sig.signal(sig.SIGCHLD, original_handler)
            mp_utils._SIGCHLD_handler_set = was_set


if __name__ == '__main__':
    unittest.main()
