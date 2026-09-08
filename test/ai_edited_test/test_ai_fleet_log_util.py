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

# [AUTO-GENERATED] Test file for paddle.distributed.fleet.utils.log_util
# Target file: paddle/distributed/fleet/utils/log_util.py
# 覆盖模块: paddle/distributed/fleet/utils/log_util.py
# Covered module: paddle/distributed/fleet/utils/log_util.py

import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from paddle.distributed.fleet.utils.log_util import (
    DistributedLogger,
    check_memory_usage,
    get_log_level_code,
    get_log_level_name,
    get_rotate_file_logger,
    get_sync_logger,
    layer_to_str,
    set_log_level,
    sync_rotate_logger,
)


class TestSetLogLevel(unittest.TestCase):
    """测试 set_log_level 函数 / Test set_log_level function"""

    def test_set_log_level_str(self):
        """测试使用字符串设置日志级别 / Test setting log level with string"""
        set_log_level("DEBUG")
        self.assertEqual(get_log_level_code(), logging.DEBUG)

    def test_set_log_level_int(self):
        """测试使用整数设置日志级别 / Test setting log level with integer"""
        set_log_level(logging.WARNING)
        self.assertEqual(get_log_level_code(), logging.WARNING)

    def test_set_log_level_info(self):
        """测试设置 INFO 日志级别 / Test setting INFO log level"""
        set_log_level("INFO")
        self.assertEqual(get_log_level_code(), logging.INFO)

    def test_set_log_level_case_insensitive(self):
        """测试日志级别字符串大小写不敏感 / Test log level string is case-insensitive"""
        set_log_level("debug")
        self.assertEqual(get_log_level_code(), logging.DEBUG)

    def test_set_log_level_error(self):
        """测试设置 ERROR 日志级别 / Test setting ERROR log level"""
        set_log_level("ERROR")
        self.assertEqual(get_log_level_code(), logging.ERROR)

    def test_set_log_level_critical(self):
        """测试设置 CRITICAL 日志级别 / Test setting CRITICAL log level"""
        set_log_level(logging.CRITICAL)
        self.assertEqual(get_log_level_code(), logging.CRITICAL)

    def test_set_log_level_invalid_type(self):
        """测试设置无效类型的日志级别抛出异常 / Test invalid type raises assertion"""
        with self.assertRaises(AssertionError):
            set_log_level(3.14)


class TestGetLogLevel(unittest.TestCase):
    """测试 get_log_level_code 和 get_log_level_name 函数
    Test get_log_level_code and get_log_level_name functions"""

    def test_get_log_level_code(self):
        """测试获取日志级别代码 / Test getting log level code"""
        set_log_level("WARNING")
        code = get_log_level_code()
        self.assertEqual(code, logging.WARNING)

    def test_get_log_level_name(self):
        """测试获取日志级别名称 / Test getting log level name"""
        set_log_level("WARNING")
        name = get_log_level_name()
        self.assertEqual(name, "WARNING")

    def test_get_log_level_name_debug(self):
        """测试获取 DEBUG 日志级别名称 / Test getting DEBUG log level name"""
        set_log_level("DEBUG")
        name = get_log_level_name()
        self.assertEqual(name, "DEBUG")

    def test_get_log_level_name_info(self):
        """测试获取 INFO 日志级别名称 / Test getting INFO log level name"""
        set_log_level("INFO")
        name = get_log_level_name()
        self.assertEqual(name, "INFO")

    def test_get_log_level_name_error(self):
        """测试获取 ERROR 日志级别名称 / Test getting ERROR log level name"""
        set_log_level("ERROR")
        name = get_log_level_name()
        self.assertEqual(name, "ERROR")


class TestLayerToStr(unittest.TestCase):
    """测试 layer_to_str 函数 / Test layer_to_str function"""

    def test_layer_to_str_no_args(self):
        """测试无参数的 layer_to_str / Test layer_to_str with no arguments"""
        result = layer_to_str("Linear")
        self.assertEqual(result, "Linear()")

    def test_layer_to_str_with_args(self):
        """测试带位置参数的 layer_to_str / Test layer_to_str with positional arguments"""
        result = layer_to_str("Linear", 10, 20)
        self.assertEqual(result, "Linear(10, 20)")

    def test_layer_to_str_with_kwargs(self):
        """测试带关键字参数的 layer_to_str / Test layer_to_str with keyword arguments"""
        result = layer_to_str("Linear", in_features=10, out_features=20)
        self.assertEqual(result, "Linear(in_features=10, out_features=20)")

    def test_layer_to_str_with_both(self):
        """测试同时带位置参数和关键字参数 / Test layer_to_str with both args and kwargs"""
        result = layer_to_str("Linear", 10, out_features=20)
        self.assertEqual(result, "Linear(10, out_features=20)")

    def test_layer_to_str_single_arg(self):
        """测试单个位置参数 / Test layer_to_str with single arg"""
        result = layer_to_str("Dropout", 0.5)
        self.assertEqual(result, "Dropout(0.5)")

    def test_layer_to_str_empty_args_kwargs(self):
        """测试空列表参数 / Test layer_to_str with empty list args"""
        result = layer_to_str("Layer")
        self.assertEqual(result, "Layer()")


class TestDistributedLogger(unittest.TestCase):
    """测试 DistributedLogger 类 / Test DistributedLogger class"""

    def test_distributed_logger_init(self):
        """测试 DistributedLogger 初始化 / Test DistributedLogger initialization"""
        logger = DistributedLogger("test_logger")
        self.assertEqual(logger.name, "test_logger")

    def test_distributed_logger_init_with_level(self):
        """测试带日志级别的 DistributedLogger 初始化
        Test DistributedLogger initialization with level"""
        logger = DistributedLogger("test_logger_level", level=logging.DEBUG)
        self.assertEqual(logger.level, logging.DEBUG)

    def test_distributed_logger_info(self):
        """测试 DistributedLogger info 方法 / Test DistributedLogger info method"""
        with patch('paddle.device.synchronize'):
            logger = DistributedLogger("test_logger_info")
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            logger.info("test message")

    def test_distributed_logger_propagate_false(self):
        """测试 DistributedLogger propagate 设置
        Test DistributedLogger propagate setting"""
        logger = DistributedLogger("test_propagate")
        logger.propagate = False
        self.assertFalse(logger.propagate)


class TestGetRotateFileLogger(unittest.TestCase):
    """测试 get_rotate_file_logger 函数 / Test get_rotate_file_logger function"""

    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._test_dir = tempfile.mkdtemp()
        os.chdir(self._test_dir)
        self._orig_env = os.environ.get("FLAGS_selected_gpus")
        os.environ["FLAGS_selected_gpus"] = "0"

    def tearDown(self):
        os.chdir(self._orig_cwd)
        if self._orig_env is not None:
            os.environ["FLAGS_selected_gpus"] = self._orig_env
        else:
            os.environ.pop("FLAGS_selected_gpus", None)
        hp_dir = os.path.join(self._test_dir, "hybrid_parallel")
        if os.path.exists(hp_dir):
            shutil.rmtree(hp_dir)
        shutil.rmtree(self._test_dir, ignore_errors=True)

    def test_get_rotate_file_logger(self):
        """测试获取轮转文件日志器 / Test getting rotating file logger"""
        logger = get_rotate_file_logger("INFO", "test_rotate")
        self.assertIsInstance(logger, DistributedLogger)
        self.assertTrue(len(logger.handlers) > 0)

    def test_get_rotate_file_logger_creates_dir(self):
        """测试 get_rotate_file_logger 创建日志目录
        Test get_rotate_file_logger creates log directory"""
        logger = get_rotate_file_logger("DEBUG", "test_dir_create")
        log_dir = os.path.join(os.getcwd(), "hybrid_parallel")
        self.assertTrue(os.path.exists(log_dir))

    def test_get_rotate_file_logger_debug_level(self):
        """测试 DEBUG 级别的轮转日志器 / Test DEBUG level rotating logger"""
        logger = get_rotate_file_logger(logging.DEBUG, "test_debug")
        self.assertEqual(logger.level, logging.DEBUG)

    def test_get_rotate_file_logger_with_device_id(self):
        """测试指定 GPU 设备 ID 的日志器 / Test logger with specified GPU device ID"""
        os.environ["FLAGS_selected_gpus"] = "3"
        logger = get_rotate_file_logger("INFO", "test_device")
        self.assertIsInstance(logger, DistributedLogger)

    def test_get_rotate_file_logger_propagate_false(self):
        """测试轮转日志器不向上传播 / Test rotating logger does not propagate"""
        logger = get_rotate_file_logger("INFO", "test_propagate")
        self.assertFalse(logger.propagate)


class TestGetSyncLogger(unittest.TestCase):
    """测试 get_sync_logger 函数 / Test get_sync_logger function"""

    @patch('paddle.device.synchronize')
    def test_get_sync_logger(self, mock_sync):
        """测试获取同步日志器 / Test getting sync logger"""
        logger = get_sync_logger()
        self.assertIsNotNone(logger)
        mock_sync.assert_called_once()


class TestSyncRotateLogger(unittest.TestCase):
    """测试 sync_rotate_logger 函数 / Test sync_rotate_logger function"""

    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._test_dir = tempfile.mkdtemp()
        os.chdir(self._test_dir)
        self._orig_env = os.environ.get("FLAGS_selected_gpus")
        os.environ["FLAGS_selected_gpus"] = "0"

    def tearDown(self):
        os.chdir(self._orig_cwd)
        if self._orig_env is not None:
            os.environ["FLAGS_selected_gpus"] = self._orig_env
        else:
            os.environ.pop("FLAGS_selected_gpus", None)
        hp_dir = os.path.join(self._test_dir, "hybrid_parallel")
        if os.path.exists(hp_dir):
            shutil.rmtree(hp_dir)
        shutil.rmtree(self._test_dir, ignore_errors=True)
        # Reset global state
        import paddle.distributed.fleet.utils.log_util as log_mod

        log_mod.g_sync_rotate_logger = None

    def test_sync_rotate_logger(self):
        """测试获取同步轮转日志器 / Test getting sync rotate logger"""
        logger = sync_rotate_logger()
        self.assertIsInstance(logger, DistributedLogger)

    def test_sync_rotate_logger_cached(self):
        """测试同步轮转日志器缓存 / Test sync rotate logger caching"""
        logger1 = sync_rotate_logger()
        logger2 = sync_rotate_logger()
        self.assertIs(logger1, logger2)


class TestCheckMemoryUsage(unittest.TestCase):
    """测试 check_memory_usage 函数 / Test check_memory_usage function"""

    @patch('paddle.distributed.fleet.utils.log_util.logger')
    @patch('subprocess.run')
    @patch('paddle.device.cuda', create=True)
    @patch('paddle.device.cpu', create=True)
    def test_check_memory_usage_basic(
        self, mock_cpu, mock_cuda, mock_run, mock_logger
    ):
        """测试基本内存使用检查 / Test basic memory usage check"""
        mock_cuda.max_memory_allocated.return_value = 0
        mock_cuda.max_memory_reserved.return_value = 0
        mock_cuda.memory_allocated.return_value = 0
        mock_cuda.memory_reserved.return_value = 0
        mock_run.return_value = MagicMock(
            stdout="              total        used        free      shared  buff/cache   available\nMem:        64000000     10000000     54000000        1000     2000000     52000000\nSwap:             0           0           0"
        )
        check_memory_usage("test_msg")
        self.assertTrue(mock_logger.info.called)

    @patch('paddle.distributed.fleet.utils.log_util.logger')
    @patch('subprocess.run')
    @patch('paddle.device.cuda', create=True)
    @patch('paddle.device.cpu', create=True)
    def test_check_memory_usage_with_pinned(
        self, mock_cpu, mock_cuda, mock_run, mock_logger
    ):
        """测试包含 pinned 内存的检查 / Test memory check with pinned memory"""
        mock_cuda.max_memory_allocated.return_value = 1024 * 1024 * 1024
        mock_cuda.max_memory_reserved.return_value = 2 * 1024 * 1024 * 1024
        mock_cuda.memory_allocated.return_value = 512 * 1024 * 1024
        mock_cuda.memory_reserved.return_value = 1024 * 1024 * 1024
        mock_cuda.max_pinned_memory_allocated.return_value = 0
        mock_cuda.max_pinned_memory_reserved.return_value = 0
        mock_cuda.pinned_memory_allocated.return_value = 0
        mock_cuda.pinned_memory_reserved.return_value = 0
        mock_run.return_value = MagicMock(
            stdout="              total        used        free      shared  buff/cache   available\nMem:        64000000     10000000     54000000        1000     2000000     52000000\nSwap:             0           0           0"
        )
        check_memory_usage("test_pinned")
        self.assertTrue(mock_logger.info.called)

    @patch('paddle.distributed.fleet.utils.log_util.logger')
    @patch('subprocess.run')
    @patch('paddle.device.cuda', create=True)
    @patch('paddle.device.cpu', create=True)
    def test_check_memory_usage_with_cpu(
        self, mock_cpu, mock_cuda, mock_run, mock_logger
    ):
        """测试包含 CPU 内存的检查 / Test memory check with CPU memory"""
        mock_cuda.max_memory_allocated.return_value = 0
        mock_cuda.max_memory_reserved.return_value = 0
        mock_cuda.memory_allocated.return_value = 0
        mock_cuda.memory_reserved.return_value = 0
        mock_cpu.max_memory_allocated.return_value = 0
        mock_cpu.max_memory_reserved.return_value = 0
        mock_cpu.memory_allocated.return_value = 0
        mock_cpu.memory_reserved.return_value = 0
        mock_run.return_value = MagicMock(
            stdout="              total        used        free      shared  buff/cache   available\nMem:        64000000     10000000     54000000        1000     2000000     52000000\nSwap:             0           0           0"
        )
        check_memory_usage("test_cpu")
        self.assertTrue(mock_logger.info.called)


if __name__ == '__main__':
    unittest.main()
