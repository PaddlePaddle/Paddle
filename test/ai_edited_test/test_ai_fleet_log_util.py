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
# 覆盖模块: paddle/distributed/fleet/utils/log_util.py
# 未覆盖行: 64,81,84,85,89,90,92,93,94,96,97,103,106,107,108,122,123,124,128,129,130,133,136,139,140,141,142,143,145,146,147,150,153,156,159,160,161,162,164,167
# Covered module: paddle/distributed/fleet/utils/log_util.py
# Uncovered lines: 64,81,84,85,89,90,92,93,94,96,97,103,106,107,108,122,123,124,128,129,130,133,136,139,140,141,142,143,145,146,147,150,153,156,159,160,161,162,164,167

import logging
import os
import shutil
import tempfile
import unittest

from paddle.distributed.fleet.utils.log_util import (
    DistributedLogger,
    get_log_level_code,
    get_log_level_name,
    get_rotate_file_logger,
    layer_to_str,
    set_log_level,
)


class TestSetLogLevel(unittest.TestCase):
    """测试 set_log_level 函数
    Test set_log_level function"""

    def test_set_log_level_str(self):
        """测试使用字符串设置日志级别
        Test setting log level with string"""
        set_log_level("DEBUG")
        self.assertEqual(get_log_level_code(), logging.DEBUG)

    def test_set_log_level_int(self):
        """测试使用整数设置日志级别
        Test setting log level with integer"""
        set_log_level(logging.WARNING)
        self.assertEqual(get_log_level_code(), logging.WARNING)

    def test_set_log_level_info(self):
        """测试设置 INFO 日志级别
        Test setting INFO log level"""
        set_log_level("INFO")
        self.assertEqual(get_log_level_code(), logging.INFO)

    def test_set_log_level_case_insensitive(self):
        """测试日志级别字符串大小写不敏感
        Test log level string is case-insensitive"""
        set_log_level("debug")
        self.assertEqual(get_log_level_code(), logging.DEBUG)


class TestGetLogLevel(unittest.TestCase):
    """测试 get_log_level_code 和 get_log_level_name 函数
    Test get_log_level_code and get_log_level_name functions"""

    def test_get_log_level_code(self):
        """测试获取日志级别代码
        Test getting log level code"""
        set_log_level("WARNING")
        code = get_log_level_code()
        self.assertEqual(code, logging.WARNING)

    def test_get_log_level_name(self):
        """测试获取日志级别名称
        Test getting log level name"""
        set_log_level("WARNING")
        name = get_log_level_name()
        self.assertEqual(name, "WARNING")

    def test_get_log_level_name_debug(self):
        """测试获取 DEBUG 日志级别名称
        Test getting DEBUG log level name"""
        set_log_level("DEBUG")
        name = get_log_level_name()
        self.assertEqual(name, "DEBUG")


class TestLayerToStr(unittest.TestCase):
    """测试 layer_to_str 函数
    Test layer_to_str function"""

    def test_layer_to_str_no_args(self):
        """测试无参数的 layer_to_str
        Test layer_to_str with no arguments"""
        result = layer_to_str("Linear")
        self.assertEqual(result, "Linear()")

    def test_layer_to_str_with_args(self):
        """测试带位置参数的 layer_to_str
        Test layer_to_str with positional arguments"""
        result = layer_to_str("Linear", 10, 20)
        self.assertEqual(result, "Linear(10, 20)")

    def test_layer_to_str_with_kwargs(self):
        """测试带关键字参数的 layer_to_str
        Test layer_to_str with keyword arguments"""
        result = layer_to_str("Linear", in_features=10, out_features=20)
        self.assertEqual(result, "Linear(in_features=10, out_features=20)")

    def test_layer_to_str_with_both(self):
        """测试同时带位置参数和关键字参数的 layer_to_str
        Test layer_to_str with both positional and keyword arguments"""
        result = layer_to_str("Linear", 10, out_features=20)
        self.assertEqual(result, "Linear(10, out_features=20)")


class TestDistributedLogger(unittest.TestCase):
    """测试 DistributedLogger 类
    Test DistributedLogger class"""

    def test_distributed_logger_init(self):
        """测试 DistributedLogger 初始化
        Test DistributedLogger initialization"""
        logger = DistributedLogger("test_logger")
        self.assertEqual(logger.name, "test_logger")

    def test_distributed_logger_info(self):
        """测试 DistributedLogger info 方法
        Test DistributedLogger info method"""
        logger = DistributedLogger("test_logger_info")
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        # Should not raise
        logger.info("test message")


class TestGetRotateFileLogger(unittest.TestCase):
    """测试 get_rotate_file_logger 函数
    Test get_rotate_file_logger function"""

    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._test_dir = tempfile.mkdtemp()
        os.chdir(self._test_dir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        if os.path.exists(os.path.join(self._test_dir, "hybrid_parallel")):
            shutil.rmtree(os.path.join(self._test_dir, "hybrid_parallel"))
        shutil.rmtree(self._test_dir)

    def test_get_rotate_file_logger(self):
        """测试获取轮转文件日志器
        Test getting rotating file logger"""
        logger = get_rotate_file_logger("INFO", "test_rotate")
        self.assertIsInstance(logger, DistributedLogger)
        self.assertTrue(len(logger.handlers) > 0)

    def test_get_rotate_file_logger_creates_dir(self):
        """测试 get_rotate_file_logger 创建日志目录
        Test get_rotate_file_logger creates log directory"""
        logger = get_rotate_file_logger("DEBUG", "test_dir_create")
        log_dir = os.path.join(os.getcwd(), "hybrid_parallel")
        self.assertTrue(os.path.exists(log_dir))


if __name__ == '__main__':
    unittest.main()
