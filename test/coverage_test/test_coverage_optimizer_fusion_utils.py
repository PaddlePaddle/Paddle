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

# [AUTO-GENERATED] Test file for paddle.optimizer.fusion_utils
# 覆盖模块: paddle/optimizer/fusion_utils.py
# Uncovered lines: 44,47,51,59,60,62-66,206-214,229,232,233,236,237,241,243,246,250,251,255,259-261,264,268-271,274,282

import unittest

import paddle
from paddle.optimizer.fusion_utils import (
    FusionStorage,
    get_align,
    get_current_device_type,
)


class TestGetDeviceType(unittest.TestCase):
    """测试 get_current_device_type 函数
    Test get_current_device_type function"""

    def test_get_device_type(self):
        """测试获取当前设备类型
        Test getting current device type"""
        device_type = get_current_device_type()
        self.assertIn(device_type, ['gpu', 'xpu', 'npu', 'cpu', 'unknown'])


class TestGetAlign(unittest.TestCase):
    """测试 get_align 函数
    Test get_align function"""

    def test_get_align_basic(self):
        """测试基本的 get_align
        Test basic get_align"""
        t = paddle.randn([10])
        result = get_align(t)
        self.assertIsInstance(result, (int, np.integer))


class TestFusionStorage(unittest.TestCase):
    """测试 FusionStorage 类
    Test FusionStorage class"""

    def test_fusion_storage_init(self):
        """测试 FusionStorage 初始化
        Test FusionStorage initialization"""
        storage = FusionStorage(
            accumulators={},
            master_weights={},
        )
        self.assertIsNotNone(storage)

    def test_fusion_storage_with_merged(self):
        """测试带 merged_model_params 的 FusionStorage
        Test FusionStorage with merged_model_params"""
        storage = FusionStorage(
            accumulators={},
            master_weights={},
            merged_model_params={},
        )
        self.assertIsNotNone(storage)

    def test_fusion_storage_invalid_type(self):
        """测试 FusionStorage 类型检查
        Test FusionStorage type checking"""
        with self.assertRaises(AssertionError):
            FusionStorage(accumulators="invalid", master_weights={})


import numpy as np

if __name__ == '__main__':
    unittest.main()
