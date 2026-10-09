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

# [AUTO-GENERATED] Test file for paddle.distributed.fleet.layers.mpu
# 覆盖模块: paddle/distributed/fleet/layers/mpu/mp_layers.py, paddle/distributed/fleet/layers/mpu/mp_ops.py, paddle/distributed/fleet/layers/mpu/random.py
# 未覆盖行: mp_layers: 38,42,45,47,214-218,227,228,230,234-236,238,241,242,246,248,249,254,255,257-259,266-268,274; mp_ops: 47,67-71,75,92,98,138,149-151,153,160,172,190,249,258,274,362-366,372,378,381,384,387; random: 50,53,75,86,198,199,201,202,205,206,208,209,213,217,222,223,236,238,239,243,244,248,249,251,252,256,266
# Covered module: paddle/distributed/fleet/layers/mpu/mp_layers.py, mp_ops.py, random.py
# Uncovered lines: mp_layers: 38-274; mp_ops: 47-387; random: 50-266

import unittest

from paddle.distributed.fleet.layers.mpu import (
    mp_layers,
    mp_ops,
    random,
)


class TestMPLayersImport(unittest.TestCase):
    """测试 mp_layers 模块导入
    Test mp_layers module import"""

    def test_mp_layers_module_importable(self):
        """测试 mp_layers 模块可导入
        Test mp_layers module is importable"""
        self.assertIsNotNone(mp_layers)

    def test_mp_ops_module_importable(self):
        """测试 mp_ops 模块可导入
        Test mp_ops module is importable"""
        self.assertIsNotNone(mp_ops)

    def test_random_module_importable(self):
        """测试 random 模块可导入
        Test random module is importable"""
        self.assertIsNotNone(random)


class TestMPULayerClasses(unittest.TestCase):
    """测试 MPU 层类
    Test MPU layer classes"""

    def test_column_parallel_linear_import(self):
        """测试 ColumnParallelLinear 可以被导入
        Test ColumnParallelLinear can be imported"""
        self.assertTrue(hasattr(mp_layers, 'ColumnParallelLinear'))

    def test_row_parallel_linear_import(self):
        """测试 RowParallelLinear 可以被导入
        Test RowParallelLinear can be imported"""
        self.assertTrue(hasattr(mp_layers, 'RowParallelLinear'))


class TestMPURandom(unittest.TestCase):
    """测试 MPU 随机数模块
    Test MPU random module"""

    def test_model_parallel_random_seed_import(self):
        """测试 model_parallel_random_seed 可以被导入
        Test model_parallel_random_seed can be imported"""
        self.assertTrue(hasattr(random, 'model_parallel_random_seed'))


if __name__ == '__main__':
    unittest.main()
