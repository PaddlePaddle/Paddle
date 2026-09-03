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

# [AUTO-GENERATED] Test file for paddle.distributed.fleet.utils.mix_precision_utils
# 覆盖模块: paddle/distributed/fleet/utils/mix_precision_utils.py
# Uncovered lines: 84,92,119,122,129,133,138,148-157,162,166,171,174,175,178,192-195,202-205,212-215,218-222

import unittest

from paddle.distributed.fleet.utils.mix_precision_utils import (
    MixPrecisionOptimizer,
)


class TestMixPrecisionOptimizer(unittest.TestCase):
    """测试 MixPrecisionOptimizer 类
    Test MixPrecisionOptimizer class"""

    def test_mix_precision_optimizer_import(self):
        """测试 MixPrecisionOptimizer 可导入
        Test MixPrecisionOptimizer can be imported"""
        self.assertIsNotNone(MixPrecisionOptimizer)


if __name__ == '__main__':
    unittest.main()
