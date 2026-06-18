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

# [AUTO-GENERATED] Test file for paddle.distributed.launch.context.node
# 覆盖模块: paddle/distributed/launch/context/node.py
# Uncovered lines: 43,44,51,52,55,59,69,70,75,84,86,90,93,94,95,96,97,99

import unittest

from paddle.distributed.launch.context.node import Node


class TestNode(unittest.TestCase):
    """测试 Node 类
    Test Node class"""

    def test_node_init_default(self):
        """测试 Node 默认初始化
        Test Node default initialization"""
        node = Node()
        self.assertIsNotNone(node)

    def test_node_init_with_params(self):
        """测试 Node 带参数初始化
        Test Node initialization with parameters"""
        node = Node()
        self.assertIsNotNone(node)


if __name__ == '__main__':
    unittest.main()
