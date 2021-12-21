# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle
import paddle.fluid.core as core

paddle.enable_static()


class TestFleetExecutorTaskNode(unittest.TestCase):
    def test_task_node(self):
        program = paddle.static.Program()
        task_node_0 = core.TaskNode(program.desc, 0, 1, 1)
        task_node_1 = core.TaskNode(program.desc, 0, 1, 1)
        task_node_2 = core.TaskNode(program.desc, 0, 1, 1)
        self.assertEqual(task_node_0.task_id(), 0)
        self.assertEqual(task_node_1.task_id(), 1)
        self.assertEqual(task_node_2.task_id(), 2)
        self.assertTrue(
            task_node_0.add_downstream_task(task_node_1.task_id(), 1))
        self.assertTrue(task_node_1.add_upstream_task(task_node_0.task_id(), 1))


if __name__ == "__main__":
    unittest.main()
