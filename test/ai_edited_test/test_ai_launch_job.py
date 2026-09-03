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

# [AUTO-GENERATED] Test file for paddle.distributed.launch.job.job
# 覆盖模块: paddle/distributed/launch/job/job.py
# 未覆盖行: 16-81
# Covered module: paddle/distributed/launch/job/job.py
# Uncovered lines: all

import unittest


class TestJobMode(unittest.TestCase):
    """测试 JobMode 常量类
    Test JobMode constant class"""

    def test_job_mode_collective(self):
        """测试 COLLECTIVE 模式常量
        Test COLLECTIVE mode constant"""
        from paddle.distributed.launch.job.job import JobMode

        self.assertEqual(JobMode.COLLECTIVE, 'collective')

    def test_job_mode_ps(self):
        """测试 PS 模式常量
        Test PS mode constant"""
        from paddle.distributed.launch.job.job import JobMode

        self.assertEqual(JobMode.PS, 'ps')

    def test_job_mode_heter(self):
        """测试 HETER 模式常量
        Test HETER mode constant"""
        from paddle.distributed.launch.job.job import JobMode

        self.assertEqual(JobMode.HETER, 'heter')


class TestJobInit(unittest.TestCase):
    """测试 Job 初始化
    Test Job initialization"""

    def test_init_default(self):
        """测试默认初始化
        Test default initialization"""
        from paddle.distributed.launch.job.job import Job

        job = Job()

        self.assertEqual(job.id, 'default')
        self.assertEqual(job.mode, 'collective')
        self.assertEqual(job.replicas, 1)
        self.assertEqual(job.replicas_min, 1)
        self.assertEqual(job.replicas_max, 1)
        self.assertFalse(job.elastic)

    def test_init_custom(self):
        """测试自定义参数初始化
        Test custom parameter initialization"""
        from paddle.distributed.launch.job.job import Job

        job = Job(jid='my_job', mode='ps', nnodes='4')

        self.assertEqual(job.id, 'my_job')
        self.assertEqual(job.mode, 'ps')
        self.assertEqual(job.replicas, 4)
        self.assertEqual(job.replicas_min, 4)
        self.assertEqual(job.replicas_max, 4)
        self.assertFalse(job.elastic)

    def test_init_single_node(self):
        """测试单节点初始化
        Test single node initialization"""
        from paddle.distributed.launch.job.job import Job

        job = Job(nnodes='1')

        self.assertEqual(job.replicas, 1)
        self.assertFalse(job.elastic)


class TestJobStr(unittest.TestCase):
    """测试 Job.__str__ 方法
    Test Job.__str__ method"""

    def test_str(self):
        """测试字符串表示
        Test string representation"""
        from paddle.distributed.launch.job.job import Job

        job = Job(jid='test_job', mode='collective', nnodes='4')

        result = str(job)

        self.assertIn('test_job', result)
        self.assertIn('collective', result)
        self.assertIn('4', result)

    def test_str_elastic(self):
        """测试弹性任务的字符串表示
        Test elastic job string representation"""
        from paddle.distributed.launch.job.job import Job

        job = Job(jid='elastic_job', mode='collective', nnodes='2:8')

        result = str(job)

        self.assertIn('elastic_job', result)
        self.assertIn('True', result)


class TestJobProperties(unittest.TestCase):
    """测试 Job 属性
    Test Job properties"""

    def test_mode_property(self):
        """测试 mode 属性
        Test mode property"""
        from paddle.distributed.launch.job.job import Job

        job = Job(mode='ps')
        self.assertEqual(job.mode, 'ps')

    def test_id_property(self):
        """测试 id 属性
        Test id property"""
        from paddle.distributed.launch.job.job import Job

        job = Job(jid='custom_id')
        self.assertEqual(job.id, 'custom_id')

    def test_replicas_property(self):
        """测试 replicas 属性
        Test replicas property"""
        from paddle.distributed.launch.job.job import Job

        job = Job(nnodes='3')
        self.assertEqual(job.replicas, 3)

    def test_replicas_min_property(self):
        """测试 replicas_min 属性
        Test replicas_min property"""
        from paddle.distributed.launch.job.job import Job

        job = Job(nnodes='3')
        self.assertEqual(job.replicas_min, 3)

    def test_replicas_max_property(self):
        """测试 replicas_max 属性
        Test replicas_max property"""
        from paddle.distributed.launch.job.job import Job

        job = Job(nnodes='3')
        self.assertEqual(job.replicas_max, 3)

    def test_elastic_property(self):
        """测试 elastic 属性
        Test elastic property"""
        from paddle.distributed.launch.job.job import Job

        job = Job(nnodes='3')
        self.assertFalse(job.elastic)


class TestJobSetReplicas(unittest.TestCase):
    """测试 Job.set_replicas 方法
    Test Job.set_replicas method"""

    def test_set_replicas_single_number(self):
        """测试设置单个数字
        Test setting single number"""
        from paddle.distributed.launch.job.job import Job

        job = Job()
        job.set_replicas('5')

        self.assertEqual(job.replicas, 5)
        self.assertEqual(job.replicas_min, 5)
        self.assertEqual(job.replicas_max, 5)
        self.assertFalse(job.elastic)

    def test_set_replicas_range(self):
        """测试设置弹性范围
        Test setting elastic range"""
        from paddle.distributed.launch.job.job import Job

        job = Job()
        job.set_replicas('2:8')

        self.assertEqual(job.replicas, 8)
        self.assertEqual(job.replicas_min, 2)
        self.assertEqual(job.replicas_max, 8)
        self.assertTrue(job.elastic)

    def test_set_replicas_integer_input(self):
        """测试整数输入
        Test integer input"""
        from paddle.distributed.launch.job.job import Job

        job = Job()
        job.set_replicas(4)

        self.assertEqual(job.replicas, 4)
        self.assertFalse(job.elastic)

    def test_set_replicas_none_input(self):
        """测试 None 输入默认为 1
        Test None input defaults to 1"""
        from paddle.distributed.launch.job.job import Job

        job = Job()
        job.set_replicas(None)

        self.assertEqual(job.replicas, 1)
        self.assertFalse(job.elastic)

    def test_set_replicas_empty_string(self):
        """测试空字符串输入默认为 1
        Test empty string input defaults to 1"""
        from paddle.distributed.launch.job.job import Job

        job = Job()
        job.set_replicas('')

        self.assertEqual(job.replicas, 1)
        self.assertFalse(job.elastic)


class TestJobReplicasSetter(unittest.TestCase):
    """测试 Job.replicas setter
    Test Job.replicas setter"""

    def test_replicas_setter(self):
        """测试直接设置 replicas
        Test directly setting replicas"""
        from paddle.distributed.launch.job.job import Job

        job = Job(nnodes='1')
        job.replicas = 10

        self.assertEqual(job.replicas, 10)
        # Note: setter only changes _replicas, not min/max
        self.assertEqual(job.replicas_min, 1)
        self.assertEqual(job.replicas_max, 1)

    def test_replicas_setter_does_not_change_elastic(self):
        """测试 setter 不改变 elastic 状态
        Test setter does not change elastic status"""
        from paddle.distributed.launch.job.job import Job

        job = Job(nnodes='2:8')
        self.assertTrue(job.elastic)

        job.replicas = 5

        self.assertTrue(job.elastic)
        self.assertEqual(job.replicas, 5)


class TestJobEdgeCases(unittest.TestCase):
    """测试 Job 边界情况
    Test Job edge cases"""

    def test_large_range(self):
        """测试大范围弹性
        Test large elastic range"""
        from paddle.distributed.launch.job.job import Job

        job = Job(nnodes='1:100')

        self.assertEqual(job.replicas_min, 1)
        self.assertEqual(job.replicas_max, 100)
        self.assertTrue(job.elastic)

    def test_same_min_max(self):
        """测试相同最小最大值（非弹性）
        Test same min and max (non-elastic)"""
        from paddle.distributed.launch.job.job import Job

        job = Job(nnodes='5:5')

        self.assertEqual(job.replicas_min, 5)
        self.assertEqual(job.replicas_max, 5)
        self.assertTrue(job.elastic)  # Still elastic since range was specified

    def test_set_replicas_after_init(self):
        """测试初始化后重新设置副本数
        Test setting replicas after initialization"""
        from paddle.distributed.launch.job.job import Job

        job = Job(nnodes='2:8')

        self.assertTrue(job.elastic)
        self.assertEqual(job.replicas, 8)

        job.set_replicas('4')

        self.assertFalse(job.elastic)
        self.assertEqual(job.replicas, 4)
        self.assertEqual(job.replicas_min, 4)
        self.assertEqual(job.replicas_max, 4)

    def test_ps_mode(self):
        """测试 PS 模式
        Test PS mode"""
        from paddle.distributed.launch.job.job import Job, JobMode

        job = Job(mode=JobMode.PS, nnodes='3')

        self.assertEqual(job.mode, JobMode.PS)
        self.assertEqual(job.replicas, 3)

    def test_heter_mode(self):
        """测试 HETER 模式
        Test HETER mode"""
        from paddle.distributed.launch.job.job import Job, JobMode

        job = Job(mode=JobMode.HETER, nnodes='2')

        self.assertEqual(job.mode, JobMode.HETER)
        self.assertEqual(job.replicas, 2)


if __name__ == '__main__':
    unittest.main()
