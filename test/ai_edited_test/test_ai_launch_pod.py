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

# [AUTO-GENERATED] Test file for paddle.distributed.launch.job.pod
# 覆盖模块: paddle/distributed/launch/job/pod.py
# 未覆盖行: 27-212
# Covered module: paddle/distributed/launch/job/pod.py
# Uncovered lines: all

import sys
import unittest
from unittest.mock import MagicMock, patch

from paddle.distributed.launch.job.pod import Pod
from paddle.distributed.launch.job.status import Status

# Get module reference for patching (paddle.distributed.launch is a function,
# not a module, so @patch string paths don't resolve through it)
_pod_mod = sys.modules['paddle.distributed.launch.job.pod']


class TestPodSpec(unittest.TestCase):
    """测试 PodSpec 类
    Test PodSpec class"""

    def test_pod_spec_init(self):
        """测试 PodSpec 初始化
        Test PodSpec initialization"""
        from paddle.distributed.launch.job.pod import PodSpec

        spec = PodSpec()

        self.assertIsInstance(spec._name, str)
        self.assertEqual(len(spec._name), 6)
        self.assertEqual(spec._init_containers, [])
        self.assertEqual(spec._containers, [])
        self.assertEqual(spec._rank, -1)
        self.assertIsNone(spec._init_timeout)
        self.assertEqual(spec._restart, -1)
        self.assertEqual(spec._replicas, 0)
        self.assertEqual(spec._exit_code, 0)


class TestPodInit(unittest.TestCase):
    """测试 Pod 初始化
    Test Pod initialization"""

    def test_pod_init(self):
        """测试 Pod 初始化
        Test Pod initialization"""
        pod = Pod()

        self.assertIsInstance(pod.name, str)
        self.assertEqual(len(pod.name), 6)
        self.assertEqual(pod.replicas, 0)
        self.assertEqual(pod.rank, -1)
        self.assertEqual(pod.restart, -1)
        self.assertEqual(pod.containers, [])
        self.assertEqual(pod.init_containers, [])

    def test_pod_str(self):
        """测试 Pod 字符串表示
        Test Pod string representation"""
        pod = Pod()

        result = str(pod)

        self.assertIn('Pod:', result)


class TestPodReplicas(unittest.TestCase):
    """测试 Pod.replicas 属性
    Test Pod.replicas property"""

    def test_replicas_setter_min_one(self):
        """测试 replicas 最小值为 1
        Test replicas minimum value is 1"""
        pod = Pod()
        pod.replicas = 0

        self.assertEqual(pod.replicas, 1)

    def test_replicas_setter_positive(self):
        """测试设置正数副本数
        Test setting positive replicas"""
        pod = Pod()
        pod.replicas = 4

        self.assertEqual(pod.replicas, 4)

    def test_replicas_setter_negative(self):
        """测试设置负数副本数
        Test setting negative replicas"""
        pod = Pod()
        pod.replicas = -5

        self.assertEqual(pod.replicas, 1)


class TestPodRank(unittest.TestCase):
    """测试 Pod.rank 属性
    Test Pod.rank property"""

    def test_rank_setter(self):
        """测试 rank setter
        Test rank setter"""
        pod = Pod()
        pod.rank = 3

        self.assertEqual(pod.rank, 3)


class TestPodContainers(unittest.TestCase):
    """测试 Pod 容器管理
    Test Pod container management"""

    def test_add_container(self):
        """测试添加容器
        Test adding container"""
        pod = Pod()
        mock_container = MagicMock()

        pod.add_container(mock_container)

        self.assertEqual(len(pod.containers), 1)
        self.assertEqual(mock_container.rank, 0)

    def test_add_multiple_containers(self):
        """测试添加多个容器
        Test adding multiple containers"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c1 = MagicMock()

        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        self.assertEqual(len(pod.containers), 2)
        self.assertEqual(mock_c0.rank, 0)
        self.assertEqual(mock_c1.rank, 1)

    def test_add_init_container(self):
        """测试添加 init 容器
        Test adding init container"""
        pod = Pod()
        mock_container = MagicMock()

        pod.add_init_container(mock_container)

        self.assertEqual(len(pod.init_containers), 1)
        self.assertEqual(mock_container.rank, 0)

    def test_add_multiple_init_containers(self):
        """测试添加多个 init 容器
        Test adding multiple init containers"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c1 = MagicMock()

        pod.add_init_container(mock_c0)
        pod.add_init_container(mock_c1)

        self.assertEqual(len(pod.init_containers), 2)
        self.assertEqual(mock_c0.rank, 0)
        self.assertEqual(mock_c1.rank, 1)


class TestPodExitCode(unittest.TestCase):
    """测试 Pod.exit_code 属性
    Test Pod.exit_code property"""

    def test_exit_code_no_containers(self):
        """测试无容器时退出码为 0
        Test exit code is 0 when no containers"""
        pod = Pod()
        self.assertEqual(pod.exit_code, 0)

    def test_exit_code_all_zero(self):
        """测试所有容器退出码为 0
        Test exit code is 0 when all containers exit with 0"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c0.exit_code = 0
        mock_c1 = MagicMock()
        mock_c1.exit_code = 0
        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        self.assertEqual(pod.exit_code, 0)

    def test_exit_code_non_zero(self):
        """测试有容器非零退出码
        Test exit code is non-zero when a container fails"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c0.exit_code = 0
        mock_c1 = MagicMock()
        mock_c1.exit_code = 1
        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        self.assertEqual(pod.exit_code, 1)


class TestPodDeploy(unittest.TestCase):
    """测试 Pod.deploy 方法
    Test Pod.deploy method"""

    def test_deploy_basic(self):
        """测试基本部署
        Test basic deploy"""
        pod = Pod()
        mock_init = MagicMock()
        mock_container = MagicMock()
        pod.add_init_container(mock_init)
        pod.add_container(mock_container)

        pod.deploy()

        mock_init.start.assert_called_once()
        mock_init.wait.assert_called_once()
        mock_container.start.assert_called_once()
        self.assertEqual(pod.restart, 0)

    def test_deploy_increment_restart(self):
        """测试部署增加重启计数
        Test deploy increments restart count"""
        pod = Pod()
        mock_container = MagicMock()
        pod.add_container(mock_container)

        self.assertEqual(pod.restart, -1)

        pod.deploy()

        self.assertEqual(pod.restart, 0)

        pod.deploy()

        self.assertEqual(pod.restart, 1)

    def test_deploy_with_timeout(self):
        """测试带 init_timeout 的部署
        Test deploy with init_timeout"""
        pod = Pod()
        pod._init_timeout = 30
        mock_init = MagicMock()
        mock_container = MagicMock()
        pod.add_init_container(mock_init)
        pod.add_container(mock_container)

        pod.deploy()

        mock_init.wait.assert_called_once_with(30)


class TestPodStop(unittest.TestCase):
    """测试 Pod.stop 方法
    Test Pod.stop method"""

    def test_stop_signal_only(self):
        """测试仅发送信号停止
        Test stop with signal only"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c1 = MagicMock()
        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        pod.stop(sigint=15)

        mock_c0.send_signal.assert_called_once_with(15)
        mock_c1.send_signal.assert_called_once_with(15)

    def test_stop_terminate_no_timeout(self):
        """测试无超时的 terminate 停止
        Test terminate stop without timeout"""
        pod = Pod()
        mock_c = MagicMock()
        pod.add_container(mock_c)

        pod.stop(sigint='TERM')

        mock_c.terminate.assert_called_once()

    def test_stop_with_timeout_join_success(self):
        """测试超时停止且 join 成功
        Test stop with timeout and join success"""
        pod = Pod()
        mock_c = MagicMock()
        pod.add_container(mock_c)

        with patch.object(pod, 'join', return_value=True):
            result = pod.stop(timeout=10)

        self.assertTrue(result)
        mock_c.terminate.assert_called_once()

    def test_stop_with_timeout_join_fail(self):
        """测试超时停止且 join 失败
        Test stop with timeout and join failure"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c1 = MagicMock()
        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        with patch.object(pod, 'join', return_value=False):
            result = pod.stop(timeout=10)

        self.assertFalse(result)
        # terminate is called first without force, then with force=True
        mock_c0.terminate.assert_called_with(force=True)
        mock_c1.terminate.assert_called_with(force=True)


class TestPodJoin(unittest.TestCase):
    """测试 Pod.join 方法
    Test Pod.join method"""

    def test_join_success(self):
        """测试 join 成功
        Test join success"""
        pod = Pod()
        mock_c = MagicMock()
        mock_c.wait.return_value = True
        pod.add_container(mock_c)

        result = pod.join(timeout=10)

        self.assertTrue(result)

    def test_join_failure(self):
        """测试 join 失败
        Test join failure"""
        pod = Pod()
        mock_c = MagicMock()
        mock_c.wait.return_value = False
        pod.add_container(mock_c)

        result = pod.join(timeout=10)

        self.assertFalse(result)

    def test_join_multiple_containers(self):
        """测试多个容器 join
        Test join with multiple containers"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c0.wait.return_value = True
        mock_c1 = MagicMock()
        mock_c1.wait.return_value = True
        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        result = pod.join()

        self.assertTrue(result)


class TestPodStatus(unittest.TestCase):
    """测试 Pod.status 属性
    Test Pod.status property"""

    def test_status_ready_no_containers(self):
        """测试无容器时状态为 COMPLETED (is_completed 对空列表返回 True)
        Test status is COMPLETED when no containers (is_completed returns True for empty)"""
        pod = Pod()
        self.assertEqual(pod.status, Status.COMPLETED)

    def test_status_failed(self):
        """测试有容器失败时状态为 FAILED
        Test status is FAILED when a container fails"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c0.status = Status.RUNNING
        mock_c1 = MagicMock()
        mock_c1.status = Status.FAILED
        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        self.assertEqual(pod.status, Status.FAILED)

    def test_status_completed(self):
        """测试所有容器完成时状态为 COMPLETED
        Test status is COMPLETED when all containers complete"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c0.status = Status.COMPLETED
        mock_c1 = MagicMock()
        mock_c1.status = Status.COMPLETED
        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        self.assertEqual(pod.status, Status.COMPLETED)

    def test_status_running(self):
        """测试所有容器运行中时状态为 RUNNING
        Test status is RUNNING when all containers running"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c0.status = Status.RUNNING
        mock_c1 = MagicMock()
        mock_c1.status = Status.RUNNING
        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        self.assertEqual(pod.status, Status.RUNNING)


class TestPodIsMethods(unittest.TestCase):
    """测试 Pod 状态检查方法
    Test Pod status check methods"""

    def test_is_failed_true(self):
        """测试 is_failed 返回 True
        Test is_failed returns True"""
        pod = Pod()
        mock_c = MagicMock()
        mock_c.status = Status.FAILED
        pod.add_container(mock_c)

        self.assertTrue(pod.is_failed())

    def test_is_failed_false(self):
        """测试 is_failed 返回 False
        Test is_failed returns False"""
        pod = Pod()
        mock_c = MagicMock()
        mock_c.status = Status.RUNNING
        pod.add_container(mock_c)

        self.assertFalse(pod.is_failed())

    def test_is_completed_true(self):
        """测试 is_completed 返回 True
        Test is_completed returns True"""
        pod = Pod()
        mock_c = MagicMock()
        mock_c.status = Status.COMPLETED
        pod.add_container(mock_c)

        self.assertTrue(pod.is_completed())

    def test_is_completed_false(self):
        """测试 is_completed 返回 False
        Test is_completed returns False"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c0.status = Status.COMPLETED
        mock_c1 = MagicMock()
        mock_c1.status = Status.RUNNING
        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        self.assertFalse(pod.is_completed())

    def test_is_running_true(self):
        """测试 is_running 返回 True
        Test is_running returns True"""
        pod = Pod()
        mock_c = MagicMock()
        mock_c.status = Status.RUNNING
        pod.add_container(mock_c)

        self.assertTrue(pod.is_running())

    def test_is_running_false(self):
        """测试 is_running 返回 False
        Test is_running returns False"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c0.status = Status.RUNNING
        mock_c1 = MagicMock()
        mock_c1.status = Status.COMPLETED
        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        self.assertFalse(pod.is_running())


class TestPodReset(unittest.TestCase):
    """测试 Pod.reset 方法
    Test Pod.reset method"""

    def test_reset(self):
        """测试重置 pod
        Test resetting pod"""
        pod = Pod()
        mock_c = MagicMock()
        pod.add_container(mock_c)
        pod.add_init_container(MagicMock())

        self.assertGreater(len(pod.containers), 0)
        self.assertGreater(len(pod.init_containers), 0)

        pod.reset()

        self.assertEqual(len(pod.containers), 0)
        self.assertEqual(len(pod.init_containers), 0)


class TestPodFailedContainer(unittest.TestCase):
    """测试 Pod.failed_container 方法
    Test Pod.failed_container method"""

    def test_failed_container_none(self):
        """测试无失败容器
        Test no failed containers"""
        pod = Pod()
        mock_c = MagicMock()
        mock_c.status = Status.RUNNING
        pod.add_container(mock_c)

        result = pod.failed_container()

        self.assertEqual(len(result), 0)

    def test_failed_container_some(self):
        """测试有失败容器
        Test some failed containers"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c0.status = Status.RUNNING
        mock_c1 = MagicMock()
        mock_c1.status = Status.FAILED
        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        result = pod.failed_container()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], mock_c1)


class TestPodLogs(unittest.TestCase):
    """测试 Pod.logs 方法
    Test Pod.logs method"""

    def test_logs_default(self):
        """测试默认日志输出
        Test default log output"""
        pod = Pod()
        mock_c = MagicMock()
        pod.add_container(mock_c)
        mock_init = MagicMock()
        pod.add_init_container(mock_init)

        pod.logs()

        mock_c.logs.assert_called_once()
        mock_init.logs.assert_called_once()

    def test_logs_with_idx(self):
        """测试指定索引日志输出
        Test log output with specific index"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c1 = MagicMock()
        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        pod.logs(idx=1)

        mock_c0.logs.assert_not_called()
        mock_c1.logs.assert_called_once()

    def test_logs_no_containers(self):
        """测试无容器时日志输出
        Test log output with no containers"""
        pod = Pod()

        pod.logs()  # should not raise


class TestPodTail(unittest.TestCase):
    """测试 Pod.tail 方法
    Test Pod.tail method"""

    def test_tail_default(self):
        """测试默认尾部输出
        Test default tail output"""
        pod = Pod()
        mock_c = MagicMock()
        pod.add_container(mock_c)

        pod.tail()

        mock_c.tail.assert_called_once()

    def test_tail_with_idx(self):
        """测试指定索引尾部输出
        Test tail output with specific index"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c1 = MagicMock()
        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        pod.tail(idx=1)

        mock_c0.tail.assert_not_called()
        mock_c1.tail.assert_called_once()


class TestPodWatch(unittest.TestCase):
    """测试 Pod.watch 方法
    Test Pod.watch method"""

    def test_watch_any_failed(self):
        """测试 watch 检测到任一容器失败
        Test watch detects any container failure"""
        pod = Pod()
        mock_c = MagicMock()
        mock_c.status = Status.FAILED
        pod.add_container(mock_c)

        with patch.object(Pod, '__init__', lambda self: None):
            pod._init_containers = []
            pod._containers = [mock_c]

        # Directly test the watch logic - first iteration finds FAILED
        # Since watch loops, we need to make it stop after detecting failed
        # Use module-level patching since @patch string paths can't resolve
        original_sleep = _pod_mod.time.sleep
        try:
            _pod_mod.time.sleep = MagicMock()
            result = pod.watch()
        finally:
            _pod_mod.time.sleep = original_sleep

        self.assertEqual(result, Status.FAILED)

    def test_watch_all_completed(self):
        """测试 watch 检测到所有容器完成
        Test watch detects all containers completed"""
        pod = Pod()
        mock_c0 = MagicMock()
        mock_c0.status = Status.COMPLETED
        mock_c1 = MagicMock()
        mock_c1.status = Status.COMPLETED
        pod.add_container(mock_c0)
        pod.add_container(mock_c1)

        original_sleep = _pod_mod.time.sleep
        try:
            _pod_mod.time.sleep = MagicMock()
            result = pod.watch()
        finally:
            _pod_mod.time.sleep = original_sleep

        self.assertEqual(result, Status.COMPLETED)

    def test_watch_init_container_failure(self):
        """测试 watch 检测到 init 容器失败
        Test watch detects init container failure"""
        pod = Pod()
        mock_init = MagicMock()
        mock_init.status = Status.FAILED
        mock_c = MagicMock()
        mock_c.status = Status.RUNNING
        pod.add_init_container(mock_init)
        pod.add_container(mock_c)

        original_sleep = _pod_mod.time.sleep
        try:
            _pod_mod.time.sleep = MagicMock()
            result = pod.watch()
        finally:
            _pod_mod.time.sleep = original_sleep

        self.assertEqual(result, Status.FAILED)

    def test_watch_custom_lists(self):
        """测试 watch 使用自定义状态列表
        Test watch with custom status lists"""
        pod = Pod()
        mock_c = MagicMock()
        mock_c.status = Status.RUNNING

        original_sleep = _pod_mod.time.sleep
        original_time = _pod_mod.time.time
        try:
            _pod_mod.time.sleep = MagicMock()
            _pod_mod.time.time = MagicMock(side_effect=[100.0, 103.0])
            # Watch should loop and eventually timeout
            result = pod.watch(
                all_list=[Status.COMPLETED],
                any_list=[Status.FAILED],
                timeout=2,
            )
        finally:
            _pod_mod.time.sleep = original_sleep
            _pod_mod.time.time = original_time

        # RUNNING is neither in all_list nor any_list, times out
        self.assertIsNone(result)

    def test_watch_with_timeout(self):
        """测试 watch 带超时
        Test watch with timeout"""
        pod = Pod()
        mock_c = MagicMock()
        mock_c.status = Status.RUNNING
        pod.add_container(mock_c)

        original_sleep = _pod_mod.time.sleep
        original_time = _pod_mod.time.time
        try:
            _pod_mod.time.sleep = MagicMock()
            _pod_mod.time.time = MagicMock(side_effect=[100.0, 103.0])
            result = pod.watch(timeout=2)
        finally:
            _pod_mod.time.sleep = original_sleep
            _pod_mod.time.time = original_time

        self.assertIsNone(result)

    def test_watch_negative_timeout(self):
        """测试 watch 带负超时（无限等待模拟）
        Test watch with negative timeout"""
        pod = Pod()
        mock_c = MagicMock()
        mock_c.status = Status.COMPLETED
        pod.add_container(mock_c)

        original_sleep = _pod_mod.time.sleep
        original_time = _pod_mod.time.time
        try:
            _pod_mod.time.sleep = MagicMock()
            _pod_mod.time.time = MagicMock(side_effect=[100.0, 101.0, 102.0])
            result = pod.watch(timeout=-1)
        finally:
            _pod_mod.time.sleep = original_sleep
            _pod_mod.time.time = original_time

        self.assertEqual(result, Status.COMPLETED)


if __name__ == '__main__':
    unittest.main()
