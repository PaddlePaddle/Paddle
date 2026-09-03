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

# [AUTO-GENERATED] Unit test for paddle.distributed.fleet.base.role_maker
# Target: cover uncovered lines 96-165, 285-329, 396-547, 549-611, 1214-1281
# in paddle/distributed/fleet/base/role_maker.py

"""
测试模块：paddle.distributed.fleet.base.role_maker
Test Module: paddle.distributed.fleet.base.role_maker

本测试覆盖以下功能：
This test covers:
1. Role - 角色常量类 / Role constants
2. Gloo - 未初始化状态的 barrier/all_reduce/all_gather
3. RoleMakerBase - 基类方法（to_string, _all_gather, _all_reduce, _barrier）
4. PaddleCloudRoleMaker - 非分布式环境下的一些路径
5. UserDefinedRoleMaker - 用户自定义角色分配
"""

import os
import unittest

from paddle.distributed.fleet.base.role_maker import (
    Gloo,
    PaddleCloudRoleMaker,
    Role,
    RoleMakerBase,
    UserDefinedRoleMaker,
)


class TestRoleConstants(unittest.TestCase):
    """测试 Role 常量值
    Test Role constant values"""

    def test_worker_value(self):
        """测试 WORKER 角色值
        Test WORKER role value"""
        self.assertEqual(Role.WORKER, 1)

    def test_server_value(self):
        """测试 SERVER 角色值
        Test SERVER role value"""
        self.assertEqual(Role.SERVER, 2)

    def test_heter_worker_value(self):
        """测试 HETER_WORKER 角色值
        Test HETER_WORKER role value"""
        self.assertEqual(Role.HETER_WORKER, 3)

    def test_all_value(self):
        """测试 ALL 角色值
        Test ALL role value"""
        self.assertEqual(Role.ALL, 4)

    def test_coordinator_value(self):
        """测试 COORDINATOR 角色值
        Test COORDINATOR role value"""
        self.assertEqual(Role.COORDINATOR, 5)


class TestGlooInit(unittest.TestCase):
    """测试 Gloo 初始化和未初始化状态
    Test Gloo initialization and uninitialized state"""

    def test_gloo_init_defaults(self):
        """测试 Gloo 初始化默认属性
        Test Gloo default attributes after init"""
        gloo = Gloo()
        self.assertFalse(gloo._is_initialized)
        self.assertIsNone(gloo._rendezvous)
        self.assertIsNone(gloo._role)
        self.assertEqual(gloo._role_id, -1)
        self.assertEqual(gloo._worker_num, -1)
        self.assertEqual(gloo._server_num, -1)

    def test_gloo_rendezvous_constants(self):
        """测试 Gloo RENDEZVOUS 常量
        Test Gloo RENDEZVOUS constants"""
        self.assertEqual(Gloo.RENDEZVOUS.HDFS, 1)
        self.assertEqual(Gloo.RENDEZVOUS.FILE, 2)
        self.assertEqual(Gloo.RENDEZVOUS.HTTP, 3)

    def test_gloo_init_invalid_rendezvous(self):
        """测试无效的 rendezvous 类型抛出异常
        Test invalid rendezvous type raises error"""
        gloo = Gloo()
        with self.assertRaises((ValueError, TypeError, AttributeError)):
            gloo.init(
                rendezvous=999,
                role=Role.WORKER,
                role_id=0,
                worker_num=1,
                server_num=1,
                kwargs={},
            )

    def test_gloo_init_hdfs_missing_args(self):
        """测试 HDFS rendezvous 缺少参数时抛出异常
        Test HDFS rendezvous with missing args raises error"""
        gloo = Gloo()
        with self.assertRaises(ValueError):
            gloo.init(
                rendezvous=Gloo.RENDEZVOUS.HDFS,
                role=Role.WORKER,
                role_id=0,
                worker_num=1,
                server_num=1,
                kwargs={},
            )

    def test_gloo_init_file_missing_args(self):
        """测试 FILE rendezvous 缺少参数时抛出异常
        Test FILE rendezvous with missing args raises error"""
        gloo = Gloo()
        with self.assertRaises(ValueError):
            gloo.init(
                rendezvous=Gloo.RENDEZVOUS.FILE,
                role=Role.WORKER,
                role_id=0,
                worker_num=1,
                server_num=1,
                kwargs={},
            )

    def test_gloo_init_http_missing_ip(self):
        """测试 HTTP rendezvous 缺少 IP 时抛出异常
        Test HTTP rendezvous with missing IP raises error"""
        gloo = Gloo()
        with self.assertRaises(ValueError):
            gloo.init(
                rendezvous=Gloo.RENDEZVOUS.HTTP,
                role=Role.WORKER,
                role_id=0,
                worker_num=1,
                server_num=1,
                kwargs={"http.port": "8080"},
            )

    def test_gloo_init_http_missing_port(self):
        """测试 HTTP rendezvous 缺少端口时抛出异常
        Test HTTP rendezvous with missing port raises error"""
        gloo = Gloo()
        with self.assertRaises(ValueError):
            gloo.init(
                rendezvous=Gloo.RENDEZVOUS.HTTP,
                role=Role.WORKER,
                role_id=0,
                worker_num=1,
                server_num=1,
                kwargs={"http.host": "127.0.0.1"},
            )

    def test_gloo_barrier_uninitialized(self):
        """测试未初始化时 barrier 发出警告
        Test barrier warns when uninitialized"""
        import warnings

        gloo = Gloo()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gloo.barrier("worker")
            self.assertTrue(len(w) > 0)

    def test_gloo_barrier_invalid_world(self):
        """测试无效的 comm_world 抛出异常
        Test invalid comm_world raises error"""
        gloo = Gloo()
        gloo._is_initialized = True
        with self.assertRaises(ValueError):
            gloo.barrier("invalid_world")

    def test_gloo_all_reduce_uninitialized(self):
        """测试未初始化时 all_reduce 发出警告
        Test all_reduce warns when uninitialized"""
        import warnings

        gloo = Gloo()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = gloo.all_reduce([1, 2, 3])
            self.assertTrue(len(w) > 0)
            self.assertEqual(result, [1, 2, 3])

    def test_gloo_all_reduce_invalid_world(self):
        """测试 all_reduce 无效 comm_world 抛出异常
        Test all_reduce invalid comm_world raises error"""
        gloo = Gloo()
        gloo._is_initialized = True
        with self.assertRaises(ValueError):
            gloo.all_reduce([1, 2, 3], comm_world="invalid")

    def test_gloo_all_gather_uninitialized(self):
        """测试未初始化时 all_gather 发出警告
        Test all_gather warns when uninitialized"""
        import warnings

        gloo = Gloo()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = gloo.all_gather([1, 2, 3])
            self.assertTrue(len(w) > 0)
            self.assertEqual(result, [1, 2, 3])

    def test_gloo_all_gather_invalid_world(self):
        """测试 all_gather 无效 comm_world 抛出异常
        Test all_gather invalid comm_world raises error"""
        gloo = Gloo()
        gloo._is_initialized = True
        with self.assertRaises(ValueError):
            gloo.all_gather([1], comm_world="invalid")


class TestRoleMakerBase(unittest.TestCase):
    """测试 RoleMakerBase 基类
    Test RoleMakerBase base class"""

    def test_init_defaults(self):
        """测试默认初始化属性
        Test default initialization attributes"""
        maker = RoleMakerBase()
        self.assertEqual(maker._worker_endpoints, [])
        self.assertEqual(maker._server_endpoints, [])
        self.assertEqual(maker._cur_endpoint, "")
        self.assertFalse(maker._role_is_generated)
        self.assertIsNone(maker._role)
        self.assertEqual(maker._current_id, -1)

    def test_get_trainer_endpoints(self):
        """测试获取 trainer 端点列表
        Test getting trainer endpoints"""
        maker = RoleMakerBase()
        maker._worker_endpoints = ["127.0.0.1:8080"]
        self.assertEqual(maker._get_trainer_endpoints(), ["127.0.0.1:8080"])

    def test_get_pserver_endpoints(self):
        """测试获取 pserver 端点列表
        Test getting pserver endpoints"""
        maker = RoleMakerBase()
        maker._server_endpoints = ["127.0.0.1:8081"]
        self.assertEqual(maker._get_pserver_endpoints(), ["127.0.0.1:8081"])

    def test_to_string(self):
        """测试 to_string 方法
        Test to_string method"""
        maker = RoleMakerBase()
        maker._role = Role.WORKER
        maker._current_id = 0
        maker._worker_endpoints = ["a:1"]
        maker._server_endpoints = ["b:2"]
        s = maker.to_string()
        self.assertIn("role:", s)
        self.assertIn("current_id:", s)

    def test_is_worker_not_implemented(self):
        """测试 _is_worker 抛出 NotImplementedError
        Test _is_worker raises NotImplementedError"""
        maker = RoleMakerBase()
        with self.assertRaises(NotImplementedError):
            maker._is_worker()

    def test_is_server_not_implemented(self):
        """测试 _is_server 抛出 NotImplementedError
        Test _is_server raises NotImplementedError"""
        maker = RoleMakerBase()
        with self.assertRaises(NotImplementedError):
            maker._is_server()

    def test_is_first_worker_not_implemented(self):
        """测试 _is_first_worker 抛出 NotImplementedError
        Test _is_first_worker raises NotImplementedError"""
        maker = RoleMakerBase()
        with self.assertRaises(NotImplementedError):
            maker._is_first_worker()

    def test_worker_num_not_implemented(self):
        """测试 _worker_num 抛出 NotImplementedError
        Test _worker_num raises NotImplementedError"""
        maker = RoleMakerBase()
        with self.assertRaises(NotImplementedError):
            maker._worker_num()

    def test_server_num_not_implemented(self):
        """测试 _server_num 抛出 NotImplementedError
        Test _server_num raises NotImplementedError"""
        maker = RoleMakerBase()
        with self.assertRaises(NotImplementedError):
            maker._server_num()

    def test_worker_index_not_implemented(self):
        """测试 _worker_index 抛出 NotImplementedError
        Test _worker_index raises NotImplementedError"""
        maker = RoleMakerBase()
        with self.assertRaises(NotImplementedError):
            maker._worker_index()

    def test_server_index_not_implemented(self):
        """测试 _server_index 抛出 NotImplementedError
        Test _server_index raises NotImplementedError"""
        maker = RoleMakerBase()
        with self.assertRaises(NotImplementedError):
            maker._server_index()

    def test_role_id_not_implemented(self):
        """测试 _role_id 抛出 NotImplementedError
        Test _role_id raises NotImplementedError"""
        maker = RoleMakerBase()
        with self.assertRaises(NotImplementedError):
            maker._role_id()

    def test_node_num_not_implemented(self):
        """测试 _node_num 抛出 NotImplementedError
        Test _node_num raises NotImplementedError"""
        maker = RoleMakerBase()
        with self.assertRaises(NotImplementedError):
            maker._node_num()

    def test_all_gather_base(self):
        """测试 _all_gather 基类方法（打印警告）
        Test _all_gather base method (prints warning)"""
        maker = RoleMakerBase()
        # 不会抛异常，只打印警告 / Should not raise, only prints warning
        maker._all_gather([1, 2, 3])

    def test_all_reduce_base(self):
        """测试 _all_reduce 基类方法（打印警告）
        Test _all_reduce base method (prints warning)"""
        maker = RoleMakerBase()
        maker._all_reduce([1, 2, 3])

    def test_barrier_base(self):
        """测试 _barrier 基类方法（打印警告）
        Test _barrier base method (prints warning)"""
        maker = RoleMakerBase()
        maker._barrier("worker")


class TestPaddleCloudRoleMaker(unittest.TestCase):
    """测试 PaddleCloudRoleMaker 类
    Test PaddleCloudRoleMaker class"""

    def setUp(self):
        # 清理环境变量 / Clean up environment variables
        self._env_backup = {}
        env_keys = [
            "PADDLE_PSERVERS_IP_PORT_LIST",
            "PADDLE_TRAINERS_NUM",
            "PADDLE_TRAINER_ENDPOINTS",
            "TRAINING_ROLE",
            "PADDLE_TRAINER_ID",
            "PADDLE_PORT",
            "POD_IP",
            "PADDLE_COORDINATOR_ENDPOINTS",
            "PADDLE_CURRENT_ENDPOINT",
            "PADDLE_RANK_IN_NODE",
            "PADDLE_LOCAL_DEVICE_IDS",
            "PADDLE_WORLD_DEVICE_IDS",
            "PADDLE_TRAINING_ROLE",
            "PADDLE_WITH_GLOO",
            "PADDLE_AUTO_PARALLEL_CONFIG",
        ]
        for key in env_keys:
            if key in os.environ:
                self._env_backup[key] = os.environ.pop(key)

    def tearDown(self):
        # 恢复环境变量 / Restore environment variables
        for key, val in self._env_backup.items():
            os.environ[key] = val
        # 清理新增的 / Clean up new ones
        env_keys = [
            "PADDLE_PSERVERS_IP_PORT_LIST",
            "PADDLE_TRAINERS_NUM",
            "PADDLE_TRAINER_ENDPOINTS",
            "TRAINING_ROLE",
            "PADDLE_TRAINER_ID",
            "PADDLE_PORT",
            "POD_IP",
            "PADDLE_COORDINATOR_ENDPOINTS",
            "PADDLE_CURRENT_ENDPOINT",
            "PADDLE_RANK_IN_NODE",
            "PADDLE_LOCAL_DEVICE_IDS",
            "PADDLE_WORLD_DEVICE_IDS",
            "PADDLE_TRAINING_ROLE",
            "PADDLE_WITH_GLOO",
            "PADDLE_AUTO_PARALLEL_CONFIG",
        ]
        for key in env_keys:
            os.environ.pop(key, None)

    def test_non_distributed(self):
        """测试非分布式模式下角色生成
        Test role generation in non-distributed mode"""
        # 不设置 PADDLE_PSERVERS_IP_PORT_LIST 时回退到非分布式
        # Falls back to non-distributed without PADDLE_PSERVERS_IP_PORT_LIST
        maker = PaddleCloudRoleMaker(is_collective=False)
        maker._generate_role()
        self.assertTrue(maker._is_non_distributed())
        self.assertTrue(maker._is_worker())
        self.assertFalse(maker._is_server())
        self.assertEqual(maker._current_id, 0)
        self.assertEqual(maker._worker_num(), 1)
        self.assertEqual(maker._node_num(), 1)

    def test_missing_trainers_num(self):
        """测试缺少 PADDLE_TRAINERS_NUM 时抛出异常
        Test missing PADDLE_TRAINERS_NUM raises error"""
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:8080"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        maker = PaddleCloudRoleMaker(is_collective=False)
        with self.assertRaises(ValueError):
            maker._generate_role()

    def test_missing_training_role(self):
        """测试缺少 TRAINING_ROLE 时抛出异常
        Test missing TRAINING_ROLE raises error"""
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:8080"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        maker = PaddleCloudRoleMaker(is_collective=False)
        with self.assertRaises(ValueError):
            maker._generate_role()

    def test_invalid_training_role(self):
        """测试无效的 TRAINING_ROLE 时抛出异常
        Test invalid TRAINING_ROLE raises error"""
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:8080"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["TRAINING_ROLE"] = "INVALID_ROLE"
        maker = PaddleCloudRoleMaker(is_collective=False)
        with self.assertRaises(ValueError):
            maker._generate_role()

    def test_gloo_init_disabled(self):
        """测试 PADDLE_WITH_GLOO=0 时不初始化 gloo
        Test gloo not initialized when PADDLE_WITH_GLOO=0"""
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:8080"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_PORT"] = "8081"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_WITH_GLOO"] = "0"
        maker = PaddleCloudRoleMaker(is_collective=False)
        maker._generate_role()
        self.assertFalse(maker._gloo._is_initialized)

    def test_heter_device_not_generated(self):
        """测试角色未生成时 _heter_device 触发生成
        Test _heter_device triggers generation when not generated"""
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:8080"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_PORT"] = "8081"
        os.environ["POD_IP"] = "127.0.0.1"
        maker = PaddleCloudRoleMaker(is_collective=False)
        # 调用 _heter_device 应触发 _generate_role
        # Calling _heter_device should trigger _generate_role
        result = maker._heter_device()
        self.assertEqual(result, "cpu")

    def test_heter_device_type(self):
        """测试 _heter_device_type 方法
        Test _heter_device_type method"""
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:8080"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_PORT"] = "8081"
        os.environ["POD_IP"] = "127.0.0.1"
        maker = PaddleCloudRoleMaker(is_collective=False)
        result = maker._heter_device_type()
        self.assertEqual(result, "cpu")

    def test_get_stage_id(self):
        """测试 _get_stage_id 方法
        Test _get_stage_id method"""
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:8080"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_PORT"] = "8081"
        os.environ["POD_IP"] = "127.0.0.1"
        maker = PaddleCloudRoleMaker(is_collective=False)
        self.assertEqual(maker._get_stage_id(), 1)

    def test_get_stage_trainers(self):
        """测试 _get_stage_trainers 方法
        Test _get_stage_trainers method"""
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:8080"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_PORT"] = "8081"
        os.environ["POD_IP"] = "127.0.0.1"
        maker = PaddleCloudRoleMaker(is_collective=False)
        self.assertEqual(maker._get_stage_trainers(), [])

    def test_get_num_stage(self):
        """测试 _get_num_stage 方法
        Test _get_num_stage method"""
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:8080"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_PORT"] = "8081"
        os.environ["POD_IP"] = "127.0.0.1"
        maker = PaddleCloudRoleMaker(is_collective=False)
        self.assertEqual(maker._get_num_stage(), 1)

    def test_heter_worker_num(self):
        """测试 _heter_worker_num 在非分布式模式下为 0
        Test _heter_worker_num is 0 in non-distributed mode"""
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:8080"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_PORT"] = "8081"
        os.environ["POD_IP"] = "127.0.0.1"
        maker = PaddleCloudRoleMaker(is_collective=False)
        maker._generate_role()
        self.assertEqual(maker._heter_worker_num(), 0)


class TestUserDefinedRoleMaker(unittest.TestCase):
    """测试 UserDefinedRoleMaker 类
    Test UserDefinedRoleMaker class"""

    def test_ps_worker_role(self):
        """测试用户定义的 PS Worker 角色
        Test user-defined PS worker role"""
        maker = UserDefinedRoleMaker(
            is_collective=False,
            current_id=0,
            role=Role.WORKER,
            worker_num=2,
            server_endpoints=["127.0.0.1:8081"],
            worker_endpoints=["127.0.0.1:8082", "127.0.0.1:8083"],
        )
        maker._generate_role()
        self.assertTrue(maker._is_worker())
        self.assertFalse(maker._is_server())
        self.assertTrue(maker._is_first_worker())
        self.assertEqual(maker._worker_num(), 2)
        self.assertEqual(maker._current_id, 0)
        self.assertEqual(maker._cur_endpoint, "127.0.0.1:8082")

    def test_ps_server_role(self):
        """测试用户定义的 PS Server 角色
        Test user-defined PS server role"""
        maker = UserDefinedRoleMaker(
            is_collective=False,
            current_id=0,
            role=Role.SERVER,
            worker_num=2,
            server_endpoints=["127.0.0.1:8081", "127.0.0.1:8084"],
            worker_endpoints=["127.0.0.1:8082", "127.0.0.1:8083"],
        )
        maker._generate_role()
        self.assertTrue(maker._is_server())
        self.assertFalse(maker._is_worker())
        self.assertEqual(maker._cur_endpoint, "127.0.0.1:8081")
        self.assertEqual(maker._server_num(), 2)

    def test_ps_worker_not_first(self):
        """测试非首个 worker
        Test non-first worker"""
        maker = UserDefinedRoleMaker(
            is_collective=False,
            current_id=1,
            role=Role.WORKER,
            worker_num=2,
            server_endpoints=["127.0.0.1:8081"],
            worker_endpoints=["127.0.0.1:8082", "127.0.0.1:8083"],
        )
        maker._generate_role()
        self.assertTrue(maker._is_worker())
        self.assertFalse(maker._is_first_worker())
        self.assertEqual(maker._cur_endpoint, "127.0.0.1:8083")

    def test_collective_role(self):
        """测试 collective 模式角色
        Test collective mode role"""
        maker = UserDefinedRoleMaker(
            is_collective=True,
            current_id=0,
            worker_endpoints=["127.0.0.1:8082", "127.0.0.1:8083"],
        )
        maker._generate_role()
        self.assertEqual(maker._worker_num(), 2)
        self.assertEqual(maker._current_id, 0)
        self.assertEqual(maker._node_num(), 1)

    def test_worker_num_from_endpoints(self):
        """测试从端点列表推断 worker 数量
        Test inferring worker_num from endpoints"""
        maker = UserDefinedRoleMaker(
            is_collective=False,
            current_id=0,
            role=Role.WORKER,
            worker_num=0,
            server_endpoints=[],
            worker_endpoints=[
                "127.0.0.1:8082",
                "127.0.0.1:8083",
                "127.0.0.1:8084",
            ],
        )
        maker._generate_role()
        self.assertEqual(maker._worker_num(), 3)

    def test_node_num_single_node(self):
        """测试单节点节点数
        Test single node node_num"""
        maker = UserDefinedRoleMaker(
            is_collective=False,
            current_id=0,
            role=Role.WORKER,
            worker_num=2,
            server_endpoints=[],
            worker_endpoints=["127.0.0.1:8082", "127.0.0.1:8083"],
        )
        maker._generate_role()
        self.assertEqual(maker._node_num(), 1)

    def test_node_num_multi_node(self):
        """测试多节点节点数
        Test multi-node node_num"""
        maker = UserDefinedRoleMaker(
            is_collective=False,
            current_id=0,
            role=Role.WORKER,
            worker_num=3,
            server_endpoints=[],
            worker_endpoints=[
                "192.168.1.1:8082",
                "192.168.1.2:8083",
                "192.168.1.1:8084",
            ],
        )
        maker._generate_role()
        self.assertEqual(maker._node_num(), 2)

    def test_role_id(self):
        """测试 _role_id 方法
        Test _role_id method"""
        maker = UserDefinedRoleMaker(
            is_collective=False,
            current_id=1,
            role=Role.WORKER,
            worker_num=2,
            server_endpoints=[],
            worker_endpoints=["127.0.0.1:8082", "127.0.0.1:8083"],
        )
        maker._generate_role()
        self.assertEqual(maker._role_id(), 1)


class TestClipGradBaseNotImplemented(unittest.TestCase):
    """测试 ClipGradBase 抽象方法
    Test ClipGradBase abstract methods"""

    def test_clip_grad_base_str(self):
        """测试 ClipGradBase.__str__ 抛出 NotImplementedError
        Test ClipGradBase.__str__ raises NotImplementedError"""
        from paddle.nn.clip import ClipGradBase

        base = ClipGradBase()
        with self.assertRaises(NotImplementedError):
            str(base)


if __name__ == '__main__':
    unittest.main()
