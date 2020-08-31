#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""Test cloud role maker."""

from __future__ import print_function
import os
import unittest
import paddle.distributed.fleet.base.role_maker as role_maker


class TestRoleMakerBase(unittest.TestCase):
    """
    Test cases for RoleMakerBase
    """

    def test_rolemaker_base(self):
        role = role_maker.RoleMakerBase()
        self.assertRaises(Exception, role.is_worker)
        self.assertRaises(Exception, role.is_server)
        self.assertRaises(Exception, role.is_first_worker)
        self.assertRaises(Exception, role.worker_num)
        self.assertRaises(Exception, role.server_num)
        self.assertRaises(Exception, role.worker_index)
        self.assertRaises(Exception, role.server_index)
        self.assertRaises(Exception, role.role_id)
        self.assertRaises(Exception, role.node_num)

        trainer_endpoints = role.get_trainer_endpoints()
        self.assertTrue(len(trainer_endpoints) == 0)
        pserver_endpoints = role.get_pserver_endpoints()
        self.assertTrue(len(pserver_endpoints) == 0)

        print(role.to_string())
        self.assertTrue(role._all_gather(role._node_type_comm, 1) is None)
        self.assertTrue(role._all_reduce(role._node_type_comm, 1) is None)
        role._barrier(role._node_type_comm)


class TestCloudRoleMaker(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMaker.
    """

    def setUp(self):
        """Set up, set envs."""
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001,127.0.0.2:36001"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001,127.0.0.2:36001"
        os.environ["POD_IP"] = "127.0.0.1"

    def test_tr_rolemaker(self):
        """Test tr rolenamer."""
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"

        try:
            import netifaces
        except:
            print("warning: no netifaces, skip test_tr_rolemaker")
            return

        ro = role_maker.PaddleCloudRoleMaker(
            is_collective=False, init_gloo=False)
        self.assertTrue(ro.is_worker())
        self.assertFalse(ro.is_server())
        self.assertEqual(ro.worker_num(), 2)
        self.assertTrue(ro.is_first_worker())
        worker_endpoints = ro.get_trainer_endpoints()
        self.assertEqual(worker_endpoints[0], '127.0.0.1:36001')
        self.assertEqual(ro.role_id(), 0)
        self.assertEqual(ro.node_num(), 2)

    def test_tr_rolemaker_collective(self):
        ro = role_maker.PaddleCloudRoleMaker(is_collective=True)
        self.assertEqual(ro.worker_num(), 2)
        self.assertEqual(ro.node_num(), 2)

    def test_ps_rolemaker(self):
        """Test ps rolemaker."""
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"

        try:
            import netifaces
        except:
            print("warning: no netifaces, skip test_ps_rolemaker")
            return

        ro = role_maker.PaddleCloudRoleMaker(
            is_collective=False, init_gloo=False)
        self.assertEqual(ro.server_index(), 0)
        self.assertFalse(ro.is_worker())
        self.assertTrue(ro.is_server())
        self.assertEqual(ro.server_num(), 2)
        pserver_endpoints = ro.get_pserver_endpoints()
        self.assertEqual(pserver_endpoints[0], '127.0.0.1:36001')
        self.assertTrue(ro._all_gather(ro._all_comm, 1) is None)
        self.assertTrue(ro._all_reduce(ro._all_comm, 1) is None)

    def test_traing_role(self):
        """Test training role."""
        os.environ["TRAINING_ROLE"] = "TEST"
        try:
            import netifaces
        except:
            print("warning: no netifaces, skip test_training_role")
            return

        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro.generate_role)


class TestUserDefinedRoleMaker(unittest.TestCase):
    """
    Test cases for UserDefinedRoleMaker.
    """

    def setUp(self):
        pass

    def test_ps_rolemaker(self):
        try:
            import netifaces
        except:
            print("warning: no netifaces, skip test_ps_rolemaker")
            return

        ro = role_maker.UserDefinedRoleMaker(
            is_collective=False,
            init_gloo=False,
            server_endpoints="127.0.0.1:36001,127.0.0.1:36001",
            role=role_maker.Role.SERVER,
            current_id=0,
            worker_num=2)
        self.assertEqual(ro.server_num(), 2)
        ro.generate_role()
        self.assertTrue(ro.is_server())
        self.assertEqual(ro.role_id(), 0)

    def test_tr_rolemaker(self):
        try:
            import netifaces
        except:
            print("warning: no netifaces, skip test_tr_rolemaker")
            return

        ro = role_maker.UserDefinedRoleMaker(
            is_collective=False,
            init_gloo=False,
            server_endpoints="127.0.0.1:36001,127.0.0.1:36001",
            role=role_maker.Role.WORKER,
            current_id=0,
            worker_num=2)
        self.assertIn("127.0.0.1:36001", ro.get_pserver_endpoints())
        self.assertTrue(ro.is_worker())
        self.assertEqual(ro.role_id(), 0)


if __name__ == "__main__":
    unittest.main()
