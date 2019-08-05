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

from __future__ import print_function

import unittest
import paddle.fluid as fluid
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
from paddle.fluid.incubate.fleet.base.role_maker import UserDefinedRoleMaker
from paddle.fluid.incubate.fleet.base.role_maker import UserDefinedCollectiveRoleMaker
from paddle.fluid.incubate.fleet.base.role_maker import Role
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet


class DistributeTranspilerConfigTest(unittest.TestCase):
    def set_runtime_split_send_recv(self, config, value):
        config.runtime_split_send_recv = value

    def set_sync_mode(self, config, value):
        config.sync_mode = value

    def testConfig(self):
        config = DistributeTranspilerConfig()
        self.assertRaises(Exception, self.set_sync_mode, config, None)
        self.assertRaises(Exception, self.set_runtime_split_send_recv, config,
                          None)
        self.assertRaises(Exception, self.set_runtime_split_send_recv, config,
                          True)
        self.set_sync_mode(config, False)
        self.assertFalse(config.sync_mode)
        self.set_runtime_split_send_recv(config, True)
        self.assertRaises(Exception, self.set_sync_mode, config, True)


class UserDefinedRoleMakerTest(unittest.TestCase):
    def createRoleMaker(self,
                        current_id=0,
                        role=Role.WORKER,
                        worker_num=0,
                        server_endpoints=None):
        role = UserDefinedRoleMaker(current_id, role, worker_num,
                                    server_endpoints)

    def testRoleMaker(self):
        self.createRoleMaker(
            current_id=0,
            role=Role.WORKER,
            worker_num=1,
            server_endpoints=["127.0.0.1:8080"])
        ## test all invalid server_endpoints
        self.assertRaises(
            Exception, self.createRoleMaker)  # server_endpoints must be as list
        self.assertRaises(
            Exception, self.createRoleMaker,
            server_endpoints=[])  # server_endpoints can't be empty
        self.assertRaises(
            Exception, self.createRoleMaker, server_endpoint=[
                3, []
            ])  # element in server_endpoints must be as string
        self.assertRaises(
            Exception,
            self.createRoleMaker,
            server_endpoint=["127.0.0.1:8080", "127.0.0.1:8080"]
        )  # element in server_endpoints can't be duplicate
        ## test all invalid current_id 
        self.assertRaises(
            Exception, self.createRoleMaker,
            current_id="0")  # current_id must be as int
        self.assertRaises(
            Exception, self.createRoleMaker,
            current_id=-1)  # current_id must be greater than or equal to 0
        self.assertRaises(
            Exception,
            self.createRoleMaker,
            current_id=1,
            role=Role.SERVER,
            server_endpoints=["127.0.0.1:8080"]
        )  # if role is server, current_id must be less than len(server_endpoints)
        ## test all invalid worker_num
        self.assertRaises(
            Exception, self.createRoleMaker,
            worker_num="1")  # worker_num must be as int
        self.assertRaises(
            Exception, self.createRoleMaker,
            worker_num=0)  # worker_num must be greater than 0
        ## test all invalid role
        self.assertRaises(
            Exception, self.createRoleMaker,
            role=3)  # role must be as Role(Role.WORKER=1, Role.SERVER=2)


class UserDefinedCollectiveRoleMakerTest(unittest.TestCase):
    def createRoleMaker(self, current_id=0, worker_endpoints=None):
        role = UserDefinedCollectiveRoleMaker(current_id, worker_endpoints)

    def testRoleMaker(self):
        self.createRoleMaker(current_id=0, worker_endpoints=["127.0.0.1:8080"])
        ## test all invalid worker_endpoints
        self.assertRaises(
            Exception, self.createRoleMaker)  # worker_endpoints must be as list
        self.assertRaises(
            Exception, self.createRoleMaker,
            worker_endpoints=[])  # worker_endpoints can't be empty
        self.assertRaises(
            Exception, self.createRoleMaker,
            worker_endpoint=[3,
                             []])  # element worker_endpoints must be as string
        self.assertRaises(
            Exception,
            self.createRoleMaker,
            worker_endpoint=["127.0.0.1:8080", "127.0.0.1:8080"]
        )  # element in worker_endpoints can't be duplicate
        ## test all invalid current_id
        self.assertRaises(
            Exception, self.createRoleMaker,
            current_id="0")  # current_id must be as int
        self.assertRaises(
            Exception, self.createRoleMaker,
            current_id=-1)  # current_id must be greater than or equal to 0
        self.assertRaises(
            Exception,
            self.createRoleMaker,
            current_id=1,
            worker_endpoints=["127.0.0.1:8080"]
        )  # current_id must be less than len(worker_endpoints)


class DistributedTranspilerTest(unittest.TestCase):
    def _distributed_optimizer(self, optimizer, config):
        optimizer = fleet.distributed_optimizer(optimizer, config)

    def testDistributedTranspiler(self):
        role = UserDefinedRoleMaker(
            current_id=0,
            role=Role.WORKER,
            worker_num=1,
            server_endpoints=["127.0.0.1:8080"])
        fleet.init(role)
        optimizer = fluid.optimizer.Adam(learning_rate=0.001)
        config = DistributeTranspilerConfig()
        self.assertRaises(Exception, self._distributed_optimizer, optimizer,
                          dict({
                              "sync_mode": False
                          }))
        self.assertRaises(Exception, self._distributed_optimizer, None, config)
        optimizer = fleet.distributed_optimizer(optimizer, config)


if __name__ == '__main__':
    unittest.main()
