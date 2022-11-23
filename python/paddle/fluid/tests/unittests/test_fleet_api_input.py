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

import unittest
import paddle
import paddle.fluid as fluid
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
from paddle.fluid.incubate.fleet.base.role_maker import UserDefinedRoleMaker
from paddle.fluid.incubate.fleet.base.role_maker import UserDefinedCollectiveRoleMaker
from paddle.fluid.incubate.fleet.base.role_maker import Role
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.parameter_server import TranspilerOptimizer
from paddle.fluid.incubate.fleet.collective import CollectiveOptimizer
from dist_simnet_bow import train_network


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


class FleetTest(unittest.TestCase):

    def testInvalidInputs(self):
        self.assertRaises(Exception, fleet.split_files, "files")
        self.assertRaises(Exception, fleet.init, "pserver")

        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = paddle.mean(hidden)
        adam = fluid.optimizer.Adam()
        adam.minimize(loss)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        pe = fluid.ParallelExecutor(use_cuda=False, loss_name=loss.name)
        self.assertRaises(Exception,
                          fleet.save_inference_model,
                          dirname='/tmp/',
                          feeded_var_names=['X'],
                          target_vars=[loss],
                          executor=pe)
        self.assertRaises(Exception,
                          fleet.save_inference_model,
                          dirname='/tmp/',
                          feeded_var_names=['X'],
                          target_vars=[loss],
                          executor="executor")
        compiled_prog = fluid.compiler.CompiledProgram(
            fluid.default_main_program())
        self.assertRaises(Exception,
                          fleet.save_inference_model,
                          dirname='/tmp/',
                          feeded_var_names=['X'],
                          target_vars=[loss],
                          executor=exe,
                          main_program=compiled_prog)
        self.assertRaises(Exception,
                          fleet.save_persistables,
                          executor=pe,
                          dirname='/tmp/')
        self.assertRaises(Exception,
                          fleet.save_persistables,
                          executor="executor",
                          dirname='/tmp/')
        self.assertRaises(Exception,
                          fleet.save_persistables,
                          executor=exe,
                          dirname='/tmp/',
                          main_program=compiled_prog)
        self.assertRaises(Exception, fleet._transpile, "config")

    def set_program(self, avg_cost, strategy):
        with fluid.scope_guard(fluid.Scope()):
            optimizer = fluid.optimizer.SGD(0.1)
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
            optimizer.minimize(avg_cost)

    def test_init_role(self):
        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.SERVER,
            worker_num=2,
            server_endpoints=["127.0.0.1:36011", "127.0.0.1:36012"])
        # for test optimizer without init(role)
        # fleet.init(role)
        batch_size = 128
        is_sparse = True
        is_distribute = False
        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = False
        strategy.geo_sgd_mode = True
        strategy.geo_sgd_need_push_nums = 5
        avg_cost, _, _ = train_network(batch_size, is_distribute, is_sparse)

        self.assertRaises(Exception, self.set_program, avg_cost, strategy)

    def test_transpile(self):
        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.SERVER,
            worker_num=2,
            server_endpoints=["127.0.0.1:36011", "127.0.0.1:36012"])
        # for test optimizer without init(role)
        fleet.init(role)
        batch_size = 128
        is_sparse = True
        is_distribute = False

        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = False
        strategy.runtime_split_send_recv = True
        avg_cost, _, _ = train_network(batch_size, is_distribute, is_sparse)

        self.set_program(avg_cost, strategy)
        strategy.runtime_split_send_recv = False
        self.set_program(avg_cost, strategy)


class TranspilerOptimizerTest(unittest.TestCase):

    def testInvalidInputs(self):
        self.assertRaises(Exception, TranspilerOptimizer, "Adam", None)
        self.assertRaises(Exception, TranspilerOptimizer,
                          fluid.optimizer.Adam(0.001), "strategy")

        transpiler = TranspilerOptimizer(fluid.optimizer.Adam(0.001))
        self.assertRaises(Exception, transpiler.minimize, loss=[])
        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = paddle.mean(hidden)
        self.assertRaises(Exception,
                          transpiler.minimize,
                          loss=loss.name,
                          startup_program=[])


class UserDefinedRoleMakerTest(unittest.TestCase):

    def createRoleMaker(self,
                        current_id=0,
                        role=Role.WORKER,
                        worker_num=1,
                        server_endpoints=["127.0.0.1:8080"]):
        role = UserDefinedRoleMaker(current_id, role, worker_num,
                                    server_endpoints)

    def testRoleMaker(self):
        self.createRoleMaker()
        # test all invalid server_endpoints
        self.assertRaises(
            Exception, self.createRoleMaker,
            server_endpoints=None)  # server_endpoints must be as list
        self.assertRaises(
            Exception, self.createRoleMaker,
            server_endpoints=[])  # server_endpoints can't be empty
        self.assertRaises(Exception,
                          self.createRoleMaker,
                          server_endpoints=[
                              3, []
                          ])  # element in server_endpoints must be as string
        self.assertRaises(Exception,
                          self.createRoleMaker,
                          server_endpoints=[
                              "127.0.0.1:8080", "127.0.0.1:8080"
                          ])  # element in server_endpoints can't be duplicate
        # test all invalid current_id
        self.assertRaises(Exception, self.createRoleMaker,
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
        # test all invalid worker_num
        self.assertRaises(Exception, self.createRoleMaker,
                          worker_num="1")  # worker_num must be as int
        self.assertRaises(Exception, self.createRoleMaker,
                          worker_num=0)  # worker_num must be greater than 0
        # test all invalid role
        self.assertRaises(
            Exception, self.createRoleMaker,
            role=3)  # role must be as Role(Role.WORKER=1, Role.SERVER=2)


class UserDefinedCollectiveRoleMakerTest(unittest.TestCase):

    def createRoleMaker(self,
                        current_id=0,
                        worker_endpoints=["127.0.0.1:8080"]):
        role = UserDefinedCollectiveRoleMaker(current_id, worker_endpoints)

    def testRoleMaker(self):
        self.createRoleMaker()
        # test all invalid worker_endpoints
        self.assertRaises(
            Exception, self.createRoleMaker,
            worker_endpoints=None)  # worker_endpoints must be as list
        self.assertRaises(
            Exception, self.createRoleMaker,
            worker_endpoints=[])  # worker_endpoints can't be empty
        self.assertRaises(Exception,
                          self.createRoleMaker,
                          worker_endpoints=[
                              3, []
                          ])  # element worker_endpoints must be as string
        self.assertRaises(Exception,
                          self.createRoleMaker,
                          worker_endpoints=[
                              "127.0.0.1:8080", "127.0.0.1:8080"
                          ])  # element in worker_endpoints can't be duplicate
        # test all invalid current_id
        self.assertRaises(Exception, self.createRoleMaker,
                          current_id="0")  # current_id must be as int
        self.assertRaises(
            Exception, self.createRoleMaker,
            current_id=-1)  # current_id must be greater than or equal to 0
        self.assertRaises(
            Exception,
            self.createRoleMaker,
            current_id=1,
            worker_endpoints=[
                "127.0.0.1:8080"
            ])  # current_id must be less than len(worker_endpoints)


class CollectiveOptimizerTest(unittest.TestCase):

    def test_ds_as_None(self):
        optimizer = fluid.optimizer.AdamOptimizer()
        dist_optimizer = CollectiveOptimizer(optimizer, strategy=None)


if __name__ == '__main__':
    unittest.main()
