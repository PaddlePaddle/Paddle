# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import os
import unittest

import paddle
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import (
    fleet,
)
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import (
    StrategyFactory,
)
from paddle.fluid.transpiler.distribute_transpiler import (
    DistributeTranspilerConfig,
    ServerRuntimeConfig,
)


class TestStrategyFactor(unittest.TestCase):
=======
import unittest
import paddle
import paddle.fluid as fluid
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig, ServerRuntimeConfig
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
import os


class TestStrategyFactor(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_sync_strategy(self):
        os.environ['CPU_NUM'] = "2"
        strategy = StrategyFactory.create_sync_strategy()
        self.assertEqual(strategy._program_config.sync_mode, False)
        self.assertEqual(strategy._program_config.runtime_split_send_recv, True)
        self.assertEqual(strategy._build_strategy.async_mode, True)
        self.assertEqual(strategy._execute_strategy.num_threads, 2)

        # test set_program_config using DistributeTranspilerConfig()
        program_config_class = DistributeTranspilerConfig()
        program_config_class.min_block_size = 81920
        strategy.set_program_config(program_config_class)
        program_config = strategy.get_program_config()
        self.assertEqual(program_config.min_block_size, 81920)

        # test set_program_config using dict
        program_config_dict = dict()
        program_config_dict['min_block_size'] = 8192
        strategy.set_program_config(program_config_dict)
        program_config = strategy.get_program_config()
        self.assertEqual(program_config.min_block_size, 8192)

        # test set_program_config exception
        program_config_dict['unknown'] = None
<<<<<<< HEAD
        self.assertRaises(
            Exception, strategy.set_program_config, program_config_dict
        )
        program_config_illegal = None
        self.assertRaises(
            Exception, strategy.set_program_config, program_config_illegal
        )

        trainer_runtime_config = strategy.get_trainer_runtime_config()
        trainer_runtime_config.runtime_configs[
            'communicator_send_queue_size'
        ] = '50'
        runtime_configs = trainer_runtime_config.get_communicator_flags()
        self.assertIn('communicator_send_queue_size', runtime_configs)
        self.assertNotIn(
            'communicator_independent_recv_thread', runtime_configs
        )
=======
        self.assertRaises(Exception, strategy.set_program_config,
                          program_config_dict)
        program_config_illegal = None
        self.assertRaises(Exception, strategy.set_program_config,
                          program_config_illegal)

        trainer_runtime_config = strategy.get_trainer_runtime_config()
        trainer_runtime_config.runtime_configs[
            'communicator_send_queue_size'] = '50'
        runtime_configs = trainer_runtime_config.get_communicator_flags()
        self.assertIn('communicator_send_queue_size', runtime_configs)
        self.assertNotIn('communicator_independent_recv_thread',
                         runtime_configs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertEqual(runtime_configs['communicator_send_queue_size'], '2')

    def test_geo_strategy(self):
        strategy = StrategyFactory.create_geo_strategy(5)
        self.assertEqual(strategy._program_config.sync_mode, False)
        self.assertEqual(strategy._program_config.runtime_split_send_recv, True)
        self.assertEqual(strategy._program_config.geo_sgd_mode, True)
        self.assertEqual(strategy._program_config.geo_sgd_need_push_nums, 5)
        self.assertEqual(strategy._build_strategy.async_mode, True)

        # test set_build_strategy using fluid.BuildStrategy
        build_strategy_class = fluid.BuildStrategy()
        build_strategy_class.memory_optimize = False
        strategy.set_build_strategy(build_strategy_class)
        build_strategy = strategy.get_build_strategy()
        self.assertEqual(build_strategy.memory_optimize, False)

        # test set_build_strategy using dict
        build_strategy_dict = dict()
        build_strategy_dict['memory_optimize'] = True
        strategy.set_build_strategy(build_strategy_dict)
        build_strategy = strategy.get_build_strategy()
        self.assertEqual(build_strategy.memory_optimize, True)

        # test set_build_strategy exception
        build_strategy_dict['unknown'] = None
<<<<<<< HEAD
        self.assertRaises(
            Exception, strategy.set_build_strategy, build_strategy_dict
        )
        build_strategy_illegal = None
        self.assertRaises(
            Exception, strategy.set_build_strategy, build_strategy_illegal
        )
=======
        self.assertRaises(Exception, strategy.set_build_strategy,
                          build_strategy_dict)
        build_strategy_illegal = None
        self.assertRaises(Exception, strategy.set_build_strategy,
                          build_strategy_illegal)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        os.environ["CPU_NUM"] = '100'
        trainer_runtime_config = strategy.get_trainer_runtime_config()
        runtime_configs = trainer_runtime_config.get_communicator_flags()
        self.assertIn('communicator_thread_pool_size', runtime_configs)
        self.assertIn('communicator_send_wait_times', runtime_configs)
<<<<<<< HEAD
        self.assertNotIn(
            'communicator_independent_recv_thread', runtime_configs
        )
=======
        self.assertNotIn('communicator_independent_recv_thread',
                         runtime_configs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_async_strategy(self):
        os.environ["CPU_NUM"] = '100'

        strategy = StrategyFactory.create_async_strategy()
        self.assertEqual(strategy._program_config.sync_mode, False)
        self.assertEqual(strategy._program_config.runtime_split_send_recv, True)
        self.assertEqual(strategy._build_strategy.async_mode, True)

        trainer_runtime_config = strategy.get_trainer_runtime_config()
        self.assertEqual(
<<<<<<< HEAD
            trainer_runtime_config.runtime_configs[
                'communicator_send_queue_size'
            ],
            '100',
        )
=======
            trainer_runtime_config.
            runtime_configs['communicator_send_queue_size'], '100')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # test set_trainer_runtime_config using dict
        trainer_runtime_config_dict = dict()
        trainer_runtime_config_dict['communicator_send_queue_size'] = '20'
        strategy.set_trainer_runtime_config(trainer_runtime_config_dict)
        trainer_runtime_config = strategy.get_trainer_runtime_config()
<<<<<<< HEAD
        trainer_communicator_flags = (
            trainer_runtime_config.get_communicator_flags()
        )
        self.assertIn(
            'communicator_send_queue_size', trainer_communicator_flags
        )
        self.assertEqual(
            trainer_communicator_flags['communicator_send_queue_size'], '20'
        )

        # test set_trainer_runtime_config exception
        trainer_runtime_config_dict['unknown'] = None
        self.assertRaises(
            Exception,
            strategy.set_trainer_runtime_config,
            trainer_runtime_config_dict,
        )
        trainer_runtime_config_illegal = None
        self.assertRaises(
            Exception,
            strategy.set_trainer_runtime_config,
            trainer_runtime_config_illegal,
        )
=======
        trainer_communicator_flags = trainer_runtime_config.get_communicator_flags(
        )
        self.assertIn('communicator_send_queue_size',
                      trainer_communicator_flags)
        self.assertEqual(
            trainer_communicator_flags['communicator_send_queue_size'], '20')

        # test set_trainer_runtime_config exception
        trainer_runtime_config_dict['unknown'] = None
        self.assertRaises(Exception, strategy.set_trainer_runtime_config,
                          trainer_runtime_config_dict)
        trainer_runtime_config_illegal = None
        self.assertRaises(Exception, strategy.set_trainer_runtime_config,
                          trainer_runtime_config_illegal)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # test set_execute_strategy using fluid.ExecutionStrategy
        exec_strategy_class = fluid.ExecutionStrategy()
        exec_strategy_class.num_threads = 4
        strategy.set_execute_strategy(exec_strategy_class)
        exec_strategy = strategy.get_execute_strategy()
        self.assertEqual(exec_strategy.num_threads, 4)

        # test set_execute_strategy using dict
        exec_strategy_dict = dict()
        exec_strategy_dict['num_threads'] = 8
        strategy.set_execute_strategy(exec_strategy_dict)
        exec_strategy = strategy.get_execute_strategy()
        self.assertEqual(exec_strategy.num_threads, 8)

        # test set_execute_strategy exception
        exec_strategy_dict['unknown'] = None
<<<<<<< HEAD
        self.assertRaises(
            Exception, strategy.set_execute_strategy, exec_strategy_dict
        )
        exec_strategy_illegal = None
        self.assertRaises(
            Exception, strategy.set_execute_strategy, exec_strategy_illegal
        )
=======
        self.assertRaises(Exception, strategy.set_execute_strategy,
                          exec_strategy_dict)
        exec_strategy_illegal = None
        self.assertRaises(Exception, strategy.set_execute_strategy,
                          exec_strategy_illegal)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_half_async_strategy(self):
        strategy = StrategyFactory.create_half_async_strategy()
        self.assertEqual(strategy._program_config.sync_mode, False)
        self.assertEqual(strategy._program_config.runtime_split_send_recv, True)
        self.assertEqual(strategy._build_strategy.async_mode, True)

        # test set_server_runtime_config using ServerRuntimeConfig
        server_runtime_config_class = ServerRuntimeConfig()
        server_runtime_config_class._rpc_send_thread_num = 24
        strategy.set_server_runtime_config(server_runtime_config_class)
        server_runtime_config = strategy.get_server_runtime_config()
        self.assertEqual(server_runtime_config._rpc_send_thread_num, 24)

        # test set_server_runtime_config using dict
        server_runtime_config_dict = dict()
        server_runtime_config_dict['_rpc_send_thread_num'] = 20
        strategy.set_server_runtime_config(server_runtime_config_dict)
        server_runtime_config = strategy.get_server_runtime_config()
        self.assertEqual(server_runtime_config._rpc_send_thread_num, 20)

        # test set_server_runtime_config exception
        server_runtime_config_dict['unknown'] = None
<<<<<<< HEAD
        self.assertRaises(
            Exception,
            strategy.set_server_runtime_config,
            server_runtime_config_dict,
        )
        server_runtime_config_illegal = None
        self.assertRaises(
            Exception,
            strategy.set_server_runtime_config,
            server_runtime_config_illegal,
        )
=======
        self.assertRaises(Exception, strategy.set_server_runtime_config,
                          server_runtime_config_dict)
        server_runtime_config_illegal = None
        self.assertRaises(Exception, strategy.set_server_runtime_config,
                          server_runtime_config_illegal)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        os.environ["CPU_NUM"] = '100'
        trainer_runtime_config = strategy.get_trainer_runtime_config()
        trainer_runtime_config.runtime_configs[
<<<<<<< HEAD
            'communicator_send_queue_size'
        ] = '50'
        runtime_configs = trainer_runtime_config.get_communicator_flags()
        self.assertIn('communicator_send_queue_size', runtime_configs)
        self.assertNotIn(
            'communicator_independent_recv_thread', runtime_configs
        )
=======
            'communicator_send_queue_size'] = '50'
        runtime_configs = trainer_runtime_config.get_communicator_flags()
        self.assertIn('communicator_send_queue_size', runtime_configs)
        self.assertNotIn('communicator_independent_recv_thread',
                         runtime_configs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertEqual(runtime_configs['communicator_send_queue_size'], '100')


class TestCreateDefaultStrategy(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_default_strategy(self):
        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.WORKER,
            worker_num=2,
<<<<<<< HEAD
            server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"],
        )
=======
            server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        fleet.init(role)

        def type_error_optimizer():
            optimizer = fluid.optimizer.SGD(0.0001)
            optimizer = fleet.distributed_optimizer(optimizer)

        self.assertRaises(TypeError, type_error_optimizer)


class TestHalfAsyncStrategy(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_half_async_strategy(self):
        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.WORKER,
            worker_num=2,
<<<<<<< HEAD
            server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"],
        )
=======
            server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        fleet.init(role)

        half_async_config = DistributeTranspilerConfig()

        half_async_config.sync_mode = False
        half_async_config.geo_sgd_mode = False
        half_async_config.runtime_split_send_recv = False

        optimizer = fluid.optimizer.SGD(0.0001)
        optimizer = fleet.distributed_optimizer(optimizer, half_async_config)


class TestDebugInfo(unittest.TestCase):
<<<<<<< HEAD
    def test_debug_info(self):
        x = paddle.static.data(name='x', shape=[-1, 1], dtype='float32')
        y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
        y_predict = paddle.static.nn.fc(x, size=1, activation=None)
        cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
=======

    def test_debug_info(self):
        x = fluid.layers.data(name='x', shape=[1], dtype='float32')
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        avg_cost = paddle.mean(cost)

        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.WORKER,
            worker_num=2,
<<<<<<< HEAD
            server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"],
        )
=======
            server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        fleet.init(role)

        optimizer = fluid.optimizer.SGD(0.0001)
        strategy = StrategyFactory.create_sync_strategy()
<<<<<<< HEAD
        strategy.set_debug_opt(
            {
                "dump_param": ["fc_0.tmp_0"],
                "dump_fields": ["fc_0.tmp_0", "fc_0.tmp_0@GRAD"],
                "dump_fields_path": "dump_text/",
            }
        )
=======
        strategy.set_debug_opt({
            "dump_param": ["fc_0.tmp_0"],
            "dump_fields": ["fc_0.tmp_0", "fc_0.tmp_0@GRAD"],
            "dump_fields_path": "dump_text/"
        })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        optimizer = fleet.distributed_optimizer(optimizer, strategy)


if __name__ == '__main__':
    unittest.main()
