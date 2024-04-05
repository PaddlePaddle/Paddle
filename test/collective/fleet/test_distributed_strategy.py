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

import os
import unittest

import paddle
from paddle import base
from paddle.distributed.transpiler.distribute_transpiler import (
    DistributeTranspilerConfig,
    ServerRuntimeConfig,
)
from paddle.incubate.distributed.fleet import role_maker
from paddle.incubate.distributed.fleet.parameter_server.distribute_transpiler import (
    fleet,
)
from paddle.incubate.distributed.fleet.parameter_server.distribute_transpiler.distributed_strategy import (
    StrategyFactory,
)


class TestStrategyFactor(unittest.TestCase):
    def test_sync_strategy(self):
        os.environ['CPU_NUM'] = "2"
        strategy = StrategyFactory.create_sync_strategy()
        self.assertEqual(strategy._program_config.sync_mode, False)
        self.assertEqual(strategy._program_config.runtime_split_send_recv, True)
        self.assertEqual(strategy._build_strategy.async_mode, True)

        # test set_program_config using DistributeTranspilerConfig()
        program_config_class = DistributeTranspilerConfig()
        program_config_class.min_block_size = 81920
        strategy.set_program_config(program_config_class)
        program_config = strategy.get_program_config()
        self.assertEqual(program_config.min_block_size, 81920)

        # test set_program_config using dict
        program_config_dict = {}
        program_config_dict['min_block_size'] = 8192
        strategy.set_program_config(program_config_dict)
        program_config = strategy.get_program_config()
        self.assertEqual(program_config.min_block_size, 8192)

        # test set_program_config exception
        program_config_dict['unknown'] = None
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
        self.assertEqual(runtime_configs['communicator_send_queue_size'], '2')

    def test_geo_strategy(self):
        strategy = StrategyFactory.create_geo_strategy(5)
        self.assertEqual(strategy._program_config.sync_mode, False)
        self.assertEqual(strategy._program_config.runtime_split_send_recv, True)
        self.assertEqual(strategy._program_config.geo_sgd_mode, True)
        self.assertEqual(strategy._program_config.geo_sgd_need_push_nums, 5)
        self.assertEqual(strategy._build_strategy.async_mode, True)

        # test set_build_strategy using base.BuildStrategy
        build_strategy_class = base.BuildStrategy()
        build_strategy_class.memory_optimize = False
        strategy.set_build_strategy(build_strategy_class)
        build_strategy = strategy.get_build_strategy()
        self.assertEqual(build_strategy.memory_optimize, False)

        # test set_build_strategy using dict
        build_strategy_dict = {}
        build_strategy_dict['memory_optimize'] = True
        strategy.set_build_strategy(build_strategy_dict)
        build_strategy = strategy.get_build_strategy()
        self.assertEqual(build_strategy.memory_optimize, True)

        # test set_build_strategy exception
        build_strategy_dict['unknown'] = None
        self.assertRaises(
            Exception, strategy.set_build_strategy, build_strategy_dict
        )
        build_strategy_illegal = None
        self.assertRaises(
            Exception, strategy.set_build_strategy, build_strategy_illegal
        )

        os.environ["CPU_NUM"] = '100'
        trainer_runtime_config = strategy.get_trainer_runtime_config()
        runtime_configs = trainer_runtime_config.get_communicator_flags()
        self.assertIn('communicator_thread_pool_size', runtime_configs)
        self.assertIn('communicator_send_wait_times', runtime_configs)
        self.assertNotIn(
            'communicator_independent_recv_thread', runtime_configs
        )

    def test_async_strategy(self):
        os.environ["CPU_NUM"] = '100'

        strategy = StrategyFactory.create_async_strategy()
        self.assertEqual(strategy._program_config.sync_mode, False)
        self.assertEqual(strategy._program_config.runtime_split_send_recv, True)
        self.assertEqual(strategy._build_strategy.async_mode, True)

        trainer_runtime_config = strategy.get_trainer_runtime_config()
        self.assertEqual(
            trainer_runtime_config.runtime_configs[
                'communicator_send_queue_size'
            ],
            '100',
        )

        # test set_trainer_runtime_config using dict
        trainer_runtime_config_dict = {}
        trainer_runtime_config_dict['communicator_send_queue_size'] = '20'
        strategy.set_trainer_runtime_config(trainer_runtime_config_dict)
        trainer_runtime_config = strategy.get_trainer_runtime_config()
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
        server_runtime_config_dict = {}
        server_runtime_config_dict['_rpc_send_thread_num'] = 20
        strategy.set_server_runtime_config(server_runtime_config_dict)
        server_runtime_config = strategy.get_server_runtime_config()
        self.assertEqual(server_runtime_config._rpc_send_thread_num, 20)

        # test set_server_runtime_config exception
        server_runtime_config_dict['unknown'] = None
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

        os.environ["CPU_NUM"] = '100'
        trainer_runtime_config = strategy.get_trainer_runtime_config()
        trainer_runtime_config.runtime_configs[
            'communicator_send_queue_size'
        ] = '50'
        runtime_configs = trainer_runtime_config.get_communicator_flags()
        self.assertIn('communicator_send_queue_size', runtime_configs)
        self.assertNotIn(
            'communicator_independent_recv_thread', runtime_configs
        )
        self.assertEqual(runtime_configs['communicator_send_queue_size'], '100')


class TestCreateDefaultStrategy(unittest.TestCase):
    def test_default_strategy(self):
        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.WORKER,
            worker_num=2,
            server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"],
        )
        fleet.init(role)

        def type_error_optimizer():
            optimizer = paddle.optimizer.SGD(0.0001)
            optimizer = fleet.distributed_optimizer(optimizer)

        self.assertRaises(TypeError, type_error_optimizer)


class TestHalfAsyncStrategy(unittest.TestCase):
    def test_half_async_strategy(self):
        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.WORKER,
            worker_num=2,
            server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"],
        )
        fleet.init(role)

        half_async_config = DistributeTranspilerConfig()

        half_async_config.sync_mode = False
        half_async_config.geo_sgd_mode = False
        half_async_config.runtime_split_send_recv = False

        optimizer = paddle.optimizer.SGD(0.0001)
        optimizer = fleet.distributed_optimizer(optimizer, half_async_config)


class TestDebugInfo(unittest.TestCase):
    def test_debug_info(self):
        x = paddle.static.data(name='x', shape=[-1, 1], dtype='float32')
        y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
        y_predict = paddle.static.nn.fc(x, size=1, activation=None)
        cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
        avg_cost = paddle.mean(cost)

        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.WORKER,
            worker_num=2,
            server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"],
        )
        fleet.init(role)

        optimizer = paddle.optimizer.SGD(0.0001)
        strategy = StrategyFactory.create_sync_strategy()
        strategy.set_debug_opt(
            {
                "dump_param": ["fc_0.tmp_0"],
                "dump_fields": ["fc_0.tmp_0", "fc_0.tmp_0@GRAD"],
                "dump_fields_path": "dump_text/",
            }
        )
        optimizer = fleet.distributed_optimizer(optimizer, strategy)


if __name__ == '__main__':
    unittest.main()
