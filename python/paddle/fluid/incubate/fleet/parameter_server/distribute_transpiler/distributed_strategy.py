#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = [
    "TrainerRuntimeConfig", "DistributedStrategy", "SyncStrategy",
    "AsyncStrategy", "HalfAsyncStrategy", "GeoStrategy", "StrategyFactory"
]

import os
import paddle.fluid as fluid
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig, ServerRuntimeConfig


class TrainerRuntimeConfig(object):
    def __init__(self):
        self.runtime_configs = {}
        # not used 
        self.runtime_configs['rpc_deadline'] = os.getenv("FLAGS_rpc_deadline",
                                                         "180000")
        self.runtime_configs['rpc_retry_times'] = os.getenv(
            "FLAGS_rpc_retry_times", "3")

    def get_communicator_flags(self):
        return self.runtime_configs

    def __repr__(self):
        raw0, raw1, length = 45, 5, 50
        h_format = "{:^45s}{:<5s}\n"
        l_format = "{:<45s}{:<5s}\n"

        border = "".join(["="] * length)
        line = "".join(["-"] * length)

        draws = ""
        draws += border + "\n"
        draws += h_format.format("TrainerRuntimeConfig Overview", "Value")
        draws += line + "\n"

        for k, v in self.get_communicator_flags().items():
            draws += l_format.format(k, v)

        draws += border

        _str = "\n{}\n".format(draws)
        return _str


class DistributedStrategy(object):
    def __init__(self):
        self._program_config = DistributeTranspilerConfig()
        self._trainer_runtime_config = TrainerRuntimeConfig()
        self._server_runtime_config = ServerRuntimeConfig()
        num_threads = int(os.getenv("CPU_NUM", "1"))

        self._execute_strategy = fluid.ExecutionStrategy()
        self._build_strategy = fluid.BuildStrategy()

        self._execute_strategy.num_threads = num_threads
        if num_threads > 1:
            self._build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        self.debug_opt = None

    def set_debug_opt(self, opt_info):
        self.debug_opt = opt_info

    def get_debug_opt(self):
        opt_info = dict()
        if self.debug_opt is not None and isinstance(self.debug_opt, dict):
            opt_info["dump_slot"] = bool(self.debug_opt.get("dump_slot", 0))
            opt_info["dump_converter"] = str(
                self.debug_opt.get("dump_converter", ""))
            opt_info["dump_fields"] = self.debug_opt.get("dump_fields", [])
            opt_info["dump_file_num"] = self.debug_opt.get("dump_file_num", 16)
            opt_info["dump_fields_path"] = self.debug_opt.get(
                "dump_fields_path", "")
            opt_info["dump_param"] = self.debug_opt.get("dump_param", [])
        return opt_info

    def get_program_config(self):
        return self._program_config

    def set_program_config(self, config):
        if isinstance(config, DistributeTranspilerConfig):
            self._program_config = config
        elif isinstance(config, dict):
            for key in config:
                if hasattr(self._program_config, key):
                    setattr(self._program_config, key, config[key])
                else:
                    raise ValueError(
                        "DistributeTranspilerConfig doesn't have key: {}".
                        format(key))
        else:
            raise TypeError(
                "program_config only accept input type: dict or DistributeTranspilerConfig"
            )

    def get_trainer_runtime_config(self):
        return self._trainer_runtime_config

    def set_trainer_runtime_config(self, config):
        if isinstance(config, TrainerRuntimeConfig):
            self._trainer_runtime_config = config
        elif isinstance(config, dict):
            for key, Value in config.items():
                if key in self._trainer_runtime_config.runtime_configs:
                    self._trainer_runtime_config.runtime_configs[key] = Value
                else:
                    raise ValueError(
                        "TrainerRuntimeConfig doesn't have key: {}".format(key))
        else:
            raise TypeError(
                "trainer_runtime_config only accept input type: dict or TrainerRuntimeConfig"
            )

    def get_server_runtime_config(self):
        return self._server_runtime_config

    def set_server_runtime_config(self, config):
        if isinstance(config, ServerRuntimeConfig):
            self._server_runtime_config = config
        elif isinstance(config, dict):
            for key in config:
                if hasattr(self._server_runtime_config, key):
                    setattr(self._server_runtime_config, key, config[key])
                else:
                    raise ValueError(
                        "ServerRuntimeConfig doesn't have key: {}".format(key))
        else:
            raise TypeError(
                "server_runtime_config only accept input type: dict or ServerRuntimeConfig"
            )

    def get_execute_strategy(self):
        return self._execute_strategy

    def set_execute_strategy(self, config):
        if isinstance(config, fluid.ExecutionStrategy):
            self._execute_strategy = config
        elif isinstance(config, dict):
            for key in config:
                if hasattr(self._execute_strategy, key):
                    setattr(self._execute_strategy, key, config[key])
                else:
                    raise ValueError(
                        "ExecutionStrategy doesn't have key: {}".format(key))
        else:
            raise TypeError(
                "execute_strategy only accept input type: dict or ExecutionStrategy"
            )

    def get_build_strategy(self):
        return self._build_strategy

    def set_build_strategy(self, config):
        if isinstance(config, fluid.BuildStrategy):
            self._build_strategy = config
        elif isinstance(config, dict):
            for key in config:
                if hasattr(self._build_strategy, key):
                    setattr(self._build_strategy, key, config[key])
                else:
                    raise ValueError(
                        "BuildStrategy doesn't have key: {}".format(key))
        else:
            raise TypeError(
                "build_strategy only accept input type: dict or BuildStrategy")


class SyncStrategy(DistributedStrategy):
    def __init__(self):
        super(SyncStrategy, self).__init__()
        self._program_config.sync_mode = False
        self._program_config.runtime_split_send_recv = True
        self._build_strategy.async_mode = True
        self._program_config.half_async = True
        self._program_config.completely_not_async = True
        self._execute_strategy.use_thread_barrier = True

        num_threads = os.getenv("CPU_NUM", "1")

        self._trainer_runtime_config.runtime_configs[
            'communicator_max_merge_var_num'] = os.getenv(
                "FLAGS_communicator_max_merge_var_num", num_threads)
        self._trainer_runtime_config.runtime_configs[
            'communicator_send_wait_times'] = os.getenv(
                "FLAGS_communicator_send_wait_times", "5")
        self._trainer_runtime_config.runtime_configs[
            'communicator_thread_pool_size'] = os.getenv(
                "FLAGS_communicator_thread_pool_size", "10")
        self._trainer_runtime_config.runtime_configs[
            'communicator_send_queue_size'] = os.getenv(
                "FLAGS_communicator_send_queue_size", num_threads)


class AsyncStrategy(DistributedStrategy):
    def __init__(self):
        super(AsyncStrategy, self).__init__()
        self._program_config.sync_mode = False
        self._program_config.runtime_split_send_recv = True
        self._build_strategy.async_mode = True

        num_threads = os.getenv("CPU_NUM", "1")

        self._trainer_runtime_config.runtime_configs[
            'communicator_max_merge_var_num'] = os.getenv(
                "FLAGS_communicator_max_merge_var_num", num_threads)
        self._trainer_runtime_config.runtime_configs[
            'communicator_independent_recv_thread'] = os.getenv(
                "FLAGS_communicator_independent_recv_thread", "0")
        self._trainer_runtime_config.runtime_configs[
            'communicator_min_send_grad_num_before_recv'] = os.getenv(
                "FLAGS_communicator_min_send_grad_num_before_recv", num_threads)
        self._trainer_runtime_config.runtime_configs[
            'communicator_thread_pool_size'] = os.getenv(
                "FLAGS_communicator_thread_pool_size", "10")
        self._trainer_runtime_config.runtime_configs[
            'communicator_send_wait_times'] = os.getenv(
                "FLAGS_communicator_send_wait_times", "5")
        self._trainer_runtime_config.runtime_configs[
            'communicator_is_sgd_optimizer'] = os.getenv(
                "FLAGS_communicator_is_sgd_optimizer", "1")
        self._trainer_runtime_config.runtime_configs[
            'communicator_send_queue_size'] = os.getenv(
                "FLAGS_communicator_send_queue_size", num_threads)


class HalfAsyncStrategy(DistributedStrategy):
    def __init__(self):
        super(HalfAsyncStrategy, self).__init__()
        self._program_config.sync_mode = False
        self._program_config.runtime_split_send_recv = True
        self._program_config.half_async = True
        self._build_strategy.async_mode = True
        self._execute_strategy.use_thread_barrier = True

        num_threads = os.getenv("CPU_NUM", "1")

        self._trainer_runtime_config.runtime_configs[
            'communicator_max_merge_var_num'] = os.getenv(
                "FLAGS_communicator_max_merge_var_num", num_threads)
        self._trainer_runtime_config.runtime_configs[
            'communicator_send_wait_times'] = os.getenv(
                "FLAGS_communicator_send_wait_times", "5")
        self._trainer_runtime_config.runtime_configs[
            'communicator_thread_pool_size'] = os.getenv(
                "FLAGS_communicator_thread_pool_size", "10")
        self._trainer_runtime_config.runtime_configs[
            'communicator_send_queue_size'] = os.getenv(
                "FLAGS_communicator_send_queue_size", num_threads)


class GeoStrategy(DistributedStrategy):
    def __init__(self, update_frequency=100):
        super(GeoStrategy, self).__init__()
        self._program_config.sync_mode = False
        self._program_config.runtime_split_send_recv = True
        self._program_config.geo_sgd_mode = True
        self._program_config.geo_sgd_need_push_nums = update_frequency
        self._build_strategy.async_mode = True

        self._trainer_runtime_config.runtime_configs[
            'communicator_thread_pool_size'] = os.getenv(
                "FLAGS_communicator_thread_pool_size", "10")
        self._trainer_runtime_config.runtime_configs[
            'communicator_send_wait_times'] = os.getenv(
                "FLAGS_communicator_send_wait_times", "5")


class StrategyFactory(object):
    def __init_(self):
        pass

    @staticmethod
    def create_sync_strategy():
        return SyncStrategy()

    @staticmethod
    def create_half_async_strategy():
        return HalfAsyncStrategy()

    @staticmethod
    def create_async_strategy():
        return AsyncStrategy()

    @staticmethod
    def create_geo_strategy(update_frequency=100):
        return GeoStrategy(update_frequency)
