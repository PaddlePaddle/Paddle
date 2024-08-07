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

__all__ = []

import os

from paddle import base
from paddle.distributed.transpiler.distribute_transpiler import (
    DistributeTranspilerConfig,
    ServerRuntimeConfig,
)
from paddle.incubate.distributed.fleet.parameter_server.mode import (
    DistributedMode,
)


class TrainerRuntimeConfig:
    def __init__(self):
        self.mode = None
        num_threads = os.getenv("CPU_NUM", "1")

        self.runtime_configs = {}
        self.runtime_configs['communicator_max_merge_var_num'] = os.getenv(
            "FLAGS_communicator_max_merge_var_num", num_threads
        )
        self.runtime_configs['communicator_send_queue_size'] = os.getenv(
            "FLAGS_communicator_send_queue_size", num_threads
        )
        self.runtime_configs[
            'communicator_independent_recv_thread'
        ] = os.getenv("FLAGS_communicator_independent_recv_thread", "1")
        self.runtime_configs[
            'communicator_min_send_grad_num_before_recv'
        ] = os.getenv(
            "FLAGS_communicator_min_send_grad_num_before_recv", num_threads
        )
        self.runtime_configs['communicator_thread_pool_size'] = os.getenv(
            "FLAGS_communicator_thread_pool_size", "5"
        )
        self.runtime_configs['communicator_send_wait_times'] = os.getenv(
            "FLAGS_communicator_send_wait_times", "5"
        )
        self.runtime_configs['communicator_is_sgd_optimizer'] = os.getenv(
            "FLAGS_communicator_is_sgd_optimizer", "1"
        )

        # not used
        self.runtime_configs['rpc_deadline'] = os.getenv(
            "FLAGS_rpc_deadline", "180000"
        )
        self.runtime_configs['rpc_retry_times'] = os.getenv(
            "FLAGS_rpc_retry_times", "3"
        )

    def get_communicator_flags(self):
        need_keys = []
        num_threads = os.getenv("CPU_NUM", "1")
        mode_str = ""
        if self.mode is None or self.mode == DistributedMode.ASYNC:
            need_keys = self.runtime_configs.keys()
            mode_str = "async"
        elif (
            self.mode == DistributedMode.SYNC
            or self.mode == DistributedMode.HALF_ASYNC
        ):
            mode_str = "sync or half_async"
            need_keys = [
                'communicator_max_merge_var_num',
                'communicator_send_wait_times',
                'communicator_thread_pool_size',
                'communicator_send_queue_size',
            ]
        elif self.mode == DistributedMode.GEO:
            mode_str = "GEO"
            need_keys = [
                'communicator_thread_pool_size',
                'communicator_send_wait_times',
                'communicator_max_merge_var_num',
                'communicator_send_queue_size',
            ]
        else:
            raise ValueError("Unsupported Mode")

        if (
            self.mode == DistributedMode.SYNC
            or self.mode == DistributedMode.HALF_ASYNC
        ):
            max_merge_var_num = self.runtime_configs[
                'communicator_max_merge_var_num'
            ]
            send_queue_size = self.runtime_configs[
                'communicator_send_queue_size'
            ]
            if max_merge_var_num != num_threads:
                print(
                    f'WARNING: In {mode_str} mode, communicator_max_merge_var_num '
                    'must be equal to CPU_NUM. But received, '
                    f'communicator_max_merge_var_num = {max_merge_var_num}, CPU_NUM = '
                    f'{num_threads}. communicator_max_merge_var_num will be forced to {num_threads}.'
                )
                self.runtime_configs[
                    'communicator_max_merge_var_num'
                ] = num_threads
            if send_queue_size != num_threads:
                print(
                    f'WARNING: In {mode_str} mode, communicator_send_queue_size '
                    'must be equal to CPU_NUM. But received, '
                    f'communicator_send_queue_size = {send_queue_size}, CPU_NUM = '
                    f'{num_threads}. communicator_send_queue_size will be forced to {num_threads}.'
                )
                self.runtime_configs[
                    'communicator_send_queue_size'
                ] = num_threads

        return {key: str(self.runtime_configs[key]) for key in need_keys}

    def display(self, configs):
        raw0, raw1, length = 45, 5, 50
        h_format = "{:^45s}{:<5s}\n"
        l_format = "{:<45s}{:<5s}\n"

        border = "".join(["="] * length)
        line = "".join(["-"] * length)

        draws = ""
        draws += border + "\n"
        draws += h_format.format("TrainerRuntimeConfig Overview", "Value")
        draws += line + "\n"

        for k, v in configs.items():
            draws += l_format.format(k, v)

        draws += border

        _str = f"\n{draws}\n"
        return _str

    def __repr__(self):
        return self.display(self.get_communicator_flags())


class PSLibRuntimeConfig:
    def __init__(self):
        self.runtime_configs = {}

    def get_runtime_configs(self):
        return self.runtime_configs


class DistributedStrategy:
    def __init__(self):
        self._program_config = DistributeTranspilerConfig()
        self._trainer_runtime_config = TrainerRuntimeConfig()
        self._pslib_runtime_config = PSLibRuntimeConfig()
        self._server_runtime_config = ServerRuntimeConfig()
        num_threads = int(os.getenv("CPU_NUM", "1"))

        self._build_strategy = base.BuildStrategy()

        if num_threads > 1:
            self._build_strategy.reduce_strategy = (
                base.BuildStrategy.ReduceStrategy.Reduce
            )
        self.debug_opt = None
        self.use_ps_gpu = False

    def set_debug_opt(self, opt_info):
        self.debug_opt = opt_info

    def get_debug_opt(self):
        opt_info = {}
        if self.debug_opt is not None and isinstance(self.debug_opt, dict):
            opt_info["dump_slot"] = bool(self.debug_opt.get("dump_slot", 0))
            opt_info["dump_converter"] = str(
                self.debug_opt.get("dump_converter", "")
            )
            opt_info["dump_fields"] = self.debug_opt.get("dump_fields", [])
            opt_info["dump_file_num"] = self.debug_opt.get("dump_file_num", 16)
            opt_info["dump_fields_path"] = self.debug_opt.get(
                "dump_fields_path", ""
            )
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
                        f"DistributeTranspilerConfig doesn't have key: {key}"
                    )
        else:
            raise TypeError(
                "program_config only accept input type: dict or DistributeTranspilerConfig"
            )
        self.check_program_config()

    def check_program_config(self):
        raise NotImplementedError(
            "check_program_config must be implemented by derived class. You should use StrategyFactory to create DistributedStrategy."
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
                        f"TrainerRuntimeConfig doesn't have key: {key}"
                    )
        else:
            raise TypeError(
                "trainer_runtime_config only accept input type: dict or TrainerRuntimeConfig"
            )
        self.check_trainer_runtime_config()

    def check_trainer_runtime_config(self):
        raise NotImplementedError(
            "check_trainer_runtime_config must be implemented by derived class. You should use StrategyFactory to create DistributedStrategy."
        )

    def get_pslib_runtime_config(self):
        return self._pslib_runtime_config

    def set_pslib_runtime_config(self, config):
        self._pslib_runtime_config.runtime_configs = config

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
                        f"ServerRuntimeConfig doesn't have key: {key}"
                    )
        else:
            raise TypeError(
                "server_runtime_config only accept input type: dict or ServerRuntimeConfig"
            )
        self.check_server_runtime_config()

    def check_server_runtime_config(self):
        raise NotImplementedError(
            "check_server_runtime_config must be implemented by derived class. You should use StrategyFactory to create DistributedStrategy."
        )

    def get_build_strategy(self):
        return self._build_strategy

    def set_build_strategy(self, config):
        if isinstance(config, base.BuildStrategy):
            self._build_strategy = config
        elif isinstance(config, dict):
            for key in config:
                if hasattr(self._build_strategy, key):
                    setattr(self._build_strategy, key, config[key])
                else:
                    raise ValueError(f"BuildStrategy doesn't have key: {key}")
        else:
            raise TypeError(
                "build_strategy only accept input type: dict or BuildStrategy"
            )
        self.check_build_strategy()

    def check_build_strategy(self):
        raise NotImplementedError(
            "check_build_strategy must be implemented by derived class. You should use StrategyFactory to create DistributedStrategy."
        )


class SyncStrategy(DistributedStrategy):
    def __init__(self):
        super().__init__()
        self.check_program_config()
        self.check_trainer_runtime_config()
        self.check_server_runtime_config()
        self.check_build_strategy()

    def check_trainer_runtime_config(self):
        self._trainer_runtime_config.mode = DistributedMode.SYNC

    def check_program_config(self):
        self._program_config.sync_mode = False
        self._program_config.runtime_split_send_recv = True
        self._program_config.half_async = True
        self._program_config.completely_not_async = True

    def check_server_runtime_config(self):
        pass

    def check_build_strategy(self):
        self._build_strategy.async_mode = True


class AsyncStrategy(DistributedStrategy):
    def __init__(self):
        super().__init__()
        self.check_program_config()
        self.check_trainer_runtime_config()
        self.check_server_runtime_config()
        self.check_build_strategy()

    def check_trainer_runtime_config(self):
        self._trainer_runtime_config.mode = DistributedMode.ASYNC

    def check_program_config(self):
        self._program_config.sync_mode = False
        self._program_config.runtime_split_send_recv = True

    def check_server_runtime_config(self):
        pass

    def check_build_strategy(self):
        self._build_strategy.async_mode = True


class HalfAsyncStrategy(DistributedStrategy):
    def __init__(self):
        super().__init__()
        self.check_program_config()
        self.check_trainer_runtime_config()
        self.check_server_runtime_config()
        self.check_build_strategy()

    def check_trainer_runtime_config(self):
        self._trainer_runtime_config.mode = DistributedMode.HALF_ASYNC

    def check_program_config(self):
        self._program_config.sync_mode = False
        self._program_config.runtime_split_send_recv = True
        self._program_config.half_async = True

    def check_server_runtime_config(self):
        pass

    def check_build_strategy(self):
        self._build_strategy.async_mode = True


class GeoStrategy(DistributedStrategy):
    def __init__(self, update_frequency=100):
        super().__init__()
        self._program_config.geo_sgd_need_push_nums = update_frequency
        self.check_program_config()
        self.check_trainer_runtime_config()
        self.check_server_runtime_config()
        self.check_build_strategy()

    def check_program_config(self):
        self._program_config.sync_mode = False
        self._program_config.runtime_split_send_recv = True
        self._program_config.geo_sgd_mode = True

    def check_trainer_runtime_config(self):
        self._trainer_runtime_config.mode = DistributedMode.GEO

        self._trainer_runtime_config.runtime_configs[
            'communicator_send_queue_size'
        ] = self._program_config.geo_sgd_need_push_nums

        self._trainer_runtime_config.runtime_configs[
            'communicator_max_merge_var_num'
        ] = self._program_config.geo_sgd_need_push_nums

    def check_server_runtime_config(self):
        pass

    def check_build_strategy(self):
        self._build_strategy.async_mode = True


class StrategyFactory:
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
