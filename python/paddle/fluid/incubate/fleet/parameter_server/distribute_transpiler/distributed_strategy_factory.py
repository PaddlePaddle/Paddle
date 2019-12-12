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

import os
import paddle.fluid as fluid
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig


class ServerRuntimeConfig(object):
    def __init__(self):
        self._rpc_send_thread_num = int(
            os.getenv("FLAGS_rpc_send_thread_num", "12"))
        self._rpc_get_thread_num = int(
            os.getenv("FLAGS_rpc_get_thread_num", "12"))
        self._rpc_prefetch_thread_num = int(
            os.getenv("FLAGS_rpc_prefetch_thread_num", "12"))

    def __str__(self):
        print_str = "rpc_send_thread_num: {}\nrpc_get_thread_num: {}\nrpc_prefetch_thread_num: {}" % (
            self._rpc_send_thread_num, self._rpc_get_thread_num,
            self._rpc_prefetch_thread_num)
        return print_str

    def __repr__(self):
        return self.__str__()


class TrainerRuntimeConfig(object):
    def __init__(self):
        self._communicator_flags = dict()
        self._communicator_flags["max_merge_var_num"] = int(
            os.getenv("FLAGS_communicator_max_merge_var_num", "20"))
        self._communicator_flags["send_queue_size"] = int(
            os.getenv("FLAGS_communicator_send_queue_size", "20"))
        self._communicator_flags["independent_recv_thread"] = bool(
            int(os.getenv("FLAGS_communicator_independent_recv_thread", "1")))
        self._communicator_flags["min_send_grad_num_before_recv"] = int(
            os.getenv("FLAGS_communicator_min_send_grad_num_before_recv", "20"))
        self._communicator_flags["thread_pool_size"] = int(
            os.getenv("FLAGS_communicator_thread_pool_size", "5"))
        self._communicator_flags["send_wait_times"] = int(
            os.getenv("FLAGS_communicator_send_wait_times", "5"))
        self._communicator_flags["fake_rpc"] = int(
            os.getenv("FLAGS_communicator_fake_rpc", "0"))
        self._communicator_flags["merge_sparse_grad"] = int(
            os.getenv("FLAGS_communicator_merge_sparse_grad", "1"))
        self._communicator_flags["is_sgd_optimizer"] = int(
            os.getenv("communicator_is_sgd_optimizer", "1"))

        self._rpc_deadline = int(os.getenv("FLAGS_rpc_deadline", "180000"))
        self._rpc_retry_times = int(os.getenv("FLAGS_rpc_retry_times", "3"))

    def __str__(self):
        print_str = "communicator_max_merge_var_num: {}\n" % self._communicator_flags[
            "max_merge_var_num"]
        print_str += "communicator_send_queue_size: {}\n" % self._communicator_flags[
            "send_queue_size"]
        print_str += "communicator_independent_recv_thread: {}\n" % self._communicator_flags[
            "independent_recv_thread"]
        print_str += "communicator_min_send_grad_num_before_recv: {}\n" % self._communicator_flags[
            "min_send_grad_num_before_recv"]
        print_str += "communicator_thread_pool_size: {}\n" % self._communicator_flags[
            "thread_pool_size"]
        print_str += "communicator_send_wait_times: {}\n" % self._communicator_flags[
            "send_wait_times"]
        print_str += "communicator_fake_rpc: {}\n" % self._communicator_flags[
            "fake_rpc"]
        print_str += "communicator_merge_sparse_grad: {}\n" % self._communicator_flags[
            "merge_sparse_grad"]
        print_str += "rpc_deadline: {}\n" % self._rpc_deadline
        print_str += "rpc_retry_times: {}" % self._rpc_retry_times
        return print_str

    def __repr__(self):
        return self.__str__()


class DistributedStrategy(object):
    def __init__(self):
        self.__program_config = DistributeTranspilerConfig()
        self.__trainer_runtime_config = TrainerRuntimeConfig()
        self.__server_runtime_config = ServerRuntimeConfig()
        self.__execute_strategy = fluid.ExecutionStrategy()
        self.__build_strategy = fluid.BuildStrategy()
        num_threads = int(os.getenv("CPU_NUM", "1"))
        self.__execute_strategy.num_threads = num_threads
        if num_threads > 1:
            self.__build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

    def get_program_config(self):
        return self.__program_config

    def set_program_config(self, config):
        assert (isinstance(config, DistributeTranspilerConfig))
        self.__program_config = config

    def get_trainer_runtime_config(self):
        return self.__trainer_runtime_config

    def set_trainer_runtime_config(self, config):
        assert (isinstance(config, TrainerRuntimeConfig))
        self.__trainer_runtime_config = config

    def get_server_runtime_config(self):
        return self.__server_runtime_config

    def set_server_runtime_config(self, config):
        assert (isinstance(config, ServerRuntimeConfig))
        self.__server_runtime_config = config

    def get_execute_strategy(self):
        return self.__execute_strategy

    def set_execute_strategy(self, config):
        assert (isinstance(config, fluid.ExecutionStrategy))
        self.__execute_strategy = config

    def get_build_trategy(self):
        return self.__build_strategy

    def set_build_strategy(self, config):
        assert (isinstance(config, fluid.BuildStrategy))
        self.__build_strategy = config

    # def set_program_config(self, **kwargs):
    #     attr_list = dir(self.__program_config)
    #     for key, v in kwargs.items():
    #         if key not in attr_list:
    #             raise ValueError("TranspilerConfig doesn't have %s attribute" % (key))
    #         if key == "sync_mode" or key == "runtime_split_send_recv" or key == "geo_sgd_mode":
    #             raise ValueError("You can't set {} attribute" % key)
    #         setattr(self.__program_config, key, v)

    # def set_trainer_runtime_config(self, **kwargs):
    #     attr_list = dir(self.__trainer_runtime_config)
    #     for key, v in kwargs.items():
    #         if key not in attr_list:
    #             raise ValueError("TrainerRuntimeConfig doesn't have %s attribute" % (key))
    #         setattr(self.__trainer_runtime_config, key, v)

    # def set_server_runtime_config(self, **kwargs):
    #     attr_list = dir(self.__server_runtime_config)
    #     for key, v in kwargs.items():
    #         if key not in attr_list:
    #             raise ValueError("ServerRuntimeConfig doesn't have %s attribute" % (key))
    #         setattr(self.__server_runtime_config, key, v)

    # def set_execute_strategy(self, **kwargs):
    #     attr_list = dir(self.__execute_strategy)
    #     for key, v in kwargs.items():
    #         if key not in attr_list:
    #             raise ValueError("ExecuteStrategy doesn't have %s attribute" % (key))
    #         setattr(self.__execute_strategy, key, v)

    # def set_build_strategy(self, **kwargs):
    #     attr_list = dir(self.__build_strategy)
    #     for key, v in kwargs.items():
    #         if key not in attr_list:
    #             raise ValueError("BuildStrategy doesn't have %s attribute" % (key))
    #         setattr(self.__build_strategy, key, v)


class SyncStrategy(DistributedStrategy):
    def __init__(self):
        super(SyncStrategy, self).__init__()
        self.__program_config.sync_mode = True
        self.__program_config.runtime_split_send_recv = False
        self.__build_strategy.async_mode = False


class AsyncStrategy(DistributedStrategy):
    def __init__(self):
        super(AsyncStrategy, self).__init__()
        self.__program_config.sync_mode = False
        self.__program_config.runtime_split_send_recv = True
        self.__build_strategy.async_mode = True


class HalfAsyncStrategy(DistributedStrategy):
    def __init__(self):
        super(HalfAsyncStrategy, self).__init__()
        self.__program_config.sync_mode = False
        self.__program_config.runtime_split_send_recv = False
        self.__build_strategy.async_mode = False


class GeoStrategy(DistributedStrategy):
    def __init__(self, update_frequency=100):
        super(GeoStrategy, self).__init__()
        self.__program_config.sync_mode = False
        self.__program_config.runtime_split_send_recv = True
        self.__program_config.geo_sgd_mode = True
        self.__program_config.geo_sgd_need_push_nums = update_frequency
        self.__build_strategy.async_mode = True


class DistributedStrategyFactory(object):
    def __init_(self):
        self.__distributed_strategy = None

    def create_sync_strategy(self):
        self.__distributed_strategy = SyncStrategy()
        return self.__distributed_strategy

    def create_half_async_strategy(self):
        self.__distributed_strategy = HalfAsyncStrategy()
        return self.__distributed_strategy

    def create_async_strategy(self):
        self.__distributed_strategy = AsyncStrategy()
        return self.__distributed_strategy

    def create_geo_strategy(self, update_frequency=100):
        self.__distributed_strategy = GeoStrategy(update_frequency)
        return self.__distributed_strategy
