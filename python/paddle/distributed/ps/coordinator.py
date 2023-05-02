# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import abc
import logging
import os
import time

from google.protobuf import text_format

import paddle
from paddle.distributed import fleet
from paddle.distributed.communicator import FLCommunicator
from paddle.distributed.fleet.proto import the_one_ps_pb2
from paddle.distributed.ps.utils.public import is_distributed_env

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s'
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


class ClientInfoAttr:
    CLIENT_ID = 0
    DEVICE_TYPE = 1
    COMPUTE_CAPACITY = 2
    BANDWIDTH = 3


class FLStrategy:
    JOIN = 0
    WAIT = 1
    FINISH = 2


class ClientSelectorBase(abc.ABC):
    def __init__(self, fl_clients_info_mp):
        self.fl_clients_info_mp = fl_clients_info_mp
        self.clients_info = {}
        self.fl_strategy = {}

    def parse_from_string(self):
        if not self.fl_clients_info_mp:
            logger.warning("fl-ps > fl_clients_info_mp is null!")

        for client_id, info in self.fl_clients_info_mp.items():
            self.fl_client_info_desc = the_one_ps_pb2.FLClientInfo()
            text_format.Parse(
                bytes(info, encoding="utf8"), self.fl_client_info_desc
            )
            self.clients_info[client_id] = {}
            self.clients_info[client_id][
                ClientInfoAttr.DEVICE_TYPE
            ] = self.fl_client_info_desc.device_type
            self.clients_info[client_id][
                ClientInfoAttr.COMPUTE_CAPACITY
            ] = self.fl_client_info_desc.compute_capacity
            self.clients_info[client_id][
                ClientInfoAttr.BANDWIDTH
            ] = self.fl_client_info_desc.bandwidth

    @abc.abstractmethod
    def select(self):
        pass


class ClientSelector(ClientSelectorBase):
    def __init__(self, fl_clients_info_mp):
        super().__init__(fl_clients_info_mp)
        self.__fl_strategy = {}

    def select(self):
        self.parse_from_string()
        for client_id in self.clients_info:
            logger.info(
                "fl-ps > client {} info : {}".format(
                    client_id, self.clients_info[client_id]
                )
            )
            # ......... to implement ...... #
            fl_strategy_desc = the_one_ps_pb2.FLStrategy()
            fl_strategy_desc.iteration_num = 99
            fl_strategy_desc.client_id = 0
            fl_strategy_desc.next_state = "JOIN"
            str_msg = text_format.MessageToString(fl_strategy_desc)
            self.__fl_strategy[client_id] = str_msg
        return self.__fl_strategy


class FLClientBase(abc.ABC):
    def __init__(self):
        pass

    def set_basic_config(self, role_maker, config, metrics):
        self.role_maker = role_maker
        self.config = config
        self.total_train_epoch = int(self.config.get("runner.epochs"))
        self.train_statical_info = {}
        self.train_statical_info['speed'] = []
        self.epoch_idx = 0
        self.worker_index = fleet.worker_index()
        self.main_program = paddle.static.default_main_program()
        self.startup_program = paddle.static.default_startup_program()
        self._client_ptr = fleet.get_fl_client()
        self._coordinators = self.role_maker._get_coordinator_endpoints()
        logger.info(f"fl-ps > coordinator enpoints: {self._coordinators}")
        self.strategy_handlers = {}
        self.exe = None
        self.use_cuda = int(self.config.get("runner.use_gpu"))
        self.place = paddle.CUDAPlace(0) if self.use_cuda else paddle.CPUPlace()
        self.print_step = int(self.config.get("runner.print_interval"))
        self.debug = self.config.get("runner.dataset_debug", False)
        self.reader_type = self.config.get("runner.reader_type", "QueueDataset")
        self.set_executor()
        self.make_save_model_path()
        self.set_metrics(metrics)

    def set_train_dataset_info(self, train_dataset, train_file_list):
        self.train_dataset = train_dataset
        self.train_file_list = train_file_list
        logger.info(
            "fl-ps > {}, data_feed_desc:\n {}".format(
                type(self.train_dataset), self.train_dataset._desc()
            )
        )

    def set_test_dataset_info(self, test_dataset, test_file_list):
        self.test_dataset = test_dataset
        self.test_file_list = test_file_list

    def set_train_example_num(self, num):
        self.train_example_nums = num

    def load_dataset(self):
        if self.reader_type == "InmemoryDataset":
            self.train_dataset.load_into_memory()

    def release_dataset(self):
        if self.reader_type == "InmemoryDataset":
            self.train_dataset.release_memory()

    def set_executor(self):
        self.exe = paddle.static.Executor(self.place)

    def make_save_model_path(self):
        self.save_model_path = self.config.get("runner.model_save_path")
        if self.save_model_path and (not os.path.exists(self.save_model_path)):
            os.makedirs(self.save_model_path)

    def set_dump_fields(self):
        # DumpField
        # TrainerDesc -> SetDumpParamVector -> DumpParam -> DumpWork
        if self.config.get("runner.need_dump"):
            self.debug = True
            dump_fields_path = "{}/epoch_{}".format(
                self.config.get("runner.dump_fields_path"), self.epoch_idx
            )
            dump_fields = self.config.get("runner.dump_fields", [])
            dump_param = self.config.get("runner.dump_param", [])
            persist_vars_list = self.main_program.all_parameters()
            persist_vars_name = [
                str(param).split(":")[0].strip().split()[-1]
                for param in persist_vars_list
            ]
            logger.info(f"fl-ps > persist_vars_list: {persist_vars_name}")

            if dump_fields_path is not None:
                self.main_program._fleet_opt[
                    'dump_fields_path'
                ] = dump_fields_path
            if dump_fields is not None:
                self.main_program._fleet_opt["dump_fields"] = dump_fields
            if dump_param is not None:
                self.main_program._fleet_opt["dump_param"] = dump_param

    def set_metrics(self, metrics):
        self.metrics = metrics
        self.fetch_vars = [var for _, var in self.metrics.items()]


class FLClient(FLClientBase):
    def __init__(self):
        super().__init__()

    def __build_fl_client_info_desc(self, state_info):
        # ......... to implement ...... #
        state_info = {
            ClientInfoAttr.DEVICE_TYPE: "Andorid",
            ClientInfoAttr.COMPUTE_CAPACITY: 10,
            ClientInfoAttr.BANDWIDTH: 100,
        }
        client_info = the_one_ps_pb2.FLClientInfo()
        client_info.device_type = state_info[ClientInfoAttr.DEVICE_TYPE]
        client_info.compute_capacity = state_info[
            ClientInfoAttr.COMPUTE_CAPACITY
        ]
        client_info.bandwidth = state_info[ClientInfoAttr.BANDWIDTH]
        str_msg = text_format.MessageToString(client_info)
        return str_msg

    def run(self):
        self.register_default_handlers()
        self.print_program()
        self.strategy_handlers['initialize_model_params']()
        self.strategy_handlers['init_worker']()
        self.load_dataset()
        self.train_loop()
        self.release_dataset()
        self.strategy_handlers['finish']()

    def train_loop(self):
        while self.epoch_idx < self.total_train_epoch:
            logger.info(f"fl-ps > curr epoch idx: {self.epoch_idx}")
            self.strategy_handlers['train']()
            self.strategy_handlers['save_model']()
            self.barrier()
            state_info = {
                "client id": self.worker_index,
                "auc": 0.9,
                "epoch": self.epoch_idx,
            }
            self.push_fl_client_info_sync(state_info)
            strategy_dict = self.pull_fl_strategy()
            logger.info(f"fl-ps > recved fl strategy: {strategy_dict}")
            # ......... to implement ...... #
            if strategy_dict['next_state'] == "JOIN":
                self.strategy_handlers['infer']()
            elif strategy_dict['next_state'] == "FINISH":
                self.strategy_handlers['finish']()

    def push_fl_client_info_sync(self, state_info):
        str_msg = self.__build_fl_client_info_desc(state_info)
        self._client_ptr.push_fl_client_info_sync(str_msg)
        return

    def pull_fl_strategy(self):
        strategy_dict = {}
        fl_strategy_str = (
            self._client_ptr.pull_fl_strategy()
        )  # block: wait for coordinator's strategy arrived
        logger.info(
            "fl-ps > fl client recved fl_strategy(str):\n{}".format(
                fl_strategy_str
            )
        )
        fl_strategy_desc = the_one_ps_pb2.FLStrategy()
        text_format.Parse(
            bytes(fl_strategy_str, encoding="utf8"), fl_strategy_desc
        )
        strategy_dict["next_state"] = fl_strategy_desc.next_state
        return strategy_dict

    def barrier(self):
        fleet.barrier_worker()

    def register_handlers(self, strategy_type, callback_func):
        self.strategy_handlers[strategy_type] = callback_func

    def register_default_handlers(self):
        self.register_handlers('train', self.callback_train)
        self.register_handlers('infer', self.callback_infer)
        self.register_handlers('finish', self.callback_finish)
        self.register_handlers(
            'initialize_model_params', self.callback_initialize_model_params
        )
        self.register_handlers('init_worker', self.callback_init_worker)
        self.register_handlers('save_model', self.callback_save_model)

    def callback_init_worker(self):
        fleet.init_worker()

    def callback_initialize_model_params(self):
        if self.exe is None or self.main_program is None:
            raise AssertionError("exe or main_program not set")
        self.exe.run(self.startup_program)

    def callback_train(self):
        epoch_start_time = time.time()
        self.set_dump_fields()
        fetch_info = [
            f"Epoch {self.epoch_idx} Var {var_name}"
            for var_name in self.metrics
        ]
        self.exe.train_from_dataset(
            program=self.main_program,
            dataset=self.train_dataset,
            fetch_list=self.fetch_vars,
            fetch_info=fetch_info,
            print_period=self.print_step,
            debug=self.debug,
        )
        self.epoch_idx += 1
        epoch_time = time.time() - epoch_start_time
        epoch_speed = self.train_example_nums / epoch_time
        self.train_statical_info["speed"].append(epoch_speed)
        logger.info("fl-ps > callback_train finished")

    def callback_infer(self):
        fetch_info = [
            f"Epoch {self.epoch_idx} Var {var_name}"
            for var_name in self.metrics
        ]
        self.exe.infer_from_dataset(
            program=self.main_program,
            dataset=self.test_dataset,
            fetch_list=self.fetch_vars,
            fetch_info=fetch_info,
            print_period=self.print_step,
            debug=self.debug,
        )

    def callback_save_model(self):
        model_dir = f"{self.save_model_path}/{self.epoch_idx}"
        if fleet.is_first_worker() and self.save_model_path:
            if is_distributed_env():
                fleet.save_persistables(self.exe, model_dir)  # save all params
            else:
                raise ValueError("it is not distributed env")

    def callback_finish(self):
        fleet.stop_worker()

    def print_program(self):
        with open(
            f"./{self.worker_index}_worker_main_program.prototxt", 'w+'
        ) as f:
            f.write(str(self.main_program))
        with open(
            f"./{self.worker_index}_worker_startup_program.prototxt",
            'w+',
        ) as f:
            f.write(str(self.startup_program))

    def print_train_statical_info(self):
        with open("./train_statical_info.txt", 'w+') as f:
            f.write(str(self.train_statical_info))


class Coordinator:
    def __init__(self, ps_hosts):
        self._communicator = FLCommunicator(ps_hosts)
        self._client_selector = None

    def start_coordinator(self, self_endpoint, trainer_endpoints):
        self._communicator.start_coordinator(self_endpoint, trainer_endpoints)

    def make_fl_strategy(self):
        logger.info("fl-ps > running make_fl_strategy(loop) in coordinator\n")
        while True:
            # 1. get all fl clients reported info
            str_map = (
                self._communicator.query_fl_clients_info()
            )  # block: wait for all fl clients info reported
            # 2. generate fl strategy
            self._client_selector = ClientSelector(str_map)
            fl_strategy = self._client_selector.select()
            # 3. save fl strategy from python to c++
            self._communicator.save_fl_strategy(fl_strategy)
            time.sleep(5)
