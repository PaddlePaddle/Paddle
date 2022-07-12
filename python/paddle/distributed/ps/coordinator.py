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

from paddle.fluid.communicator import FLCommunicator
from paddle.distributed.fleet.proto import the_one_ps_pb2
import paddle.distributed.fleet as fleet
from google.protobuf import text_format
import time
import abc


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
            print("fl-ps > fl_clients_info_mp is null!")

        for client_id, info in self.fl_clients_info_mp.items():
            self.fl_client_info_desc = the_one_ps_pb2.FLClientInfo()
            text_format.Parse(bytes(info, encoding="utf8"),
                              self.fl_client_info_desc)
            self.clients_info[client_id] = {}
            self.clients_info[client_id][
                ClientInfoAttr.
                DEVICE_TYPE] = self.fl_client_info_desc.device_type
            self.clients_info[client_id][
                ClientInfoAttr.
                COMPUTE_CAPACITY] = self.fl_client_info_desc.compute_capacity
            self.clients_info[client_id][
                ClientInfoAttr.BANDWIDTH] = self.fl_client_info_desc.bandwidth

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
            print("fl-ps > client {} info : {}".format(
                client_id, self.clients_info[client_id]))
            # ......... to implement ...... #
            fl_strategy_desc = the_one_ps_pb2.FLStrategy()
            fl_strategy_desc.iteration_num = 99
            fl_strategy_desc.client_id = 0
            fl_strategy_desc.next_state = "JOIN"
            str_msg = text_format.MessageToString(fl_strategy_desc)
            self.__fl_strategy[client_id] = str_msg
        return self.__fl_strategy


class FlClient(object):

    def __init__(self, role_maker):
        self._client_ptr = fleet.get_fl_client()
        self._coordinators = role_maker._get_coordinator_endpoints()
        print("fl-ps > coordinator enpoints: {}".format(self._coordinators))

    def __build_fl_client_info_desc(self, state_info):
        # ......... to implement ...... #
        state_info = {
            ClientInfoAttr.DEVICE_TYPE: "Andorid",
            ClientInfoAttr.COMPUTE_CAPACITY: 10,
            ClientInfoAttr.BANDWIDTH: 100
        }
        client_info = the_one_ps_pb2.FLClientInfo()
        client_info.device_type = state_info[ClientInfoAttr.DEVICE_TYPE]
        client_info.compute_capacity = state_info[
            ClientInfoAttr.COMPUTE_CAPACITY]
        client_info.bandwidth = state_info[ClientInfoAttr.BANDWIDTH]
        str_msg = text_format.MessageToString(client_info)
        return str_msg

    def push_fl_client_info_sync(self, state_info):
        str_msg = self.__build_fl_client_info_desc(state_info)
        self._client_ptr.push_fl_client_info_sync(str_msg)
        return

    def pull_fl_strategy(self):
        strategy_dict = {}
        fl_strategy_str = self._client_ptr.pull_fl_strategy(
        )  # block: wait for coordinator's strategy arrived
        print("fl-ps > fl client recved fl_strategy_str: {}".format(
            fl_strategy_str))
        fl_strategy_desc = the_one_ps_pb2.FLStrategy()
        text_format.Parse(bytes(fl_strategy_str, encoding="utf8"),
                          fl_strategy_desc)
        print("fl-ps > interation num: {}".format(
            fl_strategy_desc.iteration_num))
        strategy_dict["next_state"] = fl_strategy_desc.next_state
        return strategy_dict


class Coordinator(object):

    def __init__(self, ps_hosts):
        self._communicator = FLCommunicator(ps_hosts)
        self._client_selector = None

    def start_coordinator(self, self_endpoint, trainer_endpoints):
        self._communicator.start_coordinator(self_endpoint, trainer_endpoints)

    def make_fl_strategy(self):
        print("fl-ps > running make_fl_strategy(loop) in coordinator\n")
        while True:
            # 1. get all fl clients reported info
            str_map = self._communicator.query_fl_clients_info(
            )  # block: wait for all fl clients info reported
            # 2. generate fl strategy
            self._client_selector = ClientSelector(str_map)
            fl_strategy = self._client_selector.select()
            # 3. save fl strategy from python to c++
            self._communicator.save_fl_strategy(fl_strategy)
            time.sleep(5)
