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

from paddle.fluid.communicator import FlCommunicator
from paddle.distributed.fleet.proto import the_one_ps_pb2
import paddle.distributed.fleet as fleet
from google.protobuf import text_format
import time


class ClientSelector(object):

    def __init__(self, clients_info):
        self.clients_info = clients_info
        self.fl_strategy = {0: "WAIT", 1: "JOIN"}

    def algorithm_1(self):
        pass

    def algorithm_2(self):
        pass


class FlClient(object):

    def __init__(self, role_maker):
        self._client_ptr = fleet.get_fl_client()
        self._coordinators = role_maker._get_coordinator_endpoints()
        print(">>> coordinator enpoints: {}".format(self._coordinators))
        self.fl_res_desc = the_one_ps_pb2.FLParameter()
        self.res_str = ""

    def __build_fl_param_desc(self, dict_msg):
        self.fl_req_desc = the_one_ps_pb2.FLParameter()
        client_info = self.fl_req_desc.client_info
        client_info.device_type = "Andorid"
        client_info.compute_capacity = 10
        client_info.bandwidth = 100
        str_msg = text_format.MessageToString(self.fl_req_desc)
        return str_msg

    def push_fl_state_sync(self, dict_msg):
        str_msg = self.__build_fl_param_desc(dict_msg)
        self._client_ptr.push_fl_state_sync(str_msg)
        return

    def get_fl_strategy(self):
        while True:
            fl_strategy_str = self._client_ptr.get_fl_strategy()
            # self.fl_res_desc.ParseFromString(fl_strategy_str)
            print("trainer recved fl_strategy_str: {}".format(fl_strategy_str))
            if fl_strategy_str == "JOIN":
                return
            elif fl_strategy_str == "WAIT":
                return
            elif fl_strategy_str == "FINISH":
                return

    def wait(self):
        pass

    def stop(self):
        pass


class Coordinator(object):

    def __init__(self, ps_hosts):
        self._communicator = FlCommunicator(ps_hosts)
        self._client_selector = None

    def start_coordinator(self, self_endpoint, trainer_endpoints):
        self._communicator.start_coordinator(self_endpoint, trainer_endpoints)

    def make_fl_strategy(self):
        print(">>> entering make_fl_strategy")
        while True:
            # 1. get all clients reported info
            str_map = self._communicator.query_fl_clients_info()
            print("queried fl clients info: {}".format(str_map))
            # 2. generate fl strategy
            self._client_selector = ClientSelector(str_map)
            self._client_selector.algorithm_1()
            # 3. save fl strategy in c++
            self._communicator.save_fl_strategy(
                self._client_selector.fl_strategy)
            time.sleep(5)
