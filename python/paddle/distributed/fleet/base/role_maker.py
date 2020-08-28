#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Defination of Role Makers."""
import os
import numpy as np
from multiprocessing import Process, Manager
import paddle.fluid as fluid

__all__ = ['RoleMakerBase', 'UserDefinedRoleMaker', 'PaddleCloudRoleMaker']


class Role:
    WORKER = 1
    SERVER = 2


class RoleMakerBase(object):
    """
    RoleMakerBase is a base class for assigning a role to current process
    in distributed training.
    A paddle developer can implement RoleMakerBase to design a role maker
    for worker or pserver assignment.
    """

    def __init__(self):
        self._worker_endpoints = []
        self._server_endpoints = []
        self._role_is_generated = False
        self._role = None
        self._current_id = -1

        self._node_type = None
        self._node_type_comm = None
        self._all_comm = None

    def is_worker(self):
        """
        return is_worker() of current process
        """
        raise NotImplementedError("Please implement this method in child class")

    def is_server(self):
        """
        return is_server() of current process
        """
        raise NotImplementedError("Please implement this method in child class")

    def is_first_worker(self):
        """
        Check whether the node is the first instance of worker.
        Returns:
            bool: True if this is the first node of worker,
                  False if not.
        """
        raise NotImplementedError("Please implement this method in child class")

    def worker_num(self):
        """
        Get current total worker number.

        Returns:
            int: worker number
        """
        raise NotImplementedError("Please implement this method in child class")

    def server_num(self):
        """
        Get current total server number.

        Returns:
            int: server number
        """
        raise NotImplementedError("Please implement this method in child class")

    def worker_index(self):
        """
        Get current worker id.

        Returns:
            int: node id
        """
        raise NotImplementedError("Please implement this method in child class")

    def server_index(self):
        """
        Get current server id.

        Returns:
            int: node id
        """
        raise NotImplementedError("Please implement this method in child class")

    def role_id(self):
        """
        Get current id.

        Returns:
            int: node id
        """
        raise NotImplementedError("Please implement this method in child class")

    def node_num(self):
        """
        Get the training node number
        Returns:
            int: node num
        """
        raise NotImplementedError("Please implement this method in child class")

    def get_trainer_endpoints(self):
        """
        return trainer endpoints
        """
        return self._worker_endpoints

    def get_pserver_endpoints(self):
        """
        return pserver endpoints
        """
        return self._server_endpoints

    def to_string(self):
        return "role: {}, current_id: {}, worker_endpoints: {}, server_endpoints: {}".format(
            self._role, self._current_id, self._worker_endpoints,
            self._server_endpoints)

    def _all_gather(self, comm_world, input):
        """

        Args:
            input(int|float): input value

        Returns:
            return a list of values
        """
        print("warning: RoleMakerBase does not have all gather.")
        return None

    def _all_reduce(self, comm_world, input, mode="sum"):
        """
        Args:
            input(list/numpy.array): array of one dim
            output(list/numpy.array): array of one dim
            mode(str): "sum" or "min" or "max"
        """
        print("warning: RoleMakerBase does not have all reduce worker.")
        return None

    def _barrier(self, comm_world):
        """
        barrier between trainers if current role is TRAINER
        """
        print("warning: RoleMakerBase does not have barrier worker.")


class PaddleCloudRoleMaker(RoleMakerBase):
    def __init__(self, is_collective=False, **kwargs):
        super(PaddleCloudRoleMaker, self).__init__()
        self._is_collective = is_collective
        self._init_gloo = False  #default no init gloo
        self._kwargs = kwargs

        self._role_is_generated = False

        self._server_endpoints = None
        self._worker_endpoints = None

        self._node_type_comm = None
        self._all_comm = None

        if not self._is_collective:
            self._hdfs_name = kwargs.get("hdfs_name", "")
            self._hdfs_ugi = kwargs.get("hdfs_ugi", "")
            self._hdfs_path = kwargs.get("path", "").rstrip("/")
            self._init_timeout_seconds = kwargs.get("init_timeout_seconds",
                                                    3600)
            self._run_timeout_seconds = kwargs.get("run_timeout_seconds",
                                                   9999999)
            ip_port = kwargs.get("http_ip_port", "")
            self._http_ip_port = []
            self._http_server = None
            # if ip_port is not empty, it will use http instead of hdfs
            if ip_port != "":
                self._http_ip_port = ip_port.split(":")
                # it's for communication between processes
                self._manager = Manager()
                # global dict to store status
                self._http_server_d = self._manager.dict()
                # set running status of http server
                self._http_server_d["running"] = False
            self._iface = self.__get_default_iface()
            # this environment variable can be empty
            self._prefix = os.getenv("SYS_JOB_ID", "")

    def _barrier(self, comm_world):
        if isinstance(comm_world, fluid.core.Gloo):
            comm_world.barrier()
        else:
            print("warning: must init Gloo before using _barrier() function")

    def _all_gather(self, comm_world, input):
        if isinstance(comm_world, fluid.core.Gloo):
            self._barrier(comm_world)
            output = comm_world.all_gather(input)
            return output
        else:
            print("warning: must init Gloo before using _all_gather() function")
            return None

    def _all_reduce(self, comm_world, input, mode="sum"):
        if isinstance(comm_world, fluid.core.Gloo):

            input = np.array(input)

            input_shape = input.shape
            input_list = input.reshape(-1).tolist()

            self._barrier(comm_world)
            ans = comm_world.all_reduce(input_list, mode)
            output = np.array(ans).reshape(input_shape)
            return output
        else:
            print("warning: must init Gloo before using _all_reduce() function")
            return None

    def is_worker(self):
        """
        whether current process is worker
        """
        if not self._role_is_generated:
            self.generate_role()
        return self._role == Role.WORKER

    def is_server(self):
        """
        whether current process is server
        """
        if not self._role_is_generated:
            self.generate_role()
        return self._role == Role.SERVER

    def is_first_worker(self):
        """
        whether current process is worker of rank 0
        """
        if not self._role_is_generated:
            self.generate_role()
        return self._role == Role.WORKER and self._current_id == 0

    def worker_index(self):
        """
        get index of current worker
        """
        if not self._role_is_generated:
            self.generate_role()
        return self._current_id

    def server_index(self):
        """
        get index of current server
        """
        if not self._role_is_generated:
            self.generate_role()
        return self._current_id

    def role_id(self):
        """
        get index of current node
        """
        if self.is_server():
            return self.server_index()
        elif self.is_worker():
            return self.worker_index()

    def worker_num(self):
        """
        retrun the current number of worker
        """
        if not self._role_is_generated:
            self.generate_role()
        return self._trainers_num

    def server_num(self):
        """
        return the current number of server
        """
        if not self._role_is_generated:
            self.generate_role()
        return self._trainers_num

    def node_num(self):
        """
        return the training node number
        """
        if not self._role_is_generated:
            self.generate_role()
        return self._node_num

    def get_trainer_endpoints(self):
        """
        get endpoint of all trainers
        """
        if not self._role_is_generated:
            self.generate_role()
        return self._worker_endpoints

    def get_pserver_endpoints(self):
        """
        get endpoint of all pservers
        """
        if not self._role_is_generated:
            self.generate_role()
        return self._server_endpoints

    def _get_rank(self):
        """
        get current rank in all workers and pservers
        """
        if not self._role_is_generated:
            self.generate_role()
        return self._rank

    def _get_size(self):
        """
        get total num of all workers and pservers
        """
        if not self._role_is_generated:
            self.generate_role()
        return self._size

    def _ps_env(self):
        try:
            # Environment variable PADDLE_PSERVERS_IP_PORT_LIST must be set
            # format: string(ip:port), eg. 127.0.0.1:6001
            self._server_endpoints = os.environ[
                "PADDLE_PSERVERS_IP_PORT_LIST"].split(",")
            self._worker_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS",
                                               "").split(",")

            trainers_num = int(os.environ["PADDLE_TRAINERS_NUM"])
            training_role = os.environ["TRAINING_ROLE"]

            if training_role not in ["TRAINER", "PSERVER"]:
                raise ValueError("TRAINING_ROLE must be PSERVER or TRAINER")

            if training_role == "TRAINER":
                role = Role.WORKER
                current_id = int(os.environ["PADDLE_TRAINER_ID"])
                if len(self._worker_endpoints) > 0:
                    self._cur_endpoint = self._worker_endpoints[current_id]
            elif training_role == "PSERVER":
                role = Role.SERVER
                port = os.environ["PADDLE_PORT"]
                ip = os.environ["POD_IP"]
                self._cur_endpoint = ip + ":" + port
                current_id = self._server_endpoints.index(self._cur_endpoint)
            else:
                raise ValueError("TRAINING_ROLE must be PSERVER or TRAINER")
        except ValueError as ve:
            raise ValueError(
                "something wrong with PaddleCloud, please check environment")

        self._trainers_num = trainers_num
        self._role = role
        self._current_id = current_id
        self._node_num = len(
            set([x.split(':')[0] for x in self._worker_endpoints]))

    def _collective_env(self):
        self._current_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        self._training_role = os.getenv("PADDLE_TRAINING_ROLE", "TRAINER")
        assert (self._training_role == "TRAINER")
        self._worker_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
        self._cur_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
        assert self._worker_endpoints is not None, "can't find PADDLE_TRAINER_ENDPOINTS"
        self._worker_endpoints = self._worker_endpoints.split(",")
        self._trainers_num = len(self._worker_endpoints)
        self._node_num = len(
            set([x.split(':')[0] for x in self._worker_endpoints]))

    def _init_gloo_env(self):
        def init_gloo_instance(role="trainer"):
            role = role.lower()
            assert role in ["trainer", "pserver", "all"]
            if role == "trainer":
                all_list = self._worker_endpoints
                rank = self._current_id
            elif role == "pserver":
                all_list = self._server_endpoints
                rank = self._current_id
            else:
                all_list = self._worker_endpoints + self._server_endpoints
                rank = all_list.index(self._cur_endpoint)
            gloo = fluid.core.Gloo()
            gloo.set_rank(rank)
            gloo.set_size(len(all_list))
            gloo.set_prefix(self._prefix)
            gloo.set_iface(self._iface)
            gloo.set_timeout_seconds(self._init_timeout_seconds,
                                     self._run_timeout_seconds)
            if len(self._http_ip_port) != 0:
                gloo.set_http_store(self._http_ip_port[0],
                                    int(self._http_ip_port[1]), role)
            else:
                gloo.set_hdfs_store(self._hdfs_path + "/" + role,
                                    self._hdfs_name, self._hdfs_ugi)
            gloo.init()
            return gloo

        # paddlecloud support gloo
        if self._role == Role.WORKER:
            if self._current_id == 0 and len(self._http_ip_port) != 0:
                size_d = {
                    "trainer": len(self._worker_endpoints),
                    "pserver": len(self._server_endpoints),
                    "all":
                    len(self._worker_endpoints) + len(self._server_endpoints)
                }
                # child process for http server
                self._http_server = Process(
                    target=self.__start_kv_server,
                    args=(self._http_server_d, size_d))
                self._http_server.daemon = True
                # set running status to True
                self._http_server_d["running"] = True
                # start child process
                self._http_server.start()
            self._node_type = 1
            gloo = init_gloo_instance("trainer")
            self._node_type_comm = gloo
        else:
            assert self._role == Role.SERVER
            self._node_type = 0
            gloo = init_gloo_instance("pserver")
            self._node_type_comm = gloo

        all_list = self._worker_endpoints + self._server_endpoints
        self._rank = all_list.index(self._cur_endpoint)
        self._size = len(all_list)

        gloo = init_gloo_instance("all")
        self._all_comm = gloo

        if self._http_server is not None:
            # set running status to False
            self._http_server_d["running"] = False
            # wait until child process exits
            self._http_server.join()

    def generate_role(self):
        """
        generate role for role maker
        """
        if not self._role_is_generated:
            if not self._is_collective:
                self._ps_env()
                if "PADDLE_WITH_GLOO" in os.environ:
                    self._init_gloo = bool(os.environ["PADDLE_WITH_GLOO"])
                if self._init_gloo:
                    self._init_gloo_env()
            else:
                self._collective_env()
            self._role_is_generated = True

    def __get_default_iface(self):
        """
        get default physical interface
        """
        default1 = self.__get_default_iface_from_gateway()
        default2 = self.__get_default_iface_from_interfaces()
        return default2 if default1 == "lo" else default1

    def __get_default_iface_from_gateway(self):
        """
        get default physical interface
        """
        import netifaces
        gateways = netifaces.gateways()
        if gateways.get(netifaces.AF_INET) != None:
            gateway = gateways[netifaces.AF_INET]
            if len(gateway) > 0 and len(gateway[0]) > 1:
                return gateway[0][1]
        return "lo"

    def __get_default_iface_from_interfaces(self):
        """
        get default physical interface
        """
        import netifaces
        for intf_name in netifaces.interfaces():
            addresses = netifaces.ifaddresses(intf_name)
            if netifaces.AF_INET in addresses:
                ipv4_addresses = addresses[netifaces.AF_INET]
                for ipv4_address in ipv4_addresses:
                    if 'broadcast' in ipv4_address:
                        return intf_name
        return "lo"

    def __start_kv_server(self, http_server_d, size_d):
        from paddle.distributed.fleet.utils import KVServer
        http_server = KVServer(int(self._http_ip_port[1]), size_d)
        http_server.start()
        wait_seconds = 5
        while http_server_d.get("running",
                                False) and not http_server.shoud_stop():
            time.sleep(wait_seconds)
        http_server.stop()


class UserDefinedRoleMaker(PaddleCloudRoleMaker):
    def __init__(self, is_collective=False, init_gloo=False, **kwargs):
        super(UserDefinedRoleMaker, self).__init__(
            is_collective=is_collective, init_gloo=init_gloo, **kwargs)

    def _user_defined_ps_env(self):
        self._server_endpoints = self._kwargs.get("server_endpoints")
        self._worker_endpoints = self._kwargs.get("worker_endpoints", [])
        self._trainers_num = self._kwargs.get("worker_num", 0)

        if self._trainers_num == 0:
            assert (len(self._worker_endpoints) > 0)
            self._trainers_num = len(self._worker_endpoints)

        self._role = self._kwargs.get("role")
        self._current_id = self._kwargs.get("current_id")

        if self._role == Role.WORKER and len(
                self._worker_endpoints) > self._current_id:
            self._cur_endpoint = self._worker_endpoints[self._current_id]
        elif self._role == Role.SERVER:
            self._cur_endpoint = self._server_endpoints[self._current_id]
        self._node_num = len(
            set([x.split(':')[0] for x in self._worker_endpoints]))

    def _user_defined_collective_env(self):
        self._worker_endpoints = self._kwargs.get("worker_endpoints")
        self._current_id = self._kwargs.get("current_id")
        self._trainers_num = len(self._worker_endpoints)
        self._training_role = Role.Worker
        self._node_num = len(
            set([x.split(':')[0] for x in self._worker_endpoints]))

    def generate_role(self):
        """
        generate role for role maker
        """
        if not self._role_is_generated:
            if not self._is_collective:
                self._user_defined_ps_env()
                if self._init_gloo:
                    self._init_gloo_env()
            else:
                self._user_defined_collective_env()
            self._role_is_generated = True
