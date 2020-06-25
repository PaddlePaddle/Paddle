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

from __future__ import print_function
from multiprocessing import Process, Manager
import paddle.fluid as fluid
import os
import time

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

    def all_gather(self, input):
        """
        all gather between trainers and pservers

        Args:
            input(int|float): input value

        Returns:
            return a list of values
        """
        print("warning: RoleMakerBase does not have all gather.")
        return None

    def all_reduce_worker(self, input, output, mode="sum"):
        """
        all reduce between trainers if current role is TRAINER,
        only support array of one dim.

        Args:
            input(list/numpy.array): array of one dim
            output(list/numpy.array): array of one dim
            mode(str): "sum" or "min" or "max"
        """
        print("warning: RoleMakerBase does not have all reduce worker.")

    def barrier_worker(self):
        """
        barrier between trainers if current role is TRAINER
        """
        print("warning: RoleMakerBase does not have barrier worker.")

    def barrier_all(self):
        """
        barrier between trainers if current role is PSERVER
        """
        print("warning: RoleMakerBase does not have barrier all.")


class PaddleCloudRoleMaker(RoleMakerBase):
    """
    role maker for paddle cloud,
    base class is RoleMakerBase
    """

    def __init__(self, is_collective=False, is_heter=True):
        super(PaddleCloudRoleMaker, self).__init__()
        self._role_is_generated = False
        self._is_collective = is_collective

    def generate_role(self):
        """Generate role."""
        if not self._role_is_generated:
            if not self._is_collective:
                try:
                    # Environment variable PADDLE_PSERVERS_IP_PORT_LIST must be set
                    # format: string(ip:port), eg. 127.0.0.1:6001
                    eplist = os.environ["PADDLE_PSERVERS_IP_PORT_LIST"].split(
                        ",")
                    # note that, we usually assign the same port to different ips
                    # if we run parameter server training in local mode
                    # port should be different in environment variables

                    trainers_num = int(os.environ["PADDLE_TRAINERS_NUM"])
                    training_role = os.environ["TRAINING_ROLE"]

                    if training_role not in ["TRAINER", "PSERVER"]:
                        raise ValueError(
                            "TRAINING_ROLE must be PSERVER or TRAINER")

                    if training_role == "TRAINER":
                        role = Role.WORKER
                        current_id = int(os.environ["PADDLE_TRAINER_ID"])
                    elif training_role == "PSERVER":
                        role = Role.SERVER
                        cur_ip = os.environ["POD_IP"]
                        curr_port = os.environ["PADDLE_PORT"]
                        curr_endpoint = ":".join([cur_ip, curr_port])
                        current_id = eplist.index(curr_endpoint)
                    else:
                        raise ValueError(
                            "TRAINING_ROLE must be PSERVER or TRAINER")
                except ValueError as ve:
                    raise ValueError(
                        "something wrong with PaddleCloud, please check environment"
                    )

                self._trainers_num = trainers_num
                self._server_endpoints = eplist
                self._role = role
                self._current_id = current_id
            else:
                self._current_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
                self._training_role = os.getenv("PADDLE_TRAINING_ROLE",
                                                "TRAINER")
                assert (self._training_role == "TRAINER")
                self._worker_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
                self._current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
                assert self._worker_endpoints is not None, "can't find PADDLE_TRAINER_ENDPOINTS"
                self._worker_endpoints = self._worker_endpoints.split(",")
                self._trainers_num = len(self._worker_endpoints)

            self._role_is_generated = True

    def get_pserver_endpoints(self):
        if not self._role_is_generated:
            self.generate_role()
        return self._server_endpoints

    def is_worker(self):
        if not self._role_is_generated:
            self.generate_role()
        return self._role == Role.WORKER

    def is_server(self):
        if not self._role_is_generated:
            self.generate_role()
        return self._role == Role.SERVER

    def is_first_worker(self):
        if not self._role_is_generated:
            self.generate_role()
        return self._role == Role.WORKER and self._current_id == 0

    def worker_index(self):
        if not self._role_is_generated:
            self.generate_role()
        return self._current_id

    def server_index(self):
        if not self._role_is_generated:
            self.generate_role()
        return self._current_id

    def worker_num(self):
        if not self._role_is_generated:
            self.generate_role()
        return self._trainers_num


class UserDefinedRoleMaker(RoleMakerBase):
    """
    UserDefinedRoleMaker is designed for worker and server assignment
    under manual. Typically, a worker and a server node will be appointed
    on each physical node, It can be assign by user.
    """

    def __init__(self,
                 current_id=0,
                 role=Role.WORKER,
                 worker_num=0,
                 server_endpoints=None):
        super(UserDefinedRoleMaker, self).__init__()

        if not isinstance(server_endpoints, list):
            raise TypeError("server_endpoints must be as string list")
        elif len(server_endpoints) <= 0:
            raise ValueError(
                "the length of server_endpoints list must be greater than 0")
        elif len(server_endpoints) != len(set(server_endpoints)):
            raise ValueError("server_endpoints can't have duplicate elements")
        else:
            for server_endpoint in server_endpoints:
                if not isinstance(server_endpoint, str):
                    raise TypeError(
                        "every element in server_endpoints list must be as string"
                    )
            self._server_endpoints = server_endpoints

        if role != Role.WORKER and role != Role.SERVER:
            raise TypeError("role must be as Role")
        else:
            self._role = role

        if not isinstance(current_id, int):
            raise TypeError("current_id must be as int")
        else:
            if current_id < 0:
                raise ValueError(
                    "current_id must be greater than or equal to 0")
            elif self._role == Role.SERVER and current_id >= len(
                    server_endpoints):
                raise ValueError(
                    "if role is Role.SERVER, current_id must be less than or equal to len(server_endpoints) - 1"
                )
            self._current_id = current_id

        if not isinstance(worker_num, int):
            raise TypeError("worker_num must be as int")
        else:
            if worker_num <= 0:
                raise ValueError("worker_num must be greater than 0")
            self._worker_num = worker_num

    def generate_role(self):
        self._role_is_generated = True

    def is_worker(self):
        return self._role == Role.WORKER

    def is_server(self):
        return self._role == Role.SERVER

    def is_first_worker(self):
        return self._role == Role.WORKER and self._current_id == 0

    def worker_index(self):
        return self._current_id

    def server_index(self):
        return self._current_id

    def worker_num(self):
        return self._worker_num
