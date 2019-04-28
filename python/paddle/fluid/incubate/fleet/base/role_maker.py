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

from __future__ import print_function
from enum import Enum

__all__ = [
    'Role', 'RoleMakerBase', 'MPISymetricRoleMaker', 'UserDefinedRoleMaker'
]


class Role(Enum):
    WORKER = 1,
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


class MPIRoleMaker(RoleMakerBase):
    """
    MPIRoleMaker is a MPI-API based role maker which is a counter-part of K8SRoleMaker
    mpi4py will be used if a developer inherits MPIRoleMaker
    """

    def __init__(self):
        super(MPIRoleMaker, self).__init__()
        from mpi4py import MPI
        self.MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._node_type_comm = None
        self._ips = None
        self._ip = None

    def _get_rank(self):
        """
        return rank
        """
        self._rank = self._comm.Get_rank()
        return self._rank

    def _get_size(self):
        """
        return size
        """
        self._size = self._comm.Get_size()
        return self._size

    def _all_gather(self, obj):
        """
        all_gather(obj) will call MPI's allgather function
        """
        self._barrier_all()
        return self._comm.allgather(obj)

    def _worker_gather(self, obj):
        """
        worker_gather(obj) will call MPI's allgather function
        """
        if self.is_worker():
            self._node_type_comm.barrier()
            return self._node_type_comm.allgather(obj)
        return None

    def _barrier_all(self):
        """
        barrier_all() will call MPI's barrier_all function
        """
        self._comm.barrier()

    def _finalize(self):
        """
        finalize the current MPI instance.
        """
        pass

    def _get_ips(self):
        """
        collect current distributed job's ip list
        """
        if not self._ips:
            self._ips = self._comm.allgather(self.get_local_ip())
        return self._ips

    def get_local_ip(self):
        """
        return get local ip
        """
        import socket
        self._ip = socket.gethostbyname(socket.gethostname())
        return self._ip

    def generate_role(self):
        """
        generate_role() should be called to identify current process's role
        """
        raise NotImplementedError("Please implement this method in child class")


class MPISymetricRoleMaker(MPIRoleMaker):
    """
    MPISymetricRoleMaker is designed for worker and server assignment
    under MPI. Typically, a worker and a server node will be appointed
    on each physical node. This role maker can be only used under MPI.
    """

    def __init__(self):
        super(MPISymetricRoleMaker, self).__init__()
        self._node_type = None
        self._proc_per_node = 2

    def _check_role_generation(self):
        if not self._role_is_generated:
            raise NameError("generate_role() should be called first")
        return True

    def is_first_worker(self):
        """
        return whether current process is the first worker assigned by role maker
        """
        if self._check_role_generation():
            return self.is_worker() and 0 == self.worker_index()
        return False

    def is_worker(self):
        """
        return whether current process is worker assigned by role maker
        """
        if self._check_role_generation():
            return self._node_type == 1
        return False

    def is_server(self):
        """
        return whether current process is server assigned by role maker
        """
        if self._check_role_generation():
            return self._node_type == 0
        return False

    def _worker_num(self):
        """
        return the current number of worker
        """
        if self._check_role_generation():
            if self.is_worker():
                return self._get_size() / 2
        return 0

    def _server_num(self):
        """
        return the current number of server
        """
        if self._check_role_generation():
            if self.is_server():
                return self._get_size() / 2
        return 0

    def worker_index(self):
        """
        return the index of worker
        """
        if self._check_role_generation():
            return self._rank / self._proc_per_node
        return 0

    def server_index(self):
        """
        return the index of server
        """
        if self._check_role_generation():
            return self._rank / self._proc_per_node
        return 0

    def _barrier_worker(self):
        """
        barrier all workers in current distributed job
        """
        if self._check_role_generation():
            if self.is_worker():
                self._node_type_comm.barrier()

    def _barrier_server(self):
        """
        barrier all servers in current distributed job
        """
        if self._check_role_generation():
            if self.is_server():
                self._node_type_comm.barrier()

    def generate_role(self):
        """
        generate currently process's role
        """
        if not self._role_is_generated:
            # TODO(guru4elephant): only allow to be called once
            self._worker_endpoints = self._get_ips()
            self._server_endpoints = self._get_ips()

            if 0 == self._get_rank() % self._proc_per_node % 2:
                self._node_type = 0
            else:
                self._node_type = 1
            self._node_type_comm = self._comm.Split(self._node_type)
            self._role_is_generated = True


class UserDefinedRoleMaker(RoleMakerBase):
    def __init__(self,
                 current_id=0,
                 current_endpoint=None,
                 role=Role.WORKER,
                 worker_endpoints=None,
                 server_endpoints=None):
        """
        UserDefinedRoleMaker is designed for worker and server assignment
        under manual. Typically, a worker and a server node will be appointed
        on each physical node, It can be assign by user.
        """
        super(UserDefinedRoleMaker, self).__init__()

        self._current_id = current_id
        self._current_endpoint = current_endpoint
        self._role = role
        self._worker_endpoints = worker_endpoints
        self._server_endpoints = server_endpoints

    def is_worker(self):
        return self._role == Role.WORKER

    def is_server(self):
        return self._role == Role.SERVER

    def is_first_worker(self):
        return self._role == Role.WORKER and self._current_id == 0

    def worker_index(self):
        if self._role == Role.SERVER:
            raise Exception(
                "the role is Role.SERVER, you may invoke server_index() instead of"
            )
        return self._current_id

    def server_index(self):
        if self._role == Role.WORKER:
            raise Exception(
                "the role is Role.WORKER, you may invoke worker_index() instead of"
            )
        return self._current_id
