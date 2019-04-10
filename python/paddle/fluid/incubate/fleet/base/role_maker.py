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

import sys

from enum import Enum


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
        self.trainer_endpoints_ = []
        self.pserver_endpoints_ = []
        self.role_is_generated_ = False

    def _is_worker(self):
        """
        return is_worker() of current process
        """
        raise NotImplementedError("Please implement this method in child class")

    def _is_server(self):
        """
        return is_server() of current process
        """
        raise NotImplementedError("Please implement this method in child class")

    def _get_local_ip(self):
        """
        return get local ip
        """
        import socket
        self.ip_ = socket.gethostbyname(socket.gethostname())
        return self.ip_

    def _get_trainer_endpoints(self):
        """
        return trainer endpoints
        """
        return self.trainer_endpoints_

    def _get_pserver_endpoints(self):
        """
        return pserver endpoints
        """
        return self.pserver_endpoints_

    def _generate_role(self):
        """
        generate_role() should be called to identify current process's role
        """
        raise NotImplementedError("Please implement this method in child class")


class MPIRoleMaker(RoleMakerBase):
    """
    MPIRoleMaker is a MPI-API based role maker which is a counter-part of K8SRoleMaker
    mpi4py will be used if a developer inherits MPIRoleMaker
    """

    def __init__(self):
        super(MPIRoleMaker, self).__init__()
        from mpi4py import MPI
        self.comm_ = MPI.COMM_WORLD
        self.MPI = MPI
        self.ips_ = None

    def _get_rank(self):
        """
        return rank
        """
        self.rank_ = self.comm_.Get_rank()
        return self.rank_

    def _get_size(self):
        """
        return size
        """
        self.size_ = self.comm_.Get_size()
        return self.size_

    def _all_gather(self, obj):
        """
        all_gather(obj) will call MPI's allgather function
        """
        self._barrier_all()
        return self.comm_.allgather(obj)

    def _worker_gather(self, obj):
        """
        worker_gather(obj) will call MPI's allgather function
        """
        if self._is_worker():
            self.node_type_comm_.barrier()
            return self.node_type_comm_.allgather(obj)
        return None

    def _barrier_all(self):
        """
        barrier_all() will call MPI's barrier_all function
        """
        self.comm_.barrier()

    def _get_ips(self):
        """
        collect current distributed job's ip list
        """
        if self.ips_ == None:
            self.ips_ = self.comm_.allgather(self._get_local_ip())
        return self.ips_

    def _finalize(self):
        """
        finalize the current MPI instance.
        """
        self.comm_.finalize()


class MPISymetricRoleMaker(MPIRoleMaker):
    """
    MPISymetricRoleMaker is designed for worker and server assignment
    under MPI. Typically, a worker and a server node will be appointed
    on each physical node. This role maker can be only used under MPI.
    """

    def __init__(self):
        super(MPISymetricRoleMaker, self).__init__()
        self.node_type_ = None
        self.proc_per_node_ = 2

    def _check_role_generation(self):
        if not self.role_is_generated_:
            sys.stderr.write("generate_role() should be called first")
            sys.exit(-1)
            return False
        return True

    def _is_first_worker(self):
        """
        return whether current process is the first worker assigned by role maker
        """
        if self._check_role_generation():
            return self._is_worker() and 0 == self._worker_index()
        return False

    def _is_worker(self):
        """
        return whether current process is worker assigned by role maker
        """
        if self._check_role_generation():
            return self.node_type_ == 1
        return False

    def _is_server(self):
        """
        return whether current process is server assigned by role maker
        """
        if self._check_role_generation():
            return self.node_type_ == 0
        return False

    def _worker_num(self):
        """
        return the current number of worker
        """
        if self._check_role_generation():
            if self._is_worker():
                return self._get_size() / 2
        return 0

    def _server_num(self):
        """
        return the current number of server
        """
        if self._check_role_generation():
            if self._is_server():
                return self._get_size() / 2
        return 0

    def _worker_index(self):
        """
        return the index of worker
        """
        if self._check_role_generation():
            return self.rank_ / self.proc_per_node_
        return 0

    def _server_index(self):
        """
        return the index of server
        """
        if self._check_role_generation():
            return self.rank_ / self.proc_per_node_
        return 0

    def _barrier_worker(self):
        """
        barrier all workers in current distributed job
        """
        if self._check_role_generation():
            if self._is_worker():
                self.node_type_comm_.barrier()

    def _barrier_server(self):
        """
        barrier all servers in current distributed job
        """
        if self._check_role_generation():
            if self._is_server():
                self.node_type_comm_.barrier()

    def _generate_role(self):
        """
        generate currently process's role
        """
        if not self.role_is_generated_:
            # TODO(guru4elephant): only allow to be called once
            self.trainer_endpoints_ = self._get_ips()
            self.pserver_endpoints_ = self._get_ips()

            if 0 == self._get_rank() % self.proc_per_node_ % 2:
                self.node_type_ = 0
            else:
                self.node_type_ = 1
            self.node_type_comm_ = self.comm_.Split(self.node_type_)
            self.role_is_generated_ = True


class UserDefinedRoleMaker(RoleMakerBase):
    def __init__(self,
                 current_id=0,
                 current_endpoint=None,
                 workers=0,
                 worker_endpoints=None,
                 servers=0,
                 server_endpoints=None,
                 role=Role.WORKER):
        super(UserDefinedRoleMaker, self).__init__()

        self.current_id = current_id
        self.current_endpoint = current_endpoint
        self.workers = workers
        self.worker_endpoints = worker_endpoints
        self.servers = servers
        self.server_endpoints = server_endpoints
        self.role = role

    def _is_worker(self):
        return self.role == Role.WORKER

    def _is_server(self):
        return self.role == Role.SERVER

    def _generate_role(self):
        self.role_is_generated_ = True
