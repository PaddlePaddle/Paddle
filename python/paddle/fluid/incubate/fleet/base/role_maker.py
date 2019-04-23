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


class RoleMakerBase(object):
    """
    RoleMakerBase is a base class for assigning a role to current process
    in distributed training.
    A paddle developer can implement RoleMakerBase to design a role maker
    for worker or pserver assignment.
    """

    def __init__(self):
        self._role_maker_name = ""
        self._trainer_endpoints = []
        self._pserver_endpoints = []
        self._role_is_generated = False

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
        self._ip = socket.gethostbyname(socket.gethostname())
        return self._ip

    def _get_trainer_endpoints(self):
        """
        return trainer endpoints
        """
        return self._trainer_endpoints

    def _get_pserver_endpoints(self):
        """
        return pserver endpoints
        """
        return self._pserver_endpoints

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
        self._comm = MPI.COMM_WORLD
        self.MPI = MPI
        self._ips = None

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
        if self._is_worker():
            self._node_type_comm.barrier()
            return self._node_type_comm.allgather(obj)
        return None

    def _barrier_all(self):
        """
        barrier_all() will call MPI's barrier_all function
        """
        self._comm.barrier()

    def _get_ips(self):
        """
        collect current distributed job's ip list
        """
        if self._ips == None:
            self._ips = self._comm.allgather(self._get_local_ip())
        return self._ips

    def _finalize(self):
        """
        finalize the current MPI instance.
        """
        pass


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
            return self._node_type == 1
        return False

    def _is_server(self):
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
            return self._rank / self._proc_per_node
        return 0

    def _server_index(self):
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
            if self._is_worker():
                self._node_type_comm.barrier()

    def _barrier_server(self):
        """
        barrier all servers in current distributed job
        """
        if self._check_role_generation():
            if self._is_server():
                self._node_type_comm.barrier()

    def _generate_role(self):
        """
        generate currently process's role
        """
        if not self._role_is_generated:
            # TODO(guru4elephant): only allow to be called once
            self._trainer_endpoints = self._get_ips()
            self._pserver_endpoints = self._get_ips()

            if 0 == self._get_rank() % self._proc_per_node % 2:
                self._node_type = 0
            else:
                self._node_type = 1
            self._node_type_comm = self._comm.Split(self._node_type)
            self._role_is_generated = True
