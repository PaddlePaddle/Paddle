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
        self.role_maker_name_ = ""
        self.trainer_endpoints_ = []
        self.pserver_endpoints_ = []
        self.role_is_generated_ = False

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

    def get_local_ip(self):
        """
        return get local ip
        """
        import socket
        self.ip_ = socket.gethostbyname(socket.gethostname())
        return self.ip_

    def get_trainer_endpoints(self):
        """
        return trainer endpoints
        """
        return self.trainer_endpoints_

    def get_pserver_endpoints(self):
        """
        return pserver endpoints
        """
        return self.pserver_endpoints_

    def generate_role(self):
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

    def get_rank(self):
        """
        return rank
        """
        self.rank_ = self.comm_.Get_rank()
        return self.rank_

    def get_size(self):
        """
        return size
        """
        self.size_ = self.comm_.Get_size()
        return self.size_

    def all_gather(self, obj):
        """
        all_gather(obj) will call MPI's allgather function
        """
        self.barrier_all()
        return self.comm_.allgather(obj)

    def barrier_all(self):
        """
        barrier_all() will call MPI's barrier_all function
        """
        self.comm_.barrier()

    def get_ips(self):
        """
        collect current distributed job's ip list
        """
        if self.ips_ == None:
            self.ips_ = self.comm_.allgather(self.get_local_ip())
        return self.ips_

    def finalize(self):
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
            return self.node_type_ == 1
        return False

    def is_server(self):
        """
        return whether current process is server assigned by role maker
        """
        if self._check_role_generation():
            return self.node_type_ == 0
        return False

    def worker_num(self):
        """
        return the current number of worker
        """
        if self._check_role_generation():
            if self.is_worker():
                return self.get_size() / 2;
        return 0

    def server_num(self):
        """
        return the current number of server
        """
        if self._check_role_generation():
            if self.is_server():
                return self.get_size() / 2;
        return 0

    def worker_index(self):
        """
        return the index of worker
        """
        if self._check_role_generation():
            return self.rank_ / self.proc_per_node_
        return 0

    def server_index(self):
        """
        return the index of server
        """
        if self._check_role_generation():
            return self.rank_ / self.proc_per_node_
        return 0

    def barrier_worker(self):
        """
        barrier all workers in current distributed job
        """
        if self._check_role_generation():
            if self.is_worker():
                self.node_type_comm_.barrier()

    def barrier_server(self):
        """
        barrier all servers in current distributed job
        """
        if self._check_role_generation():
            if self.is_server():
                self.node_type_comm_.barrier()

    def generate_role(self):
        """
        generate currently process's role
        """
        if not self.role_is_generated_:
            # TODO(guru4elephant): only allow to be called once
            self.trainer_endpoints_ = self.get_ips()
            self.pserver_endpoints_ = self.get_ips()

            if 0 == self.get_rank() % self.proc_per_node_ % 2:
                self.node_type_ = 0
            else:
                self.node_type_ = 1
            self.node_type_comm_ = self.comm_.Split(self.node_type_)
            self.role_is_generated_ = True
