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


class RoleMakerBase(object):
    def __init__(self):
        self.role_maker_name_ = ""
        self.trainer_endpoints_ = []
        self.pserver_endpoints_ = []

    def is_worker(self):
        raise NotImplementedError("Please implement this method in child class")

    def is_server(self):
        raise NotImplementedError("Please implement this method in child class")

    def get_local_ip(self):
        import socket
        self.ip_ = socket.gethostbyname(socket.gethostname())
        return self.ip_

    def get_trainer_endpoints(self):
        return self.trainer_endpoints_

    def get_pserver_endpoints(self):
        return self.pserver_endpoints_

    def generate_role(self):
        raise NotImplementedError("Please implement this method in child class")


class MPIRoleMaker(RoleMakerBase):
    def __init__(self):
        from mpi4py import MPI
        self.comm_ = MPI.COMM_WORLD
        self.MPI = MPI
        self.ips_ = None

    def get_rank(self):
        self.rank_ = self.comm_.Get_rank()
        return self.rank_

    def get_size(self):
        self.size_ = self.comm_.Get_size()
        return self.size_

    def all_gather(self, obj):
        self.barrier_all()
        return self.comm_.allgather(obj)

    def barrier_all(self):
        self.comm_.barrier()

    def get_ips(self):
        if self.ips_ == None:
            self.ips_ = self.comm_.allgather(self.get_local_ip())
        return self.ips_

    def finalize(self):
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

    def is_first_worker(self):
        return self.is_worker() and 0 == self.worker_index()

    def is_worker(self):
        return self.node_type_ == 1

    def is_server(self):
        return self.node_type_ == 0

    def worker_num(self):
        if self.is_worker():
            return self.get_size()

    def server_num(self):
        if self.is_server():
            return self.get_size()

    def worker_index(self):
        return self.rank_ / self.proc_per_node_

    def server_index(self):
        return self.rank_ / self.proc_per_node_

    def barrier_worker(self):
        if self.is_worker():
            self.node_type_comm_.barrier()

    def barrier_server(self):
        if self.is_server():
            self.node_type_comm_.barrier()

    def generate_role(self):
        self.trainer_endpoints_ = self.get_ips()
        self.pserver_endpoints_ = self.get_ips()

        if 0 == self.get_rank() % self.proc_per_node_ % 2:
            self.node_type_ = 0
        else:
            self.node_type_ = 1
        self.node_type_comm_ = self.comm_.Split(self.node_type_)
