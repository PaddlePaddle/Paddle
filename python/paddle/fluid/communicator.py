# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# Copyright(c) 2019 PaddlePaddle Authors.All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http:  // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .executor import global_scope

"""
Communicator is used for async distribute training in distribute_transpiler mode.
It's a wrapper of a cpp class Communicator and should be used inside fleet API.
"""
from . import core
from paddle.fluid.incubate.fleet.parameter_server.mode import DistributedMode

__all__ = ['Communicator', 'FLCommunicator', 'LargeScaleKV']


class Communicator:
    def __init__(self, mode, kwargs=None, envs=None):
        """
        Communicator is used for async distribute training in distribute_transpiler mode.
        It's a wrapper of a cpp class Communicator and should be used inside fleet API.

        Args:
            program(Program): the trainers program after transpile of distribute_transpiler.
            It's used by communicator to extract the information to do communication.

        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                prog = fluid.Program()
                comm = fluid.communicator.Communicator(prog)
                comm.start()
                comm.stop()
        """
        # set all recv op to not_run mode

        if kwargs is None:
            if envs is None:
                envs = {}
        else:
            if mode == DistributedMode.SYNC:
                envs["pserver_endpoints"] = ','.join(
                    kwargs["pserver_endpoints"]
                )

            envs["trainers"] = str(kwargs["trainers"])
            envs["trainer_id"] = str(kwargs["trainer_id"])
            envs["need_global_step"] = str(kwargs["need_global_step"])
            envs["barrier_table_id"] = str(kwargs["barrier_table_id"])

        mode_str = None

        if mode == DistributedMode.SYNC:
            mode_str = "SYNC"
        elif mode == DistributedMode.ASYNC:
            mode_str = "ASYNC"
        elif mode == DistributedMode.HALF_ASYNC:
            mode_str = "HALF_ASYNC"
        elif mode == DistributedMode.GEO:
            mode_str = "GEO"

        self.mode = mode_str
        self.envs = envs
        self.communicator_ = None
        self.send_ctx_ = None
        self.recv_ctx_ = None

    def init_with_ctx(
        self, send_ctx, recv_ctx, proto_txt, unit64_hosts, scope=None
    ):
        if scope is None:
            scope = global_scope()
        self.communicator_ = core.DistCommunicator(
            self.mode,
            proto_txt,
            unit64_hosts,
            send_ctx,
            recv_ctx,
            scope,
            self.envs,
        )
        self.send_ctx_ = send_ctx
        self.recv_ctx_ = recv_ctx

    def create_client_to_client_connection(
        self,
        pserver_timeout_ms=500000,
        pserver_connect_timeout_ms=10000,
        max_retry=3,
    ):
        self.communicator_.create_client_to_client_connection(
            pserver_timeout_ms, pserver_connect_timeout_ms, max_retry
        )

    def get_client_info(self):
        return self.communicator_.get_client_info()

    def set_clients(self, host_list):
        self.communicator_.set_clients(host_list)

    def start(self):
        """
        Start communicator. Should call before training process.

        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                prog = fluid.Program()
                comm = fluid.communicator.Communicator(prog)
                comm.start()
                comm.stop()
        """
        if self.communicator_ is None:
            print('you must call init_with_ctx first to init comm before start')
            return
        self.communicator_.start()

    def stop(self):
        """
        Stop communicator. Should call after training process.

        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                prog = fluid.Program()
                comm = fluid.communicator.Communicator(prog)
                comm.start()
                comm.stop()
        """
        if self.communicator_ is None:
            print('you must call init_with_ctx first to init comm before stop')
            return
        self.communicator_.stop()

    def is_running(self):
        """
        Get communicator is running or stop.

        Returns:
            bool

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                prog = fluid.Program()
                comm = fluid.communicator.Communicator(prog)
                comm.is_running()
        """
        if self.communicator_ is None:
            print('you must call init_with_ctx first to init comm before stop')
            return
        self.communicator_.is_running()

    def recv(self):
        self.communicator_.recv()

    def init_params(self, context):
        self.communicator_.init_params(context)

    def pull_dense(self, context):
        self.communicator_.pull_dense(context)

    def push_sparse_param(self, var_name, table_id=-1, scope=None):
        if scope is None:
            scope = global_scope()
        if not self.is_running():
            raise ValueError(
                "Communicator should init first. Using fleet.init_worker() before push_sparse_param()"
            )
        assert isinstance(var_name, str)
        assert isinstance(table_id, int)
        if table_id == -1:
            table_id = self.send_ctx_[var_name].table_id()
        self.communicator_.push_sparse_param(var_name, table_id, scope)


class FLCommunicator(Communicator):  ## only for coordinator
    def __init__(self, ps_hosts, kwargs=None):
        mode = None
        super().__init__(mode, kwargs)
        send_ctx = {}
        dense_map = {}
        prototxt = ""
        self.mode = "WITH_COORDINATOR"
        self.init_with_ctx(send_ctx, dense_map, prototxt, ps_hosts)

    def start_coordinator(self, self_endpoint, trainer_endpoints):
        if self.communicator_ is not None:
            self.communicator_.start_coordinator(
                self_endpoint, trainer_endpoints
            )
        return

    def save_fl_strategy(self, mp):
        if self.communicator_ is not None:
            self.communicator_.save_fl_strategy(mp)
        else:
            raise ValueError("self.communicator_ is null")
        return

    def query_fl_clients_info(self):
        info_mp = {}
        if self.communicator_ is not None:
            info_mp = self.communicator_.query_fl_clients_info()
        return info_mp


class LargeScaleKV:
    def __init__(self):
        self.scale_kv = core.LargeScaleKV()

    def save(self, varname, dirname):
        self.scale_kv.save(varname, dirname)

    def load(self, varname, dirname):
        self.scale_kv.load(varname, dirname)

    def size(self, varname):
        return self.scale_kv.size(varname)


class HeterClient:
    def __init__(self, endpoint, previous_endpoint, trainer_id):
        self.heter_client_ = core.HeterClient(
            endpoint, previous_endpoint, trainer_id
        )

    def stop(self):
        self.heter_client_.stop()
