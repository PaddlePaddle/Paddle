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

from .executor import global_scope
"""
Communicator is used for async distribute training in distribute_transpiler mode.
It's a wrapper of a cpp class Communicator and should be used inside fleet API.
"""
from . import core
from .framework import Program
from .transpiler.distribute_transpiler import DistributedMode

__all__ = ['Communicator']


class Communicator(object):
    def __init__(self, program, mode, kwargs=None, envs={}):
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
        assert isinstance(program, Program)
        for op in program.block(0).ops:
            if op.type == "recv":
                op._set_attr('do_not_run', True)

        if mode == DistributedMode.GEO:
            push_vars = kwargs["push_vars"]
            push_var_names = []

            for k, vs in push_vars.items():
                varnames = "&".join(vs["var_names"])
                sections = "&".join([str(v) for v in vs["sections"]])
                endpoints = "&".join(vs["epmap"])
                is_sparse = "1" if vs["is_sparse"] == ['True'] else "0"

                push_var_names.append(k)
                envs[k] = "#".join([varnames, sections, endpoints, is_sparse])

            envs["geo_trainer_nums"] = str(kwargs["trainers"])
            envs["geo_need_push_nums"] = str(kwargs["push_nums"])
            envs["geo_send_varnames"] = '#'.join(push_var_names)

        if mode == DistributedMode.SYNC:
            envs["pserver_endpoints"] = ','.join(kwargs["pserver_endpoints"])
            envs["trainer_id"] = str(kwargs["trainer_id"])

        mode_str = None

        if mode == DistributedMode.SYNC:
            mode_str = "SYNC"
        elif mode == DistributedMode.ASYNC:
            mode_str = "ASYNC"
        elif mode == DistributedMode.HALF_ASYNC:
            mode_str = "HALF_ASYNC"
        elif mode == DistributedMode.GEO:
            mode_str = "GEO"

        self.communicator_ = core.DistCommunicator(mode_str, program.desc,
                                                   global_scope(), envs)

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
        self.communicator_.is_running()
