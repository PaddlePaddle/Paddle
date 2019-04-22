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

import sys
from .optimizer_factory import *
from google.protobuf import text_format

import paddle.fluid as fluid
from paddle.fluid.framework import Program
from paddle.fluid.optimizer import Optimizer

from ...base.fleet_base import Fleet
from ...base.fleet_base import Mode
from ...base.role_maker import MPISymetricRoleMaker
from ...base.fleet_base import DistributedOptimizer


class PSLib(Fleet):
    def __init__(self):
        super(PSLib, self).__init__(Mode.PSLIB)
        self._opt_info = None
        self.local_ip_ = 0
        self._fleet_ptr = None

    def init(self, role_maker=None):
        super(PSLib, self).init(MPISymetricRoleMaker())
        self._fleet_ptr = fluid.core.Fleet()

    def init_worker(self, executor):
        pass

    def run_worker(self, executor, main_program=None):
        """
        init_worker(): will be called by user. When a user knows current process is_server(), he/she
                    should call init_worker() to initialize global information about worker and connect
                    worker with pserver. You should run startup program before init_worker.

        Args:
            programs(Program|list): a Program or a list of Programs
            scopes(Scope|list): a Scope or  a list of Scopes, default None.
        """
        if not isinstance(main_program, Program):
            raise ValueError("main_program must be an instance of Program")

        programs = [main_program]
        scopes = [fluid.global_scope()] * len(programs)

        if len(scopes) != len(programs):
            print(
                "You should make sure len(scopes) == len(programs) or set scopes None"
            )
            sys.exit(-1)
        if self._opt_info:
            if "fleet_desc" in self._opt_info:
                self._dist_desc_str = text_format.MessageToString(
                    self._opt_info["fleet_desc"])
                self._dist_desc = self._opt_info["fleet_desc"]
            else:
                print("You should run DistributedOptimizer.minimize() first")
                sys.exit(-1)
            # barrier_all for init_server, wait for server starts
            self.role_maker_._barrier_all()
            self.all_ips_ = self.role_maker_._all_gather(self.local_ip_)
            self._fleet_ptr.init_worker(self._dist_desc_str, self.all_ips_,
                                        self.role_maker_._get_size(),
                                        self.role_maker_._get_rank())
            # barrier_all for init_worker
            self.role_maker_._barrier_all()
            # prepare for client to client communication
            info = self._fleet_ptr.get_clients_info()
            all_info = self.role_maker_._worker_gather(info[0])
            self._fleet_ptr.gather_clients(all_info)
            self._fleet_ptr.create_client2client_connection()
            # barrier for init model
            self.role_maker_._barrier_worker()
            if self.role_maker_._is_first_worker():
                tables = self._dist_desc.trainer_param.dense_table
                for prog, scope in zip(programs, scopes):
                    prog_id = str(id(prog))
                    prog_conf = self._opt_info['program_configs'][prog_id]
                    prog_tables = {}
                    for key in prog_conf:
                        if "dense" not in key:
                            continue
                        for table_id in prog_conf[key]:
                            prog_tables[int(table_id)] = 0
                    for table in tables:
                        if int(table.table_id) not in prog_tables:
                            continue
                        var_name_list = []
                        for i in range(0, len(table.dense_variable_name)):
                            var_name = table.dense_variable_name[i]
                            if scope.find_var(var_name) is None:
                                print("var " + var_name +
                                      " not found in scope, " +
                                      "you should run startup program first")
                                sys.exit(-1)
                            var_name_list.append(var_name)
                        self._fleet_ptr.init_model(scope,
                                                   int(table.table_id),
                                                   var_name_list)
            # barrier for init model done
            self.role_maker_._barrier_worker()
        else:
            raise NameError(
                "You should run DistributedOptimizer.minimize() first")

    def init_server(self, executor, model_dir=None):
        pass

    def run_server(self, executor):
        """
         init_pserver(): will be called by user. When a user knows current process is_worker(), he/she
             should call init_pserver() to initialize global information about parameter server
         """
        if self._opt_info:
            if "fleet_desc" in self._opt_info:
                self._dist_desc_str = text_format.MessageToString(
                    self._opt_info["fleet_desc"])
                self._dist_desc = self._opt_info["fleet_desc"]
            else:
                print("You should run DistributedOptimizer.minimize() first")
                sys.exit(-1)
            self._fleet_ptr.init_server(self._dist_desc_str,
                                        self.role_maker_._get_rank())
            self.local_ip_ = self._fleet_ptr.run_server()

            # barrier_all for init_server
            self.role_maker_._barrier_all()
            self.all_ips_ = self.role_maker_._all_gather(self.local_ip_)

            self._fleet_ptr.gather_servers(self.all_ips_,
                                           self.role_maker_._get_size())
            # barrier_all for init_worker, wait all workers start
            self.role_maker_._barrier_all()
        else:
            raise NameError(
                "You should run DistributedOptimizer.minimize() first")

    def stop_worker(self):
        """
        stop(): will be called after a user finishes his/her training task. Fleet instance will be
            destroyed when stop() is called.
        """
        self.role_maker_._barrier_worker()
        if self.role_maker_._is_first_worker():
            self._fleet_ptr.stop_server()
        self.role_maker_._barrier_worker()
        self.role_maker_._barrier_all()
        self.role_maker_._finalize()

    def stop(self, executor):
        """
        stop(): will be called after a user finishes his/her training task. Fleet instance will be
            destroyed when stop() is called.
        """
        self.role_maker_._barrier_worker()
        if self.role_maker_._is_first_worker():
            self._fleet_ptr.stop_server()
        self.role_maker_._barrier_worker()
        self.role_maker_._barrier_all()
        self.role_maker_._finalize()

    def distributed_optimizer(self, optimizer, strategy=None):
        if not isinstance(optimizer, Optimizer):
            raise ValueError("optimizer must be an instance of Optimizer")
        self.optimizer = DownpourOptimizer(optimizer, strategy)
        return self.optimizer

    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names=None,
                             target_vars=None,
                             main_program=None,
                             export_for_deployment=True):
        """
        save pserver model called from a worker
        """
        self._fleet_ptr.save_model(dirname)

    def save_persistables(self, executor, dirname, main_program=None):
        self._fleet_ptr.save_model(dirname)

    def _set_opt_info(self, opt_info):
        """
        this function saves the result from DistributedOptimizer.minimize()
        """
        self._opt_info = opt_info


fleet = PSLib()


class DownpourOptimizer(DistributedOptimizer):
    """
    DistributedOptimizer is a wrapper for paddle.fluid.optimizer
    A user should pass a paddle.fluid.optimizer to DistributedOptimizer
    minimize() function is implemented.
    DistributedOptimizer is the starting point for a user who wants to
    run distributed training. The optimized information will be stored in
    Fleet() instance who holds the global information about current distributed
    training.
    """

    def __init__(self, optimizer, strategy=None):
        super(DownpourOptimizer, self).__init__(optimizer, strategy)

        self._optimizer = optimizer
        self._optimizer_name = "Distributed%s" % optimizer.type.capitalize()
        if optimizer.type != "adam":
            print("Currently, distributed optimizer only supports Adam"
                  "Will config built-in adam for you."
                  "We will support more functions in DistributedOptimizer",
                  sys.stderr)
            self._optimizer_name = "DistributedAdam"

        self._distributed_optimizer = globals()[self._optimizer_name](optimizer)

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        Currently, backward function can not be called through DistributedOptimizer
        """
        raise NotImplementedError()

    def apply_gradients(self, params_grads):
        """
        Currently, apply_gradients function can not be called through DistributedOptimizer
        """
        raise NotImplementedError()

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        minimize a program through loss, loss can be a list in DistributedOptimizer
        Args:
            loss (Variable|Variable List): loss variable or loss variable list to run optimization.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables to update.
            no_grad_set (set|None): set of Variables should be ignored.
        Returns:
            tuple: (optimize_ops, params_grads) which are, list of operators appended;
            and list of (param, grad) Variables pair for optimization.
        Note that in parameter server mode, a worker will not get anything about optimize_os
        Because optmizer algorithms run on pserver side. We will make this usable in pserver
        process, but currently the optimization part is written into Fleet(). A user does not
        need to care about how to startup a pserver node.
        """
        optimize_ops, param_grads, opt_info = \
                      self._distributed_optimizer._minimize(
                          loss,
                          startup_program,
                          parameter_list,
                          no_grad_set)

        fleet._set_opt_info(opt_info)
        return [optimize_ops, param_grads]
