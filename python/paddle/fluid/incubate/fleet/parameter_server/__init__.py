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
import os
from ..base.role_maker import MPISymetricRoleMaker
from paddle.fluid.optimizer import Optimizer

# this is a temporary solution
# TODO(guru4elephant)
# will make this more flexible for more Parameter Server Archs
fleet_instance = Fleet()

init = fleet_instance.init
stop = fleet_instance.stop
init_pserver = fleet_instance.init_pserver
init_worker = fleet_instance.init_worker
init_pserver_model = fleet_instance.init_pserver_model
save_pserver_model = fleet_instance.save_pserver_model


class Fleet(object):
    """
    
    """

    def __init__(self):
        self.opt_info = None  # for fleet only
        self.role_maker_ = None

    def init(self):
        # TODO(guru4elephant)
        # this is a temporary solution
        # we will support more configurable RoleMaker for users in the future
        self.role_maker_ = MPISymetricRoleMaker()
        self.role_maker_.generate_role()
        self._fleet_ptr = core.FleetWrapper()

    def stop(self):
        self.role_maker_.barrier_worker()
        if self.role_maker_.is_first_worker():
            self._fleet_ptr.stop_server()
        self.role_maker_.barrier_worker()
        self.role_maker_.barrier_all()
        self.role_maker_.finalize()

    def init_pserver(self):
        if self._opt_info:
            if "fleet_desc" in self._opt_info:
                self._dist_desc_str = text_format.MessageToString(
                    self._opt_info["fleet_desc"])
                self._dist_desc = self._opt_info["fleet_desc"]
            else:
                print("You should run DistributedOptimizer.minimize() first")
                sys.exit(-1)
            self._fleet_ptr.init_server(self._dist_desc_str)
            ip = self._fleet_ptr.start_server()
            ips = self.role_maker_.all_gather(ip)
            self._fleet_ptr.gather_servers(ips, self.role_maker_.get_size())
            self.role_maker_.barrier_all()
        else:
            print("You should run DistributedOptimizer.minimize() first")
            sys.exit(-1)

    def init_worker(self):
        if self._opt_info:
            if "fleet_desc" in self._opt_info:
                self._dist_desc_str = text_format.MessageToString(
                    self._opt_info["fleet_desc"])
                self._dist_desc = self._opt_info["fleet_desc"]
            else:
                print("You should run DistributedOptimizer.minimize() first")
                sys.exit(-1)
            self.role_maker_.barrier_all()
            self._fleet_ptr.init_work(self.dist_desc_str_,
                                      self.role_maker.get_ips(),
                                      self.role_maker_.get_size(),
                                      self.role_maker_.get_rank())
            self.role_maker_.barrier_worker()
        else:
            print("You should run DistributedOptimizer.minimize() first")
            sys.exit(-1)

    def init_pserver_model(self):
        if self.role_maker_.is_first_worker():
            self._fleet_ptr.init_model()
        self.role_maker_.barrier_worker()

    def save_pserver_model(self, save_path):
        self._fleet_ptr.save_model(save_path)

    def _set_opt_info(self, opt_info):
        self._opt_info = opt_info


class DistributedOptimizer(paddle.fluid.Optimizer):
    def __init__(self, optimizer, dist_config={}):
        super(DistributedOptimizer, self).__init__()
        self._optimizer = optimizer
        self._optimizer_name = "Distributed%s" % optimizer.type.capitalize()
        if optimizer.type != "adam":
            print("Currently, distributed optimizer only supports Adam"
                  "Will config built-in adam for you."
                  "We will support more functions in DistributedOptimizer",
                  sys.stderr)
            self._optimizer_name = "DistributedAdam"

        self._distributed_optimizer = globals()[self._optimizer_name]()

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        pass

    def apply_gradients(self, params_grads):
        pass

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        optimize_ops, param_grads, opt_info = \
                      self._distributed_optimizer.minimize(
                          self._optimizer,
                          loss,
                          startup_program,
                          parameter_list,
                          no_grad_set)

        fleet_instance._set_opt_info(opt_info)
        return [optimize_ops, param_grads]
