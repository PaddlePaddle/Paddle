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
import logging

import paddle.fluid as fluid
import paddle.fluid.io as io
from paddle.fluid.optimizer import Optimizer
import paddle.fluid.transpiler.distribute_transpiler as dist_transpiler

from ..base.fleet_base import Fleet
from ..base.fleet_base import Mode
from ..base.fleet_base import DistributedOptimizer


class Collective(Fleet):
    def __init__(self):
        super(Collective, self).__init__(Mode.COLLECTIVE)
        self.local_ip_ = 0

    def init(self, role_maker=None):
        super(Collective, self).init(role_maker)
        self._role_maker._generate_role()

    def init_worker(self, executor):
        logging.warn(
            "You should not call 'init_worker' method for collective mode.")

    def run_worker(self, executor, main_program=None):
        logging.warn(
            "You should not call 'run_worker' method for collective mode.")

    def init_server(self, executor, model_dir=None):
        logging.warn(
            "You should not call 'init_server' method for collective mode.")

    def run_server(self, executor):
        logging.warn(
            "You should not call 'run_server' method for collective mode.")

    def stop_worker(self):
        logging.warn(
            "You should not call 'stop_worker' method for collective mode.")

    def stop(self, executor):
        """
        stop(): will be called after a user finishes his/her training task.
        """
        self.role_maker_.barrier_all()
        self.role_maker_.finalize()

    def distributed_optimizer(self, optimizer, strategy=None):
        if not isinstance(optimizer, Optimizer):
            raise ValueError("optimizer must be an instance of Optimizer")
        self.optimizer = CollectiveOptimizer(optimizer, strategy)
        return self.optimizer

    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names=None,
                             target_vars=None,
                             main_program=None,
                             export_for_deployment=True):
        io.save_inference_model(dirname, feeded_var_names, target_vars,
                                executor, main_program, None, None,
                                export_for_deployment)

    def save_persistables(self, executor, dirname, main_program=None):
        io.save_persistables(executor, dirname, main_program, None)

    def _get_worker_endpoints(self):
        return self.worker_endpoints

    def _get_current_id(self):
        return self.current_id


fleet = Collective()
init = fleet.init
distributed_optimizer = fleet.distributed_optimizer
get_worker_endpoints = fleet._get_worker_endpoints
get_current_id = fleet._get_current_id


class CollectiveOptimizer(DistributedOptimizer):
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
        super(CollectiveOptimizer, self).__init__(optimizer, strategy)
        assert strategy is None, "You cannot set 'strategy' for collective."

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        return self._optimizer.backward(loss, startup_program, parameter_list,
                                        no_grad_set, callbacks)

    def apply_gradients(self, params_grads):
        return self._optimizer.apply_gradients(params_grads)

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        minimize a program through loss
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
        optimize_ops, param_grads = self._optimizer.minimize(
            loss, startup_program, parameter_list, no_grad_set)

        worker_endpoints = fleet.worker_endpoints
        trainer_id = fleet.current_id
        current_endpoint = fleet.current_endpoint

        startup_program = startup_program if startup_program else \
            fluid.framework.default_startup_program

        # call transpiler
        config = dist_transpiler.DistributeTranspilerConfig()
        config.mode = "nccl2"
        t = dist_transpiler.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id,
            trainers=','.join(worker_endpoints),
            startup_program=startup_program,
            current_endpoint=current_endpoint)

        return optimize_ops, param_grads
