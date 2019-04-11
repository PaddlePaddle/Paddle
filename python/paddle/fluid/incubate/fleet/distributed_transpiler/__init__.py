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
import os
import sys

from paddle.fluid.executor import Executor

from paddle.fluid.framework import Program
from paddle.fluid.framework import default_main_program
from paddle.fluid.framework import default_startup_program

from paddle.fluid.optimizer import Optimizer

import paddle.fluid.io as io

from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspiler as OriginTranspiler

from python.paddle.fluid.incubate.fleet.base.role_maker import Role
from ..base.fleet_base import Fleet
from ..base.fleet_base import Mode
from ..base.fleet_base import DistributedOptimizer


class DistributedTranspiler(Fleet):
    def __init__(self):
        super(DistributedTranspiler, self).__init__(Mode.TRANSPILER)
        self._transpiler = OriginTranspiler()
        self._startup_program = None
        self._main_program = None

    def init_worker(self, executor):
        if not isinstance(executor, Executor):
            raise ValueError("executor must be an instance of Executor")

        if not self._startup_program:
            raise ValueError(
                "startup_program is None, need invoke DistributedOptimizer.minimize first"
            )

        executor.run(self._startup_program)

    def run_worker(self, executor, main_program=None):
        pass

    def init_server(self, executor, model_dir=None):
        if not isinstance(executor, Executor):
            raise ValueError("executor must be an instance of Executor")

        if not self._startup_program:
            raise ValueError(
                "startup_program is None, need invoke DistributedOptimizer.minimize first"
            )

        executor.run(self._startup_program)

        if model_dir:
            if not os.path.isdir(model_dir):
                raise ValueError("There is no directory named '%s'", model_dir)

            io.load_persistables(executor, model_dir, self._startup_program)

    def run_server(self, executor):
        if not isinstance(executor, Executor):
            raise ValueError("executor must be an instance of Executor")

        if not self._main_program:
            raise ValueError(
                "main_program is None, need invoke DistributedOptimizer.minimize first"
            )

        executor.run(self._main_program)

    def barrier_worker(self):
        pass

    def stop_worker(self):
        pass

    def stop(self, executor):
        if not isinstance(executor, Executor):
            raise ValueError("executor must be an instance of Executor")
        executor.close()

    def distributed_optimizer(self, optimizer, strategy=None):
        if not isinstance(optimizer, Optimizer):
            raise ValueError("optimizer must be an instance of Optimizer")
        self.optimizer = TranspilerOptimizer(optimizer, strategy)
        return self.optimizer

    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names,
                             target_vars,
                             main_program=None,
                             export_for_deployment=True):
        io.save_inference_model(dirname, feeded_var_names, target_vars,
                                executor, main_program, None, None,
                                export_for_deployment)

    def save_persistables(self, executor, dirname, main_program=None):
        io.save_persistables(executor, dirname, main_program, None)

    def _transpile(self, config):
        if not isinstance(config, DistributeTranspilerConfig):
            raise ValueError(
                "config must be an instance of DistributeTranspilerConfig")

        self._transpiler = OriginTranspiler(config)
        self._transpiler.transpile(
            trainer_id=fleet.worker_id(),
            pservers=fleet.server_endpoints,
            trainers=fleet.workers)

        if self.role == Role.WORKER:
            self._startup_program = default_startup_program()
            self._main_program = self._transpiler.get_trainer_program()
        else:
            self._main_program, self._startup_program = self._transpiler.get_pserver_programs(
                self.current_endpoint)


fleet = DistributedTranspiler()


class TranspilerOptimizer(DistributedOptimizer):
    def __init__(self, optimizer, strategy=None):
        super(TranspilerOptimizer, self).__init__(optimizer, strategy)

        if not isinstance(strategy, DistributeTranspilerConfig):
            raise ValueError(
                "In {} mode, strategy must be an instance of DistributeTranspilerConfig".
                format(fleet.mode))

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
        optimize_ops, params_grads = self._optimizer.minimize(
            loss, startup_program, parameter_list, no_grad_set)
        self.transpile()
        return optimize_ops, params_grads

    def transpile(self):
        fleet._transpile(OriginTranspiler(config=self._strategy))
