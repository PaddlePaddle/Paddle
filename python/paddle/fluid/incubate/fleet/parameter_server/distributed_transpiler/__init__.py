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

from ...base.role_maker import Role
from ...base.fleet_base import Fleet
from ...base.fleet_base import Mode
from ...base.fleet_base import DistributedOptimizer


class DistributedTranspiler(Fleet):
    """
    A subclass for compatibility with fluid.transpiler.DistributeTranspiler.
    """

    def __init__(self):
        super(DistributedTranspiler, self).__init__(Mode.TRANSPILER)
        self._transpiler = OriginTranspiler()
        self._startup_program = None
        self._main_program = None

    def init_worker(self, executor):
        """
        `init_worker` has many many functions to do before training,
        first, wait for all parameter servers launch completely.
        second, run executor to initialize startup program
        third, wait for all worker initialize completely.

        Args:
            executor(Executor): The executor to run for init startup program.

        Returns:
            None
        """
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
        """
        `init_server` has many many functions to do before start pserver,
        first, run executor to initialize startup program,
        second, if the `model_dir` is not empty, it will load parameters from it for increment training.

        Args:
            executor(Executor): The executor to run for init server.
            model_dir(str): The directory path.

        Returns:
            None
        """
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
        """
        `run_server` execute executor to start pserver main program.

        Args:
            executor(Executor): The executor to run for init server.

        Returns:
            None
        """
        if not isinstance(executor, Executor):
            raise ValueError("executor must be an instance of Executor")

        if not self._main_program:
            raise ValueError(
                "main_program is None, need invoke DistributedOptimizer.minimize first"
            )

        executor.run(self._main_program)

    def stop_worker(self):
        pass

    def stop(self, executor):
        """
        Close this executor.

        For the distributed training, this method would free the resource on PServers related to
        the current Trainer.

        Args:
            executor(Executor): The executor to run for init server.

        Returns:
            None
        """

        if not isinstance(executor, Executor):
            raise ValueError("executor must be an instance of Executor")
        executor.close()

    def distributed_optimizer(self, optimizer, strategy=None):
        """
        Optimizer for distributed training.

        For the distributed training, this method would rebuild a new instance of DistributedOptimizer.
        Which has basic Optimizer function and special features for distributed training.

        Args:
            optimizer(Optimizer): The executor to run for init server.
            strategy(dict): Extra properties for distributed optimizer.

        Returns:
            TranspilerOptimizer: subclass of DistributedOptimizer.
        """

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
        """
        Prune the given `main_program` to build a new program especially for inference,
        and then save it and all related parameters to given `dirname` by the `executor`.
        """
        io.save_inference_model(dirname, feeded_var_names, target_vars,
                                executor, main_program, None, None,
                                export_for_deployment)

    def save_persistables(self, executor, dirname, main_program=None):
        """
        This function filters out all variables with `persistable==True` from the
        give `main_program` and then saves these variables to the folder `dirname`
        or file `filename`.

        The `dirname` is used to specify the folder where persistable variables
        are going to be saved. If you would like to save variables in separate
        files, set `filename` None; if you would like to save all variables in a
        single file, use `filename` to specify the file name.
        """
        io.save_persistables(executor, dirname, main_program, None)

    def _transpile(self, config):
        if not isinstance(config, DistributeTranspilerConfig):
            raise ValueError(
                "config must be an instance of DistributeTranspilerConfig")

        self._transpiler = OriginTranspiler(config)
        self._transpiler.transpile(
            trainer_id=fleet.worker_id(),
            pservers=fleet.server_endpoints,
            trainers=fleet.worker_num())

        if self.role == Role.WORKER:
            self._main_program = self._transpiler.get_trainer_program()
            self._startup_program = default_startup_program()
        else:
            self._main_program, self._startup_program = \
                self._transpiler.get_pserver_programs(self.current_endpoint)


fleet = DistributedTranspiler()


class TranspilerOptimizer(DistributedOptimizer):
    def __init__(self, optimizer, strategy=None):
        super(TranspilerOptimizer, self).__init__(optimizer, strategy)

        if strategy and not isinstance(strategy, DistributeTranspilerConfig):
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
        if self._strategy is None:
            self._strategy = DistributeTranspilerConfig()

        fleet._transpile(config=self._strategy)
