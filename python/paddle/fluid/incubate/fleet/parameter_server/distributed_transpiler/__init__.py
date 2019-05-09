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

from paddle.fluid.framework import default_startup_program

from paddle.fluid.optimizer import Optimizer

import paddle.fluid.io as io

from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspiler as OriginTranspiler

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

    def init_worker(self):
        """
        `init_worker` has many many functions to do before training,
        first, wait for all parameter servers launch completely.
        second, run executor to initialize startup program
        third, wait for all worker initialize completely.

        Returns:
            None
        """
        pass

    def run_worker(self, main_programs=None, scopes=None):
        pass

    def init_server(self, model_dir=None):
        """
        `init_server` has many many functions to do before start pserver,
        first, run executor to initialize startup program,
        second, if the `model_dir` is not empty, it will load parameters from it for increment training.

        Args:
            model_dir(str): The directory path.

        Returns:
            None
        """
        if not self._startup_program:
            raise ValueError(
                "startup_program is None, need invoke DistributedOptimizer.minimize first"
            )

        self._executor.run(self._startup_program)

        if model_dir:
            if not os.path.isdir(model_dir):
                raise ValueError("There is no directory named '%s'", model_dir)

            io.load_persistables(self._executor, model_dir,
                                 self._startup_program)

    def run_server(self):
        """
        `run_server` execute executor to start pserver main program.

        Returns:
            None
        """
        if not self._main_program:
            raise ValueError(
                "main_program is None, need invoke DistributedOptimizer.minimize first"
            )

        self._executor.run(self._main_program)

    def stop_worker(self):
        pass

    def stop(self):
        """
        Close this executor.

        For the distributed training, this method would free the resource on PServers related to
        the current Trainer.

        Returns:
            None
        """
        self._executor.close()

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
        self._optimizer = TranspilerOptimizer(optimizer, strategy)
        return self._optimizer

    def save_inference_model(self,
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
                                self._executor, main_program, None, None,
                                export_for_deployment)

    def save_persistables(self, dirname, main_program=None):
        """
        This function filters out all variables with `persistable==True` from the
        give `main_program` and then saves these variables to the folder `dirname`
        or file `filename`.

        The `dirname` is used to specify the folder where persistable variables
        are going to be saved. If you would like to save variables in separate
        files, set `filename` None; if you would like to save all variables in a
        single file, use `filename` to specify the file name.
        """
        io.save_persistables(self._executor, dirname, main_program, None)

    def _transpile(self, config):
        self._transpiler = OriginTranspiler(config)
        self._transpiler.transpile(
            trainer_id=fleet.worker_index(),
            pservers=fleet.server_endpoints(to_string=True),
            trainers=fleet.worker_num())

        if self.is_worker():
            self._main_program = self._transpiler.get_trainer_program()
            self._startup_program = default_startup_program()
        else:
            self._main_program, self._startup_program = \
                self._transpiler.get_pserver_programs(self.server_endpoints(self.server_index()))


fleet = DistributedTranspiler()


class TranspilerOptimizer(DistributedOptimizer):
    """
    DistributedOptimizer is a wrapper for paddle.fluid.optimizer
    A user should pass a paddle.fluid.optimizer to DistributedOptimizer
    minimize() function is implemented.
    DistributedOptimizer is the starting point for a user who wants to
    run distributed training. The optimized information will be stored in
    Fleet() instance who holds the global information about current distributed
    training.

    Args:
        optimizer(Optimizer): subclass of Optimizer.
        strategy(DistributeTranspilerConfig): instance of DistributeTranspilerConfig.

    Returns:
        None
    """

    def __init__(self, optimizer, strategy=None):
        super(TranspilerOptimizer, self).__init__(optimizer, strategy)

        if strategy:
            if not isinstance(strategy, DistributeTranspilerConfig):
                raise ValueError(
                    "In {} mode, strategy must be an instance of DistributeTranspilerConfig".
                    format(fleet._mode))
            else:
                self._strategy = strategy
        else:
            self._strategy = DistributeTranspilerConfig()

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        First part of `minimize`, do auto-diff to append backward ops for
        the current program.

        Args:
            loss (Variable): loss variable to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables to update.
            no_grad_set (set|None): set of Variables should be ignored.
            callbacks (list|None): list of callables to run when appending backward
                operator for one parameter.

        Return:
            list: list of (param, grad) pair, grad is the output of backward.

        Examples:
            See examples in `apply_gradients`.
        """
        return self._optimizer.backward(loss, startup_program, parameter_list,
                                        no_grad_set, callbacks)

    def apply_gradients(self, params_grads):
        """
        Second part of `minimize`, appending optimization operators for
        given `params_grads` pairs.

        Args:
            params_grads (list): list of (param, grad) pair to do optimization.

        Returns:
            list: A list of operators appended to the current program.

        Examples:
            .. code-block:: python

                loss = network()
                optimizer = fluid.optimizer.SGD(learning_rate=0.1)
                params_grads = optimizer.backward(loss)
                # you may append operations for params_grads here
                # ...
                optimizer.apply_gradients(params_grads)
        """
        return self._optimizer.apply_gradients(params_grads)

    def minimize(self,
                 loss,
                 scope=None,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):

        if isinstance(loss, list):
            raise ValueError(
                "DistributedTranspiler's minimize can not accept loss with list")

        if isinstance(startup_program, list):
            raise ValueError(
                "DistributedTranspiler's minimize can not accept program with list"
            )

        optimize_ops, params_grads = self._optimizer.minimize(
            loss, startup_program, parameter_list, no_grad_set)
        fleet._transpile(config=self._strategy)
        return optimize_ops, params_grads
