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

import paddle.fluid.io as io
from paddle.fluid.communicator import Communicator
from paddle.fluid.framework import default_main_program
from paddle.fluid.framework import default_startup_program
from paddle.fluid.framework import Program
from paddle.fluid.optimizer import Optimizer
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspiler as OriginTranspiler
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

from paddle.fluid.incubate.fleet.base.fleet_base import DistributedOptimizer
from paddle.fluid.incubate.fleet.base.fleet_base import Fleet
from paddle.fluid.incubate.fleet.base.fleet_base import Mode


class DistributedTranspiler(Fleet):
    """
    A subclass for compatibility with fluid.transpiler.DistributeTranspiler.
    """

    def __init__(self):
        super(DistributedTranspiler, self).__init__(Mode.TRANSPILER)
        self._transpile_config = None
        self._transpiler = None
        self._origin_program = None
        self.startup_program = None
        self.main_program = None
        self._communicator = None

    def init_worker(self):
        """
        `init_worker` has many many functions to do before training,
        first, wait for all parameter servers launch completely.
        second, run executor to initialize startup program
        third, wait for all worker initialize completely.

        Returns:
            None
        """
        if not self._transpile_config.sync_mode:
            self._communicator = Communicator(self.main_program)
            self._communicator.start()

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
        if not self.startup_program:
            raise ValueError(
                "startup_program is None, need invoke DistributedOptimizer.minimize first"
            )

        self._executor.run(self.startup_program)

        if model_dir:
            if not os.path.isdir(model_dir):
                raise ValueError("There is no directory named '%s'", model_dir)

            io.load_persistables(self._executor, model_dir, self.main_program)

    def run_server(self):
        """
        `run_server` execute executor to start pserver main program.

        Returns:
            None
        """
        if not self.main_program:
            raise ValueError(
                "main_program is None, need invoke DistributedOptimizer.minimize first"
            )

        self._executor.run(self.main_program)

    def stop_worker(self):
        """
        Close this executor.

        For the distributed training, this method would free the resource on PServers related to
        the current Trainer.

        Returns:
            None
        """
        if not self._transpile_config.sync_mode:
            self._communicator.stop()
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
        if main_program is not None:
            io.save_inference_model(dirname, feeded_var_names, target_vars,
                                    executor, main_program, None, None,
                                    export_for_deployment)
        else:
            io.save_inference_model(dirname, feeded_var_names, target_vars,
                                    executor, self._origin_program, None, None,
                                    export_for_deployment, True)

            model_basename = "__model__"
            model_filename = os.path.join(dirname, model_basename)

            with open(model_filename, "rb") as f:
                program_desc_str = f.read()

            program = Program.parse_from_string(program_desc_str)
            program._copy_dist_param_info_from(self.main_program)
            self.save_persistables(executor, dirname, program)

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

        if main_program is None:
            main_program = self.main_program

        if not main_program._is_distributed:
            raise ValueError(
                "main_program is for local, may not use fleet.save_persistables")

        io.save_persistables(executor, dirname, main_program, None)

    def _transpile(self, config):
        if not isinstance(config, DistributeTranspilerConfig):
            raise ValueError(
                "config must be an instance of DistributeTranspilerConfig")

        if not config.sync_mode:
            config.runtime_split_send_recv = True

        # _origin_program is a deep copy for default_main_program, for inference
        self._origin_program = default_main_program().clone(for_test=False)

        self._transpile_config = config
        self._transpiler = OriginTranspiler(config)

        if self.is_worker():
            self._transpiler.transpile(
                trainer_id=fleet.worker_index(),
                pservers=fleet.server_endpoints(to_string=True),
                trainers=fleet.worker_num(),
                sync_mode=config.sync_mode)
            self.main_program = self._transpiler.get_trainer_program()
            self.startup_program = default_startup_program()
        else:
            self._transpiler.transpile(
                trainer_id=fleet.worker_index(),
                pservers=fleet.server_endpoints(to_string=True),
                trainers=fleet.worker_num(),
                sync_mode=config.sync_mode,
                current_endpoint=self.server_endpoints()[self.server_index()])
            self.main_program, self.startup_program = \
                self._transpiler.get_pserver_programs(self.server_endpoints()[self.server_index()])


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
