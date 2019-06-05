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

import logging

import paddle.fluid as fluid
import paddle.fluid.io as io
import paddle.fluid.transpiler.distribute_transpiler as dist_transpiler

from ..base.fleet_base import Fleet
from ..base.fleet_base import Mode
from ..base.fleet_base import DistributedOptimizer


class DistributedStrategy(object):
    def __init__(self):
        self.use_fp16 = False
        self.use_fp32 = False
        self.local_sgd = False
        self.dgc = False
        self.hierachical_allreduce = False

    def build(self):
        if self.use_fp32 and self.use_fp16:
            self.use_fp16 = False
        if self.local_sgd and self.dgc:
            self.local_sgd = False


class DistributedOptimizerFactory(object):
    def strategy_to_optimizer_map(self):
        pattern = {}
        pattern["fp16"] = [
            MixedPrecisionOptimizer, MixedPrecisionLocalSGDOptimizer
        ]
        pattern["fp32"] = [LocalSGDOptimizer]
        pattern[
            "localsgd"] = [MixedPrecisionLocalSGDOptimizer, LocalSGDOptimizer]

    def create_by_strategy(self, strategy):
        pass


class Collective(Fleet):
    def __init__(self):
        super(Collective, self).__init__(Mode.COLLECTIVE)
        self._local_ip = 0

    def init_worker(self):
        logging.warn(
            "You should not call 'init_worker' method for collective mode.")

    def run_worker(self, main_programs=None, scopes=None):
        logging.warn(
            "You should not call 'run_worker' method for collective mode.")

    def init_server(self, model_dir=None):
        logging.warn(
            "You should not call 'init_server' method for collective mode.")

    def run_server(self):
        logging.warn(
            "You should not call 'run_server' method for collective mode.")

    def stop_worker(self):
        logging.warn(
            "You should not call 'stop_worker' method for collective mode.")

    def distributed_optimizer(self, optimizer, strategy=None):
        #self._optmizer = DistributedOptimizerFactory.create_by_strategy(
        #strategy)
        self._optimizer = FullPrecisionOptimizer()
        return self._optimizer

    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names=None,
                             target_vars=None,
                             main_program=None,
                             export_for_deployment=True):
        io.save_inference_model(dirname, feeded_var_names, target_vars,
                                self._executor, main_program, None, None,
                                export_for_deployment)

    def save_persistables(self, executor, dirname, main_program=None):
        io.save_persistables(self._executor, dirname, main_program, None)


fleet = Collective()


class CollectiveOpBasedOptimizer(DistributedOptimizer):
    """
    TBA
    """

    def _transpile_program(self, startup_program=None):
        startup_program = startup_program if startup_program else \
                          fluid.framework.default_startup_program
        worker_endpoints = fleet.worker_endpoints()
        trainer_id = fleet.worker_index()
        current_endpoint = fleet.worker_endpoints()[trainer_id]
        # call transpiler
        config = dist_transpiler.DistributeTranspilerConfig()
        config.mode = "collective"
        t = dist_transpiler.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id,
            trainers=','.join(worker_endpoints),
            startup_program=startup_program,
            current_endpoint=current_endpoint)

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


class MixedPrecisionOptimizer(CollectiveOpBasedOptimizer):
    """
    TBA
    """

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        pass


class FullPrecisionOptimizer(CollectiveOpBasedOptimizer):
    """
    TBA
    """

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        self._transpile_program(startup_program)

        param_grads = self.backward(loss, startup_program, parameter_list,
                                    no_grad_set)
        train_program.global_block().append_op(type='c_sync_compute_stream')
        data_parallel_param_grads = []
        for p, g in param_grads:
            # NOTE: scale will be done on loss scale
            # in multi_devices_graph_pass using nranks.
            reduced_g = fluid.layers.collective._allreduce(g, g)
            data_parallel_param_grads.append([p, reduced_g])
        train_program.global_block().append_op(type='c_sync_comm_stream')
        self.apply_gradients(data_parallel_param_grads)


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
        self.strategy = strategy

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

        worker_endpoints = fleet.worker_endpoints()
        trainer_id = fleet.worker_index()
        current_endpoint = fleet.worker_endpoints()[trainer_id]

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
