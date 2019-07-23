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

from paddle.fluid.incubate.fleet.base.fleet_base import Fleet
from paddle.fluid.incubate.fleet.base.fleet_base import Mode
from paddle.fluid.incubate.fleet.base.fleet_base import DistributedOptimizer


class DistributedStrategy(object):
    def __init__(self):
        # precision configs
        self.use_fp16 = False
        self.use_fp32 = True
        # algorithmic communication
        self.local_sgd = False
        self.dgc = False
        # communication topology configs
        self.h_allreduce = False

    def build(self):
        self.strategy_map = {}
        # make sure we set single precision config True
        if self.use_fp32 and self.use_fp16:
            self.use_fp16 = False
        # make sure we set single algorithmic communication True
        if self.local_sgd and self.dgc:
            self.local_sgd = False
        self.strategy_map["fp16"] = self.use_fp16
        self.strategy_map["fp32"] = self.use_fp32
        self.strategy_map["localsgd"] = self.local_sgd
        self.strategy_map["dgc"] = self.dgc
        self.strategy_map["h_allreduce"] = self.h_allreduce


class DistributedOptimizerFactory(object):
    def __init__(self):
        self.strategy_to_optimizer_map()

    def strategy_to_optimizer_map(self):
        pattern = {}
        pattern["fp16"] = ["FP16SGDOptimizer", "FP16LocalSGDOptimizer"]
        pattern["fp32"] = ["FP32SGDOptimizer", "FP32LocalSGDOptimizer"]
        pattern["localsgd"] = ["FP16LocalSGDOptimizer", "FP32LocalSGDOptimizer"]
        pattern["h_allreduce"] = [
            "FP32SGDOptimizer",
            "FP32LocalSGDOptimizer",
            "FP16SGDOptimizer",
            "FP16LocalSGDOptimizer",
        ]
        self.pattern = pattern

    def create_by_strategy(self, optimizer, strategy):
        if strategy == None:
            strategy = DistributedStrategy()
        strategy.build()
        strategy_list = []
        for key in strategy.strategy_map:
            if strategy.strategy_map[key]:
                strategy_list.append(self.pattern[key])
        classname = list(set.intersection(*map(set, strategy_list)))[0]
        return globals()[classname](optimizer, strategy)


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
        optimizer_factory = DistributedOptimizerFactory()

        self._optimizer = \
            optimizer_factory.create_by_strategy(optimizer, strategy)
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
    Collective Operator Base Class For Distributed Optimizer
    The class is invisible to a user
    """

    def __init__(self, optimizer, strategy=None):
        super(CollectiveOpBasedOptimizer, self).__init__(optimizer, strategy)

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


class FP16SGDOptimizer(CollectiveOpBasedOptimizer):
    """
    do all reduce within every minibatch
    """

    def __init__(self, optimizer, strategy=None):
        super(FP16SGDOptimizer, self).__init__(optimizer, strategy)

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        pass


class FP32LocalSGDOptimizer(CollectiveOpBasedOptimizer):
    def __init__(self, optimizer, strategy=None):
        super(FP32LocalSGDOptimizer, self).__init__(optimizer, strategy)

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        opts, param_and_grads = self._optimizer.minimize(loss)
        config = fluid.DistributeTranspilerConfig()
        config.mode = 'collective'
        config.collective_mode = 'local_sgd'
        t = fluid.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id=fleet.worker_index(),
            trainers=fleet.worker_endpoints(),
            current_endpoint=fleet.worker_endpoints()[fleet.worker_index()],
            startup_program=startup_program,
            program=loss.block.program)
        return opts, param_and_grads


class FP32SGDOptimizer(CollectiveOpBasedOptimizer):
    def __init__(self, optimizer, strategy=None):
        super(FP32SGDOptimizer, self).__init__(optimizer, strategy)

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        opts, param_and_grads = self._optimizer.minimize(loss)
        config = fluid.DistributeTranspilerConfig()
        config.mode = 'collective'
        config.collective_mode = 'grad_allreduce'
        t = fluid.DistributeTranspiler(config=config)

        t.transpile(
            trainer_id=fleet.worker_index(),
            trainers=fleet.worker_endpoints(),
            current_endpoint=fleet.worker_endpoints()[fleet.worker_index()],
            startup_program=startup_program,
            program=loss.block.program)
        return opts, param_and_grads


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
