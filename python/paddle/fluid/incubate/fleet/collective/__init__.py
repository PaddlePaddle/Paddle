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

class DGCConfig(Object):
    def __init__(self):
        self.learning_rate=0.001
        self.momentum=None
        self.rampup_begin_step=0
        self.rampup_step=1
        self.sparsity=[0.999]
        self.use_nesterov=False
        self.local_grad_clip_norm=None
        self.num_trainers=None
        self.regularization=None
        self.name=None

class LocalSGCConfig(Object):
    def __init__(self):
        pass

class LambConfig(Object):
    def __init__(self):
        pass

class DistFCConfig(Object):
    def __init__(self):
        pass

class DistributedStrategy(fluid.BuildStratey):
    """
    Init function of DistributedStrategy
    """
    def __init__(self):
        #self.build_stratey = None
        """
        self.nccl_comm_num=1
        self.use_hierarchical_allreduce=False

        self.fuse_all_reduce_ops = False
        """
        self.fuse_memory_size = -1
        self.fuse_layer_size  = 1

        self.use_local_sgd = False
        self.use_dgc = False
        self.use_lamb = False
        self.use_dist_fc = False

        self.local_sgd_config = None  # LocalSGDConfig
        self.dgc_config = None   # DGCConfig
        self.lamb_config = None  # LambConfig
        self.dist_fc_config = None  # DistFCConfig


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
        self._optimizer = \
            CollectiveOptimizer(optimizer, strategy)
        return self._optimizer

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


fleet = Collective()


class CollectiveOpBasedOptimizer(DistributedOptimizer):
    """
    Collective Operator Base Class For Distributed Optimizer
    The class is invisible to a user
    """
    def __init__(self, optimizer, strategy=None):
        assert isinstance(
            strategy,DistributedStrategy), "strategy must be DistributedStrategy"
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

class DistributedDGCOptimizer(Object):
    def __init__(self):
        pass

class DistributedLambOptimizer(object)
    def __init__(self):
        pass

def _create_dgc_optimizer(optimizer, dgc_config):
    return fluid.DGCOptimizer(
        learning_rate = dgc_config.learning_rate,
        momentum = optimizer,
        rampup_begin_step = dgc_config.rampup_begin_step,
        rampup_step=dgc_config.rampup_step,
        sparsity=dgc_config.sparsity,
        use_nesterov=dgc_config.use_nesterov,
        local_grad_clip_norm=dgc_config.local_grad_clip_norm,
        num_trainers=dgc_config.num_trainers,
        regularization=dgc_config.regularization,
        name=dgc_config.name
    )

def _create_lamb_optimizer(optimizer, lamb_config):
    pass


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
        dist_optimizer = self._get_create_optimizer(optimizer, strategy)
        super(CollectiveOptimizer, self).__init__(dist_optimizer, strategy)

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

    @staticmethod
    def _get_create_optimizer(optimizer, strategy):
        error_string="You can use only one of 'use_dgc use_lamb use_dist_fc use_local_sgd' option"
        if strategy.use_local_sgd:
            #check conditions
            assert not (strategy.use_dgc
                        or strategy.use_lamb or strategy.use_dist_fc), error_string

            return optimizer

        if strategy.use_dgc:
            #check conditions
            assert not (strategy.use_local_sgd
                        or strategy.use_lamb or strategy.use_dist_fc), error_string
            assert strategy.dgc_config is not None, "DistributedStrategy.dgc_config should be set"
            return _create_dgc_optimizer(optimizer, strategy.dgc_config)

        if strategy.use_lamb:
            #check conditions
            assert not (strategy.use_local_sgd
                        or strategy.use_dgc or strategy.use_dist_fc), error_string
            assert strategy.dgc_config is not None, "DistributedStrategy.lamb_config should be set"
            return _create_lamb_optimizer(optimizer, strategy.lamb_config)

        if strategy.use_dist_fc:
            #check conditions
            assert not (strategy.use_local_sgd
                        or strategy.use_dgc or strategy.use_lamb), error_string
            return optimizer

    def _transpiler():
        if self._strategy.fuse_all_reduce_ops:
            os.environ['FLAGS_fuse_parameter_memory_size'] = self.fuse_memory_size
            os.environ['FLAGS_fuse_parameter_groups_size'] = self.fuse_layer_size

        worker_endpoints = fleet.worker_endpoints()
        trainer_id = fleet.worker_index()
        current_endpoint = fleet.worker_endpoints()[trainer_id]

        startup_program = startup_program if startup_program else \
            fluid.framework.default_startup_program

        main_program = loss.block.program

        # call transpiler
        config = dist_transpiler.DistributeTranspilerConfig()
        config.mode = "nccl2"
        t = dist_transpiler.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id,
            trainers=','.join(worker_endpoints),
            startup_program=startup_program,
            program = main_program,
            current_endpoint=current_endpoint)


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
        self._transpiler()

        optimize_ops, param_grads = self._optimizer.minimize(
            loss, startup_program, parameter_list, no_grad_set)

        return optimize_ops, param_grads
