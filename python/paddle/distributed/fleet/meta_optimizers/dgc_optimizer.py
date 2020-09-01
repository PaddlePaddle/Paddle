#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.optimizer import Momentum, DGCMomentumOptimizer
from .meta_optimizer_base import MetaOptimizerBase
import logging


class DGCOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(DGCOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.dgc_opt = None
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []
        self.meta_optimizers_black_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(DGCOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)

        opt = self.inner_opt
        if not isinstance(opt, Momentum):
            return

        configs = self.user_defined_strategy.dgc_configs
        if len(configs['sparsity']) == 0:
            # default is [0.999]
            configs['sparsity'] = [0.999]

        self.dgc_opt = DGCMomentumOptimizer(
            learning_rate=opt._learning_rate,
            momentum=opt._momentum,
            rampup_begin_step=configs['rampup_begin_step'],
            rampup_step=configs['rampup_step'],
            sparsity=configs['sparsity'],
            parameter_list=opt._parameter_list,
            use_nesterov=opt._use_nesterov,
            num_trainers=self.role_maker.worker_num(),
            regularization=opt.regularization,
            grad_clip=opt._grad_clip,
            name=opt._name)

    def _can_apply(self):
        if self.user_defined_strategy.dgc:
            if not isinstance(self.inner_opt, Momentum):
                logging.warn("dgc only works on Momentum optimizer")
                return False
            if self.role_maker.worker_num() <= 1:
                logging.warn("dgc only works on multi cards")
                return False

            return True

        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.dgc = False
        dist_strategy.dgc_configs = {}

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        return self.dgc_opt.backward(loss, startup_program, parameter_list,
                                     no_grad_set, callbacks)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        optimize_ops, params_grads = \
            self.dgc_opt.minimize(loss, startup_program,
                                      parameter_list, no_grad_set)
        return optimize_ops, params_grads
