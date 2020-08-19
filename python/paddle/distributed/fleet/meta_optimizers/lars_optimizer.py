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

from paddle.fluid.optimizer import Momentum, LarsMomentumOptimizer
from .meta_optimizer_base import MetaOptimizerBase
import logging

__all__ = ["LarsOptimizer"]


class LarsOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(LarsOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.lars_opt = None
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(LarsOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)

        opt = self.inner_opt
        if not isinstance(opt, Momentum):
            return

        configs = self.user_defined_strategy.lars_configs

        self.lars_opt = LarsMomentumOptimizer(
            learning_rate=opt._learning_rate,
            momentum=opt._momentum,
            lars_coeff=configs['lars_coeff'],
            lars_weight_decay=configs['lars_weight_decay'],
            parameter_list=opt._parameter_list,
            regularization=opt.regularization,
            grad_clip=opt._grad_clip,
            name=opt._name)

    def _can_apply(self):
        if self.user_defined_strategy.lars:
            if not isinstance(self.inner_opt, Momentum):
                logging.warn(
                    "lars need the inner optimizer to be Momentum optimizer.")
                return False
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.lars = False
        dist_strategy.lars_configs = {}

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        return self.lars_opt.backward(loss, startup_program, parameter_list,
                                      no_grad_set, callbacks)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        optimize_ops, params_grads = \
            self.lars_opt.minimize(loss, startup_program,
                                      parameter_list, no_grad_set)
        return optimize_ops, params_grads
