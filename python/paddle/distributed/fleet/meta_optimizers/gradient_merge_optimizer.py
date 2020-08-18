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

from paddle.fluid.optimizer import GradientMergeOptimizer as GM
from .meta_optimizer_base import MetaOptimizerBase

__all__ = ["GradientMergeOptimizer"]


class GradientMergeOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(GradientMergeOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.wrapped_opt = GM(optimizer)
        self.meta_optimizers_white_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(GradientMergeOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)
        self.wrapped_opt._set_k_steps(
            self.user_defined_strategy.gradient_merge_configs["k_steps"])
        self.wrapped_opt._set_avg(
            self.user_defined_strategy.gradient_merge_configs["avg"])

    def _can_apply(self):
        can_apply = (self.user_defined_strategy.gradient_merge == True) and \
                  self.user_defined_strategy.gradient_merge_configs["k_steps"] > 1
        return can_apply

    def _disable_strategy(self, dist_strategy):
        dist_strategy.gradient_merge = False
        dist_strategy.gradient_merge_configs = {}

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        optimize_ops, params_grads = \
            self.wrapped_opt.minimize(loss, startup_program,
                                      parameter_list, no_grad_set)
        return optimize_ops, params_grads
