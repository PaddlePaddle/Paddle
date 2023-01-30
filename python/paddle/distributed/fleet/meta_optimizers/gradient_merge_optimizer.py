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
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from .meta_optimizer_base import MetaOptimizerBase

__all__ = []


class GradientMergeOptimizer(MetaOptimizerBase):
<<<<<<< HEAD
    def __init__(self, optimizer):
        super().__init__(optimizer)
=======

    def __init__(self, optimizer):
        super(GradientMergeOptimizer, self).__init__(optimizer)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.inner_opt = optimizer
        self.wrapped_opt = None
        self.meta_optimizers_white_list = [
            "AMPOptimizer",
            "LarsOptimizer",
            "LambOptimizer",
            "GraphExecutionOptimizer",
            "RecomputeOptimizer",
        ]
        self.meta_optimizers_black_list = []

<<<<<<< HEAD
    def _set_basic_info(
        self, loss, role_maker, user_defined_optimizer, user_defined_strategy
    ):
        super()._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy
        )
=======
    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(GradientMergeOptimizer,
              self)._set_basic_info(loss, role_maker, user_defined_optimizer,
                                    user_defined_strategy)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _init_wrapped_opt(self):
        config = self.user_defined_strategy.gradient_merge_configs
        self.wrapped_opt = GM(self.inner_opt)
        self.wrapped_opt._set_k_steps(
<<<<<<< HEAD
            self.user_defined_strategy.gradient_merge_configs["k_steps"]
        )
        self.wrapped_opt._set_avg(
            self.user_defined_strategy.gradient_merge_configs["avg"]
        )
=======
            self.user_defined_strategy.gradient_merge_configs["k_steps"])
        self.wrapped_opt._set_avg(
            self.user_defined_strategy.gradient_merge_configs["avg"])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

<<<<<<< HEAD
        can_apply = (
            self.user_defined_strategy.gradient_merge
        ) and self.user_defined_strategy.gradient_merge_configs["k_steps"] > 1
=======
        can_apply = (self.user_defined_strategy.gradient_merge == True) and \
            self.user_defined_strategy.gradient_merge_configs["k_steps"] > 1
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return can_apply

    def _disable_strategy(self, dist_strategy):
        dist_strategy.gradient_merge = False
        dist_strategy.gradient_merge_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        # we currently do not support auto-enable GradientMerge
        return

<<<<<<< HEAD
    def minimize_impl(
        self, loss, startup_program=None, parameter_list=None, no_grad_set=None
    ):
        self._init_wrapped_opt()
        optimize_ops, params_grads = self.wrapped_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set
        )
=======
    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self._init_wrapped_opt()
        optimize_ops, params_grads = \
            self.wrapped_opt.minimize(loss, startup_program,
                                      parameter_list, no_grad_set)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return optimize_ops, params_grads
