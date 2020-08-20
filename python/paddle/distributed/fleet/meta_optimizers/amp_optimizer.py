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

import paddle.fluid.contrib.mixed_precision as mixed_precision
from .meta_optimizer_base import MetaOptimizerBase

__all__ = ["AMPOptimizer"]


class AMPOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(AMPOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.amp_opt = None
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(AMPOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)

    def _can_apply(self):
        if self.user_defined_strategy.amp:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.amp = False
        dist_strategy.amp_configs = {}

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        if self.amp_opt is None:
            config = self.user_defined_strategy.amp_configs
            custom_white_list = set(config['custom_white_list'])
            custom_black_list = set(config['custom_black_list'])
            custom_black_varnames = set(config['custom_black_varnames'])
            amp_lists = mixed_precision.AutoMixedPrecisionLists(
                custom_white_list, custom_black_list, custom_black_varnames)

            self.amp_opt = mixed_precision.decorate(
                self.inner_opt, amp_lists, config['init_loss_scaling'],
                config['incr_every_n_steps'], config['decr_every_n_nan_or_inf'],
                config['incr_ratio'], config['decr_ratio'],
                config['use_dynamic_loss_scaling'])

        optimize_ops, params_grads = \
            self.amp_opt.minimize(loss, startup_program,
                                  parameter_list, no_grad_set)
        return optimize_ops, params_grads
