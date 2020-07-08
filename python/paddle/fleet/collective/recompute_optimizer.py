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

import paddle.fluid.optimizer.RecomputeOptimizer as RO


class RecomputeOptimizer(object):
    def __init__(self, optimizer):
        self.inner_opt = RO(optimizer)
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        self.loss = loss
        self.role_maker = role_maker
        self.user_defined_optimizer = user_defined_optimizer
        self.user_defined_strategy = user_defined_strategy

    def _update_inner_optimier(self, optimizer):
        self.inner_opt = optimizer

    def _can_apply(self):
        if self.user_defined_strategy.recompute == True:
            if self.userd_defined_strategy.auto == True:
                return True
            elif len(self.user_defined_strategy.checkpoints) == 0:
                return False
            else:
                return True

    def _can_update(self, optimizer):
        if str(optimizer.__class__.__name__) in self.meta_optimizers_white_list:
            return True

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        return self.inner_opt.backward(loss, startup_program, parameter_list,
                                       no_grad_set, callbacks)

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        optimize_ops, params_grads = \
            self.inner_opt.minimize(loss, startup_program,
                                    parameter_list, no_grad_set)
