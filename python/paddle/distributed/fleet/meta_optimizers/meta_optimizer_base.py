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
# limitations under the License.

__all__ = ["MetaOptimizerBase"]


class MetaOptimizerBase(object):
    def __init__(self, optimizer):
        pass

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        self.loss = loss
        self.role_maker = role_maker
        self.user_defined_optimizer = user_defined_optimizer
        self.user_defined_strategy = user_defined_strategy

    def _update_inner_optimier(self, optimizer):
        self.inner_opt = optimizer

    def _can_apply(self):
        return False

    def _is_graph_out(self):
        return False

    def _can_update(self, optimizer):
        if str(optimizer.__class__.__name__) in self.meta_optimizers_white_list:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        raise NotImplementedError("you should implement disable strategy in {}".
                                  format(type(self).__name__))

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        raise NotImplementedError("meta optimizer not implemented")

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        optimize_ops, params_grads = self.minimize_impl(
            loss, startup_program, parameter_list, no_grad_set)
        return optimize_ops, params_grads
