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

from paddle.fluid.optimizer import PipelineOptimizer as PO
from .meta_optimizer_base import MetaOptimizerBase

__all__ = ["PipelineOptimizer"]


class PipelineOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(PipelineOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(PipelineOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)
        num_microbatches = user_defined_strategy.pipeline_configs['micro_batch']
        self.wrapped_opt = PO(self.inner_opt, num_microbatches=num_microbatches)

    def _can_apply(self):
        if self.user_defined_strategy.pipeline == True:
            return True
        return False

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        return self.wrapped_opt.backward(loss, startup_program, parameter_list,
                                         no_grad_set, callbacks)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        optimize_ops, params_grads, prog_list = \
            self.wrapped_opt.minimize(loss, startup_program,
                                      parameter_list, no_grad_set)
        return optimize_ops, params_grads
