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

from __future__ import print_function
from paddle.fleet import RoleMakerBase
from . import obj_creator
from strategy_compiler import StrategyCompiler
from meta_optimizer import MetaOptimizerFactory

__all__ = ['Fleet']


class Fleet(object):
    """
    Unified API for distributed training of PaddlePaddle
    Fleet is initialized through init function
    """

    def __init__(self):
        pass

    def init(self, role_maker):
        self.role_maker = role_maker
        self.strategy_compiler = StrategyCompiler()

    def distributed_optimizer(self, optimizer, strategy):
        self.user_defined_optimizer = optimizer
        self.user_defined_strategy = strategy

    def minimize(self, loss):
        distributed_optimizer_list = \
            MetaOptimizerFactory()._get_valid_meta_optimizers()
        valid_optimizer_list = []
        # recall meta optimizers for ranking
        for opt in distributed_optimizer_list:
            if opt.can_apply(loss, self.role_maker, self.user_defined_optimizer,
                             self.user_defined_strategy):
                valid_optimizer_list.append(opt)
        # combine recalled meta optimizers to be a valid meta optimizer
        meta_optimizer, compiled_strategy = \
                self.strategy_compiler.generate_optimizer(
                    loss, self.role_maker, self.optimizer,
                    self.strategy, valid_optimizer_list)
        optimize_ops, params_grads = meta_optimizer.minimize(loss)
        return optimize_ops, params_grads
