# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.optimizer import Optimizer
from ...utils.hybrid_parallel_util import fused_allreduce_gradients
from ...base.topology import ParallelMode
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid import framework
from paddle.fluid.framework import Variable


class HybridParallelOptimizer:
    def __init__(self, optimizer, hcg, strategy):
        self._inner_opt = optimizer
        self._strategy = strategy
        self._hcg = hcg
        self._is_mp = (
            self._hcg.get_parallel_mode() == ParallelMode.MODEL_PARALLEL)
        self._need_dp = (self._hcg.get_data_parallel_world_size() > 1)

    @imperative_base.no_grad
    @framework.dygraph_only
    def step(self):
        if self._is_mp and self._need_dp:
            fused_allreduce_gradients(
                list(self._inner_opt._parameter_list), self._hcg)
        self._inner_opt.step()

    @imperative_base.no_grad
    def minimize(self,
                 loss,
                 startup_program=None,
                 parameters=None,
                 no_grad_set=None):
        assert isinstance(loss, Variable), "The loss should be an Tensor."

        parameter_list = parameters if parameters \
            else self._parameter_list

        if self._is_mp and self._need_dp:
            fused_allreduce_gradients(list(parameter_list), self._hcg)

        return self._inner_opt.minimize(loss, startup_program, parameters,
                                        no_grad_set)

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)
