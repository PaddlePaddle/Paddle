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

from __future__ import print_function
import sys
import paddle
from paddle.optimizer import Optimizer
from paddle.fluid.clip import ClipGradByGlobalNorm
from ...utils.hybrid_parallel_util import fused_allreduce_gradients, sharding_reduce_gradients
from ...base.topology import ParallelMode
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid import framework
from paddle.fluid.framework import Variable
from ...utils.log_util import logger
from paddle.fluid import core
from paddle.fluid import layers

__all__ = []


def _obtain_optimizer_parameters_list(optimizer):
    if getattr(optimizer, '_param_groups', None) and isinstance(
            optimizer._param_groups[0], dict):
        parameters_list = []
        for group in optimizer._param_groups:
            for param in group['params']:
                parameters_list.append(param)
    else:
        parameters_list = [param for param in optimizer._parameter_list]

    return parameters_list


class HybridParallelClipGrad:
    def __init__(self, clip, hcg):
        self._clip = clip
        self._hcg = hcg

    @imperative_base.no_grad
    def _dygraph_clip(self, params_grads):
        params_and_grads = []
        sum_square_list_dist = []
        sum_square_list_not_dist = []
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = layers.merge_selected_rows(g)
                merge_grad = layers.get_tensor_from_selected_rows(merge_grad)
            square = layers.square(merge_grad)
            sum_square = layers.reduce_sum(square)

            if p.is_distributed:
                sum_square_list_dist.append(sum_square)
            else:
                sum_square_list_not_dist.append(sum_square)

        # all parameters have been filterd out
        if len(sum_square_list_dist) + len(sum_square_list_not_dist) == 0:
            return params_grads

        global_norm_var_dist = layers.concat(sum_square_list_dist) if len(
            sum_square_list_dist) != 0 else layers.concat(
                [paddle.to_tensor([0.])])
        global_norm_var_dist = layers.reduce_sum(global_norm_var_dist)
        global_norm_var_not_dist = layers.concat(
            sum_square_list_not_dist) if len(
                sum_square_list_not_dist) != 0 else layers.concat(
                    [paddle.to_tensor([0.])])
        global_norm_var_not_dist = layers.reduce_sum(global_norm_var_not_dist)

        # add all reduce to get global norm of distributed params_and_grads in world size
        # all reduce is not needed while getting global norm of non-distributed params_and_grads
        paddle.distributed.all_reduce(
            global_norm_var_dist, group=self._hcg.get_check_parallel_group())

        global_norm_var = layers.sqrt(global_norm_var_dist +
                                      global_norm_var_not_dist)

        max_global_norm = layers.fill_constant(
            shape=[1], dtype=global_norm_var.dtype, value=self.clip_norm)
        clip_var = layers.elementwise_div(
            x=max_global_norm,
            y=layers.elementwise_max(
                x=global_norm_var, y=max_global_norm))
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                params_and_grads.append((p, g))
                continue
            new_grad = layers.elementwise_mul(x=g, y=clip_var)
            params_and_grads.append((p, new_grad))

        return params_and_grads

    def __getattr__(self, item):
        return getattr(self._clip, item)

    def __call__(self, params_grads):
        return self._dygraph_clip(params_grads)


class HybridParallelOptimizer:
    # adapter wrapper for optimizer
    def __init__(self, optimizer, hcg, strategy):
        self._inner_opt = optimizer
        self._strategy = strategy
        self._hcg = hcg

        self._use_dp_mode = (
            self._hcg.get_parallel_mode() == ParallelMode.DATA_PARALLEL)

        self._need_dp = (self._hcg.get_data_parallel_world_size() > 1)

        # NOTE(shenliang03): Because of the pure DataParallel mode, the gradient synchronization 
        # is achieved through reducer, so there is no need to call fuse_allreduce in optimizer. 
        self._dp_enable = not self._use_dp_mode and self._need_dp

        self._sharding_enable = (
            self._hcg.get_sharding_parallel_world_size() > 1)

        if isinstance(self._inner_opt._grad_clip,
                      ClipGradByGlobalNorm) and not self._use_dp_mode:
            logger.warning("using ClipGradByGlobalNorm in TensorParallel, the origin " \
                  "optmizer'grad clip will be changed.")

            self._inner_opt._grad_clip = HybridParallelClipGrad(
                self._inner_opt._grad_clip, hcg)

    @imperative_base.no_grad
    @framework.dygraph_only
    def step(self):
        parameters_list = _obtain_optimizer_parameters_list(self._inner_opt)
        if self._sharding_enable:
            sharding_reduce_gradients(list(parameters_list), self._hcg)

        if self._dp_enable:
            fused_allreduce_gradients(list(parameters_list), self._hcg)

        self._inner_opt.step()

    @imperative_base.no_grad
    def minimize(self,
                 loss,
                 startup_program=None,
                 parameters=None,
                 no_grad_set=None):

        # minimize does not support parameters in the form of param_group, 
        # so no need use _obtain_optimizer_parameters_list
        parameter_list = parameters if parameters \
            else self._inner_opt._parameter_list

        # Here sharding should use global parameter list
        if self._sharding_enable:
            sharding_reduce_gradients(list(parameter_list), self._hcg)

        if self._dp_enable:
            fused_allreduce_gradients(list(parameter_list), self._hcg)

        return self._inner_opt.minimize(loss, startup_program, parameter_list,
                                        no_grad_set)

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)
