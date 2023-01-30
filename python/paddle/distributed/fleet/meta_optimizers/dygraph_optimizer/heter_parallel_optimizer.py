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

<<<<<<< HEAD
import paddle.autograd as imperative_base
from paddle import framework
=======
import paddle
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid import framework
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

__all__ = []


def _obtain_optimizer_parameters_list(optimizer):
    if getattr(optimizer, '_param_groups', None) and isinstance(
<<<<<<< HEAD
        optimizer._param_groups[0], dict
    ):
=======
            optimizer._param_groups[0], dict):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        parameters_list = []
        for group in optimizer._param_groups:
            for param in group['params']:
                parameters_list.append(param)
    else:
        parameters_list = [param for param in optimizer._parameter_list]

    return parameters_list


class HeterParallelOptimizer:
    # adapter wrapper for optimizer
    def __init__(self, optimizer, strategy):
        self._inner_opt = optimizer
        self._strategy = strategy

        # NOTE(liubo48): In pure DataParallel mode,
        # the gradient synchronization is achieved through reducer.

<<<<<<< HEAD
    @imperative_base.no_grad()
=======
    @imperative_base.no_grad
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @framework.dygraph_only
    def step(self):
        parameters_list = _obtain_optimizer_parameters_list(self._inner_opt)
        self._inner_opt.step()

<<<<<<< HEAD
    @imperative_base.no_grad()
    def minimize(
        self, loss, startup_program=None, parameters=None, no_grad_set=None
    ):

        # minimize does not support parameters in the form of param_group,
        # so no need use _obtain_optimizer_parameters_list
        parameter_list = (
            parameters if parameters else self._inner_opt._parameter_list
        )

        return self._inner_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set
        )
=======
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

        return self._inner_opt.minimize(loss, startup_program, parameter_list,
                                        no_grad_set)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)
