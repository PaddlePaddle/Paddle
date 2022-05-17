# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from .optimizer import Optimizer
from ..fluid import core
from ..fluid import framework
from ..fluid.framework import Variable

__all__ = []


class ASGD(Optimizer):
    r"""
    """
    _avg_parameter_acc_str = "_avg_parameter"
    _current_step_acc_str = "_current_step"

    def __init__(self,
                 learning_rate,
                 t0=1e6,
                 parameters=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None):
        assert learning_rate is not None
        super(ASGD, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name)
        self.type = "asgd"
        self._t0 = t0
        self._default_dict = {
            't0': t0,
        }

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")
        if isinstance(parameters, dict):
            parameters = parameters.get('params')

        self._averaged_parameters = []
        for p in parameters:
            avg_param = self._add_accumulator(self._avg_parameter_acc_str, p)
            self._averaged_parameters.append(avg_param)
            self._add_accumulator(self._current_step_acc_str, p, shape=[1], device='cpu')

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        avg_parameter = self._get_accumulator(
            self._avg_parameter_acc_str, param_and_grad[0])

        current_step = self._get_accumulator(
            self._current_step_acc_str, param_and_grad[0])

        asgd_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad),
                "AvgParam": avg_parameter,
                "CurrentStep": current_step,
            },
            outputs={"ParamOut": param_and_grad[0],
                     "AvgParamOut": avg_parameter,
                     "CurrentStepOut": current_step,
                     },
            attrs={'t0': self._t0},
            stop_gradient=True)

        return asgd_op

    def _update_param_group(self, parameters):
        self._t0 = parameters.get('t0', self._default_dict['t0'])
        parameters = parameters.get('params')
        return parameters

    def averaged_parameters(self):
        return self._averaged_parameters

