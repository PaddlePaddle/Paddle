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

import warnings

from paddle import zeros_like

from ..base import framework
from ..base.dygraph import no_grad
from ..base.framework import in_dynamic_or_pir_mode
from .optimizer import Optimizer

__all__ = []


class Rprop(Optimizer):
    r"""
    TODO
    """

    def __init__(
        self,
        initial_lr=0.001,
        lower_lr=1e-6,
        upper_lr=50,
        parameters=None,
        weight_decay=None,
        eta_divide=0.5,
        eta_multiply=1.2,
        grad_clip=None,
        multi_precision=False,
        name=None,
    ):
        if initial_lr is None:
            raise ValueError("learning_rate is not set")
        super().__init__(
            learning_rate=initial_lr,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )
        self.type = "rprop"
        self._multi_precision = multi_precision
        self._master_weights = {}
        self._prevs = []
        self._lrs = []
        self._lower_lr = lower_lr
        self._upper_lr = upper_lr
        self._eta_divide = eta_divide
        self._eta_multiply = eta_multiply
        for p in parameters:
            prev = zeros_like(p)
            self._prevs.append(prev)
            lr = p.new().resize_as_(p).fill_(initial_lr)
            self._lrs.append(lr)

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            if p.name in self._already_create_accumulater:
                continue
            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
                self._already_create_accumulater.add(p.name)
                continue
            if (
                self._is_dtype_fp16_or_bf16(p.dtype)
                and not self._multi_precision
            ):
                warnings.warn(
                    "Accumulating with FP16/BF16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Adam optimizer."
                )

    @no_grad
    def _append_optimize_op(self, block, param_and_grad):
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        params = param_and_grad[0]
        grads = param_and_grad[1]
        prevs = self._prevs
        lrs = self._lrs
        lower_lr = self._lower_lr
        upper_lr = self._upper_lr
        eta_divide = self._eta_divide
        eta_multiply = self._eta_multiply
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            params.dtype
        )
        master_weight = (
            self._master_weights[params.name] if find_master else None
        )

        if in_dynamic_or_pir_mode():
            for i, param in enumerate(params):
                grad = grads[i]
                prev = prevs[i]
                lr = lrs[i]
                sign = grad.mul(prev).sign()
                sign[sign.gt(0)] = eta_multiply
                sign[sign.lt(0)] = eta_divide
                sign[sign.eq(0)] = 1

                lr.mul_(sign).clamp_(lower_lr, upper_lr)

                grad = grad.clone()
                grad[sign.eq(eta_divide)] = 0

                param.addcmul_(grad.sign(), lr, value=-1)
                prev.copy_(grad)

            return None
        else:
            assert isinstance(block, framework.Block)
            # create the optimize op
            inputs = {
                "Param": params,
                "Grad": grads,
                "LearningRate": lrs,
            }

            outputs = {"ParamOut": params}

            attrs = {"multi_precision": find_master}

            if find_master:
                inputs["MasterParam"] = master_weight
                outputs["MasterParamOut"] = master_weight

            rprop_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
                stop_gradient=True,
            )

            return rprop_op

    def _update_param_group(self, parameters):
        parameters = parameters.get('params')
        return parameters
