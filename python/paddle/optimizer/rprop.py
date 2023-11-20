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

from paddle import _C_ops
from paddle.tensor.creation import zeros_like

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
        learning_rate=0.01,
        learning_rate_range=(1e-5, 50),
        parameters=None,
        etas=(0.5, 1.2),
        grad_clip=None,
        multi_precision=False,
        name=None,
    ):
        if learning_rate is None:
            raise ValueError("learning_rate is not set")
        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=0.0,
            grad_clip=grad_clip,
            name=name,
        )
        self.type = "rprop"
        self._multi_precision = multi_precision
        self._master_weights = {}
        self._prevs = []
        self._learning_rates = []
        self._learning_rate_range = learning_rate_range
        self._etas = etas
        for p in parameters:
            prev = zeros_like(p)
            self._prevs.append(prev)
            lr = p.new().resize_as_(p).fill_(learning_rate)
            self._learning_rates.append(lr)

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

        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            param_and_grad[0].dtype
        )
        master_weight = (
            self._master_weights[param_and_grad[0].name]
            if find_master
            else None
        )

        if in_dynamic_or_pir_mode():
            _C_ops.rprop_(
                param_and_grad[0],
                param_and_grad[1],
                self._prevs,
                self._learning_rates,
                master_weight,
                self._learning_rate_range,
                self._etas,
                find_master,
            )

            return None
        else:
            assert isinstance(block, framework.Block)
            # create the optimize op

            inputs = {
                "param": param_and_grad[0],
                "grad": param_and_grad[1],
                "prev": self._prevs,
                "learning_rate": self._learning_rates,
                "learning_rate_range": self._learning_rate_range,
                "etas": self._etas,
            }

            outputs = {
                "param_out": param_and_grad[0],
                "prev_out": self._prevs,
                "learning_rate_out": self._learning_rates,
            }

            attrs = {"multi_precision": find_master}

            if find_master:
                inputs["master_param"] = master_weight
                outputs["master_param_out"] = master_weight

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
