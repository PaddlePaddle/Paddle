# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base.libpaddle import DataType

from ..base import core, framework
from ..base.framework import (
    in_dynamic_or_pir_mode,
    in_pir_mode,
)
from .optimizer import Optimizer

__all__ = []


class NAdam(Optimizer):
    """
    >>> import paddle

    >>> inp = paddle.rand([10,10], dtype="float32")
    >>> linear = paddle.nn.Linear(10, 10)
    >>> out = linear(inp)
    >>> loss = paddle.mean(out)

    >>> nadam = paddle.optimizer.NAdam(learning_rate=0.1,
    ...                     parameters=linear.parameters())
    >>> out.backward()
    >>> nadam.step()
    >>> nadam.clear_grad()


    """

    _momentum_decay_pow_acc_str = "momentum_decay_pow"
    _beta2_pow_acc_str = "beta2_pow"
    _mu_product_acc_str = "mu_product"
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-8,
        momentum_decay=0.004,
        momentum_decay_base=0.96,
        parameters=None,
        weight_decay=None,
        grad_clip=None,
        name=None,
    ):
        if isinstance(learning_rate, (float, int)) and not 0.0 <= learning_rate:
            raise ValueError(
                f"Invalid learning rate: {learning_rate}, expect learning_rate >= 0."
            )
        if not 0.0 <= epsilon:
            raise ValueError(
                f"Invalid epsilon value: {epsilon}, expect epsilon >= 0."
            )
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(
                f"Invalid beta1: {beta1}, expect 0. <= beta1 < 1.0."
            )
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(
                f"Invalid beta2: {beta2}, expect 0. <= beta2 < 1.0."
            )
        if not 0.0 <= momentum_decay:
            raise ValueError(
                f"Invalid momentum_decay value: {momentum_decay}, expect momentum_decay >= 0."
            )
        if not 0.0 <= momentum_decay_base:
            raise ValueError(
                f"Invalid momentum_decay_base value: {momentum_decay_base}, expect momentum_decay_base >= 0."
            )

        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )

        self.type = "nadam"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._momentum_decay_base = momentum_decay_base
        self._momentum_decay = momentum_decay
        self._multi_precision = False
        self._master_weights = {}
        self._default_dict = {
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            'momentum_decay': momentum_decay,
            'momentum_decay_base': momentum_decay_base,
        }

    def _add_moments_pows(self, p):
        acc_dtype = p.dtype
        if self._is_dtype_fp16_or_bf16(acc_dtype):
            acc_dtype = (
                DataType.FLOAT32 if in_pir_mode() else core.VarDesc.VarType.FP32
            )

        self._add_accumulator(
            name=self._momentum_decay_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=1.0,
            shape=[1],
            device='cpu',
        )
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=1.0,
            shape=[1],
            device='cpu',
        )
        self._add_accumulator(
            name=self._mu_product_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=1.0,
            shape=[1],
            device='cpu',
        )

        self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
        self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype)

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        if isinstance(parameters, dict):
            parameters = parameters.get('params')

        for p in parameters:
            if p.name in self._already_create_accumulator:
                continue

            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
                self._add_moments_pows(master_p)
                self._already_create_accumulator.add(p.name)
                continue
            if (
                self._is_dtype_fp16_or_bf16(p.dtype)
                and not self._multi_precision
            ):
                warnings.warn(
                    "Accumulating with FP16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Lars optimizer."
                )
            self._add_moments_pows(p)
            self._already_create_accumulator.add(p.name)

    def _append_optimize_op(self, block, param_and_grad):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        momentum_decay_pow_acc = self._get_accumulator_master(
            self._momentum_decay_pow_acc_str, param_and_grad[0]
        )
        beta2_pow_acc = self._get_accumulator_master(
            self._beta2_pow_acc_str, param_and_grad[0]
        )
        mu_product_acc = self._get_accumulator_master(
            self._mu_product_acc_str, param_and_grad[0]
        )
        moment1_acc = self._get_accumulator_master(
            self._moment1_acc_str, param_and_grad[0]
        )
        moment2_acc = self._get_accumulator_master(
            self._moment2_acc_str, param_and_grad[0]
        )
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            param_and_grad[0].dtype
        )
        master_weight = (
            self._master_weights[param_and_grad[0].name]
            if find_master
            else None
        )

        if in_dynamic_or_pir_mode():
            _C_ops.nadam_(
                param_and_grad[0],
                param_and_grad[1],
                self._create_param_lr(param_and_grad),
                momentum_decay_pow_acc,
                beta2_pow_acc,
                mu_product_acc,
                moment1_acc,
                moment2_acc,
                master_weight,
                self._beta1,
                self._beta2,
                self._epsilon,
                self._momentum_decay,
                self._momentum_decay_base,
                find_master,
            )
            return None
        else:
            inputs = {
                "param": param_and_grad[0],
                "grad": param_and_grad[1],
                "momentum_decay_pow": momentum_decay_pow_acc,
                "beta2_pow": beta2_pow_acc,
                "mu_product": mu_product_acc,
                "moment1": moment1_acc,
                "moment2": moment2_acc,
                "learning_rate": self._create_param_lr(param_and_grad),
            }

            outputs = {
                "param_out": param_and_grad[0],
                "momentum_decay_pow_out": momentum_decay_pow_acc,
                "beta2_pow_out": beta2_pow_acc,
                "mu_product_out": mu_product_acc,
                "moment1_out": moment1_acc,
                "moment2_out": moment2_acc,
            }

            if find_master:
                inputs["master_param"] = master_weight
                outputs["master_param_out"] = master_weight
            nadam_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs={
                    "epsilon": self._epsilon,
                    "beta1": self._beta1,
                    "beta2": self._beta2,
                    "momentum_decay": self._momentum_decay,
                    "momentum_decay_base": self._momentum_decay_base,
                },
                stop_gradient=True,
            )

            return nadam_op

    def _update_param_group(self, parameters):
        self._epsilon = parameters.get('epsilon', self._default_dict['epsilon'])
        self._beta1 = parameters.get('beta1', self._default_dict['beta1'])
        self._beta2 = parameters.get('beta2', self._default_dict['beta2'])
        self._momentum_decay = parameters.get(
            'momentum_decay', self._default_dict['momentum_decay']
        )
        self._momentum_decay_base = parameters.get(
            'momentum_decay_base', self._default_dict['momentum_decay_base']
        )
        parameters = parameters.get('params')
        return parameters
