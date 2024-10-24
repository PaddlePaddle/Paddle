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

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from paddle import _C_ops, pir
from paddle.base.libpaddle import DataType

from ..base import core, framework
from ..base.framework import (
    in_dynamic_or_pir_mode,
    in_pir_mode,
)
from .optimizer import Optimizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import NotRequired

    from paddle import Tensor
    from paddle.nn.clip import GradientClipBase
    from paddle.optimizer.lr import LRScheduler
    from paddle.regularizer import WeightDecayRegularizer

    from .optimizer import _ParameterConfig

    class _RAdamParameterConfig(_ParameterConfig):
        beta1: NotRequired[float | Tensor]
        beta2: NotRequired[float | Tensor]
        epsilon: NotRequired[float]


__all__ = []


class RAdam(Optimizer):
    r"""
    The RAdam optimizer is implemented based on the Adam Optimization
    in paper `On the Variance of the Adaptive Learning Rate and Beyond <https://arxiv.org/abs/1908.03265>`_.
    RAdam improved the initial stability of training by modifying Adam's momentum term.

    .. math::

        \begin{aligned}
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{6mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{6mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{6mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{6mm}\rho_t \leftarrow \rho_{\infty} -
                2 t \beta^t_2 /\big(1-\beta_2^t \big)                                    \\
            &\hspace{6mm}\textbf{if} \: \rho_t > 5                                               \\
            &\hspace{12mm} l_t \leftarrow \frac{\sqrt{ (1-\beta^t_2) }}{ \sqrt{v_t} +\epsilon  } \\
            &\hspace{12mm} r_t \leftarrow
        \sqrt{\frac{(\rho_t-4)(\rho_t-2)\rho_{\infty}}{(\rho_{\infty}-4)(\rho_{\infty}-2) \rho_t}} \\
            &\hspace{12mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t} r_t l_t        \\
            &\hspace{6mm}\textbf{else}                                                           \\
            &\hspace{12mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}                \\
            &\hspace{0mm} \text{ with: } \gamma_t \text{ (lr)}, \: \beta_1,\beta_2 \text{ (betas)}, \: \theta_t \text{ (params)} \\
            &\hspace{0mm} \rho_{\infty} \leftarrow 2/(1-\beta_2) -1
        \end{aligned}

    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a LRScheduler. The default value is 0.001.
        parameters (list|tuple|None, optional): List/Tuple of ``Tensor`` names to update to minimize ``loss``.
            This parameter is required in dygraph mode. And you can specify different options for
            different parameter groups such as the learning rate, weight decay, etc,
            then the parameters are list of dict. Note that the learning_rate in parameter groups
            represents the scale of base learning_rate.
            The default value is None in static graph mode, at this time all parameters will be updated.
        beta1 (float|Tensor, optional): The exponential decay rate for the 1st moment estimates.
            It should be a float number or a 0-D Tensor with shape [] and data type as float32.
            The default value is 0.9.
        beta2 (float|Tensor, optional): The exponential decay rate for the 2nd moment estimates.
            It should be a float number or a 0-D Tensor with shape [] and data type as float32.
            The default value is 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            The default value is 1e-08.
        weight_decay (int|float|Tensor|WeightDecayRegularizer|None, optional): The weight decay coefficient, it can be int, float or Tensor.
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase|None, optional): Gradient clipping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three clipping strategies
            ( :ref:`api_paddle_nn_ClipGradByGlobalNorm` , :ref:`api_paddle_nn_ClipGradByNorm` ,
            :ref:`api_paddle_nn_ClipGradByValue` ). Default None, meaning there is no gradient clipping.
        name (str|None, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Note:
        Currently, RAdam doesn't support sparse parameter optimization.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> inp = paddle.rand([10,10], dtype="float32")
            >>> linear = paddle.nn.Linear(10, 10)
            >>> out = linear(inp)
            >>> loss = paddle.mean(out)

            >>> radam = paddle.optimizer.RAdam(
            ...     learning_rate=0.1,
            ...     parameters=linear.parameters()
            ... )
            >>> out.backward()
            >>> radam.step()
            >>> radam.clear_grad()

            >>> # Note that the learning_rate of linear_2 is 0.01.
            >>> linear_1 = paddle.nn.Linear(10, 10)
            >>> linear_2 = paddle.nn.Linear(10, 10)
            >>> inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            >>> out = linear_1(inp)
            >>> out = linear_2(out)
            >>> loss = paddle.mean(out)
            >>> opt = paddle.optimizer.RAdam(
            ...     learning_rate=0.1,
            ...     parameters=[{  # type: ignore
            ...         'params': linear_1.parameters()
            ...     }, {
            ...         'params': linear_2.parameters(),
            ...         'weight_decay': 0.001,
            ...         'learning_rate': 0.1,
            ...         'beta1': 0.8
            ...     }],
            ...     weight_decay=0.01,
            ...     beta1=0.9
            ... )
            >>> loss.backward()
            >>> opt.step()
            >>> opt.clear_grad()

    """

    _beta1_pow_acc_str = "beta1_pow"
    _beta2_pow_acc_str = "beta2_pow"
    _rho_acc_str = "rho"
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"

    def __init__(
        self,
        learning_rate: float | LRScheduler = 0.001,
        beta1: float | Tensor = 0.9,
        beta2: float | Tensor = 0.999,
        epsilon: float = 1.0e-8,
        parameters: (
            Sequence[Tensor] | Sequence[_RAdamParameterConfig] | None
        ) = None,
        weight_decay: float | Tensor | WeightDecayRegularizer | None = None,
        grad_clip: GradientClipBase | None = None,
        name: str | None = None,
    ) -> None:
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

        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )

        self.type = "radam"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._multi_precision = False
        self._master_weights = {}
        self._default_dict = {
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
        }

    def _add_moments_pows(self, p):
        acc_dtype = p.dtype
        if self._is_dtype_fp16_or_bf16(acc_dtype):
            acc_dtype = (
                DataType.FLOAT32 if in_pir_mode() else core.VarDesc.VarType.FP32
            )

        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=1.0,
        )
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=1.0,
        )
        self._add_accumulator(
            name=self._rho_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=1.0,
        )

        self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
        self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype)

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, (framework.Block, pir.Block)):
            raise TypeError("block is not instance of Block.")

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
        if not isinstance(block, (framework.Block, pir.Block)):
            raise TypeError("block is not instance of Block.")

        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        beta1_pow_acc = self._get_accumulator_master(
            self._beta1_pow_acc_str, param_and_grad[0]
        )
        beta2_pow_acc = self._get_accumulator_master(
            self._beta2_pow_acc_str, param_and_grad[0]
        )
        rho_acc = self._get_accumulator_master(
            self._rho_acc_str, param_and_grad[0]
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
            _C_ops.radam_(
                param_and_grad[0],
                param_and_grad[1],
                self._create_param_lr(param_and_grad),
                beta1_pow_acc,
                beta2_pow_acc,
                rho_acc,
                moment1_acc,
                moment2_acc,
                master_weight,
                self._beta1,
                self._beta2,
                self._epsilon,
                find_master,
            )
            return None
        else:
            inputs = {
                "param": param_and_grad[0],
                "grad": param_and_grad[1],
                "beta1_pow": beta1_pow_acc,
                "beta2_pow": beta2_pow_acc,
                "rho": rho_acc,
                "moment1": moment1_acc,
                "moment2": moment2_acc,
                "learning_rate": self._create_param_lr(param_and_grad),
            }

            outputs = {
                "param_out": param_and_grad[0],
                "beta1_pow_out": beta1_pow_acc,
                "beta2_pow_out": beta2_pow_acc,
                "rho_out": rho_acc,
                "moment1_out": moment1_acc,
                "moment2_out": moment2_acc,
            }

            if find_master:
                inputs["master_param"] = master_weight
                outputs["master_param_out"] = master_weight
            radam_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs={
                    "epsilon": self._epsilon,
                    "beta1": self._beta1,
                    "beta2": self._beta2,
                },
                stop_gradient=True,
            )

            return radam_op

    def _update_param_group(self, parameters):
        self._epsilon = parameters.get('epsilon', self._default_dict['epsilon'])
        self._beta1 = parameters.get('beta1', self._default_dict['beta1'])
        self._beta2 = parameters.get('beta2', self._default_dict['beta2'])
        parameters = parameters.get('params')
        return parameters
