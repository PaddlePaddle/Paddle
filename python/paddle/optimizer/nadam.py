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
    r"""
    The NAdam optimizer is implemented based on the Adam Optimization
    in paper `Incorporating Nesterov Momentum into Adam <https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ>`_.
    The main improvement is to combine the advantages of Nesterov momentum and Adam adaptive learning rate.

    .. math::

       \begin{aligned}
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm} \mu_t \leftarrow \beta_1 \big(1 - \frac{1}{2}  \rho ^{t \psi} \big)     \\
            &\hspace{5mm} \mu_{t+1} \leftarrow \beta_1 \big(1 - \frac{1}{2} 0.96 ^{(t+1)\psi}\big)\\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow \mu_{t+1} m_t/(1-\prod_{i=1}^{t+1}\mu_i) + (1-\mu_t) g_t /(1-\prod_{i=1}^{t} \mu_{i})                         \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\hspace{0mm} \text{ with: } \gamma_t \text{ (lr)}, \: \beta_1,\beta_2 \text{ (betas)}, \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (objective)} \\
            &\hspace{0mm} \: \lambda \text{ (weight decay)}, \:\psi \text{ (momentum decay)} \\
       \end{aligned}

    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a LRScheduler. The default value is 0.002.
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` names to update to minimize ``loss``.
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
        weight_decay (float|Tensor, optional): The weight decay coefficient, it can be float or Tensor.
            Default None, meaning there is no regularization.
        momentum_decay (float, optional): momentum momentum_decay. The default value is 0.004.
        grad_clip (GradientClipBase, optional): Gradient clipping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three clipping strategies
            ( :ref:`api_paddle_nn_ClipGradByGlobalNorm` , :ref:`api_paddle_nn_ClipGradByNorm` ,
            :ref:`api_paddle_nn_ClipGradByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Notes:
        Currently, NAdam doesn't support sparse parameter optimization.

    Examples:
        .. code-block:: python

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

            >>> # Note that the learning_rate of linear_2 is 0.01.
            >>> linear_1 = paddle.nn.Linear(10, 10)
            >>> linear_2 = paddle.nn.Linear(10, 10)
            >>> inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            >>> out = linear_1(inp)
            >>> out = linear_2(out)
            >>> loss = paddle.mean(out)
            >>> opt = paddle.optimizer.NAdam(
            ...     learning_rate=0.1,
            ...     parameters=[{
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

    _momentum_decay_pow_acc_str = "momentum_decay_pow"
    _beta2_pow_acc_str = "beta2_pow"
    _mu_product_acc_str = "mu_product"
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"

    def __init__(
        self,
        learning_rate=0.002,
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-8,
        momentum_decay=0.004,
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
        self._momentum_decay = momentum_decay
        self._multi_precision = False
        self._master_weights = {}
        self._default_dict = {
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            'momentum_decay': momentum_decay,
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
        )
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=1.0,
        )
        self._add_accumulator(
            name=self._mu_product_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=1.0,
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
        parameters = parameters.get('params')
        return parameters
