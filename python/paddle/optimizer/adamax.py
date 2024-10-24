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
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from paddle import _C_ops, pir

from ..base import core, framework
from ..base.dygraph import no_grad
from ..base.framework import name_scope
from .optimizer import Optimizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import NotRequired

    from paddle import Tensor
    from paddle.nn.clip import GradientClipBase
    from paddle.regularizer import WeightDecayRegularizer

    from .lr import LRScheduler
    from .optimizer import _ParameterConfig

    class _AdamaxParameterConfig(_ParameterConfig):
        beta1: NotRequired[float | Tensor]
        beta2: NotRequired[float | Tensor]
        epsilon: NotRequired[float | Tensor]


__all__ = []


class Adamax(Optimizer):
    r"""
    The Adamax optimizer is implemented based on the Adamax Optimization
    in Section 7 of `Adam paper <https://arxiv.org/abs/1412.6980>`_.
    The Adamax algorithm is a variant of the Adam algorithm based on the infinite norm,
    which makes the learning rate update algorithm more stable and simple.

    The parameter ``param_out`` update rule with gradient ``grad``:

    .. math::

        t & = t + 1

        moment\_out & = {\beta}_1 * moment + (1 - {\beta}_1) * grad

        inf\_norm\_out & = max({\beta}_2 * inf\_norm + \epsilon, |grad|)

        learning\_rate & = \frac{learning\_rate}{1 - {\beta}_1^t}

        param\_out & = param - learning\_rate * \frac{moment\_out}{inf\_norm\_out}

    Related paper: `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_

    The original paper does not have an ``epsilon`` attribute,
    it is added here for numerical stability to prevent the division by 0 error.

    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a LRScheduler. The default value is 0.001.
        beta1 (float|Tensor, optional): The exponential decay rate for the 1st moment estimates.
            It should be a float number or a 0-D Tensor with shape [] and data type as float32.
            The default value is 0.9.
        beta2 (float|Tensor, optional): The exponential decay rate for the 2nd moment estimates.
            It should be a float number or a 0-D Tensor with shape [] and data type as float32.
            The default value is 0.999.
        epsilon (float|Tensor, optional): A small float value for numerical stability.
            The default value is 1e-08.
        parameters (list|tuple|None, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``.
            This parameter is required in dygraph mode. And you can specify different options for
            different parameter groups such as the learning rate, weight decay, etc,
            then the parameters are list of dict. Note that the learning_rate in parameter groups
            represents the scale of base learning_rate.
            The default value is None in static graph mode, at this time all parameters will be updated.
        weight_decay (int|float|WeightDecayRegularizer|None, optional): The strategy of regularization.
            It can be a int or float value as coeff of L2 regularization or
            :ref:`api_paddle_regularizer_L1Decay`, :ref:`api_paddle_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_paddle_ParamAttr` already,
            the regularization setting here in optimizer will be ignored for this parameter.
            Otherwise, the regularization setting here in optimizer will take effect.
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase|None, optional): Gradient clipping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three clipping strategies
            ( :ref:`api_paddle_nn_ClipGradByGlobalNorm` , :ref:`api_paddle_nn_ClipGradByNorm` ,
            :ref:`api_paddle_nn_ClipGradByValue` ). Default None, meaning there is no gradient clipping.
        name (str|None, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    **Notes**:
        **Currently, Adamax doesn't support sparse parameter optimization.**

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> inp = paddle.uniform([10, 10], dtype="float32", min=-0.1, max=0.1)
            >>> linear = paddle.nn.Linear(10, 10)
            >>> inp = paddle.to_tensor(inp)
            >>> out = linear(inp)
            >>> loss = paddle.mean(out)

            >>> beta1 = paddle.to_tensor([0.9], dtype="float32")
            >>> beta2 = paddle.to_tensor([0.99], dtype="float32")

            >>> adamax = paddle.optimizer.Adamax(
            ...     learning_rate=0.1,
            ...     parameters=linear.parameters(),
            ...     beta1=beta1,
            ...     beta2=beta2,
            ...     weight_decay=0.01
            ... )
            >>> out.backward()
            >>> adamax.step()
            >>> adamax.clear_grad()


            >>> # Note that the learning_rate of linear_2 is 0.01.
            >>> linear_1 = paddle.nn.Linear(10, 10)
            >>> linear_2 = paddle.nn.Linear(10, 10)
            >>> inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            >>> out = linear_1(inp)
            >>> out = linear_2(out)
            >>> loss = paddle.mean(out)
            >>> adamax = paddle.optimizer.Adamax(
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
            >>> out.backward()
            >>> adamax.step()
            >>> adamax.clear_grad()
    """

    type: str
    _moment_acc_str = "moment"
    _inf_norm_acc_str = "inf_norm"
    _beta1_pow_acc_str = "beta1_pow_acc"

    def __init__(
        self,
        learning_rate: float | LRScheduler = 0.001,
        beta1: float | Tensor = 0.9,
        beta2: float | Tensor = 0.999,
        epsilon: float | Tensor = 1e-8,
        parameters: (
            Sequence[Tensor] | Sequence[_AdamaxParameterConfig] | None
        ) = None,
        weight_decay: float | WeightDecayRegularizer | None = None,
        grad_clip: GradientClipBase | None = None,
        name: str | None = None,
    ) -> None:
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        if not 0 <= beta1 < 1:
            raise ValueError("Invalid value of beta1, expect beta1 in [0,1).")
        if not 0 <= beta2 < 1:
            raise ValueError("Invalid value of beta2, expect beta2 in [0,1).")
        if not 0 <= epsilon:
            raise ValueError("Invalid value of epsilon, expect epsilon >= 0.")
        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )
        self.type = "adamax"
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
            acc_dtype = core.VarDesc.VarType.FP32

        self._add_accumulator(self._moment_acc_str, p, dtype=acc_dtype)
        self._add_accumulator(self._inf_norm_acc_str, p, dtype=acc_dtype)
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            fill_value=self._beta1,
            shape=[1],
        )

    def _create_accumulators(self, block, parameters):
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        # Create accumulator tensors for first moment and infinity norm
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
                    "Accumulating with FP16/BF16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Adam optimizer."
                )
            self._add_moments_pows(p)
            self._already_create_accumulator.add(p.name)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        moment = self._get_accumulator_master(
            self._moment_acc_str, param_and_grad[0]
        )
        inf_norm = self._get_accumulator_master(
            self._inf_norm_acc_str, param_and_grad[0]
        )

        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            param_and_grad[0].dtype
        )
        master_weight = (
            self._master_weights[param_and_grad[0].name]
            if find_master
            else None
        )

        beta1_pow_acc = self._get_accumulator_master(
            self._beta1_pow_acc_str, param_and_grad[0]
        )
        if framework.in_dynamic_or_pir_mode():
            _C_ops.adamax_(
                param_and_grad[0],
                param_and_grad[1],
                self._create_param_lr(param_and_grad),
                moment,
                inf_norm,
                beta1_pow_acc,
                master_weight,
                self._beta1,
                self._beta2,
                self._epsilon,
                find_master,
            )

        else:
            # create the adamax optimize op
            inputs = {
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad),
                "Moment": moment,
                "InfNorm": inf_norm,
                "Beta1Pow": beta1_pow_acc,
            }
            outputs = {
                "ParamOut": param_and_grad[0],
                "MomentOut": moment,
                "InfNormOut": inf_norm,
            }
            if find_master:
                inputs["MasterParam"] = master_weight
                outputs["MasterParamOut"] = master_weight
            attrs = {
                "beta1": self._beta1,
                "beta2": self._beta2,
                "epsilon": self._epsilon,
                "multi_precision": find_master,
            }
            adamax_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
                stop_gradient=True,
            )

            return adamax_op

    def _finish_update(self, block, parameters_and_grads):
        """Update Beta1 Power accumulator"""
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(parameters_and_grads, list):
            for param, grad in parameters_and_grads:
                if grad is None or param.stop_gradient is True:
                    continue
                if framework.in_dygraph_mode():
                    beta1_pow_acc = self._get_accumulator_master(
                        self._beta1_pow_acc_str, param
                    )
                    with no_grad():
                        tmp = _C_ops.scale(
                            beta1_pow_acc, self._beta1, 0.0, True
                        )
                        beta1_pow_acc.copy_(tmp, False)
                elif framework.in_pir_mode():
                    with param.block.program._optimized_guard([param, grad]):
                        beta1_pow_acc = self._get_accumulator_master(
                            self._beta1_pow_acc_str, param
                        )
                        _C_ops.scale_(beta1_pow_acc, self._beta1, 0.0, True)
                else:
                    with param.block.program._optimized_guard(
                        [param, grad]
                    ), name_scope('adamax'):
                        beta1_pow_acc = self._get_accumulator_master(
                            self._beta1_pow_acc_str, param
                        )
                        block.append_op(
                            type="scale",
                            inputs={"X": beta1_pow_acc},
                            outputs={"Out": beta1_pow_acc},
                            attrs={"scale": self._beta1},
                            stop_gradient=True,
                        )
        else:
            for param, grad in parameters_and_grads['params']:
                if grad is None or param.stop_gradient is True:
                    continue
                if framework.in_dygraph_mode():
                    beta1_pow_acc = self._get_accumulator_master(
                        self._beta1_pow_acc_str, param
                    )
                    self._beta1 = parameters_and_grads.get(
                        'beta1', self._default_dict['beta1']
                    )
                    with no_grad():
                        tmp = _C_ops.scale(
                            beta1_pow_acc, self._beta1, 0.0, True
                        )
                        beta1_pow_acc.copy_(tmp, False)
                else:
                    with param.block.program._optimized_guard(
                        [param, grad]
                    ), name_scope('adamax'):
                        beta1_pow_acc = self._get_accumulator_master(
                            self._beta1_pow_acc_str, param
                        )
                        self._beta1 = parameters_and_grads.get(
                            'beta1', self._default_dict['beta1']
                        )
                        block.append_op(
                            type="scale",
                            inputs={"X": beta1_pow_acc},
                            outputs={"Out": beta1_pow_acc},
                            attrs={"scale": self._beta1},
                            stop_gradient=True,
                        )

    def _update_param_group(self, parameters):
        self._beta1 = parameters.get('beta1', self._default_dict['beta1'])
        self._beta2 = parameters.get('beta2', self._default_dict['beta2'])
        self._epsilon = parameters.get('epsilon', self._default_dict['epsilon'])
        parameters = parameters.get('params')
        return parameters
