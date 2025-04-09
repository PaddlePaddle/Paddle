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
from collections import defaultdict
from typing import TYPE_CHECKING

import paddle
from paddle import _C_ops, pir
from paddle.base.libpaddle import DataType
from paddle.pir import Value

from ..base import core, framework
from ..base.dygraph import base as imperative_base
from ..base.framework import (
    Variable,
    in_dygraph_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)
from .optimizer import Optimizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import NotRequired

    from paddle import Tensor
    from paddle.nn.clip import GradientClipBase
    from paddle.regularizer import WeightDecayRegularizer

    from .lr import LRScheduler
    from .optimizer import _ParameterConfig

    class _AdamParameterConfig(_ParameterConfig):
        beta1: NotRequired[float | Tensor]
        beta2: NotRequired[float | Tensor]
        epsilon: NotRequired[float | Tensor]
        lazy_mode: NotRequired[bool]


__all__ = []


class Adam(Optimizer):
    r"""
    The Adam optimizer uses an optimization described at the end
    of section 2 of `Adam paper <https://arxiv.org/abs/1412.6980>`_ ,
    it can dynamically adjusts the learning rate of each parameter using
    the 1st moment estimates and the 2nd moment estimates of the gradient.

    The parameter ``param_out`` update rule with gradient ``grad``:

    .. math::

        \begin{aligned}
            &\hspace{5mm} t = t + 1 \\
            &\hspace{5mm} moment\_1\_out = {\beta}_1 * moment\_1 + (1 - {\beta}_1) * grad \\
            &\hspace{5mm} moment\_2\_out = {\beta}_2 * moment\_2 + (1 - {\beta}_2) * grad * grad \\
            &\hspace{5mm} learning\_rate = learning\_rate * \frac{\sqrt{1 - {\beta}_2^t}}{1 - {\beta}_1^t} \\
            &\hspace{5mm}\textbf{if} \: \textit{amsgrad}: \\
            &\hspace{15mm} moment\_2\_max\_out = max(moment\_2\_out, moment\_2\_max) \\
            &\hspace{15mm} param\_out = param - learning\_rate * \frac{moment\_1\_out}{\sqrt{moment\_2\_max\_out} + \epsilon} \\
            &\hspace{5mm}\textbf{else}: \: \\
            &\hspace{15mm} param\_out = param - learning\_rate * \frac{moment\_1\_out}{\sqrt{moment\_2\_out} + \epsilon} \\
        \end{aligned}

    Related paper: `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_

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
            It should be a float number or a 0-D Tensor with shape [] and data type as float32.
            The default value is 1e-08.
        parameters (list|tuple|None, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``.
            This parameter is required in dygraph mode. And you can specify different options for
            different parameter groups such as the learning rate, weight decay, etc,
            then the parameters are list of dict. Note that the learning_rate in parameter groups
            represents the scale of base learning_rate.
            The default value is None in static graph mode, at this time all parameters will be updated.
        weight_decay (int|float|WeightDecayRegularizer|None, optional): The strategy of regularization.
            It canbe a int or float value as coeff of L2 regularization or
            :ref:`api_paddle_regularizer_L1Decay`, :ref:`api_paddle_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_paddle_ParamAttr` already,
            the regularization setting here in optimizer will be ignored for this parameter.
            Otherwise, the regularization setting here in optimizer will take effect.
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase|None, optional): Gradient clipping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three clipping strategies
            ( :ref:`api_paddle_nn_ClipGradByGlobalNorm` , :ref:`api_paddle_nn_ClipGradByNorm` ,
            :ref:`api_paddle_nn_ClipGradByValue` ). Default None, meaning there is no gradient clipping.
        lazy_mode (bool, optional): The official Adam algorithm has two moving-average accumulators.
            The accumulators are updated at every step. Every element of the two moving-average
            is updated in both dense mode and sparse mode. If the size of parameter is very large,
            then the update may be very slow. The lazy mode only update the element that has
            gradient in current mini-batch, so it will be much more faster. But this mode has
            different semantics with the original Adam algorithm and may lead to different result.
            The default value is False.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating. Default is false.
        use_multi_tensor (bool, optional): Whether to use multi-tensor strategy to update all parameters at once . Default is false.
        amsgrad (bool, optional): Whether to use the AMSGrad variant of this algorithm from the paper
            `On the Convergence of Adam and Beyond <https://openreview.net/forum?id=ryQu7f-RZ>`_. Default is false.
        name (str|None, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Examples:
        .. code-block:: python
            :name: code-example1

            >>> import paddle

            >>> linear = paddle.nn.Linear(10, 10)
            >>> inp = paddle.rand([10,10], dtype="float32")
            >>> out = linear(inp)
            >>> loss = paddle.mean(out)
            >>> adam = paddle.optimizer.Adam(
            ...     learning_rate=0.1,
            ...     parameters=linear.parameters()
            ... )
            >>> loss.backward()
            >>> adam.step()
            >>> adam.clear_grad()

        .. code-block:: python
            :name: code-example2

            >>> # Adam with beta1/beta2 as Tensor and weight_decay as float
            >>> import paddle

            >>> linear = paddle.nn.Linear(10, 10)
            >>> inp = paddle.rand([10,10], dtype="float32")
            >>> out = linear(inp)
            >>> loss = paddle.mean(out)
            >>> beta1 = paddle.to_tensor([0.9], dtype="float32")
            >>> beta2 = paddle.to_tensor([0.99], dtype="float32")
            >>> adam = paddle.optimizer.Adam(
            ...     learning_rate=0.1,
            ...     parameters=linear.parameters(),
            ...     beta1=beta1,
            ...     beta2=beta2,
            ...     weight_decay=0.01
            ... )
            >>> loss.backward()
            >>> adam.step()
            >>> adam.clear_grad()

            >>> # Note that the learning_rate of linear_2 is 0.01.
            >>> linear_1 = paddle.nn.Linear(10, 10)
            >>> linear_2 = paddle.nn.Linear(10, 10)
            >>> inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            >>> out = linear_1(inp)
            >>> out = linear_2(out)
            >>> loss = paddle.mean(out)
            >>> adam = paddle.optimizer.Adam(
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
            >>> adam.step()
            >>> adam.clear_grad()

    """

    type: str
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _moment2_acc_max_str = "moment2_max"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"

    def __init__(
        self,
        learning_rate: float | LRScheduler = 0.001,
        beta1: float | Tensor = 0.9,
        beta2: float | Tensor = 0.999,
        epsilon: float | Tensor = 1e-8,
        parameters: (
            Sequence[Tensor] | Sequence[_AdamParameterConfig] | None
        ) = None,
        weight_decay: float | WeightDecayRegularizer | None = None,
        grad_clip: GradientClipBase | None = None,
        lazy_mode: bool = False,
        multi_precision: bool = False,
        use_multi_tensor: bool = False,
        amsgrad: bool = False,
        name: str | None = None,
    ) -> None:
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        if not isinstance(beta1, (Variable, Value)):
            if not 0 <= beta1 < 1:
                raise ValueError(
                    "Invalid value of beta1, expect beta1 in [0,1)."
                )
        if not isinstance(beta2, (Variable, Value)):
            if not 0 <= beta2 < 1:
                raise ValueError(
                    "Invalid value of beta2, expect beta2 in [0,1)."
                )
        if not isinstance(epsilon, (Variable, Value)):
            if not 0 <= epsilon:
                raise ValueError(
                    "Invalid value of epsilon, expect epsilon >= 0."
                )
        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )
        self.type = "adam"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._lazy_mode = lazy_mode
        self._multi_precision = multi_precision
        self._master_weights = {}
        self._default_dict = {
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            'lazy_mode': lazy_mode,
        }

        self._use_multi_tensor = use_multi_tensor
        if self._use_multi_tensor:
            self._param_dict = self._create_multi_tensor_dict()
            self._moment1_dict = self._create_multi_tensor_dict()
            self._moment2_dict = self._create_multi_tensor_dict()
            self._moment2_max_dict = (
                self._create_multi_tensor_dict() if amsgrad else None
            )
            self._beta1_pow_acc_dict = self._create_multi_tensor_dict()
            self._beta2_pow_acc_dict = self._create_multi_tensor_dict()
            self._master_weight_dict = self._create_multi_tensor_dict()
            self._master_weight_dict['FP32_LODTensor'] = None

        # whether to use AMSGrad
        self._amsgrad = amsgrad

    def _add_moments_pows(self, p):
        acc_dtype = p.dtype
        if self._is_dtype_fp16_or_bf16(acc_dtype):
            if in_pir_mode():
                acc_dtype = DataType.FLOAT32
            else:
                acc_dtype = core.VarDesc.VarType.FP32
        self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
        self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype)
        if self._amsgrad:
            self._add_accumulator(self._moment2_acc_max_str, p, dtype=acc_dtype)
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=(
                0.9
                if isinstance(self._beta1, (Variable, Value))
                else self._beta1
            ),
            shape=[1],
            type=core.VarDesc.VarType.DENSE_TENSOR,
            device='cpu',
        )
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=(
                0.999
                if isinstance(self._beta2, (Variable, Value))
                else self._beta2
            ),
            shape=[1],
            type=core.VarDesc.VarType.DENSE_TENSOR,
            device='cpu',
        )

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, (framework.Block, paddle.pir.Block))
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        # Create accumulator tensors for first and second moments
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
                    "Accumulating with FP16 or BF16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Adam optimizer."
                )
            self._add_moments_pows(p)
            self._already_create_accumulator.add(p.name)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, (framework.Block, paddle.pir.Block))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        moment1 = self._get_accumulator_master(
            self._moment1_acc_str, param_and_grad[0]
        )
        moment2 = self._get_accumulator_master(
            self._moment2_acc_str, param_and_grad[0]
        )
        moment2_max = (
            self._get_accumulator_master(
                self._moment2_acc_max_str, param_and_grad[0]
            )
            if self._amsgrad
            else None
        )
        beta1_pow_acc = self._get_accumulator_master(
            self._beta1_pow_acc_str, param_and_grad[0]
        )
        beta2_pow_acc = self._get_accumulator_master(
            self._beta2_pow_acc_str, param_and_grad[0]
        )
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            param_and_grad[0].dtype
        )
        master_weight = (
            self._master_weights[param_and_grad[0].name]
            if find_master
            else None
        )
        lr = self._create_param_lr(param_and_grad)
        # create the adam optimize op

        if in_dynamic_or_pir_mode():
            _beta1 = (
                self._beta1
                if not isinstance(self._beta1, Variable)
                else self._beta1.item(0)
            )
            _beta2 = (
                self._beta2
                if not isinstance(self._beta2, Variable)
                else self._beta2.item(0)
            )
            found_inf = (
                self._get_auxiliary_var('found_inf') if in_pir_mode() else None
            )

            _, _, _, _, _, _, _ = _C_ops.adam_(
                param_and_grad[0],
                param_and_grad[1],
                lr,
                moment1,
                moment2,
                moment2_max,
                beta1_pow_acc,
                beta2_pow_acc,
                master_weight,
                found_inf,
                _beta1,
                _beta2,
                self._epsilon,
                self._lazy_mode,
                1000,
                find_master,
                False,
                self._amsgrad,
            )

            return None
        else:
            inputs = {
                "Param": [param_and_grad[0]],
                "Grad": [param_and_grad[1]],
                "LearningRate": [lr],
                "Moment1": [moment1],
                "Moment2": [moment2],
                "Beta1Pow": [beta1_pow_acc],
                "Beta2Pow": [beta2_pow_acc],
            }

            # Pass found_inf to adam, to skip update for not only param, but also momentum and beta_pow
            found_inf = self._get_auxiliary_var('found_inf')

            if found_inf:
                inputs['SkipUpdate'] = found_inf

            outputs = {
                "ParamOut": [param_and_grad[0]],
                "Moment1Out": [moment1],
                "Moment2Out": [moment2],
                "Beta1PowOut": [beta1_pow_acc],
                "Beta2PowOut": [beta2_pow_acc],
            }
            attrs = {
                "lazy_mode": self._lazy_mode,
                "min_row_size_to_use_multithread": 1000,
                "multi_precision": find_master,
                "amsgrad": self._amsgrad,
            }

            if isinstance(self._beta1, Variable):
                inputs['Beta1Tensor'] = self._beta1
            else:
                attrs['beta1'] = self._beta1
            if isinstance(self._beta2, Variable):
                inputs['Beta2Tensor'] = self._beta2
            else:
                attrs['beta2'] = self._beta2
            if isinstance(self._epsilon, Variable):
                inputs['EpsilonTensor'] = self._epsilon
            else:
                attrs['epsilon'] = self._epsilon

            if self._amsgrad:
                inputs['Moment2Max'] = [moment2_max]
                outputs["Moment2MaxOut"] = [moment2_max]

            if find_master:
                inputs["MasterParam"] = master_weight
                outputs["MasterParamOut"] = master_weight

            adam_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
                stop_gradient=True,
            )

            return adam_op

    @imperative_base.no_grad
    @framework.non_static_only
    def step(self) -> None:
        """
        Execute the optimizer and update parameters once.

        Returns:
            None

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> a = paddle.rand([2,13], dtype="float32")
                >>> linear = paddle.nn.Linear(13, 5)
                >>> # This can be any optimizer supported by dygraph.
                >>> adam = paddle.optimizer.Adam(learning_rate = 0.01,
                ...                             parameters = linear.parameters())
                >>> out = linear(a)
                >>> out.backward()
                >>> adam.step()
                >>> adam.clear_grad()
        """
        if paddle.base.dygraph.base.in_to_static_mode():
            self._declarative_step()
            return

        if not isinstance(self._parameter_list[0], dict):
            params_grads = []
            for param in self._parameter_list:
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    grad_var = param._grad_ivar()
                    if in_dygraph_mode():
                        if (
                            hasattr(grad_var, "is_selected_rows")
                            and grad_var.is_selected_rows()
                            and self.regularization is not None
                        ):
                            raise RuntimeError(
                                "Adam don't support weight_decay with sparse parameters, please set it to None."
                            )
                    else:
                        if (
                            hasattr(grad_var, "_is_sparse")
                            and grad_var._is_sparse()
                            and self.regularization is not None
                        ):
                            raise RuntimeError(
                                "Adam don't support weight_decay with sparse parameters, please set it to None."
                            )
                    params_grads.append((param, grad_var))

            optimize_ops = self._apply_optimize(
                loss=None,
                startup_program=None,
                params_grads=params_grads,
                param_group_idx=0,
            )
        else:
            # optimize parameters in groups
            for idx, param_group in enumerate(self._param_groups):
                params_grads = defaultdict(lambda: [])
                for param in param_group['params']:
                    if param.stop_gradient:
                        continue
                    if param._grad_ivar() is not None:
                        grad_var = param._grad_ivar()
                        params_grads['params'].append((param, grad_var))
                params_grads.update(
                    {k: v for k, v in param_group.items() if k != 'params'}
                )
                self._apply_optimize(
                    loss=None,
                    startup_program=None,
                    params_grads=params_grads,
                    param_group_idx=idx,
                )

    def _multi_tensor_init(self, target_block, parameters, param_group_idx):
        """
        All parameters used for optimizer (such as: parameters, master_weight, velocity_acc for momentum) calculations are grouped into a python list by data type (bfloat16, float16, float32).
        This function will be overridden in the corresponding optimizer file.
        Args:
            target_block: the block in which the loss tensor is present
            parameters: list of parameter tensors for the optimizer
        """
        self._create_accumulators(target_block, parameters)
        for param in parameters:
            moment1 = self._get_accumulator_master(self._moment1_acc_str, param)
            moment2 = self._get_accumulator_master(self._moment2_acc_str, param)
            moment2_max = (
                self._get_accumulator_master(self._moment2_acc_max_str, param)
                if self._amsgrad
                else None
            )
            beta1_pow_acc = self._get_accumulator_master(
                self._beta1_pow_acc_str, param
            )
            beta2_pow_acc = self._get_accumulator_master(
                self._beta2_pow_acc_str, param
            )

            if param.dtype == paddle.float32:
                self._param_dict['FP32_LODTensor'][param_group_idx].append(
                    param
                )
                self._moment1_dict['FP32_LODTensor'][param_group_idx].append(
                    moment1
                )
                self._moment2_dict['FP32_LODTensor'][param_group_idx].append(
                    moment2
                )
                if self._amsgrad:
                    self._moment2_max_dict['FP32_LODTensor'][
                        param_group_idx
                    ].append(moment2_max)
                self._beta1_pow_acc_dict['FP32_LODTensor'][
                    param_group_idx
                ].append(beta1_pow_acc)
                self._beta2_pow_acc_dict['FP32_LODTensor'][
                    param_group_idx
                ].append(beta2_pow_acc)
            elif self._is_dtype_fp16_or_bf16(param.dtype):
                self._param_dict['FP16_LODTensor'][param_group_idx].append(
                    param
                )
                self._moment1_dict['FP16_LODTensor'][param_group_idx].append(
                    moment1
                )
                self._moment2_dict['FP16_LODTensor'][param_group_idx].append(
                    moment2
                )
                if self._amsgrad:
                    self._moment2_max_dict['FP16_LODTensor'][
                        param_group_idx
                    ].append(moment2_max)
                self._beta1_pow_acc_dict['FP16_LODTensor'][
                    param_group_idx
                ].append(beta1_pow_acc)
                self._beta2_pow_acc_dict['FP16_LODTensor'][
                    param_group_idx
                ].append(beta2_pow_acc)
                if self._multi_precision:
                    self._master_weight_dict['FP16_LODTensor'][
                        param_group_idx
                    ].append(self._master_weights[param.name])
                else:
                    self._master_weight_dict['FP16_LODTensor'] = None
            else:
                raise ValueError(
                    "Now multi_tensor_momentum only support fp32, fp16 or bf16 parameters and grad is DENSE_TENSOR."
                )

    def _append_optimize_multi_tensor_op(
        self,
        target_block,
        parameters_and_grads,
        param_group_idx,
    ):
        """
        For Multi Tensor, append optimize merged_operator to block.
        """
        assert isinstance(target_block, (framework.Block, pir.Block))

        grad_dict = {'FP32_LODTensor': [], 'FP16_LODTensor': []}
        lr_dict = {'FP32_LODTensor': [], 'FP16_LODTensor': []}

        if isinstance(parameters_and_grads, list):
            if framework.in_dygraph_mode():
                params = [pair[0] for pair in parameters_and_grads]
                grads_types = core.eager.get_grads_types(params)
                for index, tp in enumerate(grads_types):
                    if tp == core.DataType.FLOAT32:
                        grad_dict['FP32_LODTensor'].append(
                            parameters_and_grads[index][1]
                        )
                        lr = self._create_param_lr(parameters_and_grads[index])
                        lr_dict['FP32_LODTensor'].append(lr)
                    elif (
                        tp == core.DataType.FLOAT16
                        or tp == core.DataType.BFLOAT16
                    ):
                        grad_dict['FP16_LODTensor'].append(
                            parameters_and_grads[index][1]
                        )
                        lr = self._create_param_lr(parameters_and_grads[index])
                        lr_dict['FP16_LODTensor'].append(lr)
            elif in_pir_mode():
                for param_and_grad in parameters_and_grads:
                    if param_and_grad[1] is None:
                        continue
                    if param_and_grad[0].stop_gradient is False:
                        if (
                            param_and_grad[0].dtype == DataType.FLOAT32
                            and param_and_grad[1].is_dense_tensor_type()
                        ):
                            grad_dict['FP32_LODTensor'].append(
                                param_and_grad[1]
                            )
                            lr = self._create_param_lr(param_and_grad)
                            lr_dict['FP32_LODTensor'].append(lr)
                        elif (
                            self._is_dtype_fp16_or_bf16(param_and_grad[0].dtype)
                            and param_and_grad[1].is_dense_tensor_type()
                        ):
                            grad_dict['FP16_LODTensor'].append(
                                param_and_grad[1]
                            )
                            lr = self._create_param_lr(param_and_grad)
                            lr_dict['FP16_LODTensor'].append(lr)
            else:
                for param_and_grad in parameters_and_grads:
                    if param_and_grad[1] is None:
                        continue
                    if param_and_grad[0].stop_gradient is False:
                        if (
                            param_and_grad[0].dtype == paddle.float32
                            and param_and_grad[1].type
                            == core.VarDesc.VarType.DENSE_TENSOR
                        ):
                            grad_dict['FP32_LODTensor'].append(
                                param_and_grad[1]
                            )
                            lr = self._create_param_lr(param_and_grad)
                            lr_dict['FP32_LODTensor'].append(lr)
                        elif (
                            self._is_dtype_fp16_or_bf16(param_and_grad[0].dtype)
                            and param_and_grad[1].type
                            == core.VarDesc.VarType.DENSE_TENSOR
                        ):
                            grad_dict['FP16_LODTensor'].append(
                                param_and_grad[1]
                            )
                            lr = self._create_param_lr(param_and_grad)
                            lr_dict['FP16_LODTensor'].append(lr)
        else:
            for param_and_grad in parameters_and_grads['params']:
                if param_and_grad[1] is None:
                    continue
                if param_and_grad[0].stop_gradient is False:
                    param_grad_dict = {}
                    param_grad_dict['params'] = param_and_grad
                    param_grad_dict.update(
                        {
                            k: v
                            for k, v in parameters_and_grads.items()
                            if k != 'params'
                        }
                    )
                    param_and_grad = self._update_param_group(param_grad_dict)
                    if in_pir_mode():
                        if (
                            param_and_grad[0].dtype == DataType.FLOAT32
                            and param_and_grad[1].is_dense_tensor_type()
                        ):
                            grad_dict['FP32_LODTensor'].append(
                                param_and_grad[1]
                            )
                            lr = self._create_param_lr(param_and_grad)
                            lr_dict['FP32_LODTensor'].append(lr)
                        elif (
                            self._is_dtype_fp16_or_bf16(param_and_grad[0].dtype)
                            and param_and_grad[1].is_dense_tensor_type()
                        ):
                            grad_dict['FP16_LODTensor'].append(
                                param_and_grad[1]
                            )
                            lr = self._create_param_lr(param_and_grad)
                            lr_dict['FP16_LODTensor'].append(lr)
                    else:
                        if (
                            param_and_grad[0].dtype == paddle.float32
                            and param_and_grad[1].type
                            == core.VarDesc.VarType.DENSE_TENSOR
                        ):
                            grad_dict['FP32_LODTensor'].append(
                                param_and_grad[1]
                            )
                            lr = self._create_param_lr(param_and_grad)
                            lr_dict['FP32_LODTensor'].append(lr)
                        elif (
                            self._is_dtype_fp16_or_bf16(param_and_grad[0].dtype)
                            and param_and_grad[1].type
                            == core.VarDesc.VarType.DENSE_TENSOR
                        ):
                            grad_dict['FP16_LODTensor'].append(
                                param_and_grad[1]
                            )
                            lr = self._create_param_lr(param_and_grad)
                            lr_dict['FP16_LODTensor'].append(lr)

        multi_tensor_list = ['FP32_LODTensor', 'FP16_LODTensor']
        for key in multi_tensor_list:
            if len(self._param_dict[key][param_group_idx]) > 0:
                find_master = self._multi_precision and key == 'FP16_LODTensor'

                _beta1 = (
                    self._beta1
                    if not isinstance(self._beta1, Variable)
                    else self._beta1.item(0)
                )
                _beta2 = (
                    self._beta2
                    if not isinstance(self._beta2, Variable)
                    else self._beta2.item(0)
                )

                if in_dygraph_mode():
                    master_weight = self._master_weight_dict[key]
                    master_weight = (
                        master_weight[param_group_idx]
                        if master_weight is not None
                        else None
                    )
                    found_inf = self._get_auxiliary_var('found_inf')
                    if found_inf:
                        if isinstance(
                            found_inf, (core.eager.Tensor, pir.Value)
                        ):
                            self._set_auxiliary_var('found_inf', True)
                    else:
                        if isinstance(
                            found_inf, (core.eager.Tensor, pir.Value)
                        ):
                            self._set_auxiliary_var('found_inf', False)
                        _, _, _, _, _, _, _ = _C_ops.merged_adam_(
                            self._param_dict[key][param_group_idx],
                            grad_dict[key],
                            lr_dict[key],
                            self._moment1_dict[key][param_group_idx],
                            self._moment2_dict[key][param_group_idx],
                            (
                                self._moment2_max_dict[key][param_group_idx]
                                if self._amsgrad
                                else None
                            ),
                            self._beta1_pow_acc_dict[key][param_group_idx],
                            self._beta2_pow_acc_dict[key][param_group_idx],
                            master_weight,
                            _beta1,
                            _beta2,
                            self._epsilon,
                            find_master,
                            False,
                            self._amsgrad,
                        )
                elif in_pir_mode():
                    master_weight = self._master_weight_dict[key]
                    master_weight = (
                        master_weight[param_group_idx]
                        if master_weight is not None
                        else None
                    )
                    _, _, _, _, _, _, _ = _C_ops.merged_adam_(
                        self._param_dict[key][param_group_idx],
                        grad_dict[key],
                        lr_dict[key],
                        self._moment1_dict[key][param_group_idx],
                        self._moment2_dict[key][param_group_idx],
                        (
                            self._moment2_max_dict[key][param_group_idx]
                            if self._amsgrad
                            else None
                        ),
                        self._beta1_pow_acc_dict[key][param_group_idx],
                        self._beta2_pow_acc_dict[key][param_group_idx],
                        master_weight,
                        _beta1,
                        _beta2,
                        self._epsilon,
                        find_master,
                        False,
                        self._amsgrad,
                    )
                else:
                    inputs = {
                        "Param": self._param_dict[key][param_group_idx],
                        "Grad": grad_dict[key],
                        "LearningRate": lr_dict[key],
                        "Moment1": self._moment1_dict[key][param_group_idx],
                        "Moment2": self._moment2_dict[key][param_group_idx],
                        "Beta1Pow": self._beta1_pow_acc_dict[key][
                            param_group_idx
                        ],
                        "Beta2Pow": self._beta2_pow_acc_dict[key][
                            param_group_idx
                        ],
                    }
                    outputs = {
                        "ParamOut": self._param_dict[key][param_group_idx],
                        "Moment1Out": self._moment1_dict[key][param_group_idx],
                        "Moment2Out": self._moment2_dict[key][param_group_idx],
                        "Beta1PowOut": self._beta1_pow_acc_dict[key][
                            param_group_idx
                        ],
                        "Beta2PowOut": self._beta2_pow_acc_dict[key][
                            param_group_idx
                        ],
                    }
                    attrs = {
                        "epsilon": self._epsilon,
                        "beta1": _beta1,
                        "beta2": _beta2,
                        "amsgrad": self._amsgrad,
                    }

                    if self._amsgrad:
                        inputs["Moment2Max"] = self._moment2_max_dict[key][
                            param_group_idx
                        ]
                        outputs["Moment2MaxOut"] = self._moment2_max_dict[key][
                            param_group_idx
                        ]

                    if find_master:
                        inputs["MasterParam"] = self._master_weight_dict[key][
                            param_group_idx
                        ]
                        outputs["MasterParamOut"] = self._master_weight_dict[
                            key
                        ][param_group_idx]
                        attrs["multi_precision"] = find_master

                    target_block.append_op(
                        type="merged_adam",
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs,
                        stop_gradient=True,
                    )

    def _update_param_group(self, parameters):
        self._beta1 = parameters.get('beta1', self._default_dict['beta1'])
        self._beta2 = parameters.get('beta2', self._default_dict['beta2'])
        self._epsilon = parameters.get('epsilon', self._default_dict['epsilon'])
        self._lazy_mode = parameters.get(
            'lazy_mode', self._default_dict['lazy_mode']
        )
        parameters = parameters.get('params')
        return parameters
