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

from ..base import framework
from ..base.dygraph import no_grad
from ..base.framework import in_dynamic_or_pir_mode
from .optimizer import Optimizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor
    from paddle.nn.clip import GradientClipBase
    from paddle.regularizer import WeightDecayRegularizer

    from .lr import LRScheduler
    from .optimizer import _ParameterConfig

__all__ = []


class SGD(Optimizer):
    r"""
    Optimizer of the stochastic gradient descent algorithm.

    .. math::

        param\_out = param - learning\_rate * grad

    Parameters:
        learning_rate (float|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a LRScheduler. The default value is 0.001.
        parameters (list|tuple|None, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static graph mode, at this time all parameters will be updated.
        weight_decay (int|float|WeightDecayRegularizer|None, optional): The strategy of regularization. \
            It can be a int or float value as coeff of L2 regularization or \
            :ref:`api_paddle_regularizer_L1Decay`, :ref:`api_paddle_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_paddle_ParamAttr` already, \
            the regularization setting here in optimizer will be ignored for this parameter. \
            Otherwise, the regularization setting here in optimizer will take effect. \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase|None, optional): Gradient clipping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three clipping strategies
            ( :ref:`api_paddle_nn_ClipGradByGlobalNorm` , :ref:`api_paddle_nn_ClipGradByNorm` ,
            :ref:`api_paddle_nn_ClipGradByValue` ). Default None, meaning there is no gradient clipping.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating.
        name (str|None, optional): The default value is None. Normally there is no need for user
                to set this property. For more information, please refer to
                :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> inp = paddle.uniform(min=-0.1, max=0.1, shape=[10, 10], dtype='float32')
            >>> linear = paddle.nn.Linear(10, 10)
            >>> inp = paddle.to_tensor(inp)
            >>> out = linear(inp)
            >>> loss = paddle.mean(out)
            >>> sgd = paddle.optimizer.SGD(
            ...     learning_rate=0.1,
            ...     parameters=linear.parameters(),
            ...     weight_decay=0.01
            ... )
            >>> out.backward()
            >>> sgd.step()
            >>> sgd.clear_grad()

    """

    type: str

    def __init__(
        self,
        learning_rate: float | LRScheduler = 0.001,
        parameters: Sequence[Tensor] | Sequence[_ParameterConfig] | None = None,
        weight_decay: float | WeightDecayRegularizer | None = None,
        grad_clip: GradientClipBase | None = None,
        multi_precision: bool = False,
        name: str | None = None,
    ) -> None:
        if learning_rate is None:
            raise ValueError("learning_rate is not set")
        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )
        self.type = "sgd"
        self._multi_precision = multi_precision
        self._master_weights = {}

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            if p.name in self._already_create_accumulator:
                continue
            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
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

        lr = self._create_param_lr(param_and_grad)
        if in_dynamic_or_pir_mode():
            _C_ops.sgd_(
                param_and_grad[0],
                lr,
                param_and_grad[1],
                master_weight,
                find_master,
            )
            return None
        else:
            assert isinstance(block, framework.Block)
            # create the optimize op
            inputs = {
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": lr,
            }

            outputs = {"ParamOut": param_and_grad[0]}

            attrs = {"multi_precision": find_master}

            if find_master:
                inputs["MasterParam"] = master_weight
                outputs["MasterParamOut"] = master_weight

            sgd_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
                stop_gradient=True,
            )

            return sgd_op

    def _update_param_group(self, parameters):
        parameters = parameters.get('params')
        return parameters
