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
from paddle.tensor.creation import to_tensor

from ..base import framework
from ..base.dygraph import no_grad
from ..base.framework import in_dynamic_or_pir_mode
from .optimizer import Optimizer

__all__ = []


class Rprop(Optimizer):
    r"""
    **Notes: This optimizer is only applicable to full-batch training.**
    Optimizer of the Rprop algorithm.Please refer to this for details:
    `A direct adaptive method for faster backpropagation learning : The RPROP algorithm <https://ieeexplore.ieee.org/document/298623>`_.

    .. math::

       \begin{aligned}
            &\hspace{0mm} For\ all\ weights\ and\ biases\{                                                                                                  \\
            &\hspace{5mm} \textbf{if} \: (\frac{\partial E}{\partial w_{ij}}(t-1)*\frac{\partial E}{\partial w_{ij}}(t)> 0)\ \textbf{then} \: \{            \\
            &\hspace{10mm} learning\_rate_{ij}(t)=\mathrm{minimum}(learning\_rate_{ij}(t-1)*\eta^{+},learning\_rate_{max})                                  \\
            &\hspace{10mm} \Delta w_{ij}(t)=-sign(\frac{\partial E}{\partial w_{ij}}(t))*learning\_rate_{ij}(t)                                             \\
            &\hspace{10mm} w_{ij}(t+1)=w_{ij}(t)+\Delta w_{ij}(t)                                                                                           \\
            &\hspace{5mm} \}                                                                                                                                \\
            &\hspace{5mm} \textbf{else if} \: (\frac{\partial E}{\partial w_{ij}}(t-1)*\frac{\partial E}{\partial w_{ij}}(t)< 0)\ \textbf{then} \: \{       \\
            &\hspace{10mm} learning\_rate_{ij}(t)=\mathrm{maximum}(learning\_rate_{ij}(t-1)*\eta^{-},learning\_rate_{min})                                  \\
            &\hspace{10mm} w_{ij}(t+1)=w_{ij}(t)                                                                                                            \\
            &\hspace{10mm} \frac{\partial E}{\partial w_{ij}}(t)=0                                                                                          \\
            &\hspace{5mm} \}                                                                                                                                \\
            &\hspace{5mm} \textbf{else if} \: (\frac{\partial E}{\partial w_{ij}}(t-1)*\frac{\partial E}{\partial w_{ij}}(t)= 0)\ \textbf{then} \: \{       \\
            &\hspace{10mm} \Delta w_{ij}(t)=-sign(\frac{\partial E}{\partial w_{ij}}(t))*learning\_rate_{ij}(t)                                             \\
            &\hspace{10mm} w_{ij}(t+1)=w_{ij}(t)+\Delta w_{ij}(t)                                                                                           \\
            &\hspace{5mm} \}                                                                                                                                \\
            &\hspace{0mm} \}                                                                                                                                \\
       \end{aligned}

    Parameters:
        learning_rate (float|Tensor|LearningRateDecay, optional): The initial learning rate used to update ``Parameter``.
            It can be a float value, a ``Tensor`` with a float type or a LearningRateDecay. The default value is 0.001.
        learning_rate_range (tuple, optional): The range of learning rate.
            Learning rate cannot be smaller than the first element of the tuple;
            learning rate cannot be larger than the second element of the tuple.
            The default value is (1e-5, 50).
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``.
            This parameter is required in dygraph mode.
            The default value is None in static graph mode, at this time all parameters will be updated.
        etas (tuple, optional): Tuple used to update learning rate.
            The first element of the tuple is the multiplicative decrease factor;
            the second element of the tuple is the multiplicative increase factor.
            The default value is (0.5, 1.2).
        grad_clip (GradientClipBase, optional): Gradient clipping strategy, it's an instance of some derived class of ``GradientClipBase`` .
            There are three clipping strategies ( :ref:`api_paddle_nn_ClipGradByGlobalNorm` , :ref:`api_paddle_nn_ClipGradByNorm` , :ref:`api_paddle_nn_ClipGradByValue` ).
            Default None, meaning there is no gradient clipping.
        multi_precision (bool, optional): In mixed precision training scenarios based on GPU,
            this parameter is mainly used to ensure the numerical stability of gradient updates.
            When it is set to True, the optimizer will save a backup of FP32 type parameters with an equal value for FP16 type parameters.
            When updating gradients, first increase the gradient type to FP32, and then assign it to the FP32 type parameter backup.
            Finally, the updated FP32 type value will be converted to FP16 type first,
            and then assigned to the actual FP16 type parameters participating in the calculation.
            The default value is False.
        name (str, optional): The default value is None. Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> inp = paddle.uniform(min=-0.1, max=0.1, shape=[1, 100], dtype='float32')
            >>> linear = paddle.nn.Linear(100, 10)
            >>> inp = paddle.to_tensor(inp)
            >>> out = linear(inp)
            >>> loss = paddle.mean(out)
            >>> rprop = paddle.optimizer.Rprop(learning_rate=0.001, learning_rate_range=(0.0001,0.1), parameters=linear.parameters(), etas=(0.5,1.2))
            >>> out.backward()
            >>> rprop.step()
            >>> rprop.clear_grad()
    """
    _prevs_acc_str = "prevs"
    _learning_rates_acc_str = "learning_rates"

    def __init__(
        self,
        learning_rate=0.001,
        learning_rate_range=(1e-5, 50),
        parameters=None,
        etas=(0.5, 1.2),
        grad_clip=None,
        multi_precision=False,
        name=None,
    ):
        if learning_rate is None:
            raise ValueError("learning_rate is not set")
        if (
            not 0.0
            < learning_rate_range[0]
            <= learning_rate
            <= learning_rate_range[1]
        ):
            raise ValueError(
                "'0.0 < learning_rate_range[0] <= learning_rate <= learning_rate_range[1]' must be true"
            )
        if not 0.0 < etas[0] < 1.0 < etas[1]:
            raise ValueError("'0.0 < etas[0] < 1.0 < etas[1]' must be true")
        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=0.0,
            grad_clip=grad_clip,
            name=name,
        )
        self.type = "rprop"
        self._initial_learning_rate = learning_rate
        self._multi_precision = multi_precision
        self._master_weights = {}
        self._learning_rate_range = [learning_rate_range]
        self._etas = [etas]
        self._sign = True

    def _to_tensor(self, block, dtype):
        assert isinstance(block, framework.Block)
        self._learning_rate_range = to_tensor(
            self._learning_rate_range, dtype=dtype
        )
        self._etas = to_tensor(self._etas, dtype=dtype)

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            if p.name in self._already_create_accumulator:
                continue
            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
                self._add_accumulator(
                    self._prevs_acc_str,
                    master_p,
                    p.dtype,
                    0,
                )
                self._add_accumulator(
                    self._learning_rates_acc_str,
                    master_p,
                    p.dtype,
                    self._initial_learning_rate,
                )
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
            self._add_accumulator(
                self._prevs_acc_str,
                p,
                p.dtype,
                0,
            )
            self._add_accumulator(
                self._learning_rates_acc_str,
                p,
                p.dtype,
                fill_value=self._initial_learning_rate,
            )
            self._already_create_accumulator.add(p.name)

    @no_grad
    def _append_optimize_op(self, block, param_and_grad):
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        if self._sign:
            self._to_tensor(block, param_and_grad[0][0].dtype)
            self._sign = False

        prevs = self._get_accumulator_master(
            self._prevs_acc_str, param_and_grad[0]
        )

        learning_rates = self._get_accumulator_master(
            self._learning_rates_acc_str, param_and_grad[0]
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
            _C_ops.rprop_(
                param_and_grad[0],
                param_and_grad[1],
                prevs,
                learning_rates,
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
                "prev": prevs,
                "learning_rate": learning_rates,
                "learning_rate_range": self._learning_rate_range,
                "etas": self._etas,
            }

            outputs = {
                "param_out": param_and_grad[0],
                "prev_out": prevs,
                "learning_rate_out": learning_rates,
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
