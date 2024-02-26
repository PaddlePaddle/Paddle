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

import paddle
from paddle import _C_ops
from paddle.tensor.creation import to_tensor

from ..base import framework
from ..base.dygraph import no_grad
from ..base.framework import in_dygraph_mode, in_pir_mode
from .optimizer import Optimizer

__all__ = []


class ASGD(Optimizer):
    r"""
    Optimizer of the ASGD algorithm.Please refer to this for details:
    `Minimizing Finite Sums with the Stochastic Average Gradient <https://hal.science/hal-00860051v2>`_.

    .. math::

       \begin{aligned}
            &\hspace{0mm} d=0,\ y_i=0\ \textbf{for}\ i=1,2,...,n                            \\
            &\hspace{0mm} \textbf{for}\  \: m=0,1,...\ \textbf{do} \:                       \\
            &\hspace{5mm} i=m\ \%\ n                                                        \\
            &\hspace{5mm} d=d-y_i+f_i{}'(x)                                                 \\
            &\hspace{5mm} y_i=f_i{}'(x)                                                     \\
            &\hspace{5mm} x=x-learning\_rate(\frac{d}{\mathrm{min}(m+1,\ n)}+\lambda x)     \\
            &\hspace{0mm} \textbf{end for}                                                  \\
       \end{aligned}

    Parameters:
        learning_rate (float|Tensor|LearningRateDecay, optional): The learning rate used to update ``Parameter``.
            It can be a float value, a ``Tensor`` with a float type or a LearningRateDecay. The default value is 0.001.
        batch_num (int, optional): The number of batches needed to complete one epoch.
            Assuming the total number of samples is ``all``,
            it is recommended to set ``batch_num`` to ``all`` / ``batch_size``.
            In situations where the graphics memory is tight,
            it is possible to reduce the batch_num appropriately.
            The default value is 1.
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``.
            This parameter is required in dygraph mode.
            The default value is None in static graph mode, at this time all parameters will be updated.
        weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization.
            It can be a float value as coeff of L2 regularization or :ref:`api_paddle_regularizer_L1Decay`, :ref:`api_paddle_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_paddle_ParamAttr` already,
            the regularization setting here in optimizer will be ignored for this parameter.
            Otherwise, the regularization setting here in optimizer will take effect.
            Default None, meaning there is no regularization.
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

            >>> inp = paddle.uniform(min=-0.1, max=0.1, shape=[10, 10], dtype='float32')
            >>> linear = paddle.nn.Linear(10, 10)
            >>> inp = paddle.to_tensor(inp)
            >>> out = linear(inp)
            >>> loss = paddle.mean(out)
            >>> asgd = paddle.optimizer.ASGD(learning_rate=0.001, batch_num=10, parameters=linear.parameters(), weight_decay=0.01)
            >>> out.backward()
            >>> asgd.step()
            >>> asgd.clear_grad()
    """
    _d_acc_str = "d"
    _y_acc_str = "y"
    _m_acc_str = "m"

    def __init__(
        self,
        learning_rate=0.001,
        batch_num=1,
        parameters=None,
        weight_decay=None,
        grad_clip=None,
        multi_precision=False,
        name=None,
    ):
        if learning_rate is None:
            raise ValueError("learning_rate should not be none")
        if batch_num is None:
            raise ValueError("batch_num should not be none")
        if not 0 < batch_num:
            raise ValueError("batch_num should be greater than 0")
        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )
        self.type = "asgd"
        self._multi_precision = multi_precision
        self._master_weights = {}
        self._n = batch_num
        self._n_tensor = None

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        for p in parameters:
            if p.name in self._already_create_accumulator:
                continue
            p_new = p
            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
                p_new = master_p
            if (
                self._is_dtype_fp16_or_bf16(p.dtype)
                and not self._multi_precision
            ):
                warnings.warn(
                    "Accumulating with FP16/BF16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Adam optimizer."
                )

            self._add_accumulator(
                self._d_acc_str,
                p_new,
                p.dtype,
                0,
            )

            # Sometimes p.shape is a tuple, so we need to change it to a list
            self._add_accumulator(
                self._y_acc_str,
                p_new,
                p.dtype,
                0,
                [self._n] + list(p.shape),
            )

            self._add_accumulator(
                self._m_acc_str,
                p_new,
                "int64",
                0,
                [1],
            )

            self._already_create_accumulator.add(p.name)

    def _assign_accumulator_master(
        self, block, name, param, assign_value, index
    ):
        if self._name is not None:
            name = self._name + "_" + name
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            param.dtype
        )
        target_param = (
            self._master_weights[param.name] if find_master else param
        )
        target_name = target_param.name
        if (
            name not in self._accumulators
            or target_name not in self._accumulators[name]
        ):
            raise Exception(
                f"Accumulator {name} does not exist for parameter {target_name}"
            )

        if in_pir_mode():
            if index is None:
                self._accumulators[name][target_name] = paddle.assign(
                    assign_value
                )
            else:
                self._accumulators[name][target_name][index] = paddle.assign(
                    assign_value
                )
        else:
            assert isinstance(block, framework.Block)

            assign_inputs = {
                "X": assign_value,
            }

            assign_outputs = {
                "Out": self._accumulators[name][target_name],
            }

            block.append_op(
                type="assign",
                inputs=assign_inputs,
                outputs=assign_outputs,
            )

    @no_grad
    def _append_optimize_op(self, block, param_and_grad):
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        if self._n_tensor is None:
            self._n_tensor = to_tensor(
                [self._n],
            )

        d = self._get_accumulator_master(self._d_acc_str, param_and_grad[0])

        m = self._get_accumulator_master(self._m_acc_str, param_and_grad[0])

        ys = self._get_accumulator_master(self._y_acc_str, param_and_grad[0])
        index = paddle.mod(m, self._n_tensor).item()
        y = paddle.assign(ys[index])

        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            param_and_grad[0].dtype
        )
        master_weight = (
            self._master_weights[param_and_grad[0].name]
            if find_master
            else None
        )

        lr = self._create_param_lr(param_and_grad)

        if in_dygraph_mode():
            m.add_(to_tensor([1], dtype=m.dtype))

            _C_ops.asgd_(
                param_and_grad[0],
                param_and_grad[1],
                lr,
                d,
                ys[index],
                paddle.fmin(m, self._n_tensor),
                master_weight,
                find_master,
            )

            return None
        elif in_pir_mode():
            m = paddle.assign(paddle.add(m, to_tensor([1], dtype=m.dtype)))
            self._assign_accumulator_master(
                block, self._m_acc_str, param_and_grad[0], m, None
            )

            # The y in the static graph has one more dimension than the y in the dynamic graph.
            # So we should unify the shape of y in both dynamic and static graph.
            # eg:
            #   dynamic graph: y.shape is [2, 2]
            #   static graph: y.shape is [1, 2, 2]
            # so we should do
            #   static graph: y = y[0]
            y = y[0]

            _C_ops.asgd_(
                param_and_grad[0],
                param_and_grad[1],
                lr,
                d,
                y,
                paddle.fmin(m, self._n_tensor),
                master_weight,
                find_master,
            )

            self._assign_accumulator_master(
                block, self._y_acc_str, param_and_grad[0], y, index
            )

            return None
        else:
            assert isinstance(block, framework.Block)
            # create the optimize op
            add_inputs = {
                "X": m,
                "Y": to_tensor([1], dtype=m.dtype),
            }

            add_outputs = {
                "Out": m,
            }

            block.append_op(
                type="elementwise_add",
                inputs=add_inputs,
                outputs=add_outputs,
            )

            # The y in the static graph has one more dimension than the y in the dynamic graph.
            # So we should unify the shape of y in both dynamic and static graph.
            # eg:
            #   dynamic graph: y.shape is [2, 2]
            #   static graph: y.shape is [1, 2, 2]
            # so we should do
            #   static graph: y = y[0]
            y = y[0]

            asgd_inputs = {
                "param": param_and_grad[0],
                "grad": param_and_grad[1],
                "learning_rate": lr,
                "d": d,
                "y": y,
                "n": paddle.fmin(m, self._n_tensor),
            }

            asgd_outputs = {
                "param_out": param_and_grad[0],
                "d_out": d,
                "y_out": y,
            }

            asgd_attrs = {"multi_precision": find_master}

            if find_master:
                asgd_inputs["master_param"] = master_weight
                asgd_outputs["master_param_out"] = master_weight

            asgd_op = block.append_op(
                type=self.type,
                inputs=asgd_inputs,
                outputs=asgd_outputs,
                attrs=asgd_attrs,
                stop_gradient=True,
            )

            ys = paddle.static.setitem(ys, index, y)

            self._assign_accumulator_master(
                block, self._y_acc_str, param_and_grad[0], ys, None
            )

            return asgd_op

    def _update_param_group(self, parameters):
        parameters = parameters.get('params')
        return parameters
