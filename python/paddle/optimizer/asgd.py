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
from ..base.framework import in_dygraph_mode, in_dynamic_or_pir_mode
from .optimizer import Optimizer

__all__ = []


class ASGD(Optimizer):
    r"""
    TODO
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
            raise ValueError("learning_rate is not set")
        if batch_num is None:
            raise ValueError("batch_num is not set")
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
        self._n = [batch_num]
        self._n_tensor = None

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        for p in parameters:
            if p.name in self._already_create_accumulater:
                continue
            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
                self._add_accumulator(
                    self._d_acc_str,
                    master_p,
                    p.dtype,
                    0,
                )
                # Sometimes p.shape is a tuple, so we need to change it to a list
                self._add_accumulator(
                    self._y_acc_str,
                    master_p,
                    p.dtype,
                    0,
                    self._n + list(p.shape),
                )
                self._add_accumulator(
                    self._m_acc_str,
                    master_p,
                    "int64",
                    0,
                    [1],
                )
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
            self._add_accumulator(
                self._d_acc_str,
                p,
                p.dtype,
                0,
            )
            # Sometimes p.shape is a tuple, so we need to change it to a list
            self._add_accumulator(
                self._y_acc_str,
                p,
                p.dtype,
                0,
                self._n + list(p.shape),
            )
            self._add_accumulator(
                self._m_acc_str,
                p,
                "int64",
                0,
                [1],
            )
            self._already_create_accumulater.add(p.name)

    @no_grad
    def _append_optimize_op(self, block, param_and_grad):
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        if self._n_tensor is None:
            self._n_tensor = to_tensor(
                self._n,
            )

        d = self._get_accumulator_master(self._d_acc_str, param_and_grad[0])

        m = self._get_accumulator_master(self._m_acc_str, param_and_grad[0])

        ys = self._get_accumulator_master(self._y_acc_str, param_and_grad[0])
        y = paddle.assign(ys[paddle.mod(m, self._n_tensor).item()])

        m = paddle.assign(paddle.add(m, to_tensor([1], dtype=m.dtype)))

        # The y in the static graph has one more dimension than the y in the dynamic graph.
        # So we should unify the shape of y in both dynamic and static graph.
        # eg:
        #   dynamic graph: y.shape is [2, 2]
        #   static graph: y.shape is [1, 2, 2]
        # so we should do
        #   static graph: y = y[0]
        if not in_dygraph_mode():
            y = y[0]

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
            _C_ops.asgd_(
                param_and_grad[0],
                lr,
                param_and_grad[1],
                d,
                y,
                paddle.fmin(m, self._n_tensor),
                master_weight,
                find_master,
            )
            return None
        else:
            assert isinstance(block, framework.Block)
            # create the optimize op
            inputs = {
                "param": param_and_grad[0],
                "learning_rate": lr,
                "grad": param_and_grad[1],
                "d": d,
                "y": y,
                "n": paddle.fmin(m, self._n_tensor),
            }

            outputs = {
                "param_out": param_and_grad[0],
                "d_out": d,
                "y_out": y,
            }

            attrs = {"multi_precision": find_master}

            if find_master:
                inputs["master_param"] = master_weight
                outputs["master_param_out"] = master_weight

            asgd_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
                stop_gradient=True,
            )

            return asgd_op

    def _update_param_group(self, parameters):
        parameters = parameters.get('params')
        return parameters
