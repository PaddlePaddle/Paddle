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

from .optimizer import Optimizer
from ..fluid import core
from ..fluid import framework
from ..fluid.framework import Variable, in_dygraph_mode

import paddle
from paddle import _C_ops, _legacy_C_ops

__all__ = []

GRAD_TYPES = [int(paddle.float32), int(paddle.float16)]


class MultiTensorBase(Optimizer):

    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"

    def __init__(
        self,
        learning_rate=0.001,
        parameters=None,
        weight_decay=None,
        grad_clip=None,
        name=None,
    ):

        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )

        if self._use_multi_tensor:
            self._param_dict = self._create_multi_tensor_dict()
            self._moment1_dict = self._create_multi_tensor_dict()
            self._moment2_dict = self._create_multi_tensor_dict()
            self._beta1_pow_acc_dict = self._create_multi_tensor_dict()
            self._beta2_pow_acc_dict = self._create_multi_tensor_dict()
            self._master_weight_dict = self._create_multi_tensor_dict()
            self._master_weight_dict['FP32_LODTensor'] = None

    def _multi_tensor_init(self, target_block, parameters, param_group_idx):
        """
        All parameters used for optimizer (such as: parameters, master_weight, velocity_acc for momentum) calculations are grouped into a python list by data type (float16, float32).
        This function will be overridden in the corresponding optimizer file.
        Args:
            target_block: the block in which the loss tensor is present
            parameters: list of parameter tensors for the optimizer
        """
        self._create_accumulators(target_block, parameters)
        for param in parameters:
            moment1 = self._get_accumulator(self._moment1_acc_str, param)
            moment2 = self._get_accumulator(self._moment2_acc_str, param)
            beta1_pow_acc = self._get_accumulator(
                self._beta1_pow_acc_str, param
            )
            beta2_pow_acc = self._get_accumulator(
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
                self._beta1_pow_acc_dict['FP32_LODTensor'][
                    param_group_idx
                ].append(beta1_pow_acc)
                self._beta2_pow_acc_dict['FP32_LODTensor'][
                    param_group_idx
                ].append(beta2_pow_acc)
            elif param.dtype == paddle.float16:
                self._param_dict['FP16_LODTensor'][param_group_idx].append(
                    param
                )
                self._moment1_dict['FP16_LODTensor'][param_group_idx].append(
                    moment1
                )
                self._moment2_dict['FP16_LODTensor'][param_group_idx].append(
                    moment2
                )
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
                    "Now multi_tensor_momentum only support fp32 and fp16 parameters and grad is LOD_TENSOR."
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
        assert isinstance(target_block, framework.Block)

        grad_dict = {'FP32_LODTensor': [], 'FP16_LODTensor': []}
        lr_dict = {'FP32_LODTensor': [], 'FP16_LODTensor': []}

        if isinstance(parameters_and_grads, list):
            if framework.in_dygraph_mode():
                params = [pair[0] for pair in parameters_and_grads]
                grads_types = core.eager.get_grads_types(params)
                for index, tp in enumerate(grads_types):
                    if tp == GRAD_TYPES[0]:
                        grad_dict['FP32_LODTensor'].append(
                            parameters_and_grads[index][1]
                        )
                        lr = self._create_param_lr(parameters_and_grads[index])
                        lr_dict['FP32_LODTensor'].append(lr)
                    elif tp == GRAD_TYPES[1]:
                        grad_dict['FP16_LODTensor'].append(
                            parameters_and_grads[index][1]
                        )
                        lr = self._create_param_lr(parameters_and_grads[index])
                        lr_dict['FP16_LODTensor'].append(lr)
            else:
                for param_and_grad in parameters_and_grads:
                    if param_and_grad[1] is None:
                        continue
                    if param_and_grad[0].stop_gradient is False:
                        if (
                            param_and_grad[0].dtype == paddle.float32
                            and param_and_grad[1].type
                            == core.VarDesc.VarType.LOD_TENSOR
                        ):
                            grad_dict['FP32_LODTensor'].append(
                                param_and_grad[1]
                            )
                            lr = self._create_param_lr(param_and_grad)
                            lr_dict['FP32_LODTensor'].append(lr)
                        elif (
                            param_and_grad[0].dtype == paddle.float16
                            and param_and_grad[1].type
                            == core.VarDesc.VarType.LOD_TENSOR
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
                    param_grad_dict = dict()
                    param_grad_dict['params'] = param_and_grad
                    param_grad_dict.update(
                        {
                            k: v
                            for k, v in parameters_and_grads.items()
                            if k != 'params'
                        }
                    )
                    param_and_grad = self._update_param_group(param_grad_dict)
                    if (
                        param_and_grad[0].dtype == paddle.float32
                        and param_and_grad[1].type
                        == core.VarDesc.VarType.LOD_TENSOR
                    ):
                        grad_dict['FP32_LODTensor'].append(param_and_grad[1])
                        lr = self._create_param_lr(param_and_grad)
                        lr_dict['FP32_LODTensor'].append(lr)
                    elif (
                        param_and_grad[0].dtype == paddle.float16
                        and param_and_grad[1].type
                        == core.VarDesc.VarType.LOD_TENSOR
                    ):
                        grad_dict['FP16_LODTensor'].append(param_and_grad[1])
                        lr = self._create_param_lr(param_and_grad)
                        lr_dict['FP16_LODTensor'].append(lr)

        multi_tensor_list = ['FP32_LODTensor', 'FP16_LODTensor']
        for key in multi_tensor_list:
            if len(self._param_dict[key][param_group_idx]) > 0:
                find_master = self._multi_precision and key == 'FP16_LODTensor'

                _beta1 = (
                    self._beta1
                    if not isinstance(self._beta1, Variable)
                    else self._beta1.numpy().item(0)
                )
                _beta2 = (
                    self._beta2
                    if not isinstance(self._beta2, Variable)
                    else self._beta2.numpy().item(0)
                )

                i = 0
                use_multi_tensor_adam = True
                for beta1_pow, beta2_pow, lr in zip(
                    self._beta1_pow_acc_dict[key][param_group_idx],
                    self._beta2_pow_acc_dict[key][param_group_idx],
                    lr_dict[key],
                ):
                    if i == 0:
                        lr_first = lr
                    if lr_first != lr:
                        use_multi_tensor_adam = False

                if framework._non_static_mode():

                    master_weight = self._master_weight_dict[key]
                    master_weight = (
                        master_weight[param_group_idx]
                        if master_weight is not None
                        else None
                    )
                    if in_dygraph_mode():
                        if use_multi_tensor_adam:
                            found_inf = self._get_auxiliary_var('found_inf')
                            _, _, _, _, _, _ = _C_ops.multi_tensor_adam_(
                                self._param_dict[key][param_group_idx],
                                grad_dict[key],
                                lr_dict[key][0],
                                self._moment1_dict[key][param_group_idx],
                                self._moment2_dict[key][param_group_idx],
                                self._beta1_pow_acc_dict[key][param_group_idx][
                                    0
                                ],
                                self._beta2_pow_acc_dict[key][param_group_idx][
                                    0
                                ],
                                master_weight,
                                found_inf,
                                _beta1,
                                _beta2,
                                self._epsilon,
                                self._chunk_size,
                                self._weight_decay,
                                self._use_adamw,
                                find_master,
                                False,
                            )
                        else:
                            _, _, _, _, _, _ = _C_ops.merged_adam_(
                                self._param_dict[key][param_group_idx],
                                grad_dict[key],
                                lr_dict[key],
                                self._moment1_dict[key][param_group_idx],
                                self._moment2_dict[key][param_group_idx],
                                self._beta1_pow_acc_dict[key][param_group_idx],
                                self._beta2_pow_acc_dict[key][param_group_idx],
                                master_weight,
                                _beta1,
                                _beta2,
                                self._epsilon,
                                find_master,
                                False,
                            )
                    else:
                        if use_multi_tensor_adam:
                            _, _, _, _, _, _ = _legacy_C_ops.multi_tensor_adam(
                                self._param_dict[key][param_group_idx],
                                grad_dict[key],
                                lr_dict[key][0],
                                self._moment1_dict[key][param_group_idx],
                                self._moment2_dict[key][param_group_idx],
                                self.beta1_pow_acc[param_group_idx],
                                self.beta2_pow_acc[param_group_idx],
                                master_weight,
                                self._param_dict[key][param_group_idx],
                                self._moment1_dict[key][param_group_idx],
                                self._moment2_dict[key][param_group_idx],
                                self._beta1_pow_acc_dict[key][param_group_idx][
                                    0
                                ],
                                self._beta2_pow_acc_dict[key][param_group_idx][
                                    0
                                ],
                                master_weight,
                                'epsilon',
                                self._epsilon,
                                'beta1',
                                _beta1,
                                'beta2',
                                _beta2,
                                'chunk_size',
                                self._chunk_size,
                                'weight_decay',
                                self._weight_decay,
                                'use_adamw',
                                self._use_adamw,
                                'multi_precision',
                                find_master,
                            )
                        else:
                            _, _, _, _, _, _ = _legacy_C_ops.merged_adam(
                                self._param_dict[key][param_group_idx],
                                grad_dict[key],
                                lr_dict[key],
                                self._moment1_dict[key][param_group_idx],
                                self._moment2_dict[key][param_group_idx],
                                self._beta1_pow_acc_dict[key][param_group_idx],
                                self._beta2_pow_acc_dict[key][param_group_idx],
                                master_weight,
                                self._param_dict[key][param_group_idx],
                                self._moment1_dict[key][param_group_idx],
                                self._moment2_dict[key][param_group_idx],
                                self._beta1_pow_acc_dict[key][param_group_idx],
                                self._beta2_pow_acc_dict[key][param_group_idx],
                                master_weight,
                                'epsilon',
                                self._epsilon,
                                'beta1',
                                _beta1,
                                'beta2',
                                _beta2,
                                'multi_precision',
                                find_master,
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
                    }
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
        return None
