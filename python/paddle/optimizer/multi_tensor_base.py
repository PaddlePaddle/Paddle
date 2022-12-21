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

import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.fluid.framework import default_main_program

from ..fluid import core, framework
from ..fluid.framework import Variable, in_dygraph_mode
from .optimizer import Optimizer

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

        self._lr_first = []
        self._use_multi_tensor_adam = True

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

        if self.__class__.__name__ == 'AdamW':
            if self._lr_ratio is not None:
                raise ValueError(
                    "lr_ratio must be None, when using multi_tensor."
                )
            lr_ratio_ = 1.0

        if isinstance(parameters_and_grads, list):
            if framework.in_dygraph_mode():
                params = [pair[0] for pair in parameters_and_grads]
                grads_types = core.eager.get_grads_types(params)
                for index, tp in enumerate(grads_types):
                    if tp == GRAD_TYPES[0]:
                        grad_dict['FP32_LODTensor'].append(
                            parameters_and_grads[index][1]
                        )
                        lr = self._create_multi_param_lr(
                            parameters_and_grads[index], param_group_idx
                        )
                        lr_dict['FP32_LODTensor'].append(lr)
                    elif tp == GRAD_TYPES[1]:
                        grad_dict['FP16_LODTensor'].append(
                            parameters_and_grads[index][1]
                        )
                        lr = self._create_multi_param_lr(
                            parameters_and_grads[index], param_group_idx
                        )
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
                            lr = self._create_multi_param_lr(
                                param_and_grad, param_group_idx
                            )
                            lr_dict['FP32_LODTensor'].append(lr)
                        elif (
                            param_and_grad[0].dtype == paddle.float16
                            and param_and_grad[1].type
                            == core.VarDesc.VarType.LOD_TENSOR
                        ):
                            grad_dict['FP16_LODTensor'].append(
                                param_and_grad[1]
                            )
                            lr = self._create_multi_param_lr(
                                param_and_grad, param_group_idx
                            )
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
                        lr = self._create_multi_param_lr(
                            param_and_grad, param_group_idx
                        )
                        lr_dict['FP32_LODTensor'].append(lr)
                    elif (
                        param_and_grad[0].dtype == paddle.float16
                        and param_and_grad[1].type
                        == core.VarDesc.VarType.LOD_TENSOR
                    ):
                        grad_dict['FP16_LODTensor'].append(param_and_grad[1])
                        lr = self._create_multi_param_lr(
                            param_and_grad, param_group_idx
                        )
                        lr_dict['FP16_LODTensor'].append(lr)

        multi_tensor_list = ['FP32_LODTensor', 'FP16_LODTensor']
        for key in multi_tensor_list:
            if len(self._param_dict[key][param_group_idx]) > 0:

                with_decay = True
                if self.__class__.__name__ == 'AdamW':
                    for i in range(len(self._param_dict[key][param_group_idx])):
                        if (
                            self._apply_decay_param_fun is not None
                            and not self._apply_decay_param_fun(
                                self._param_dict[key][param_group_idx][i].name
                            )
                        ):
                            raise ValueError(
                                "when using multi_tensor, all params will be updated with weight decay, but some params will not use weight decay now."
                            )

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
                for lr in lr_dict[key]:
                    if i == 0:
                        lr_first = lr
                    if framework._non_static_mode():
                        if lr_first != lr:
                            self._use_multi_tensor_adam = False
                    else:
                        self._use_multi_tensor_adam = False
                    i = i + 1

                if framework._non_static_mode():

                    master_weight = self._master_weight_dict[key]
                    master_weight = (
                        master_weight[param_group_idx]
                        if master_weight is not None
                        else None
                    )
                    found_inf = self._get_auxiliary_var('found_inf')
                    if in_dygraph_mode():
                        if self._use_multi_tensor_adam:
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
                            if self.__class__.__name__ == 'AdamW':
                                for i in range(
                                    len(self._param_dict[key][param_group_idx])
                                ):

                                    _, _, _, _, _, _ = _C_ops.adamw_(
                                        self._param_dict[key][param_group_idx][
                                            i
                                        ],
                                        grad_dict[key][i],
                                        lr_dict[key][i],
                                        self._moment1_dict[key][
                                            param_group_idx
                                        ][i],
                                        self._moment2_dict[key][
                                            param_group_idx
                                        ][i],
                                        self._beta1_pow_acc_dict[key][
                                            param_group_idx
                                        ][i],
                                        self._beta2_pow_acc_dict[key][
                                            param_group_idx
                                        ][i],
                                        master_weight[i]
                                        if master_weight
                                        else None,
                                        found_inf,
                                        _beta1,
                                        _beta2,
                                        self._epsilon,
                                        lr_ratio_,
                                        self._weight_decay,
                                        with_decay,
                                        self._lazy_mode,
                                        1000,
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
                                    self._beta1_pow_acc_dict[key][
                                        param_group_idx
                                    ],
                                    self._beta2_pow_acc_dict[key][
                                        param_group_idx
                                    ],
                                    master_weight,
                                    _beta1,
                                    _beta2,
                                    self._epsilon,
                                    find_master,
                                    False,
                                )
                    else:
                        if self._use_multi_tensor_adam:
                            _, _, _, _, _, _ = _legacy_C_ops.multi_tensor_adam(
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
                                'beta1',
                                _beta1,
                                'beta2',
                                _beta2,
                                'epsilon',
                                self._epsilon,
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
                            if self.__class__.__name__ == 'AdamW':
                                for i in range(
                                    len(self._param_dict[key][param_group_idx])
                                ):
                                    _, _, _, _, _, _ = _legacy_C_ops.adamw(
                                        self._param_dict[key][param_group_idx][
                                            i
                                        ],
                                        grad_dict[key][i],
                                        lr_dict[key][i],
                                        self._moment1_dict[key][
                                            param_group_idx
                                        ][i],
                                        self._moment2_dict[key][
                                            param_group_idx
                                        ][i],
                                        self._beta1_pow_acc_dict[key][
                                            param_group_idx
                                        ][i],
                                        self._beta2_pow_acc_dict[key][
                                            param_group_idx
                                        ][i],
                                        master_weight[i]
                                        if master_weight
                                        else None,
                                        self._param_dict[key][param_group_idx][
                                            i
                                        ],
                                        self._moment1_dict[key][
                                            param_group_idx
                                        ][i],
                                        self._moment2_dict[key][
                                            param_group_idx
                                        ][i],
                                        self._beta1_pow_acc_dict[key][
                                            param_group_idx
                                        ][i],
                                        self._beta2_pow_acc_dict[key][
                                            param_group_idx
                                        ][i],
                                        master_weight[i]
                                        if master_weight
                                        else None,
                                        'epsilon',
                                        self._epsilon,
                                        'lazy_mode',
                                        self._lazy_mode,
                                        'min_row_size_to_use_multithread',
                                        1000,
                                        'beta1',
                                        _beta1,
                                        'beta2',
                                        _beta2,
                                        "with_decay",
                                        with_decay,
                                        'coeff',
                                        self._weight_decay,
                                        'multi_precision',
                                        find_master,
                                        'lr_ratio',
                                        lr_ratio_,
                                    )
                            else:
                                _, _, _, _, _, _ = _legacy_C_ops.merged_adam(
                                    self._param_dict[key][param_group_idx],
                                    grad_dict[key],
                                    lr_dict[key],
                                    self._moment1_dict[key][param_group_idx],
                                    self._moment2_dict[key][param_group_idx],
                                    self._beta1_pow_acc_dict[key][
                                        param_group_idx
                                    ],
                                    self._beta2_pow_acc_dict[key][
                                        param_group_idx
                                    ],
                                    master_weight,
                                    self._param_dict[key][param_group_idx],
                                    self._moment1_dict[key][param_group_idx],
                                    self._moment2_dict[key][param_group_idx],
                                    self._beta1_pow_acc_dict[key][
                                        param_group_idx
                                    ],
                                    self._beta2_pow_acc_dict[key][
                                        param_group_idx
                                    ],
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
                    if self.__class__.__name__ == 'AdamW':
                        for i in range(
                            len(self._param_dict[key][param_group_idx])
                        ):
                            inputs = {
                                "Param": [
                                    self._param_dict[key][param_group_idx][i]
                                ],
                                "Grad": [grad_dict[key][i]],
                                "LearningRate": [lr_dict[key][i]],
                                "Moment1": [
                                    self._moment1_dict[key][param_group_idx][i]
                                ],
                                "Moment2": [
                                    self._moment2_dict[key][param_group_idx][i]
                                ],
                                "Beta1Pow": [
                                    self._beta1_pow_acc_dict[key][
                                        param_group_idx
                                    ][i]
                                ],
                                "Beta2Pow": [
                                    self._beta2_pow_acc_dict[key][
                                        param_group_idx
                                    ][i]
                                ],
                            }

                            # Pass found_inf to adamw, to skip update for not only param, but also momentum and beta_pow
                            found_inf = self._get_auxiliary_var('found_inf')

                            if found_inf:
                                inputs['SkipUpdate'] = found_inf

                            outputs = {
                                "ParamOut": [
                                    self._param_dict[key][param_group_idx][i]
                                ],
                                "Moment1Out": [
                                    self._moment1_dict[key][param_group_idx][i]
                                ],
                                "Moment2Out": [
                                    self._moment2_dict[key][param_group_idx][i]
                                ],
                                "Beta1PowOut": [
                                    self._beta1_pow_acc_dict[key][
                                        param_group_idx
                                    ][i]
                                ],
                                "Beta2PowOut": [
                                    self._beta2_pow_acc_dict[key][
                                        param_group_idx
                                    ][i]
                                ],
                            }
                            attrs = {
                                "lazy_mode": self._lazy_mode,
                                "min_row_size_to_use_multithread": 1000,
                                "multi_precision": find_master,
                                "with_decay": with_decay,
                                "coeff": self._weight_decay,
                                "lr_ratio": 1.0,
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

                            if find_master:
                                inputs[
                                    "MasterParam"
                                ] = self._master_weight_dict[key][
                                    param_group_idx
                                ][
                                    i
                                ]
                                outputs[
                                    "MasterParamOut"
                                ] = self._master_weight_dict[key][
                                    param_group_idx
                                ][
                                    i
                                ]

                            target_block.append_op(
                                type=self.type,
                                inputs=inputs,
                                outputs=outputs,
                                attrs=attrs,
                                stop_gradient=True,
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
                            "Moment1Out": self._moment1_dict[key][
                                param_group_idx
                            ],
                            "Moment2Out": self._moment2_dict[key][
                                param_group_idx
                            ],
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
                            inputs["MasterParam"] = self._master_weight_dict[
                                key
                            ][param_group_idx]
                            outputs[
                                "MasterParamOut"
                            ] = self._master_weight_dict[key][param_group_idx]
                            attrs["multi_precision"] = find_master
                        target_block.append_op(
                            type="merged_adam",
                            inputs=inputs,
                            outputs=outputs,
                            attrs=attrs,
                            stop_gradient=True,
                        )
        return None

    def _create_multi_param_lr(self, param_and_grad, param_group_idx):
        # create learning rate tensor for every parameter
        param = param_and_grad[0]
        if hasattr(param, 'optimize_attr'):
            param_lr = param.optimize_attr['learning_rate']
            if len(self._lr_first) == param_group_idx:
                self._lr_first.append(param_lr)
            if param_lr != self._lr_first[param_group_idx]:
                self._use_multi_tensor_adam = False
            if type(param_lr) == Variable:
                return param_lr
            else:
                if param_lr == 1.0:
                    return self._global_learning_rate()
                else:
                    with default_main_program()._lr_schedule_guard(
                        is_with_opt=True
                    ), framework.name_scope('scale_with_param_lr'):
                        return self._global_learning_rate() * param_lr
        else:
            return self._global_learning_rate()
