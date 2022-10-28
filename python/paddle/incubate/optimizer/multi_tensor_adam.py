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

from paddle.optimizer import Adam, AdamW
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.framework import Variable, in_dygraph_mode

from paddle import _C_ops, _legacy_C_ops

__all__ = []


class MultiTensorAdam(Adam):
    r"""
    The Adam optimizer is implemented based on the Adam and Adam Optimization
    in Section 7 of `Adam paper <https://arxiv.org/abs/1412.6980>`_ and in paper
    `DECOUPLED WEIGHT DECAY REGULARIZATION <https://arxiv.org/pdf/1711.05101.pdf>`_.
    The MultiTensorAdam optimizer can deal with multiple tensor optimizations using
    Adam at once.

    The parameter ``param_out`` update rule with gradient ``grad``:

    .. math::

        t & = t + 1

        moment\_1\_out & = {\beta}_1 * moment\_1 + (1 - {\beta}_1) * grad

        moment\_2\_out & = {\beta}_2 * moment\_2 + (1 - {\beta}_2) * grad * grad

        learning\_rate & = learning\_rate * \
                            \frac{\sqrt{1 - {\beta}_2^t}}{1 - {\beta}_1^t}

        param\_out & = param - learning\_rate * \frac{moment\_1}{\sqrt{moment\_2} + \epsilon}

    Related paper: `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_

    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a LRScheduler. In MultiTensorAdam, learning_rate is same for all tensors.
            The default value is 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates.
            In MultiTensorAdam, beta1 is same for all tensors.
            The default value is 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates.
            In MultiTensorAdam, beta1 is same for all tensors.
            The default value is 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            In MultiTensorAdam, beta1 is same for all tensors.
            The default value is 1e-08.
    parameters (list|tuple, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``. \
        This parameter is required in dygraph mode.  \
        The default value is None in static mode, at this time all parameters will be updated.
    weight_decay (float, optional): weight decay (L2 penalty)
        The default value is 0.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating. Default is false.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    **Notes**:
        **Currently, MultiTensorAdam doesn't support sparse parameter optimization.**

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
            linear = paddle.nn.Linear(10, 10)
            inp = paddle.to_tensor(inp)
            out = linear(inp)
            loss = paddle.mean(out)

            beta1 = paddle.to_tensor([0.9], dtype="float32")
            beta2 = paddle.to_tensor([0.99], dtype="float32")

            multi_tensor_adam = paddle.incubate.optimizer.MultiTensorAdam(learning_rate=0.1,
                    parameters=linear.parameters(),
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=0.01)
            out.backward()
            multi_tensor_adam.step()
            multi_tensor_adam.clear_grad()
    """
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        parameters=None,
        weight_decay=0.0,
        grad_clip=None,
        multi_precision=False,
        name=None,
    ):
        super(MultiTensorAdam, self).__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            multi_precision=multi_precision,
            use_multi_tensor=True,
            name=name,
        )

        use_adamw = False
        self.type = "MultiTensorAdam"
        self._weight_decay = weight_decay
        self.use_adamw = use_adamw
        self.lr = 0.0

        n = len(self._param_groups) if self._param_groups is not None else 1
        self.beta1_pow_acc = [[] for _ in range(n)]
        self.beta2_pow_acc = [[] for _ in range(n)]


class MultiTensorAdamW(AdamW):
    r"""
    The Adam optimizer is implemented based on the Adam and Adam Optimization
    in Section 7 of `Adam paper <https://arxiv.org/abs/1412.6980>`_ and in paper
    `DECOUPLED WEIGHT DECAY REGULARIZATION <https://arxiv.org/pdf/1711.05101.pdf>`_.
    The MultiTensorAdamW optimizer can deal with multiple tensor optimizations using
    AdamW at once.

    The parameter ``param_out`` update rule with gradient ``grad``:

    .. math(AdamW)::

        t & = t + 1

        moment\_1\_out & = {\beta}_1 * moment\_1 + (1 - {\beta}_1) * grad

        moemnt\_2\_out & = {\beta}_2 * moment\_2 + (1 - {\beta}_2) * grad * grad

        learning\_rate & = learning\_rate *
                            \frac{\sqrt{1 - {\beta}_2^t}}{1 - {beta}_1^t}

        param\_out & = param - learning\_rate * (\frac{moment\_1}{\sqrt{moment\_2} + \epsilon} + \lambda * param)

    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a LRScheduler. In MultiTensorAdamW, learning_rate is same for all tensors.
            The default value is 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates.
            In MultiTensorAdamW, beta1 is same for all tensors.
            The default value is 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates.
            In MultiTensorAdamW, beta1 is same for all tensors.
            The default value is 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            In MultiTensorAdamW, beta1 is same for all tensors.
            The default value is 1e-08.
    parameters (list|tuple, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``. \
        This parameter is required in dygraph mode.  \
        The default value is None in static mode, at this time all parameters will be updated.
    weight_decay (float, optional): weight decay (L2 penalty)
        The default value is 0.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating. Default is false.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    **Notes**:
        **Currently, MultiTensorAdamW doesn't support sparse parameter optimization.**

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
            linear = paddle.nn.Linear(10, 10)
            inp = paddle.to_tensor(inp)
            out = linear(inp)
            loss = paddle.mean(out)

            beta1 = paddle.to_tensor([0.9], dtype="float32")
            beta2 = paddle.to_tensor([0.99], dtype="float32")

            multi_tensor_adam = paddle.incubate.optimizer.MultiTensorAdamW(learning_rate=0.1,
                    parameters=linear.parameters(),
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=0.01)
            out.backward()
            multi_tensor_adam.step()
            multi_tensor_adam.clear_grad()
    """
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        parameters=None,
        weight_decay=0.0,
        grad_clip=None,
        multi_precision=False,
        name=None,
    ):

        super(MultiTensorAdamW, self).__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            parameters=parameters,
            grad_clip=grad_clip,
            multi_precision=multi_precision,
            name=name,
        )

        use_adamw = True
        self.type = "MultiTensorAdamW"
        self.use_adamw = use_adamw
        self.use_multi_tensor = True
        self._weight_decay = weight_decay
        self._grad_clip = grad_clip
        self.lr = 0.0

        self._param_groups = []
        if self._parameter_list and isinstance(self._parameter_list[0], dict):
            for param_group in self._parameter_list:
                self._add_param_group(param_group.copy())
        else:
            self._param_groups = self._parameter_list

        self._param_dict = self._create_multi_tensor_dict()
        self._moment1_dict = self._create_multi_tensor_dict()
        self._moment2_dict = self._create_multi_tensor_dict()
        self._beta1_pow_acc_dict = self._create_multi_tensor_dict()
        self._beta2_pow_acc_dict = self._create_multi_tensor_dict()
        self._master_weight_dict = self._create_multi_tensor_dict()
        self._master_weight_dict['FP32_LODTensor'] = None

        n = len(self._param_groups) if self._param_groups is not None else 1
        self.beta1_pow_acc = [[] for _ in range(n)]
        self.beta2_pow_acc = [[] for _ in range(n)]


def _multi_tensor_init(optimizer, target_block, parameters, param_group_idx):
    """
    All parameters used for optimizer (such as: parameters, master_weight, velocity_acc for momentum) calculations are grouped into a python list by data type (float16, float32).
    This function will be overridden in the corresponding optimizer file.
    Args:
        target_block: the block in which the loss tensor is present
        parameters: list of parameter tensors for the optimizer
    """

    optimizer._create_accumulators(target_block, parameters)

    optimizer.beta1_pow_acc[param_group_idx] = optimizer._get_accumulator(
        optimizer._beta1_pow_acc_str, parameters[0]
    )
    optimizer.beta2_pow_acc[param_group_idx] = optimizer._get_accumulator(
        optimizer._beta2_pow_acc_str, parameters[0]
    )
    optimizer.lr = optimizer._create_param_lr(parameters)

    for param in parameters:
        moment1 = optimizer._get_accumulator(optimizer._moment1_acc_str, param)
        moment2 = optimizer._get_accumulator(optimizer._moment2_acc_str, param)

        if param.dtype == paddle.float32:
            optimizer._param_dict['FP32_LODTensor'][param_group_idx].append(
                param
            )
            optimizer._moment1_dict['FP32_LODTensor'][param_group_idx].append(
                moment1
            )
            optimizer._moment2_dict['FP32_LODTensor'][param_group_idx].append(
                moment2
            )
        elif param.dtype == paddle.float16:
            optimizer._param_dict['FP16_LODTensor'][param_group_idx].append(
                param
            )
            optimizer._moment1_dict['FP16_LODTensor'][param_group_idx].append(
                moment1
            )
            optimizer._moment2_dict['FP16_LODTensor'][param_group_idx].append(
                moment2
            )
            if optimizer._multi_precision:
                optimizer._master_weight_dict['FP16_LODTensor'][
                    param_group_idx
                ].append(optimizer._master_weights[param.name])
            else:
                optimizer._master_weight_dict['FP16_LODTensor'] = None
        else:
            raise ValueError(
                "Now multi_tensor_momentum only support fp32 and fp16 parameters and grad is LOD_TENSOR."
            )


def _append_optimize_multi_tensor_op(
    optimizer, target_block, parameters_and_grads, param_group_idx
):
    """
    For Multi Tensor Adam, append optimize merged_operator to block.
    """
    assert isinstance(target_block, framework.Block)

    grad_dict = {'FP32_LODTensor': [], 'FP16_LODTensor': []}

    if isinstance(parameters_and_grads, list):

        for param_and_grad in parameters_and_grads:
            if param_and_grad[1] is None:
                continue
            if param_and_grad[0].stop_gradient is False:
                if (
                    param_and_grad[0].dtype == paddle.float32
                    and param_and_grad[1].type
                    == core.VarDesc.VarType.LOD_TENSOR
                ):
                    grad_dict['FP32_LODTensor'].append(param_and_grad[1])
                elif (
                    param_and_grad[0].dtype == paddle.float16
                    and param_and_grad[1].type
                    == core.VarDesc.VarType.LOD_TENSOR
                ):
                    grad_dict['FP16_LODTensor'].append(param_and_grad[1])
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
                param_and_grad = _update_param_group(optimizer, param_grad_dict)
                if (
                    param_and_grad[0].dtype == paddle.float32
                    and param_and_grad[1].type
                    == core.VarDesc.VarType.LOD_TENSOR
                ):
                    grad_dict['FP32_LODTensor'].append(param_and_grad[1])
                elif (
                    param_and_grad[0].dtype == paddle.float16
                    and param_and_grad[1].type
                    == core.VarDesc.VarType.LOD_TENSOR
                ):
                    grad_dict['FP16_LODTensor'].append(param_and_grad[1])

    multi_tensor_list = ['FP32_LODTensor', 'FP16_LODTensor']
    for key in multi_tensor_list:
        if len(optimizer._param_dict[key][param_group_idx]) > 0:
            find_master = optimizer._multi_precision and key == 'FP16_LODTensor'

            _beta1 = (
                optimizer._beta1
                if not isinstance(optimizer._beta1, Variable)
                else optimizer._beta1.numpy().item(0)
            )
            _beta2 = (
                optimizer._beta2
                if not isinstance(optimizer._beta2, Variable)
                else optimizer._beta2.numpy().item(0)
            )

            if framework._non_static_mode():
                master_weight = optimizer._master_weight_dict[key]
                master_weight = (
                    master_weight[param_group_idx]
                    if master_weight is not None
                    else None
                )

                if in_dygraph_mode():
                    found_inf = optimizer._get_auxiliary_var('found_inf')
                    _, _, _, _, _, _ = _C_ops.multi_tensor_adam_(
                        optimizer._param_dict[key][param_group_idx],
                        grad_dict[key],
                        optimizer.lr,
                        optimizer._moment1_dict[key][param_group_idx],
                        optimizer._moment2_dict[key][param_group_idx],
                        optimizer.beta1_pow_acc[param_group_idx],
                        optimizer.beta2_pow_acc[param_group_idx],
                        master_weight,
                        found_inf,
                        _beta1,
                        _beta2,
                        optimizer._epsilon,
                        2048 * 32,
                        optimizer._weight_decay,
                        optimizer.use_adamw,
                        find_master,
                        False,
                    )

                    return None

                else:
                    _, _, _, _, _, _ = _legacy_C_ops.multi_tensor_adam(
                        optimizer._param_dict[key][param_group_idx],
                        grad_dict[key],
                        optimizer.lr,
                        optimizer._moment1_dict[key][param_group_idx],
                        optimizer._moment2_dict[key][param_group_idx],
                        optimizer.beta1_pow_acc[param_group_idx],
                        optimizer.beta2_pow_acc[param_group_idx],
                        master_weight,
                        optimizer._param_dict[key][param_group_idx],
                        optimizer._moment1_dict[key][param_group_idx],
                        optimizer._moment2_dict[key][param_group_idx],
                        optimizer.beta1_pow_acc[param_group_idx],
                        optimizer.beta2_pow_acc[param_group_idx],
                        master_weight,
                        'epsilon',
                        optimizer._epsilon,
                        'beta1',
                        _beta1,
                        'beta2',
                        _beta2,
                        'chunk_size',
                        2048 * 32,
                        'weight_decay',
                        optimizer._weight_decay,
                        'use_adamw',
                        optimizer.use_adamw,
                        'multi_precision',
                        find_master,
                    )

                    return None

            else:
                inputs = {
                    "Param": optimizer._param_dict[key][param_group_idx],
                    "Grad": grad_dict[key],
                    "Moment1": optimizer._moment1_dict[key][param_group_idx],
                    "Moment2": optimizer._moment2_dict[key][param_group_idx],
                    "Beta1Pow": [optimizer.beta1_pow_acc[param_group_idx]],
                    "Beta2Pow": [optimizer.beta2_pow_acc[param_group_idx]],
                    "LearningRate": [optimizer.lr],
                }
                outputs = {
                    "ParamOut": optimizer._param_dict[key][param_group_idx],
                    "Moment1Out": optimizer._moment1_dict[key][param_group_idx],
                    "Moment2Out": optimizer._moment2_dict[key][param_group_idx],
                    "Beta1PowOut": [optimizer.beta1_pow_acc[param_group_idx]],
                    "Beta2PowOut": [optimizer.beta2_pow_acc[param_group_idx]],
                }
                attrs = {
                    "epsilon": optimizer._epsilon,
                    "beta1": _beta1,
                    "beta2": _beta2,
                    "use_adamw": optimizer.use_adamw,
                    "multi_precision": find_master,
                    "weight_decay": optimizer._weight_decay,
                }
                if find_master:
                    inputs["MasterParam"] = optimizer._master_weight_dict[key][
                        param_group_idx
                    ]
                    outputs["MasterParamsOut"] = optimizer._master_weight_dict[
                        key
                    ][param_group_idx]
                    attrs["multi_precision"] = find_master
                multi_tensor_adam_op = target_block.append_op(
                    type="multi_tensor_adam",
                    inputs=inputs,
                    outputs=outputs,
                    attrs=attrs,
                    stop_gradient=True,
                )
    return multi_tensor_adam_op
