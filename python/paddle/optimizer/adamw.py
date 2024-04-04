# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from collections import defaultdict
from collections.abc import Callable

import paddle
from paddle import pir
from paddle.base.libpaddle import DataType
from paddle.pir import Value

from .. import _C_ops
from ..base import core, framework
from ..base.dygraph import base as imperative_base
from ..base.framework import (
    Parameter,
    Variable,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)
from ..nn.clip import GradientClipBase
from .lr import LRScheduler
from .optimizer import Optimizer

__all__ = []


class AdamW(Optimizer):
    r"""
    The AdamW optimizer is implemented based on the AdamW Optimization
    in paper `DECOUPLED WEIGHT DECAY REGULARIZATION <https://arxiv.org/pdf/1711.05101.pdf>`_.
    it can resolves the problem of L2 regularization failure in the Adam optimizer.

    .. math::

        t & = t + 1

        moment\_1\_out & = {\beta}_1 * moment\_1 + (1 - {\beta}_1) * grad

        moment\_2\_out & = {\beta}_2 * moment\_2 + (1 - {\beta}_2) * grad * grad

        learning\_rate & = learning\_rate *
            \frac{\sqrt{1 - {\beta}_2^t}}{1 - {beta}_1^t}

        param\_out & = param - learning\_rate * (\frac{moment\_1}{\sqrt{moment\_2} + \epsilon} + \lambda * param)


    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a LRScheduler. The default value is 0.001.
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
        weight_decay (float|Tensor, optional): The weight decay coefficient, it can be float or Tensor. The default value is 0.01.
        lr_ratio (function|None, optional): If it is not None,
            the learning rate will be updated with layer-wise learning rate ratio.
            Otherwise, the learning rate is the original.
            Default: None.
        apply_decay_param_fun (function|None, optional): If it is not None,
            only tensors that makes apply_decay_param_fun(Tensor.name)==True
            will be updated with weight decay. It only works when we want to specify tensors.
            Default: None.
        grad_clip (GradientClipBase, optional): Gradient clipping strategy, it's an instance of
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
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.
    Notes:
        **Currently, AdamW doesn't support sparse parameter optimization.**

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> linear = paddle.nn.Linear(10, 10)
            >>> inp = paddle.rand([10,10], dtype="float32")
            >>> out = linear(inp)
            >>> loss = paddle.mean(out)

            >>> beta1 = paddle.to_tensor([0.9], dtype="float32")
            >>> beta2 = paddle.to_tensor([0.99], dtype="float32")

            >>> opt = paddle.optimizer.AdamW(learning_rate=0.1,
            ...         parameters=linear.parameters(),
            ...         beta1=beta1,
            ...         beta2=beta2,
            ...         weight_decay=0.01
            ... )
            >>> loss.backward()
            >>> opt.step()
            >>> opt.clear_grad()


            >>> # Note that the learning_rate of linear_2 is 0.01.
            >>> linear_1 = paddle.nn.Linear(10, 10)
            >>> linear_2 = paddle.nn.Linear(10, 10)
            >>> inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            >>> out = linear_1(inp)
            >>> out = linear_2(out)
            >>> loss = paddle.mean(out)
            >>> opt = paddle.optimizer.AdamW(
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
        weight_decay=0.01,
        lr_ratio=None,
        apply_decay_param_fun=None,
        grad_clip=None,
        lazy_mode=False,
        multi_precision=False,
        name=None,
    ):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        if not isinstance(beta1, Value) and not 0 <= beta1 < 1:
            raise ValueError("Invalid value of beta1, expect beta1 in [0,1).")
        if not isinstance(beta2, Value) and not 0 <= beta2 < 1:
            raise ValueError("Invalid value of beta2, expect beta2 in [0,1).")
        if not isinstance(epsilon, Value) and not 0 <= epsilon:
            raise ValueError("Invalid value of epsilon, expect epsilon >= 0.")
        if not isinstance(weight_decay, float) and not isinstance(
            weight_decay, (framework.Variable, Value)
        ):
            raise TypeError("weight_decay should be float or Tensor.")
        if lr_ratio is not None:
            assert isinstance(lr_ratio, Callable)
            if (
                not core.is_compiled_with_cuda()
                and not core.is_compiled_with_xpu()
                and paddle.device.get_device().split(":")[0]
                not in paddle.device.get_all_custom_device_type()
            ):
                raise NotImplementedError("'lr_ratio' is unimplemented in CPU.")

        if parameters is not None:
            # paddle.Tensor is also iterable, so here we don't check whether
            # the input is iterable, if the input is paddle.Tensor, the
            # list(paddle.Tensor) will be a error value
            if isinstance(parameters, (paddle.Tensor, core.eager.Tensor)):
                raise TypeError(
                    "`parameters` argument given to the optimizer should be "
                    f"an iterable of paddle Tensors, but got argument type is `{type(parameters)}`."
                )
            if isinstance(parameters, dict):
                raise TypeError(
                    "`parameters` argument should not get dict type, "
                    "if parameter groups is needed, please set `parameters`"
                    " as list of dict"
                )
            self._parameter_list = list(parameters)
        else:
            self._parameter_list = None

        self._name = name
        if framework.in_dygraph_mode():
            if self._parameter_list is None:
                raise AttributeError(
                    "parameters argument given to the Optimizer should not be None in dygraph mode."
                )

        if not isinstance(learning_rate, (float, LRScheduler)):
            raise TypeError(
                "learning rate should be float or LRScheduler, got %s here"
                % type(learning_rate)
            )
        if grad_clip is not None:
            if not isinstance(grad_clip, GradientClipBase):
                raise TypeError(
                    "'grad_clip' should be an instance of GradientClipBase's derived class"
                )

        self._dtype = None
        # Infer the dtype form parameter
        if self._parameter_list:
            if isinstance(self._parameter_list[0], dict):
                for param_group in self._parameter_list:
                    assert (
                        'params' in param_group
                    ), 'params should be set in parameters if parameter groups are optimized in different options'
                self._dtype = self._parameter_list[0]['params'][0].dtype
            else:
                self._dtype = self._parameter_list[0].dtype

        # each program should have a independent learning rate
        # program -> tensor(learning_rate)
        self._learning_rate_map = {}
        # Dictionary of accumulators. Some optimizer subclasses need to
        # allocate and manage extra tensors associated with the parameters
        # to train. These tensors are called accumulators.
        # {accum_name : { parameter_name : accumulator_for_parameter, ...}, ...}
        self._accumulators = defaultdict(lambda: {})
        self.helper = None
        self._opti_name_list = []
        self._accumulators_holder = {}
        self._param_device_map = {}
        self.clear_gradients = self.clear_grad

        self.type = "adamw"
        self._learning_rate = learning_rate
        self._params_name = set()
        self._apply_decay_param_fun = apply_decay_param_fun
        self._weight_decay = weight_decay
        self._grad_clip = grad_clip
        self._lr_ratio = lr_ratio
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._lazy_mode = lazy_mode
        self._multi_precision = multi_precision
        self._master_weights = {}

        self._default_dict = {
            'weight_decay': weight_decay,
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            'lazy_mode': lazy_mode,
            'grad_clip': grad_clip,
        }

        self._param_groups = []
        if self._parameter_list and isinstance(self._parameter_list[0], dict):
            for param_group in self._parameter_list:
                self._add_param_group(param_group.copy())
        else:
            self._param_groups = self._parameter_list

        self._use_multi_tensor = None
        self.regularization = None
        self._auxiliary_vars = {}
        self._already_create_accumulator = set()

        self._create_master_grad_states()

    def _set_auxiliary_var(self, key, val):
        self._auxiliary_vars[key] = val

    def _get_auxiliary_var(self, key):
        if key in self._auxiliary_vars:
            return self._auxiliary_vars[key]
        else:
            return None

    def _add_param_group(self, param_group):
        """
        Add a param group to parameter_list.

        Args:
            param_group (dict): The group of Tensors to be optimized with
            different optimization options.
        """
        params = param_group['params']
        if isinstance(params, (Parameter, pir.core.ParameterMeta)):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters should be in ordered collections,"
                "but received set, please use list instead."
            )
        else:
            param_group['params'] = list(params)

        # Update optimization options for each groups
        for k, v in self._default_dict.items():
            param_group.setdefault(k, v)

        param_set = set()
        for group in self._param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError(
                "some parameters appear in more than one parameter group"
            )

        for param in param_group['params']:
            param.optimize_attr['learning_rate'] = param_group.get(
                'learning_rate', 1.0
            )

        self._param_groups.append(param_group)

    def _add_moments_pows(self, p):
        acc_dtype = p.dtype
        if self._is_dtype_fp16_or_bf16(acc_dtype):
            acc_dtype = (
                DataType.FLOAT32 if in_pir_mode() else core.VarDesc.VarType.FP32
            )
        if core.is_compiled_with_xpu():
            import os

            xpu_adamw_moment_dtype = os.getenv(
                "xpu_adamw_moment_dtype", default="fp32"
            )
            if xpu_adamw_moment_dtype == "fp16":
                self._add_accumulator(
                    self._moment1_acc_str, p, dtype=core.VarDesc.VarType.FP16
                )
                self._add_accumulator(
                    self._moment2_acc_str, p, dtype=core.VarDesc.VarType.FP16
                )
            else:
                self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
                self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype)
        else:
            self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
            self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype)
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.9
            if isinstance(self._beta1, (Variable, Value))
            else self._beta1,
            shape=[1],
            type=core.VarDesc.VarType.LOD_TENSOR,
            device='cpu',
        )
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.999
            if isinstance(self._beta2, (Variable, Value))
            else self._beta2,
            shape=[1],
            type=core.VarDesc.VarType.LOD_TENSOR,
            device='cpu',
        )

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
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)
        param, grad = param_and_grad

        # Whether we should do weight decay for the parameter.
        with_decay = True
        if (
            self._apply_decay_param_fun is not None
            and not self._apply_decay_param_fun(param.name)
        ):
            with_decay = False

        moment1 = self._get_accumulator_master(
            self._moment1_acc_str, param_and_grad[0]
        )
        moment2 = self._get_accumulator_master(
            self._moment2_acc_str, param_and_grad[0]
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

        # create the adamw optimize op
        if in_dynamic_or_pir_mode():
            lr_ratio_ = (
                1.0
                if self._lr_ratio is None
                else self._lr_ratio(param_and_grad[0])
            )

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

            _, _, _, _, _, _ = _C_ops.adamw_(
                param_and_grad[0],
                param_and_grad[1],
                lr,
                moment1,
                moment2,
                beta1_pow_acc,
                beta2_pow_acc,
                master_weight,
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

            # Pass found_inf to adamw, to skip update for not only param, but also momentum and beta_pow
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
                "with_decay": with_decay,
                "coeff": self._weight_decay,
                "lr_ratio": 1.0
                if self._lr_ratio is None
                else self._lr_ratio(param_and_grad[0]),
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
                inputs["MasterParam"] = master_weight
                outputs["MasterParamOut"] = master_weight

            adamw_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
                stop_gradient=True,
            )

            return adamw_op

    def __str__(self):
        return " ".join(["Weight Decay, params:", ",".join(self._params_name)])

    @imperative_base.no_grad
    @framework.non_static_only
    def step(self):
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
                >>> opt = paddle.optimizer.AdamW(learning_rate = 0.01,
                ...                             parameters = linear.parameters())
                >>> out = linear(a)
                >>> out.backward()
                >>> opt.step()
                >>> opt.clear_grad()
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
                    if framework.in_dygraph_mode():
                        if (
                            hasattr(grad_var, "is_selected_rows")
                            and grad_var.is_selected_rows()
                            and self.regularization is not None
                        ):
                            raise RuntimeError(
                                "AdamW don't support weight_decay with sparse parameters, please set it to None."
                            )
                    else:
                        if (
                            hasattr(grad_var, "_is_sparse")
                            and grad_var._is_sparse()
                            and self.regularization is not None
                        ):
                            raise RuntimeError(
                                "AdamW don't support weight_decay with sparse parameters, please set it to None."
                            )
                    params_grads.append((param, grad_var))

            optimize_ops = self._apply_optimize(
                loss=None, startup_program=None, params_grads=params_grads
            )
        else:
            # optimize parameters in groups
            for param_group in self._param_groups:
                params_grads = defaultdict(lambda: [])
                for param in param_group['params']:
                    if param.stop_gradient:
                        continue
                    if param._grad_ivar() is not None:
                        grad_var = param._grad_ivar()
                        if framework.in_dygraph_mode():
                            if (
                                hasattr(grad_var, "is_selected_rows")
                                and grad_var.is_selected_rows()
                                and self.regularization is not None
                            ):
                                raise RuntimeError(
                                    "AdamW don't support weight_decay with sparse parameters, please set it to None."
                                )
                        else:
                            if (
                                hasattr(grad_var, "_is_sparse")
                                and grad_var._is_sparse()
                                and self.regularization is not None
                            ):
                                raise RuntimeError(
                                    "AdamW don't support weight_decay with sparse parameters, please set it to None."
                                )
                        params_grads['params'].append((param, grad_var))
                params_grads.update(
                    {k: v for k, v in param_group.items() if k != 'params'}
                )
                self._apply_optimize(
                    loss=None, startup_program=None, params_grads=params_grads
                )

    def _update_param_group(self, parameters):
        self._beta1 = parameters.get('beta1', self._default_dict['beta1'])
        self._beta2 = parameters.get('beta2', self._default_dict['beta2'])
        self._epsilon = parameters.get('epsilon', self._default_dict['epsilon'])
        self._lazy_mode = parameters.get(
            'lazy_mode', self._default_dict['lazy_mode']
        )
        self._weight_decay = parameters.get(
            'weight_decay', self._default_dict['weight_decay']
        )
        parameters = parameters.get('params')

        return parameters
