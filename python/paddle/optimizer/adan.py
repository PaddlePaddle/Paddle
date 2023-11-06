# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import _C_ops
from paddle.fluid import core, framework
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid.framework import Parameter, Variable
from paddle.nn.clip import GradientClipBase
from paddle.optimizer import Optimizer
from paddle.optimizer.lr import LRScheduler

__all__ = []


class Adan(Optimizer):
    r"""
    The Adan optimizer is implemented based on the Adan Optimization
    in paper `Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models <https://arxiv.org/abs/2208.06677>`_.

    Adan
    .. math::

        t &= t + 1

        grad\_diff & = grad - pre\_grad

        updata & = grad + {\beta}_2 * grad\_diff

        moment1\_out & = {\beta}_1 * moment1 + (1 - {\beta}_1) * grad

        moemnt2\_out & = {\beta}_2 * moment2 + (1 - {\beta}_2) * grad\_diff

        moemnt3\_out  & = {\beta}_3 * moment3 + (1 - {\beta}_3) * updata * updata

        denom & = \frac{\sqrt{moment3\_out}} {1 - {\beta}3\_pow} + epsilon

        update & = \frac{\frac{moment1\_out} {1 - beta1\_pow} + {\beta}_2 * \frac{moment2\_out} {1.0 - beta2\_pow}}{denom}

        param\_out & = param * (1 - learning\_rate* weight\_decay) - update * learning\_rate

    Vinilla Adan
    .. math::

        t & = t + 1 \

        grad\_diff & = grad - pre\_grad

        updata & = grad + {\beta}_2 * grad\_diff

        moment1\_out & = {\beta}_1 * moment1 + (1 - {\beta}_1) * grad + {\beta}_2 *(1 - {\beta}_2) * grad\_diff

        moemnt3\_out & = {\beta}_3 * moment3 + (1 - {\beta}_3) * updata * updata

        denom & = \frac{\sqrt{moment3\_out}} {1 - {\beta}3\_pow} + epsilon

        update & = \frac{\frac{moment1\_out} {1 - beta1\_pow}}{denom}

        param\_out & = param * (1 - learning\_rate* weight\_decay) - update * learning\_rate

    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a LRScheduler. The default value is 0.001.
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` names to update to minimize ``loss``.
            This parameter is required in dygraph mode. And you can specify different options for
            different parameter groups such as the learning rate, weight decay, etc,
            then the parameters are list of dict. Note that the learning_rate in paramter groups
            represents the scale of base learning_rate.
            The default value is None in static graph mode, at this time all parameters will be updated.
        beta1 (float|Tensor, optional): The exponential decay rate for the 1st moment estimates.
            It should be a float number or a Tensor with shape [1] and data type as float32.
            The default value is 0.98.
        beta2 (float|Tensor, optional): The exponential decay rate for the 2nd moment estimates.
            It should be a float number or a Tensor with shape [1] and data type as float32.
            The default value is 0.92.
        beta3 (float|Tensor, optional): The exponential decay rate for the 3rd moment estimates.
            It should be a float number or a Tensor with shape [1] and data type as float32.
            The default value is 0.99.
        epsilon (float, optional): A small float value for numerical stability.
            The default value is 1e-08.
        weight_decay (float|Tensor, optional): The weight decay coefficient, it can be float or Tensor. The default value is 0.0.
        no_prox (bool, optional): How to perform the decoupled weight decay. The default value is false.
        apply_decay_param_fun (function|None, optional): If it is not None,
            only tensors that makes apply_decay_param_fun(Tensor.name)==True
            will be updated with weight decay. It only works when we want to specify tensors.
            Default: None.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating. Default is false.
        vanilla (bool, optional): Whether to use vanilla adan. Default is false.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Examples:
        .. code-block:: python

            import paddle

            linear = paddle.nn.Linear(10, 10)
            inp = paddle.rand([10,10], dtype="float32")
            out = linear(inp)
            loss = paddle.mean(out)

            beta1 = paddle.to_tensor([0.98], dtype="float32")
            beta2 = paddle.to_tensor([0.92], dtype="float32")
            beta3 = paddle.to_tensor([0.99], dtype="float32")

            opt = paddle.optimizer.Adan(learning_rate=0.1,
                    parameters=linear.parameters(),
                    beta1=beta1,
                    beta2=beta2,
                    beta2=beta3,
                    weight_decay=0.01)
            loss.backward()
            opt.step()
            opt.clear_grad()


            #Note that the learning_rate of linear_2 is 0.01.
            linear_1 = paddle.nn.Linear(10, 10)
            linear_2 = paddle.nn.Linear(10, 10)
            inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            out = linear_1(inp)
            out = linear_2(out)
            loss = paddle.mean(out)
            opt = paddle.optimizer.AdamW(
                learning_rate=0.1,
                parameters=[{
                    'params': linear_1.parameters()
                }, {
                    'params': linear_2.parameters(),
                    'weight_decay': 0.001,
                    'learning_rate': 0.1,
                    'beta1': 0.8
                }],
                weight_decay=0.01,
                beta1=0.9)
            loss.backward()
            opt.step()
            opt.clear_grad()

    """

    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _moment3_acc_str = "moment3"
    _pre_grad_str = "pre_grad"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"
    _beta3_pow_acc_str = "beta3_pow_acc"

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.98,
        beta2=0.92,
        beta3=0.99,
        epsilon=1e-8,
        parameters=None,
        weight_decay=0.0,
        no_prox=False,
        apply_decay_param_fun=None,
        grad_clip=None,
        multi_precision=False,
        is_vanilla=False,
        name=None,
    ):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert beta3 is not None
        assert epsilon is not None
        if not 0 <= beta1 < 1:
            raise ValueError("Invaild value of beta1, expect beta1 in [0,1).")
        if not 0 <= beta2 < 1:
            raise ValueError("Invaild value of beta2, expect beta2 in [0,1).")
        if not 0 <= beta3 < 1:
            raise ValueError("Invaild value of beta2, expect beta2 in [0,1).")
        if not 0 <= epsilon:
            raise ValueError("Invaild value of epsilon, expect epsilon >= 0.")
        if not isinstance(weight_decay, float) and not isinstance(
            weight_decay, framework.Variable
        ):
            raise TypeError("weight_decay should be float or Tensor.")

        if parameters is not None:
            # paddle.Tensor is also iterable, so here we don't check whether
            # the input is iterable, if the input is paddle.Tensor, the
            # list(paddle.Tensor) will be a error value
            if isinstance(parameters, (paddle.Tensor, core.eager.Tensor)):
                raise TypeError(
                    "`parameters` argument given to the optimizer should be "
                    "an iterable of paddle Tensors, but got argument type is `{}`.".format(
                        type(parameters)
                    )
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
        # {accum_name : { paramter_name : accumulator_for_parameter, ...}, ...}
        self._accumulators = defaultdict(lambda: {})
        self.helper = None
        self._opti_name_list = []
        self._accumulators_holder = {}
        self._param_device_map = {}
        self.clear_gradients = self.clear_grad

        self.type = "adan"
        self._learning_rate = learning_rate
        self._params_name = set()
        self._apply_decay_param_fun = apply_decay_param_fun
        self._weight_decay = weight_decay
        self._grad_clip = grad_clip
        self._beta1 = beta1
        self._beta2 = beta2
        self._beta3 = beta3
        self._no_prox = no_prox
        self._epsilon = epsilon
        self._multi_precision = multi_precision
        self._is_vanilla = is_vanilla
        self._master_weights = {}

        self._default_dict = {
            'weight_decay': weight_decay,
            'beta1': beta1,
            'beta2': beta2,
            'beta3': beta3,
            'no_prox': no_prox,
            'vanilla': is_vanilla,
            'epsilon': epsilon,
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
        self._already_create_accumulater = set()

        self._create_master_grad_states()

    def _create_master_grad_states(self):
        # master gradients states
        self._master_grads = {}
        self._master_grad = False

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
            param_group (dict): The group of Tensors to be optimzed with
            different optimization options.
        """
        params = param_group['params']
        if isinstance(params, Parameter):
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
            acc_dtype = core.VarDesc.VarType.FP32
        self._add_accumulator(self._pre_grad_str, p, dtype=acc_dtype)

        self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
        if not self._is_vanilla:
            self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype)
        self._add_accumulator(self._moment3_acc_str, p, dtype=acc_dtype)
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.98
            if isinstance(self._beta1, Variable)
            else self._beta1,
            shape=[1],
            type=acc_dtype,
        )
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.92
            if isinstance(self._beta2, Variable)
            else self._beta2,
            shape=[1],
            type=acc_dtype,
        )
        self._add_accumulator(
            name=self._beta3_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.99
            if isinstance(self._beta3, Variable)
            else self._beta3,
            shape=[1],
            type=acc_dtype,
        )

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            if p.name in self._already_create_accumulater:
                continue
            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
                self._add_moments_pows(master_p)
                self._already_create_accumulater.add(p.name)
                continue
            if (
                self._is_dtype_fp16_or_bf16(p.dtype)
                and not self._multi_precision
            ):
                warnings.warn(
                    "Accumulating with FP16 or BF16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Adan optimizer."
                )
            self._add_moments_pows(p)
            self._already_create_accumulater.add(p.name)

    def _set_accumulator_master(self, name, param, val, deep_copy=True):
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
                "Accumulator {} does not exist for parameter {}".format(
                    name, target_name
                )
            )
        if deep_copy:
            self._accumulators[name][target_name].copy_(val, False)
        else:
            self._accumulators[name][target_name] = val

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)
        param, grad = param_and_grad

        # Whether we should do weight decay for the parameter.
        _weight_decay = self._weight_decay
        if (
            self._apply_decay_param_fun is not None
            and not self._apply_decay_param_fun(param.name)
        ):
            _weight_decay = 0.0

        pre_grad = self._get_accumulator_master(
            self._pre_grad_str, param_and_grad[0]
        )
        moment1 = self._get_accumulator_master(
            self._moment1_acc_str, param_and_grad[0]
        )
        if not self._is_vanilla:
            moment2 = self._get_accumulator_master(
                self._moment2_acc_str, param_and_grad[0]
            )
        else:
            moment2 = None
        moment3 = self._get_accumulator_master(
            self._moment3_acc_str, param_and_grad[0]
        )

        beta1_pow_acc = self._get_accumulator_master(
            self._beta1_pow_acc_str, param_and_grad[0]
        )
        beta2_pow_acc = self._get_accumulator_master(
            self._beta2_pow_acc_str, param_and_grad[0]
        )
        beta3_pow_acc = self._get_accumulator_master(
            self._beta3_pow_acc_str, param_and_grad[0]
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

        # create the adan optimize op
        if framework.in_dygraph_mode():
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
            _beta3 = (
                self._beta3
                if not isinstance(self._beta3, Variable)
                else self._beta3.item(0)
            )

            _, _, _, _, _, _, _, _, _ = _C_ops.adan_(
                param_and_grad[0],
                param_and_grad[1],
                lr,
                pre_grad,
                moment1,
                moment3,
                beta1_pow_acc,
                beta2_pow_acc,
                beta3_pow_acc,
                moment2,
                master_weight,
                _beta1,
                _beta2,
                _beta3,
                self._epsilon,
                _weight_decay,
                self._no_prox,
                find_master,
                False,
                self._is_vanilla,
            )
            return None
        else:
            inputs = {
                "param": param_and_grad[0],
                "grad": param_and_grad[1],
                "learning_rate": lr,
                "pregrad": pre_grad,
                "moment1": moment1,
                "moment3": moment3,
                "beta1_pow": beta1_pow_acc,
                "beta2_pow": beta2_pow_acc,
                "beta3_pow": beta3_pow_acc,
            }

            outputs = {
                "param_out": param_and_grad[0],
                "pregrad_out": pre_grad,
                "moment1_out": moment1,
                "moment3_out": moment3,
                "beta1_pow_out": beta1_pow_acc,
                "beta2_pow_out": beta2_pow_acc,
                "beta3_pow_out": beta3_pow_acc,
            }
            attrs = {
                "weight_decay": _weight_decay,
                "no_prox": self._no_prox,
                "multi_precision": find_master,
                "vanilla": self._is_vanilla,
            }

            if isinstance(self._beta1, Variable):
                inputs['Beta1Tensor'] = self._beta1
            else:
                attrs['beta1'] = self._beta1
            if isinstance(self._beta2, Variable):
                inputs['Beta2Tensor'] = self._beta2
            else:
                attrs['beta2'] = self._beta2
            if isinstance(self._beta3, Variable):
                inputs['Beta3Tensor'] = self._beta3
            else:
                attrs['beta3'] = self._beta3
            if isinstance(self._epsilon, Variable):
                inputs['EpsilonTensor'] = self._epsilon
            else:
                attrs['epsilon'] = self._epsilon

            if not self._is_vanilla:
                inputs["moment2"] = moment2
                outputs["moment2_out"] = moment2

            if find_master:
                inputs["master_param"] = master_weight
                outputs["master_param_out"] = master_weight

            adan_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
                stop_gradient=True,
            )

            return adan_op

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

                import paddle

                a = paddle.rand([2,13], dtype="float32")
                linear = paddle.nn.Linear(13, 5)
                # This can be any optimizer supported by dygraph.
                opt = paddle.optimizer.Adan(learning_rate = 0.01,
                                            parameters = linear.parameters())
                out = linear(a)
                out.backward()
                opt.step()
                opt.clear_grad()
        """
        if paddle.fluid.dygraph.base.in_declarative_mode():
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
                                "Adan don't support weight_decay with sparse parameters, please set it to None."
                            )
                    else:
                        if (
                            hasattr(grad_var, "_is_sparse")
                            and grad_var._is_sparse()
                            and self.regularization is not None
                        ):
                            raise RuntimeError(
                                "Adan don't support weight_decay with sparse parameters, please set it to None."
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
                                    "Adan don't support weight_decay with sparse parameters, please set it to None."
                                )
                        else:
                            if (
                                hasattr(grad_var, "_is_sparse")
                                and grad_var._is_sparse()
                                and self.regularization is not None
                            ):
                                raise RuntimeError(
                                    "Adan don't support weight_decay with sparse parameters, please set it to None."
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
        self._beta3 = parameters.get('beta3', self._default_dict['beta3'])
        self._no_prox = parameters.get('no_prox', self._default_dict['no_prox'])
        self._is_vanilla = parameters.get(
            'vanilla', self._default_dict['vanilla']
        )
        self._epsilon = parameters.get('epsilon', self._default_dict['epsilon'])
        self._weight_decay = parameters.get(
            'weight_decay', self._default_dict['weight_decay']
        )
        parameters = parameters.get('params')

        return parameters
