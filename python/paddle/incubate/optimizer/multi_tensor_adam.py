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

from paddle.optimizer.optimizer import Optimizer
from paddle.fluid import core
from paddle.fluid.framework import Variable, Parameter
from paddle.fluid.clip import GradientClipBase
from paddle.optimizer.lr import LRScheduler
from paddle.fluid import framework
from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid import layers
from paddle.fluid import unique_name
from paddle.fluid.layer_helper import LayerHelper
import warnings
from paddle.fluid.dygraph import base as imperative_base
from collections import defaultdict

from paddle import _C_ops, _legacy_C_ops

__all__ = []


class MultiTensorAdam(Optimizer):
    r"""
    The Adam optimizer is implemented based on the Adam and Adam Optimization
    in Section 7 of `Adam paper <https://arxiv.org/abs/1412.6980>`_ and in paper
    `DECOUPLED WEIGHT DECAY REGULARIZATION <https://arxiv.org/pdf/1711.05101.pdf>`_.
    The MultiTensorAdam optimizer can deal with multiple tensor optimizations using Adam or
    AdamW at once.

    The parameter ``param_out`` update rule with gradient ``grad``:

     .. math::

        t & = t + 1

        moment\_1\_out & = {\beta}_1 * moment\_1 + (1 - {\beta}_1) * grad

        moment\_2\_out & = {\beta}_2 * moment\_2 + (1 - {\beta}_2) * grad * grad

        learning\_rate & = learning\_rate * \
                            \frac{\sqrt{1 - {\beta}_2^t}}{1 - {\beta}_1^t}

        param\_out & = param - learning\_rate * \frac{moment\_1}{\sqrt{moment\_2} + \epsilon}

    Related paper: `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_

    .. math(AdamW)::

        t & = t + 1

        moment\_1\_out & = {\beta}_1 * moment\_1 + (1 - {\beta}_1) * grad

        moemnt\_2\_out & = {\beta}_2 * moment\_2 + (1 - {\beta}_2) * grad * grad

        learning\_rate & = learning\_rate *
                            \frac{\sqrt{1 - {\beta}_2^t}}{1 - {beta}_1^t}

        param\_out & = param - learning\_rate * (\frac{moment\_1}{\sqrt{moment\_2} + \epsilon} + \lambda * param)

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
        use_adamw (boolean, optional): Apply Adam orAdamW.
            True for decoupled weight decay(also known as AdamW) (default: False)
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

            multi_tensor_adam = paddle.optimizer.MultiTensorAdam(learning_rate=0.1,
                    parameters=linear.parameters(),
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=0.01,
                    use_adamw=True)
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
        use_adamw=False,
        grad_clip=None,
        multi_precision=False,
        name=None,
    ):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        if not isinstance(beta1, Variable):
            if not 0 <= beta1 < 1:
                raise ValueError(
                    "Invaild value of beta1, expect beta1 in [0,1)."
                )
        if not isinstance(beta2, Variable):
            if not 0 <= beta2 < 1:
                raise ValueError(
                    "Invaild value of beta2, expect beta2 in [0,1)."
                )
        if not isinstance(epsilon, Variable):
            if not 0 <= epsilon:
                raise ValueError(
                    "Invaild value of epsilon, expect epsilon >= 0."
                )
        if not isinstance(weight_decay, float) and not isinstance(
            weight_decay, Variable
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

        if framework._non_static_mode():
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
        if not use_adamw:
            super(MultiTensorAdam, self).__init__(
                learning_rate=learning_rate,
                parameters=parameters,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
                name=name,
            )
        else:
            super(MultiTensorAdam, self).__init__(
                learning_rate=learning_rate,
                parameters=parameters,
                grad_clip=grad_clip,
                name=name,
            )

        self._accumulators = defaultdict(lambda: dict())
        self.helper = None

        self.type = "multi_tensor_adam"
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self.use_adamw = use_adamw
        self._multi_precision = multi_precision
        self._weight_decay = weight_decay
        self._grad_clip = grad_clip
        self._master_weights = {}
        self._default_dict = {
            'weight_decay': weight_decay,
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            'use_adamw': use_adamw,
            'grad_clip': grad_clip,
        }

        self._param_dict = {'FP32_LODTensor': [], 'FP16_LODTensor': []}
        self._moment1_dict = {'FP32_LODTensor': [], 'FP16_LODTensor': []}
        self._moment2_dict = {'FP32_LODTensor': [], 'FP16_LODTensor': []}
        self._master_weight_dict = {
            'FP32_LODTensor': None,
            'FP16_LODTensor': [],
        }

        self._param_groups = []
        if self._parameter_list and isinstance(self._parameter_list[0], dict):
            for param_group in self._parameter_list:
                self._add_param_group(param_group.copy())
        else:
            self._param_groups = self._parameter_list

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

            if 'use_adamw' in param_group:
                if not param_group['use_adamw']:
                    weight_decay = param_group['weight_decay']
                    if isinstance(weight_decay, float):
                        from paddle.fluid.regularizer import L2Decay

                        regularization = L2Decay(weight_decay)
                    else:
                        regularization = weight_decay
                    param.regularizer = regularization

            param.optimize_attr['learning_rate'] = param_group.get(
                'learning_rate', 1.0
            )

        self._param_groups.append(param_group)

    def _create_master_weight(self, param):
        if param.name in self._master_weights:
            var = self._master_weights[param.name]
        else:
            assert isinstance(self.helper, LayerHelper)

            var_name = param.name + "_fp32_master"
            var_name = unique_name.generate(var_name)
            var = layers.create_global_var(
                name=var_name,
                shape=param.shape,
                value=0,
                dtype='float32',
                persistable=True,
            )
            block = self.helper.startup_program.global_block()
            block.append_op(
                type="cast",
                inputs={"X": [param]},
                outputs={"Out": [var]},
                attrs={
                    "in_dtype": param.dtype,
                    "out_dtype": core.VarDesc.VarType.FP32,
                },
            )
            self._master_weights[param.name] = var
        return var

    def _get_accumulator(self, name, param):
        """Utility function to fetch an accumulator for a parameter
        Args:
            name: name of the accumulator
            param: parameter variable for which accumulator is to be fetched
        Returns:
            accumulator variable for the parameter
        """
        if self._name is not None:
            name = self._name + "_" + name
        find_master = (
            self._multi_precision and param.dtype == core.VarDesc.VarType.FP16
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
        return self._accumulators[name][target_name]

    def _add_moments_pows(self, p):
        acc_dtype = p.dtype
        if acc_dtype == core.VarDesc.VarType.FP16:
            acc_dtype = core.VarDesc.VarType.FP32
        self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
        self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype)
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.9
            if isinstance(self._beta1, Variable)
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
            if isinstance(self._beta2, Variable)
            else self._beta2,
            shape=[1],
            type=core.VarDesc.VarType.LOD_TENSOR,
            device='cpu',
        )

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            if self._multi_precision and p.dtype == core.VarDesc.VarType.FP16:
                master_p = self._create_master_weight(p)
                self._add_moments_pows(master_p)
                continue
            if (
                p.dtype == core.VarDesc.VarType.FP16
                and not self._multi_precision
            ):
                warnings.warn(
                    "Accumulating with FP16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the MultiTensorAdam optimizer."
                )
            self._add_moments_pows(p)

    @imperative_base.no_grad
    @framework.dygraph_only
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
                multi_tensor_adam = paddle.optimizer.MultiTensorAdam(learning_rate = 0.01,
                                            parameters = linear.parameters())
                out = linear(a)
                out.backward()
                multi_tensor_adam.step()
                multi_tensor_adam.clear_grad()
        """
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
                                "MultiTensorAdam don't support weight_decay with sparse parameters, please set it to None."
                            )
                    else:
                        if (
                            hasattr(grad_var, "_is_sparse")
                            and grad_var._is_sparse()
                            and self.regularization is not None
                        ):
                            raise RuntimeError(
                                "MultiTensorAdam don't support weight_decay with sparse parameters, please set it to None."
                            )
                    params_grads.append((param, grad_var))

            optimize_ops = self._apply_optimize(
                loss=None, startup_program=None, params_grads=params_grads
            )
        else:
            # optimize parameters in groups
            for param_group in self._param_groups:
                params_grads = defaultdict(lambda: list())
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
                                    "MultiTensorAdam don't support weight_decay with sparse parameters, please set it to None."
                                )
                        else:
                            if (
                                hasattr(grad_var, "_is_sparse")
                                and grad_var._is_sparse()
                                and self.regularization is not None
                            ):
                                raise RuntimeError(
                                    "MultiTensorAdam don't support weight_decay with sparse parameters, please set it to None."
                                )
                        params_grads['params'].append((param, grad_var))
                params_grads.update(
                    {k: v for k, v in param_group.items() if k != 'params'}
                )

                self._apply_optimize(
                    loss=None, startup_program=None, params_grads=params_grads
                )

    def _create_optimization_pass(self, parameters_and_grads):
        """Add optimization operators to update gradients to tensors.

        Args:
          parameters_and_grads(list(tuple(Tensor, Tensor))):
            a list of (tensor, gradient) pair to update.

        Returns:
          return_op_list: a list of operators that will complete one step of
            optimization. This will include parameter update ops, global step
            update ops and any other custom ops required by subclasses to manage
            their internal state.
        """
        # This is a default implementation of create_optimization_pass that
        # can be shared by most optimizers. This implementation assumes that
        # the subclass will implement the _append_optimize_op method and the
        #  _initialize_tensors method. The subclass can extend the
        # _create_accumulators method if it needs to create accumulators
        # for parameters and extend _finish_update method to add custom ops.

        # Allways called under program_guard use global block as loss block
        # But if current block is in control flow, append optimize op in the
        # grad block of current block

        global_block = framework.default_main_program().global_block()
        target_block = global_block
        current_block = framework.default_main_program().current_block()
        if current_block.idx != global_block.idx:
            assert (
                current_block.backward_block_idx != -1
            ), "current block is not global_block, but it doesn't have backward block."
            target_block = framework.default_main_program().blocks[
                current_block.backward_block_idx
            ]

        start = len(target_block.ops)
        self.helper = LayerHelper(self.__class__.__name__)

        self._create_global_learning_rate()

        self._param_dict['FP32_LODTensor'].clear()
        self._moment1_dict['FP32_LODTensor'].clear()
        self._moment2_dict['FP32_LODTensor'].clear()
        self._param_dict['FP16_LODTensor'].clear()
        self._moment1_dict['FP16_LODTensor'].clear()
        self._moment2_dict['FP16_LODTensor'].clear()
        if self._multi_precision:
            self._master_weight_dict['FP16_LODTensor'].clear()

        if (
            len(self._param_dict['FP32_LODTensor']) == 0
            and len(self._param_dict['FP16_LODTensor']) == 0
        ):
            if isinstance(parameters_and_grads, list):
                self._multi_tensor_adam_init(
                    target_block,
                    [
                        p[0]
                        for p in parameters_and_grads
                        if not p[0].stop_gradient
                    ],
                )
            else:
                self._update_param_group(parameters_and_grads)
                self._multi_tensor_adam_init(
                    target_block,
                    [
                        p[0]
                        for p in parameters_and_grads['params']
                        if not p[0].stop_gradient
                    ],
                )
        if framework._non_static_mode():
            self._append_optimize_multi_tensor_adam_op(
                target_block, parameters_and_grads
            )
        else:
            self._update_param_device_map(parameters_and_grads, target_block)
            # NOTE: Multi Tensor requires all parameters to be in the same device and program.
            # param_grad_list = [p_0,g_0,p_1,g_1,....]
            param_grad_list = []
            for param_and_grad in parameters_and_grads:
                if (
                    not param_and_grad[0].stop_gradient
                    and param_and_grad[1] is not None
                ):
                    param_grad_list.append(param_and_grad[0])
                    param_grad_list.append(param_and_grad[1])
            with param_grad_list[0].block.program._optimized_guard(
                param_grad_list
            ), name_scope("optimizer"):
                device = self._get_device_for_param(param_grad_list[0].name)
                with device_guard(device):
                    self._append_optimize_multi_tensor_adam_op(
                        target_block, parameters_and_grads
                    )

        # Get custom finish ops for subclasses
        # FIXME: Need to fix this once we figure out how to handle dependencies
        self._finish_update(target_block, parameters_and_grads)

        end = len(target_block.ops)
        return target_block._slice_ops(start, end)

    def _multi_tensor_adam_init(self, target_block, parameters):
        """
        All parameters used for optimizer (such as: parameters, master_weight, velocity_acc for momentum) calculations are grouped into a python list by data type (float16, float32).
        This function will be overridden in the corresponding optimizer file.
        Args:
            target_block: the block in which the loss tensor is present
            parameters: list of parameter tensors for the optimizer
        """

        self._create_accumulators(target_block, parameters)

        self.beta1_pow_acc = self._get_accumulator(
            self._beta1_pow_acc_str, parameters[0]
        )
        self.beta2_pow_acc = self._get_accumulator(
            self._beta2_pow_acc_str, parameters[0]
        )
        self.lr = self._create_param_lr(parameters)

        for param in parameters:
            moment1 = self._get_accumulator(self._moment1_acc_str, param)
            moment2 = self._get_accumulator(self._moment2_acc_str, param)

            if param.dtype == paddle.float32:
                self._param_dict['FP32_LODTensor'].append(param)
                self._moment1_dict['FP32_LODTensor'].append(moment1)
                self._moment2_dict['FP32_LODTensor'].append(moment2)
            elif param.dtype == paddle.float16:
                self._param_dict['FP16_LODTensor'].append(param)
                self._moment1_dict['FP16_LODTensor'].append(moment1)
                self._moment2_dict['FP16_LODTensor'].append(moment2)
                if self._multi_precision:
                    self._master_weight_dict['FP16_LODTensor'].append(
                        self._master_weights[param.name]
                    )
                else:
                    self._master_weight_dict['FP16_LODTensor'] = None
            else:
                raise ValueError(
                    "Now multi_tensor_momentum only support fp32 and fp16 parameters and grad is LOD_TENSOR."
                )

    def _append_optimize_multi_tensor_adam_op(
        self, target_block, parameters_and_grads
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
                    param_and_grad = self._update_param_group(param_grad_dict)
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
            if len(self._param_dict[key]) > 0:
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

                if framework._non_static_mode():
                    if in_dygraph_mode():
                        found_inf = self._get_auxiliary_var('found_inf')

                        _, _, _, _, _, _ = _C_ops.multi_tensor_adam_(
                            self._param_dict[key],
                            grad_dict[key],
                            self.lr,
                            self._moment1_dict[key],
                            self._moment2_dict[key],
                            self.beta1_pow_acc,
                            self.beta2_pow_acc,
                            self._master_weight_dict[key],
                            found_inf,
                            _beta1,
                            _beta2,
                            self._epsilon,
                            2048 * 32,
                            self._weight_decay,
                            self.use_adamw,
                            find_master,
                            False,
                        )

                        return None

                    else:
                        _, _, _, _, _, _ = _legacy_C_ops.multi_tensor_adam(
                            self._param_dict[key],
                            grad_dict[key],
                            self.lr,
                            self._moment1_dict[key],
                            self._moment2_dict[key],
                            self.beta1_pow_acc,
                            self.beta2_pow_acc,
                            self._master_weight_dict[key],
                            self._param_dict[key],
                            self._moment1_dict[key],
                            self._moment2_dict[key],
                            self.beta1_pow_acc,
                            self.beta2_pow_acc,
                            self._master_weight_dict[key],
                            'epsilon',
                            self._epsilon,
                            'beta1',
                            _beta1,
                            'beta2',
                            _beta2,
                            'chunk_size',
                            2048 * 32,
                            'weight_decay',
                            self._weight_decay,
                            'use_adamw',
                            self.use_adamw,
                            'multi_precision',
                            find_master,
                        )

                        return None

                else:
                    inputs = {
                        "Param": self._param_dict[key],
                        "Grad": grad_dict[key],
                        "Moment1": self._moment1_dict[key],
                        "Moment2": self._moment2_dict[key],
                        "Beta1Pow": [self.beta1_pow_acc],
                        "Beta2Pow": [self.beta2_pow_acc],
                        "LearningRate": [self.lr],
                    }
                    outputs = {
                        "ParamOut": self._param_dict[key],
                        "Moment1Out": self._moment1_dict[key],
                        "Moment2Out": self._moment2_dict[key],
                        "Beta1PowOut": [self.beta1_pow_acc],
                        "Beta2PowOut": [self.beta2_pow_acc],
                    }
                    attrs = {
                        "epsilon": self._epsilon,
                        "beta1": _beta1,
                        "beta2": _beta2,
                        "use_adamw": self.use_adamw,
                        "multi_precision": find_master,
                        "weight_decay": self._weight_decay,
                    }
                    if find_master:
                        inputs["MasterParams"] = self._master_weight_dict[key]
                        outputs["MasterParamsOut"] = self._master_weight_dict[
                            key
                        ]
                        attrs["multi_precision"] = find_master
                    multi_tensor_adam_op = target_block.append_op(
                        type="multi_tensor_adam",
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs,
                        stop_gradient=True,
                    )
        return multi_tensor_adam_op

    def _update_param_group(self, parameters):
        self._beta1 = parameters.get('beta1', self._default_dict['beta1'])
        self._beta2 = parameters.get('beta2', self._default_dict['beta2'])
        self._epsilon = parameters.get('epsilon', self._default_dict['epsilon'])
        self.use_adamw = parameters.get(
            'use_adamw', self._default_dict['use_adamw']
        )
        self._weight_decay = parameters.get(
            'weight_decay', self._default_dict['weight_decay']
        )
        parameters = parameters.get('params')
        return parameters
