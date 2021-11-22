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

from .optimizer import Optimizer
from ..fluid import core
from ..fluid import framework
from ..fluid.framework import Variable, name_scope
from ..fluid.layer_helper import LayerHelper
from ..fluid import unique_name
from ..fluid import layers
import paddle.fluid as fluid
from paddle.fluid.regularizer import L2DecayRegularizer
from paddle import _C_ops
import paddle

__all__ = []


class MultiTensorMomentum(Optimizer):
    r"""

    Simple Momentum optimizer with velocity state

    This optimizer has a flag for Nestrov Momentum.

    The update equations are as follows:

    .. math::

        & velocity = mu * velocity + gradient

        & if (use\_nesterov):

        &\quad   param = param - (gradient + mu * velocity) * learning\_rate

        & else:

        &\quad   param = param - learning\_rate * velocity

    Parameters:

        learning_rate (float|Tensor|LearningRateDecay, optional): The learning rate used to update ``Parameter``.
            It can be a float value, a ``Tensor`` with a float type or a LearningRateDecay. The default value is 0.001.
        momentum (float): Momentum factor. The default value is 0.9.
        parameters (list|tuple, optional): List|Tuple of ``Tensor`` to update to minimize ``loss``. \
            This parameter is required in dygraph mode. And you can specify different options for \
            different parameter groups such as the learning rate, weight decay, etc, \
            then the parameters are list of dict. Note that the learning_rate in paramter groups \
            represents the scale of base learning_rate. \
            The default value is None in static mode, at this time all parameters will be updated.
        weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. \
            It canbe a float value as coeff of L2 regularization or \
            :ref:`api_fluid_regularizer_L1Decay`, :ref:`api_fluid_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_fluid_ParamAttr` already, \
            the regularization setting here in optimizer will be ignored for this parameter. \
            Otherwise, the regularization setting here in optimizer will take effect. \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating. Default is false.
        rescale_grad (float, optional): Multiply the gradient with `rescale_grad` before updating. \
            Often choose to be ``1.0/batch_size``.
        name (str, optional): The default value is None. Normally there is no need for user
                to set this property. For more information, please refer to
                :ref:`api_guide_Name` .

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
            momentum = paddle.optimizer.Momentum(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
            back = out.backward()
            momentum.step()
            momentum.clear_grad()

            #Note that the learning_rate of linear_2 is 0.01.
            linear_1 = paddle.nn.Linear(10, 10)
            linear_2 = paddle.nn.Linear(10, 10)
            inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            out = linear_1(inp)
            out = linear_2(out)
            loss = paddle.mean(out)
            momentum = paddle.optimizer.Momentum(
                learning_rate=0.1,
                parameters=[{
                    'params': linear_1.parameters()
                }, {
                    'params': linear_2.parameters(),
                    'weight_decay': 0.001,
                    'learning_rate': 0.1
                }],
                weight_decay=0.01,
                momentum=0.9)                   
            out.backward()
            momentum.step()
            momentum.clear_grad()

    """
    _velocity_acc_str = "velocity"

    def __init__(self,
                 learning_rate=0.001,
                 momentum=0.9,
                 parameters=None,
                 use_nesterov=False,
                 weight_decay=None,
                 grad_clip=None,
                 multi_precision=False,
                 rescale_grad=1.0,
                 name=None):
        if learning_rate is None:
            raise ValueError("learning_rate is not set")
        if momentum is None:
            raise ValueError("momentum is not set")

        predicate = lambda regular: isinstance(regular, (L2DecayRegularizer, float))
        if isinstance(parameters, list):
            if isinstance(parameters[0], dict):
                for param_group in parameters:
                    decay = param_group[
                        'weight_decay'] if 'weight_decay' in param_group else weight_decay
                    reg_method, reg_coeff = self._update_regularization(decay)
                    param_group['regularization_methods'] = reg_method
                    param_group['regularization_coeffs'] = reg_coeff
                    py_regular = None if predicate(decay) else decay
                    param_group['weight_decay'] = py_regular

        py_regular = None if predicate(weight_decay) else weight_decay
        super(MultiTensorMomentum, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=py_regular,
            grad_clip=grad_clip,
            name=name)
        self.type = "multi_tensor_momentum"
        self._momentum = momentum
        self._use_nesterov = bool(use_nesterov)
        self._regularization_method, self._regularization_coeff = self._update_regularization(
            weight_decay)
        self._multi_precision = multi_precision
        self._rescale_grad = rescale_grad
        self._master_weights = {}

        self._default_dict = {
            'momentum': momentum,
            'use_nesterov': use_nesterov,
            'rescale_grad': rescale_grad,
            'regularization_methods': self._regularization_method,
            'regularization_coeffs': self._regularization_coeff,
        }

        # self._create_global_learning_rate()

        self._multi_tensor_init()

    def _multi_tensor_init(self):
        self._all_parameters = []
        if self._parameter_list is None or not isinstance(
                self._parameter_list[0], dict):
            for p in self._parameter_list:
                self._all_parameters.append(p)
        else:
            for param_group in self._param_groups:
                for p in param_group['params']:
                    self._all_parameters.append(p)

        self.param_dict = {'FP32_LODTensor': [], 'FP16_LODTensor': []}
        self.velocity_dict = {'FP32_LODTensor': [], 'FP16_LODTensor': []}
        self.master_weight_dict = {'FP32_LODTensor': None, 'FP16_LODTensor': []}
        self.regularization_method_dict = {
            'FP32_LODTensor': [],
            'FP16_LODTensor': []
        }
        self.regularization_coeff_dict = {
            'FP32_LODTensor': [],
            'FP16_LODTensor': []
        }

        global_block = framework.default_main_program().global_block()
        target_block = global_block
        current_block = framework.default_main_program().current_block()
        if current_block.idx != global_block.idx:
            assert current_block.backward_block_idx != -1, \
                "current block is not global_block, but it doesn't have backward block."
            target_block = framework.default_main_program().blocks[
                current_block.backward_block_idx]
        self.helper = LayerHelper(self.__class__.__name__)
        for p in self._all_parameters:
            if p.stop_gradient is False:
                if p.dtype == paddle.float32:
                    # param
                    self.param_dict['FP32_LODTensor'].append(p)
                    # velocity
                    velocity_acc = self._create_accumulators(target_block, p)
                    self.velocity_dict['FP32_LODTensor'].append(velocity_acc)
                    # fp32 no master weight
                    # regularization
                    regularization_method = self._regularization_method
                    regularization_coeff = self._regularization_coeff
                    if hasattr(p, 'regularizer'):
                        # we skip param's l2decay before, so fuse it with momentum here.
                        if isinstance(p.regularizer, L2DecayRegularizer):
                            regularization_method = "l2_decay"
                            regularization_coeff = p.regularizer._regularization_coeff
                        # the param's regularization has been done before, we avoid do l2decay in momentum.
                        elif p.regularizer is not None:
                            regularization_method = ""
                            regularization_coeff = 0
                    self.regularization_method_dict['FP32_LODTensor'].append(
                        regularization_method)
                    self.regularization_coeff_dict['FP32_LODTensor'].append(
                        regularization_coeff)
                elif p.dtype == paddle.float16:
                    # param
                    self.param_dict['FP16_LODTensor'].append(p)
                    # velocity
                    velocity_acc = self._create_accumulators(target_block, p)
                    self.velocity_dict['FP16_LODTensor'].append(velocity_acc)
                    # master weight
                    # master weight
                    if self._multi_precision:
                        self.master_weight_dict['FP16_LODTensor'].append(
                            self._master_weights[p.name])
                    else:
                        self.master_weight_dict['FP16_LODTensor'] = None
                    # regularization
                    regularization_method = self._regularization_method
                    regularization_coeff = self._regularization_coeff
                    if hasattr(p, 'regularizer'):
                        # we skip param's l2decay before, so fuse it with momentum here.
                        if isinstance(p.regularizer, L2DecayRegularizer):
                            regularization_method = "l2_decay"
                            regularization_coeff = p.regularizer._regularization_coeff
                        # the param's regularization has been done before, we avoid do l2decay in momentum.
                        elif p.regularizer is not None:
                            regularization_method = ""
                            regularization_coeff = 0
                    self.regularization_method_dict['FP16_LODTensor'].append(
                        regularization_method)
                    self.regularization_coeff_dict['FP16_LODTensor'].append(
                        regularization_coeff)
                else:
                    raise ValueError(
                        "Now multi_tensor_momentum only support fp32 and fp16 parameters."
                    )

    def _update_regularization(self, weight_decay):
        reg_method = ""
        reg_coeff = 0

        if (isinstance(weight_decay, L2DecayRegularizer)):
            reg_method = "l2_decay"
            reg_coeff = weight_decay._regularization_coeff
        if (isinstance(weight_decay, float)):
            reg_method = "l2_decay"
            reg_coeff = weight_decay
        return reg_method, reg_coeff

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
                persistable=True)
            block = self.helper.startup_program.global_block()
            block.append_op(
                type="cast",
                inputs={"X": [param]},
                outputs={"Out": [var]},
                attrs={
                    "in_dtype": param.dtype,
                    "out_dtype": core.VarDesc.VarType.FP32
                })
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
        find_master = self._multi_precision and param.dtype == core.VarDesc.VarType.FP16
        target_param = self._master_weights[
            param.name] if find_master else param
        target_name = target_param.name
        if (name not in self._accumulators or
                target_name not in self._accumulators[name]):
            raise Exception("Accumulator {} does not exist for parameter {}".
                            format(name, target_name))
        return self._accumulators[name][target_name]

    # def _create_accumulators(self, block, parameters):
    #     '''
    #     if framework.in_dygraph_mode():
    #         return
    #     '''
    #     assert isinstance(block, framework.Block)

    #     if isinstance(parameters, dict):
    #         parameters = self._update_param_group(parameters)

    #     for p in parameters:
    #         if self._multi_precision and p.dtype == core.VarDesc.VarType.FP16:
    #             master_p = self._create_master_weight(p)
    #             var = self._add_accumulator(self._velocity_acc_str, master_p)
    #             continue
    #         if p.dtype == core.VarDesc.VarType.FP16 and not self._multi_precision:
    #             warnings.warn(
    #                 "Accumulating with FP16 in optimizer can lead to poor accuracy or slow convergence."
    #                 "Consider using multi_precision=True option of the Momentum optimizer."
    #             )
    #         var = self._add_accumulator(self._velocity_acc_str, p)
    #     return var

    def _create_accumulators(self, block, p):
        '''
        if framework.in_dygraph_mode():
            return
        '''
        assert isinstance(block, framework.Block)

        if self._multi_precision and p.dtype == core.VarDesc.VarType.FP16:
            master_p = self._create_master_weight(p)
            var = self._add_accumulator(self._velocity_acc_str, master_p)
        elif p.dtype == core.VarDesc.VarType.FP16 and not self._multi_precision:
            warnings.warn(
                "Accumulating with FP16 in optimizer can lead to poor accuracy or slow convergence."
                "Consider using multi_precision=True option of the Momentum optimizer."
            )
            var = self._add_accumulator(self._velocity_acc_str, p)
        else:
            var = self._add_accumulator(self._velocity_acc_str, p)

        return var

    def _create_regularization_of_grad(self, param, grad, regularization=None):
        """ Create and add backward regularization Operators
    
        Function helper of append_regularization_ops.
        """
        # If ParamAttr is set to L2Decay, we skip doing regularization here. And then we fused
        # L2Decay with momentum which can refer to _append_optimize_op below.
        if hasattr(param, 'regularizer') and isinstance(param.regularizer,
                                                        L2DecayRegularizer):
            return grad
        return super(MultiTensorMomentum, self)._create_regularization_of_grad(
            param, grad, regularization)

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
            assert current_block.backward_block_idx != -1, \
                "current block is not global_block, but it doesn't have backward block."
            target_block = framework.default_main_program().blocks[
                current_block.backward_block_idx]

        start = len(target_block.ops)
        self.helper = LayerHelper(self.__class__.__name__)

        # params_grads_device_map = parameters_and_grads['params'] if isinstance(
        #     parameters_and_grads, dict) else parameters_and_grads
        # self._update_param_device_map(params_grads_device_map, target_block)

        # lr
        self._create_global_learning_rate()

        # if isinstance(parameters_and_grads, list):
        #     self._create_accumulators(
        #         target_block,
        #         [p[0] for p in parameters_and_grads if not p[0].stop_gradient])

        # else:
        #     params_acc_dict = parameters_and_grads.copy()
        #     params_acc_dict['params'] = [
        #         p[0] for p in params_acc_dict['params']
        #         if not p[0].stop_gradient
        #     ]
        #     self._create_accumulators(target_block, params_acc_dict)

        self._append_optimize_op(target_block, parameters_and_grads)

        # Get custom finish ops for subclasses
        # FIXME: Need to fix this once we figure out how to handle dependencies
        self._finish_update(target_block, parameters_and_grads)

        end = len(target_block.ops)
        return target_block._slice_ops(start, end)

    def _append_optimize_op(self, target_block, parameters_and_grads):
        assert isinstance(target_block, framework.Block)

        # 声明参数，参数分成4类：grad_vars可能为framework::LoDTensor或者framework::SelectedRows，find_master可能为True或false
        # params_FP64_LODTensor = []
        # grads_FP64_LODTensor = []
        # velocity_acc_FP64_LODTensor = []
        # lr_FP64_LODTensor = []
        # regularization_method_FP64_LODTensor = []
        # regularization_coeff_FP64_LODTensor = []

        # params_FP32_LODTensor = []
        # grads_FP32_LODTensor = []
        self.grad_dict = {'FP32_LODTensor': [], 'FP16_LODTensor': []}
        # velocity_acc_FP32_LODTensor = []
        self.lr_dict = {'FP32_LODTensor': [], 'FP16_LODTensor': []}
        # lr_FP32_LODTensor = []
        # regularization_method_FP32_LODTensor = []
        # regularization_coeff_FP32_LODTensor = []

        # params_FP16_LODTensor = []
        # grads_FP16_LODTensor = []
        # velocity_acc_FP16_LODTensor = []
        # lr_FP16_LODTensor = []
        # master_weight_FP16_LODTensor = []
        # regularization_method_FP16_LODTensor = []
        # regularization_coeff_FP16_LODTensor = []

        if framework.in_dygraph_mode():
            if isinstance(parameters_and_grads, list):
                for param_and_grad in parameters_and_grads:
                    if param_and_grad[1] is None:
                        continue
                    if param_and_grad[0].stop_gradient is False:
                        if param_and_grad[
                                0].dtype == paddle.float32 and param_and_grad[
                                    1].type == core.VarDesc.VarType.LOD_TENSOR:
                            # param
                            # params_FP32_LODTensor.append(param_and_grad[0])
                            # grad
                            self.grad_dict['FP32_LODTensor'].append(
                                param_and_grad[1])
                            # grads_FP32_LODTensor.append(param_and_grad[1])
                            # velocity
                            # velocity_acc = self._create_accumulators(
                            #     target_block, param_and_grad[0])
                            # velocity_acc_FP32_LODTensor.append(velocity_acc)
                            # lr
                            lr = self._create_param_lr(param_and_grad)
                            # lr_FP32_LODTensor.append(lr)
                            self.lr_dict['FP32_LODTensor'].append(lr)
                            # For fusion of momentum and l2decay 
                            # param = param_and_grad[0]
                            # regularization_method = self._regularization_method
                            # regularization_coeff = self._regularization_coeff
                            # if hasattr(param, 'regularizer'):
                            #     # we skip param's l2decay before, so fuse it with momentum here.
                            #     if isinstance(param.regularizer,
                            #                   L2DecayRegularizer):
                            #         regularization_method = "l2_decay"
                            #         regularization_coeff = param.regularizer._regularization_coeff
                            #     # the param's regularization has been done before, we avoid do l2decay in momentum.
                            #     elif param.regularizer is not None:
                            #         regularization_method = ""
                            #         regularization_coeff = 0
                            # regularization_method_FP32_LODTensor.append(
                            #     regularization_method)
                            # regularization_coeff_FP32_LODTensor.append(
                            #     regularization_coeff)
                        elif param_and_grad[
                                0].dtype == paddle.float16 and param_and_grad[
                                    1].type == core.VarDesc.VarType.LOD_TENSOR:
                            # param
                            # params_FP16_LODTensor.append(param_and_grad[0])
                            # grad
                            # grads_FP16_LODTensor.append(param_and_grad[1])
                            self.grad_dict['FP16_LODTensor'].append(
                                param_and_grad[1])
                            # velocity
                            # velocity_acc = self._create_accumulators(
                            #     target_block, param_and_grad[0])
                            # velocity_acc_FP16_LODTensor.append(velocity_acc)
                            # lr
                            lr = self._create_param_lr(param_and_grad)
                            # lr_FP16_LODTensor.append(lr)
                            self.lr_dict['FP16_LODTensor'].append(lr)
                            # master weight
                            # if self._multi_precision:
                            #     master_weight_FP16_LODTensor.append(
                            #         self._master_weights[param_and_grad[0]
                            #                              .name])
                            # else:
                            #     master_weight_FP16_LODTensor = None
                            # For fusion of momentum and l2decay 
                            # param = param_and_grad[0]
                            # regularization_method = self._regularization_method
                            # regularization_coeff = self._regularization_coeff
                            # if hasattr(param, 'regularizer'):
                            #     # we skip param's l2decay before, so fuse it with momentum here.
                            #     if isinstance(param.regularizer,
                            #                   L2DecayRegularizer):
                            #         regularization_method = "l2_decay"
                            #         regularization_coeff = param.regularizer._regularization_coeff
                            #     # the param's regularization has been done before, we avoid do l2decay in momentum.
                            #     elif param.regularizer is not None:
                            #         regularization_method = ""
                            #         regularization_coeff = 0
                            # regularization_method_FP16_LODTensor.append(
                            #     regularization_method)
                            # regularization_coeff_FP16_LODTensor.append(
                            #     regularization_coeff)
            else:
                for param_and_grad in parameters_and_grads['params']:
                    if param_and_grad[1] is None:
                        continue
                    if param_and_grad[0].stop_gradient is False:
                        param_grad_dict = dict()
                        param_grad_dict['params'] = param_and_grad
                        param_grad_dict.update({
                            k: v
                            for k, v in parameters_and_grads.items()
                            if k != 'params'
                        })
                        param_and_grad = self._update_param_group(
                            param_grad_dict)
                        self._update_regularization(param_and_grad[
                            'weight_decay'])
                        if param_and_grad[
                                0].dtype == paddle.float32 and param_and_grad[
                                    1].type == core.VarDesc.VarType.LOD_TENSOR:
                            # param
                            # params_FP32_LODTensor.append(param_and_grad[0])
                            # grad
                            # grads_FP32_LODTensor.append(param_and_grad[1])
                            self.grad_dict['FP32_LODTensor'].append(
                                param_and_grad[1])
                            # velocity
                            # velocity_acc = self._create_accumulators(
                            #     target_block, param_and_grad[0])
                            # velocity_acc_FP32_LODTensor.append(velocity_acc)
                            # lr
                            lr = self._create_param_lr(param_and_grad)
                            # lr_FP32_LODTensor.append(lr)
                            self.lr_dict['FP32_LODTensor'].append(lr)
                            # For fusion of momentum and l2decay 
                            # param = param_and_grad[0]
                            # regularization_method = self._regularization_method
                            # regularization_coeff = self._regularization_coeff
                            # if hasattr(param, 'regularizer'):
                            #     # we skip param's l2decay before, so fuse it with momentum here.
                            #     if isinstance(param.regularizer,
                            #                   L2DecayRegularizer):
                            #         regularization_method = "l2_decay"
                            #         regularization_coeff = param.regularizer._regularization_coeff
                            #     # the param's regularization has been done before, we avoid do l2decay in momentum.
                            #     elif param.regularizer is not None:
                            #         regularization_method = ""
                            #         regularization_coeff = 0
                            # regularization_method_FP32_LODTensor.append(
                            #     regularization_method)
                            # regularization_coeff_FP32_LODTensor.append(
                            #     regularization_coeff)
                        elif param_and_grad[
                                0].dtype == paddle.float16 and param_and_grad[
                                    1].type == core.VarDesc.VarType.LOD_TENSOR:
                            # param
                            # params_FP16_LODTensor.append(param_and_grad[0])
                            # grad
                            # grads_FP16_LODTensor.append(param_and_grad[1])
                            self.grad_dict['FP16_LODTensor'].append(
                                param_and_grad[1])
                            # velocity
                            # velocity_acc = self._create_accumulators(
                            #     target_block, param_and_grad[0])
                            # velocity_acc_FP16_LODTensor.append(velocity_acc)
                            # lr
                            lr = self._create_param_lr(param_and_grad)
                            # lr_FP16_LODTensor.append(lr)
                            self.lr_dict['FP16_LODTensor'].append(lr)
                            # master weight
                            # if self._multi_precision:
                            #     master_weight_FP16_LODTensor.append(
                            #         self._master_weights[param_and_grad[0]
                            #                              .name])
                            # else:
                            #     master_weight_FP16_LODTensor = None
                            # For fusion of momentum and l2decay 
                            # param = param_and_grad[0]
                            # regularization_method = self._regularization_method
                            # regularization_coeff = self._regularization_coeff
                            # if hasattr(param, 'regularizer'):
                            #     # we skip param's l2decay before, so fuse it with momentum here.
                            #     if isinstance(param.regularizer,
                            #                   L2DecayRegularizer):
                            #         regularization_method = "l2_decay"
                            #         regularization_coeff = param.regularizer._regularization_coeff
                            #     # the param's regularization has been done before, we avoid do l2decay in momentum.
                            #     elif param.regularizer is not None:
                            #         regularization_method = ""
                            #         regularization_coeff = 0
                            # regularization_method_FP16_LODTensor.append(
                            #     regularization_method)
                            # regularization_coeff_FP16_LODTensor.append(
                            #     regularization_coeff)

        if framework.in_dygraph_mode():
            # if len(params_FP64_LODTensor) > 0:
            #     #print("params_FP64_LODTensor")
            #     _, _, _ = _C_ops.multi_tensor_momentum(
            #         params_FP64_LODTensor, grads_FP64_LODTensor, velocity_acc_FP64_LODTensor, lr_FP64_LODTensor,
            #         None, params_FP64_LODTensor, velocity_acc_FP64_LODTensor, None,
            #         'mu', self._momentum, 'use_nesterov', self._use_nesterov,
            #         'regularization_methods', regularization_method_FP64_LODTensor,
            #         'regularization_coeffs', regularization_coeff_FP64_LODTensor, 'multi_precision',
            #         False)
            print("========>")
            print(self.param_dict['FP32_LODTensor'])
            multi_tensor_list = ['FP32_LODTensor', 'FP16_LODTensor']
            for key in multi_tensor_list:
                if len(self.param_dict[key]) > 0:
                    if key == 'FP32_LODTensor':
                        self._multi_precision = False
                    self.param_dict[key], self.velocity_dict[
                        key], self.master_weight_dict[
                            key] = _C_ops.multi_tensor_momentum(
                                self.param_dict[key], self.grad_dict[key],
                                self.velocity_dict[key], self.lr_dict[key],
                                self.master_weight_dict[key],
                                self.param_dict[key], self.velocity_dict[key],
                                self.master_weight_dict[key], 'mu',
                                self._momentum, 'use_nesterov',
                                self._use_nesterov, 'regularization_methods',
                                self.regularization_method_dict[key],
                                'regularization_coeffs',
                                self.regularization_coeff_dict[key],
                                'multi_precision', self._multi_precision)

            # if len(params_FP32_LODTensor) > 0:
            #     #print("params_FP32_LODTensor")
            #     _, _, _ = _C_ops.multi_tensor_momentum(
            #         params_FP32_LODTensor, grads_FP32_LODTensor,
            #         velocity_acc_FP32_LODTensor, lr_FP32_LODTensor, None,
            #         params_FP32_LODTensor, velocity_acc_FP32_LODTensor, None,
            #         'mu', self._momentum, 'use_nesterov', self._use_nesterov,
            #         'regularization_methods',
            #         regularization_method_FP32_LODTensor,
            #         'regularization_coeffs',
            #         regularization_coeff_FP32_LODTensor, 'multi_precision',
            #         False)
            # if len(params_FP16_LODTensor) > 0:
            #     #print("params_FP16_LODTensor")
            #     _, _, _ = _C_ops.multi_tensor_momentum(
            #         params_FP16_LODTensor, grads_FP16_LODTensor,
            #         velocity_acc_FP16_LODTensor, lr_FP16_LODTensor,
            #         master_weight_FP16_LODTensor, params_FP16_LODTensor,
            #         velocity_acc_FP16_LODTensor, master_weight_FP16_LODTensor,
            #         'mu', self._momentum, 'use_nesterov', self._use_nesterov,
            #         'regularization_methods',
            #         regularization_method_FP16_LODTensor,
            #         'regularization_coeffs',
            #         regularization_coeff_FP16_LODTensor, 'multi_precision',
            #         self._multi_precision)

            return None

    def _update_param_group(self, parameters):
        self._momentum = parameters.get('momentum',
                                        self._default_dict['momentum'])
        self._use_nesterov = parameters.get('use_nesterov',
                                            self._default_dict['use_nesterov'])
        self._rescale_grad = parameters.get('rescale_grad',
                                            self._default_dict['rescale_grad'])
        self._regularization_method = parameters.get(
            'regularization_methods',
            self._default_dict['regularization_methods'])
        self._regularization_coeff = parameters.get(
            'regularization_coeffs',
            self._default_dict['regularization_coeffs'])
        parameters = parameters.get('params')
        return parameters
