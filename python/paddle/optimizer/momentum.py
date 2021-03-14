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
from ..fluid.framework import Variable, name_scope
from ..fluid.layer_helper import LayerHelper
from ..fluid import unique_name
from ..fluid import layers
import paddle.fluid as fluid
from paddle.fluid.regularizer import L2DecayRegularizer
__all__ = ["Momentum"]


class Momentum(Optimizer):
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
        parameters (list, optional): List of ``Tensor`` to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
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
        py_regular = None if predicate(weight_decay) else weight_decay
        super(Momentum, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=py_regular,
            grad_clip=grad_clip,
            name=name)
        self.type = "momentum"
        self._momentum = momentum
        self._use_nesterov = bool(use_nesterov)
        self._regularization_method = ""
        self._regularization_coeff = 0
        if (isinstance(weight_decay, L2DecayRegularizer)):
            self._regularization_method = "l2_decay"
            self._regularization_coeff = weight_decay._regularization_coeff
        if (isinstance(weight_decay, float)):
            self._regularization_method = "l2_decay"
            self._regularization_coeff = weight_decay
        self._multi_precision = multi_precision
        self._rescale_grad = rescale_grad
        self._master_weights = {}

        if framework.in_dygraph_mode():
            self.helper = LayerHelper(self.__class__.__name__)
            for p in parameters:
                self._add_accumulator(self._velocity_acc_str, p)

    def _create_master_weight(self, param):
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

    def _create_accumulators(self, block, parameters):
        if framework.in_dygraph_mode():
            return

        assert isinstance(block, framework.Block)
        for p in parameters:
            if self._multi_precision and p.dtype == core.VarDesc.VarType.FP16:
                master_p = self._create_master_weight(p)
                self._add_accumulator(self._velocity_acc_str, master_p)
                continue
            if p.dtype == core.VarDesc.VarType.FP16 and not self._multi_precision:
                warnings.warn(
                    "Accumulating with FP16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Momentum optimizer."
                )
            self._add_accumulator(self._velocity_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        velocity_acc = self._get_accumulator(self._velocity_acc_str,
                                             param_and_grad[0])
        lr = self._create_param_lr(param_and_grad)

        if framework.in_dygraph_mode():
            _, _ = core.ops.momentum(
                param_and_grad[0], param_and_grad[1], velocity_acc, lr,
                param_and_grad[0], velocity_acc, 'mu', self._momentum,
                'use_nesterov', self._use_nesterov, 'regularization_method',
                self._regularization_method, 'regularization_coeff',
                self._regularization_coeff)
            return None

        find_master = self._multi_precision and param_and_grad[
            0].dtype == core.VarDesc.VarType.FP16
        master_weight = (self._master_weights[param_and_grad[0].name]
                         if find_master else None)

        attrs = {
            "mu": self._momentum,
            "use_nesterov": self._use_nesterov,
            "regularization_method": self._regularization_method,
            "regularization_coeff": self._regularization_coeff,
            "multi_precision": find_master,
            "rescale_grad": self._rescale_grad
        }

        inputs = {
            "Param": [param_and_grad[0]],
            "Grad": [param_and_grad[1]],
            "Velocity": [velocity_acc],
            "LearningRate": [lr]
        }

        outputs = {
            "ParamOut": [param_and_grad[0]],
            "VelocityOut": [velocity_acc]
        }

        if find_master:
            inputs["MasterParam"] = master_weight
            outputs["MasterParamOut"] = master_weight

        # create the momentum optimize op
        momentum_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return momentum_op
