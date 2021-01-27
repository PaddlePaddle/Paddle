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
from .adam import Adam
from ..fluid import framework
from ..fluid.dygraph import base as imperative_base
import paddle

__all__ = ['AdamW']


class AdamW(Adam):
    r"""
    The AdamW optimizer is implemented based on the AdamW Optimization
    in paper `DECOUPLED WEIGHT DECAY REGULARIZATION <https://arxiv.org/pdf/1711.05101.pdf>`_.
    it can resolves the problem of L2 regularization failure in the Adam optimizer.

    .. math::

        t & = t + 1

        moment\_1\_out & = {\\beta}_1 * moment\_1 + (1 - {\\beta}_1) * grad

        moemnt\_2\_out & = {\\beta}_2 * moment\_2 + (1 - {\\beta}_2) * grad * grad

        learning\_rate & = learning\_rate * \\
            \\frac{\sqrt{1 - {\\beta}_2^t}}{1 - {beta}_1^t}

        param\_out & = param - learning\_rate * (\\frac{moment\_1}{\sqrt{moment\_2} + \epsilon} + \lambda * param)


    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a LRScheduler. The default value is 0.001.
	parameters (list, optional): List of ``Tensor`` names to update to minimize ``loss``. \
	    This parameter is required in dygraph mode. \
	    The default value is None in static mode, at this time all parameters will be updated.
        beta1 (float|Tensor, optional): The exponential decay rate for the 1st moment estimates.
            It should be a float number or a Tensor with shape [1] and data type as float32.
            The default value is 0.9.
        beta2 (float|Tensor, optional): The exponential decay rate for the 2nd moment estimates.
            It should be a float number or a Tensor with shape [1] and data type as float32.
            The default value is 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            The default value is 1e-08.
        weight_decay (float|Tensor, optional): The weight decay coefficient, it can be float or Tensor. The default value is 0.01.
        apply_decay_param_fun (function|None, optional): If it is not None,
            only tensors that makes apply_decay_param_fun(Tensor.name)==True
            will be updated. It only works when we want to specify tensors.
            Default: None.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
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
    **Notes**:
        **Currently, AdamW doesn't support sparse parameter optimization.**

    Examples:
        .. code-block:: python
            
            import paddle

            linear = paddle.nn.Linear(10, 10)
            inp = paddle.rand([10,10], dtype="float32")
            out = linear(inp)
            loss = paddle.mean(out)

            beta1 = paddle.to_tensor([0.9], dtype="float32")
            beta2 = paddle.to_tensor([0.99], dtype="float32")

            adam = paddle.optimizer.AdamW(learning_rate=0.1,
                    parameters=linear.parameters(),
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=0.01)
            out.backward()
            adam.step()
            adam.clear_grad()

    """

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 parameters=None,
                 weight_decay=0.01,
                 apply_decay_param_fun=None,
                 grad_clip=None,
                 lazy_mode=False,
                 multi_precision=False,
                 name=None):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        if not 0 <= beta1 < 1:
            raise ValueError("Invaild value of beta1, expect beta1 in [0,1).")
        if not 0 <= beta2 < 1:
            raise ValueError("Invaild value of beta2, expect beta2 in [0,1).")
        if not 0 <= epsilon:
            raise ValueError("Invaild value of epsilon, expect epsilon >= 0.")
        coeff = weight_decay
        if not isinstance(coeff, float) and \
                not isinstance(coeff, framework.Variable):
            raise TypeError("coeff should be float or Tensor.")
        self._params_name = set()
        self._apply_decay_param_fun = apply_decay_param_fun
        self._coeff = coeff
        self._lr_to_coeff = dict()
        super(AdamW, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            grad_clip=grad_clip,
            name=name,
            lazy_mode=lazy_mode,
            multi_precision=multi_precision)

    def _append_decoupled_weight_decay(self, block, param_and_grad):
        """
        Add decoupled weight decay op.
            parameter = parameter - parameter * coeff * lr

        Args:
            block: block in which variable is to be created
            param_and_grad: (parameters, gradients) pairs,
                the parameters need to decay.
        Raises:
            Exception: The type of coeff and parameter is not consistent.
        """
        param, grad = param_and_grad

        if self._apply_decay_param_fun is not None \
                and not self._apply_decay_param_fun(param.name):
            return

        if isinstance(self._learning_rate, float):
            learning_rate = self._learning_rate
        else:
            # NOTE. We add this function to the _append_optimize_op(),
            # for we must make sure _create_param_lr() be called after
            # optimizer._create_global_learning_rate().
            learning_rate = self._create_param_lr(param_and_grad)

        with block.program._optimized_guard(
            [param, grad]), framework.name_scope('weight decay'):
            self._params_name.add(param.name)

            # If it has been calculated, the result will be reused.
            # NOTE(wangxi): In dygraph mode, apply_gradient will be executed
            # every step, so need clear _lr_to_coeff every step,
            # we do this in _create_optimization_pass
            decay_coeff = self._lr_to_coeff.get(learning_rate, None)
            if decay_coeff is None:
                decay_coeff = 1.0 - learning_rate * self._coeff
                self._lr_to_coeff[learning_rate] = decay_coeff

            scaled_param = param * decay_coeff
            paddle.fluid.layers.assign(input=scaled_param, output=param)

    def _append_optimize_op(self, block, param_and_grad):
        self._append_decoupled_weight_decay(block, param_and_grad)
        return super(AdamW, self)._append_optimize_op(block, param_and_grad)

    def _create_optimization_pass(self, parameters_and_grads):
        optimize_ops = super(
            AdamW, self)._create_optimization_pass(parameters_and_grads)
        # In dygraph mode, clear _lr_to_coeff after applied gradient
        self._lr_to_coeff = dict()
        return optimize_ops

    def __str__(self):
        return " ".join(["Weight Decay, params:", ",".join(self._params_name)])
