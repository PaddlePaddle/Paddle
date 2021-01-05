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
from ..fluid.framework import program_guard
from ..fluid.regularizer import append_regularization_ops
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
        super(AdamW, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            grad_clip=grad_clip,
            name=name,
            lazy_mode=lazy_mode)

    def _decoupled_weight_decay(self, params_grads):
        """
        Adds decoupled weight decay ops.
            parameter = parameter - parameter * coeff * lr

        Args:
            params_grads: A list of (parameters, gradients) pairs,
                the parameters need to decay.
        Raises:
            Exception: The type of coeff and parameter is not consistent.
        """

        for param, grad in params_grads:
            # If no gradient then we don't need to do anything
            if grad is None:
                continue

            if self._apply_decay_param_fun is not None \
                    and not self._apply_decay_param_fun(param.name):
                continue

            if isinstance(self._coeff, float):
                assert param.dtype == paddle.fluid.core.VarDesc.VarType.FP32, \
                    "the type of coeff(float) and parameter(%s) is not consistent."%(param.dtype)
            else:
                assert self._coeff.dtype == param.dtype, \
                    "the type of coeff(%s) and parameter(%s) is not consistent."%(self._coeff.dtype, param.dtype)
            if isinstance(self._learning_rate, float):
                learning_rate = self._learning_rate
            else:
                learning_rate = self._learning_rate()

            with param.block.program._optimized_guard(
                [param, grad]), framework.name_scope('weight decay'):
                decay_coeff = 1.0 - self._coeff * learning_rate
                scaled_param = param * decay_coeff
                paddle.fluid.layers.assign(input=scaled_param, output=param)

    def apply_gradients(self, params_grads):
        """
        Second part of `minimize`, appending optimization operators for
        given `params_grads` pairs.

        Args:
            params_grads (list): list of (param, grad) pair to do optimization.

        Returns:
            list: A list of operators appended to the current program.

        Examples:
            .. code-block:: python

                import paddle
                import numpy as np

                inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
                linear = paddle.nn.Linear(10, 10)
                inp = paddle.to_tensor(inp)
                out = linear(inp)
                loss = paddle.mean(out)
                optimizer = paddle.optimizer.Adam(learning_rate=0.1,
                        parameters=linear.parameters())
                params_grads = optimizer.backward(loss)
                optimizer.apply_gradients(params_grads)

        """
        self._decoupled_weight_decay(params_grads)
        return super(AdamW, self).apply_gradients(params_grads)

    def _apply_optimize(self, loss, startup_program, params_grads):
        """
        Second part of `minimize`, appending optimization operators for
        given `params_grads` pairs.
        Args:
            loss (Tensor): loss tensor to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameters`.
            params_grads (list): list of (param, grad) pair to do optimization.
        Returns:
            list: A list of operators appended to the current program.
        """
        if framework.in_dygraph_mode():
            with program_guard(framework.default_main_program(),
                               framework.default_startup_program()):
                self._decoupled_weight_decay(params_grads)

                if self._grad_clip is not None:
                    params_grads = self._grad_clip(params_grads)
                params_grads = append_regularization_ops(params_grads,
                                                         self.regularization)
                optimize_ops = self._create_optimization_pass(params_grads)
        else:
            program = loss.block.program
            with program_guard(program, startup_program):
                optimize_ops = self.apply_gradients(params_grads)
        return optimize_ops

    def __str__(self):
        return " ".join(["Weight Decay, params:", ",".join(self._params_name)])
