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
from ..fluid import core, layers
from ..fluid import framework
from ..fluid.framework import Variable
from ..fluid.regularizer import append_regularization_ops
from ..fluid.clip import append_gradient_clip_ops

__all__ = ["Lamb"]


class Lamb(Optimizer):
    """
    LAMB (Layer-wise Adaptive Moments optimizer for Batching training) Optimizer.

    LAMB Optimizer is designed to scale up the batch size of training without losing
    accuracy, which supports adaptive element-wise updating and accurate layer-wise
    correction. For more information, please refer to `Large Batch Optimization for
    Deep Learning: Training BERT in 76 minutes <https://arxiv.org/abs/1904.00962>`_ .

    The updating of parameters follows:

    ..  math::

        m_t &= \\beta_1 m_{t - 1}+ (1 - \\beta_1)g_t

        v_t &= \\beta_2 v_{t - 1}  + (1 - \\beta_2)g_t^2

        r_t &= \\frac{m_t}{\\sqrt{v_t}+\\epsilon}

        w_t &= w_{t-1} -\\eta_t \\frac{\\left \| w_{t-1}\\right \|}{\\left \| r_t + \\lambda w_{t-1}\\right \|} (r_t + \\lambda w_{t-1})


    where :math:`m` is the 1st moment, and :math:`v` the 2nd moment, :math:`\\eta` the
    learning rate, :math:`\\lambda` the LAMB weight decay rate.

    Args:
        learning_rate (float|Variable, optional): the learning rate used to update parameters. \
            Can be a float value or a Variable with data type float32. Default 0.001.
        lamb_weight_decay (float, optional): The LAMB weight decay rate. Default 0.01. Remind that weight_decay should be None.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates.
            Default 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates.
            Default 0.999.
        epsilon (float, optional): A small float value for numerical stability. Default 1e-6.
        parameters (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name(str|None): For detailed information, please refer to
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.
    Examples:
        .. code-block:: python
            import paddle
            import numpy as np
            inp = paddle.uniform(min=-0.1, max=0.1, shape=[10, 10], dtype='float32')
            linear = paddle.nn.Linear(10, 10)
            out = linear(inp)
            loss = paddle.mean(out)
            beta1 = paddle.to_tensor([0.9], dtype="float32")
            beta2 = paddle.to_tensor([0.85], dtype="float32")
            lamb = paddle.optimizer.Lamb(learning_rate=0.002, parameters=linear.parameters(), lamb_weight_decay=0.01)
            back = out.backward()
            lamb.step()
            lamb.clear_grad()
    """
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    # these two not used in op temporarily
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"

    def __init__(self,
                 learning_rate=0.001,
                 lamb_weight_decay=0.01,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-6,
                 parameters=None,
                 grad_clip=None,
                 exclude_from_weight_decay_fn=None,
                 name=None):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(Lamb, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=None,
            grad_clip=grad_clip,
            name=name)
        self.type = "lamb"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._lamb_weight_decay = lamb_weight_decay
        self._exclude_from_weight_decay_fn = exclude_from_weight_decay_fn

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            self._add_accumulator(self._moment1_acc_str, p)
            self._add_accumulator(self._moment2_acc_str, p)
            self._add_accumulator(
                name=self._beta1_pow_acc_str,
                param=p,
                fill_value=0.9 if isinstance(self._beta1, Variable) \
                        else self._beta1,
                shape=[1],
                type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')
            self._add_accumulator(
                name=self._beta2_pow_acc_str,
                param=p,
                fill_value=0.999 if isinstance(self._beta2, Variable) \
                        else self._beta2,
                shape=[1],
                type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        block.program._use_lamb = True

        moment1 = self._get_accumulator(self._moment1_acc_str,
                                        param_and_grad[0])
        moment2 = self._get_accumulator(self._moment2_acc_str,
                                        param_and_grad[0])
        beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                              param_and_grad[0])
        beta2_pow_acc = self._get_accumulator(self._beta2_pow_acc_str,
                                              param_and_grad[0])

        if self._exclude_from_weight_decay_fn is not None \
            and self._exclude_from_weight_decay_fn(param_and_grad[0]):
            weight_decay = 0.0
        else:
            weight_decay = self._lamb_weight_decay

        # create the lamb optimize op
        lamb_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad),
                "Moment1": moment1,
                "Moment2": moment2,
                "Beta1Pow": beta1_pow_acc,
                "Beta2Pow": beta2_pow_acc
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "Moment1Out": moment1,
                "Moment2Out": moment2,
                "Beta1PowOut": beta1_pow_acc,
                "Beta2PowOut": beta2_pow_acc
            },
            attrs={
                "beta1": self._beta1,
                "beta2": self._beta2,
                "epsilon": self._epsilon,
                "weight_decay": weight_decay
            },
            stop_gradient=True)

        return lamb_op

    def apply_gradients(self, params_grads):
        params_grads = sorted(params_grads, key=lambda x: x[0].name)

        params_grads = self._clip_grad_by_global_norm(params_grads)

        # 'optimizer(grad_clip)' or 'set_gradient_clip'
        if self._grad_clip is not None:
            params_grads = self._grad_clip(params_grads)
        else:

            params_grads = append_gradient_clip_ops(params_grads)

        # Add regularization if any
        params_grads = append_regularization_ops(params_grads,
                                                 self.regularization)

        optimize_ops = self._create_optimization_pass(params_grads)
        return optimize_ops

    def _clip_grad_by_global_norm(self, params_grads):
        with framework.name_scope('global_norm_clip'):
            sum_square_list = []
            for p, g in params_grads:
                if g is None:
                    continue
                merge_grad = g
                with p.block.program._optimized_guard([p, g]):
                    if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                        merge_grad = layers.merge_selected_rows(g)
                        merge_grad = layers.get_tensor_from_selected_rows(
                            merge_grad)

                    square = layers.square(merge_grad)
                    sum_square = layers.reduce_sum(input=square)
                    sum_square_list.append(sum_square)
            if len(sum_square_list) == 0:
                return params_grads

            with p.block.program._optimized_guard([p, g]):
                global_norm_var = layers.sums(sum_square_list)
                global_norm_var = layers.sqrt(x=global_norm_var)
                max_global_norm = layers.fill_constant(
                    shape=[1], dtype=global_norm_var.dtype, value=1.0)
                scale_var = layers.elementwise_div(
                    x=max_global_norm,
                    y=layers.elementwise_max(
                        x=max_global_norm, y=global_norm_var))

            for p, g in params_grads:
                if g is None:
                    continue

                with p.block.program._optimized_guard([p, g]):
                    new_grad = layers.elementwise_mul(x=g, y=scale_var)
                    layers.assign(new_grad, g)
        return params_grads
