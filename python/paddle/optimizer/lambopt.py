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
from ..fluid import core, framework, layers
from ..fluid.clip import append_gradient_clip_ops, _correct_clip_op_role_var
from ..fluid.framework import Variable
from ..fluid.initializer import Constant
from ..tensor.math import multiply, pow, sqrt
from ..tensor.linalg import norm
from ..tensor.logic import greater_than
from ..tensor.search import where
from ..fluid.regularizer import append_regularization_ops
from ..tensor.creation import assign, ones, zeros

__all__ = ["LAMBOptimizer"]


class LAMBOptimizer(Optimizer):
    r"""
    A LAMB optimizer that includes "correct" L2 weight decay.
    """
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 parameters=None,
                 grad_clip=None,
                 exclude_from_weight_decay=None,
                 name=None):
        super(LAMBOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=None,
            grad_clip=grad_clip,
            name=name)
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._weight_decay_rate = weight_decay_rate
        self._exclude_from_weight_decay = exclude_from_weight_decay

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            self._add_accumulator(self._moment1_acc_str, p)
            self._add_accumulator(self._moment2_acc_str, p)
            self._add_accumulator(
                name=self._beta1_pow_acc_str,
                param=p,
                fill_value=0.9 if isinstance(self._beta_1, Variable) \
                        else self._beta_1,
                shape=[1],
                type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')
            self._add_accumulator(
                name=self._beta2_pow_acc_str,
                param=p,
                fill_value=0.999 if isinstance(self._beta_2, Variable) \
                        else self._beta_2,
                shape=[1],
                type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        block.program._use_lamb = True

        m = moment1 = self._get_accumulator(self._moment1_acc_str,
                                            param_and_grad[0])
        v = self._get_accumulator(self._moment2_acc_str, param_and_grad[0])
        beta_1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                               param_and_grad[0])
        beta_2_pow_acc = self._get_accumulator(self._beta2_pow_acc_str,
                                               param_and_grad[0])

        beta_1 = layers.fill_constant(
            dtype='float32', shape=[1], value=self._beta_1, name='lamb_beta_1')
        beta_2 = layers.fill_constant(
            dtype='float32', shape=[1], value=self._beta_2, name='lamb_beta_2')
        epsilon = layers.fill_constant(
            dtype='float32', shape=[1], value=self._epsilon, name='epsilon')

        one = ones(shape=[1]).astype('float32')
        zero = zeros(shape=[1]).astype('float32')

        next_m = multiply(m, beta_1) + multiply(param_and_grad[1], one - beta_1)
        next_v = multiply(v, beta_2) + multiply(
            pow(param_and_grad[1], 2), one - beta_2)

        beta1_correction = one - beta_1_pow_acc
        beta2_correction = one - beta_2_pow_acc

        next_m_unbiased = next_m / beta1_correction
        next_v_unbiased = next_v / beta2_correction

        update = next_m_unbiased / (sqrt(next_v_unbiased) + epsilon)

        if self._exclude_from_weight_decay is not None and self._exclude_from_weight_decay(
                param_and_grad[0]):
            self._weight_decay_rate = 0.0
        update += self._weight_decay_rate * param_and_grad[0]

        w_norm = norm(param_and_grad[0], p=2)
        g_norm = norm(update, p=2)

        learning_rate = self._create_param_lr(param_and_grad)

        ratio = where(
            greater_than(w_norm, zero),
            where(greater_than(g_norm, zero), (w_norm / g_norm), one), one)
        update_with_lr = ratio * learning_rate * update
        next_param = param_and_grad[0] - update_with_lr

        beta_1_pow_acc *= beta_1
        beta_2_pow_acc *= beta_2

        assign(next_m, m)
        assign(next_v, v)
        assign(next_param, param_and_grad[0])

        return None

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
            #params_and_grads = []
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

            param_new_grad_name_dict = dict()
            for p, g in params_grads:
                if g is None:
                    continue

                with p.block.program._optimized_guard([p, g]):
                    new_grad = layers.elementwise_mul(x=g, y=scale_var)
                    layers.assign(new_grad, g)
        return params_grads
