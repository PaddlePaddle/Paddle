# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except jin compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import paddle
import numpy as np
from collections import OrderedDict, defaultdict
from functools import reduce
from paddle.fluid import core, framework
from paddle.fluid.dygraph import parallel_helper, layers
from paddle.regularizer import L1Decay, L2Decay
from paddle.utils import deprecated, unique_name
from paddle.fluid.layers import collective, tensor
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid.initializer import Constant
from paddle import _C_ops
from paddle.distributed.collective import _get_default_group
from paddle.optimizer.lr import LRScheduler
from .optimizer import Optimizer

__all__ = ["DGCMomentumOptimizer"]


@imperative_base.no_grad
@framework.dygraph_only
def sync_params_buffers(params, group=None, src_rank=0):
    if len(params) == 0:
        return

    for param in params:
        if not isinstance(param, core.VarBase):
            raise TypeError("The data type of '%s' must be VarBase" %
                            param.name)
    for param in params:
        paddle.distributed.broadcast(param,
                                     src=src_rank,
                                     group=group,
                                     use_calc_stream=True)


class DGCMomentumOptimizer(Optimizer):

    def __init__(self,
                 parameters,
                 learning_rate=0.01,
                 momentum=0.9,
                 rampup_begin_step=10,
                 rampup_step=1,
                 sparsity=[0.999],
                 use_nesterov=False,
                 weight_decay=None,
                 grad_clip=None,
                 group=None,
                 name=None):
        if learning_rate is None:
            raise ValueError("learning_rate is not set")
        if momentum is None:
            raise ValueError("momentum is not set")

        super(DGCMomentumOptimizer, self).__init__(learning_rate=learning_rate,
                                                   parameters=parameters,
                                                   weight_decay=None,
                                                   grad_clip=grad_clip,
                                                   name=name)
        self._parameters = parameters
        self._momentum = momentum
        self._use_nesterov = use_nesterov
        self._grad_clip = grad_clip
        self.trainable_params = []
        self.reduce_order = 0
        self.reduce_count = -1
        self._dgc_clip_norm = None
        self._learning_rate = learning_rate
        self._global_learning_rate_var = paddle.to_tensor([
            learning_rate
            if isinstance(self._learning_rate, float) else learning_rate()
        ],
                                                          dtype="float32")
        self._not_ready_param = {}
        self._reduce_var_ready = {}

        self._weight_decay = weight_decay
        self._u_velocity_acc_str = "_dgc_u_"
        self._v_velocity_acc_str = "_dgc_v_"
        self._dgc_vars = defaultdict(lambda: list())

        self._reduce_var_index = {}
        self._index_to_param = {}
        self._reduce_grad_res = {}
        self.helper = LayerHelper(self.__class__.__name__)

        ranks = list(range(paddle.distributed.get_world_size())
                     ) if group is None else group.ranks
        self._num_trainers = len(ranks)

        if grad_clip is not None:
            if not isinstance(grad_clip, paddle.nn.ClipGradByNorm):
                raise TypeError(
                    "The type of grad_clip should be 'ClipGradByNorm', because DGCModel only support ClipGradByNorm"
                )

            assert len(ranks) > 0, "dgc only works on multi cards."

            self._dgc_clip_norm = grad_clip.clip_norm * (self._num_trainers**
                                                         -0.5)
        self.regular_type, self.regular_coeff = self._get_regularization_param(
            weight_decay)

        self.new_group = paddle.distributed.new_group(ranks)

        # dgc config
        self._rampup_begin_step = self._add_cpu_var("rampup_begin_step",
                                                    rampup_begin_step)
        self._rampup_step = rampup_step
        self._sparsity = sparsity

        if len(ranks) > 1:
            # check the environment
            assert parallel_helper.__parallel_ctx__clz__ is not None, \
            "ParallelContext must be initialized before. You should use init_parallel_env() before" \
            "constructing the DGC Parallel."

            # sync buffer and params
            sync_params_buffers(self._parameters, group=self.new_group)

            self.init_params()
            self.register_bw_hooks()
        else:
            raise Exception("dgc only works on multi cards.")

        print(
            "  dgc working: rampup_begin_step{}  rampup_step{}  sparsity{}...".
            format(rampup_begin_step, rampup_step, sparsity))

    def register_bw_hooks(self):
        for param in self.trainable_params:
            #NOTE: clip grad and add dgc op in backward hook.
            reduce_function = self._get_reduce_fn(param)
            param.register_hook(reduce_function)

    def _get_reduce_fn(self, pp):

        @imperative_base.no_grad
        def deal_grad(param, grad):
            is_use_dgc = self._is_use_dgc(param, grad)
            clip_var = grad

            # local gradient clipping
            if self._dgc_clip_norm is not None and is_use_dgc and self._global_step_var[
                    0] >= self._rampup_begin_step:
                clip_var = self._clip_by_norm(grad,
                                              max_norm=self._dgc_clip_norm,
                                              name=param.name + "_grad")

            u_var, v_var = self._accumulators[self._u_velocity_acc_str][
                param.name], self._accumulators[self._v_velocity_acc_str][
                    param.name]

            k_var, encoded_var, gather_var = self._get_auxiliary_var(param)

            # momentum correction and local regularization
            self._dgc_op(param, clip_var, grad, u_var, v_var, k_var,
                         encoded_var, gather_var, is_use_dgc)

            self._reduce_grad_res[param.name] = grad

            self.reduce_order += 1

            if self.reduce_order == self.reduce_count:
                _C_ops.dgc_wait_comm([grad], self.new_group.id)

        @imperative_base.no_grad
        def reduce_grad(gg):
            self._reduce_var_ready[pp.name] = 1
            comm_grad = gg.scale(1.0 / self.new_group.nranks)
            order = self._reduce_var_index[pp.name]
            if order == self.reduce_order:
                deal_grad(pp, comm_grad)
                for index in range(self.reduce_order, self.reduce_count):
                    param_name = self._index_to_param[index]
                    if self._reduce_var_ready[
                            param_name] and param_name in self._not_ready_param.keys(
                            ):
                        p, g = self._not_ready_param[param_name]
                        deal_grad(p, g)
                        self._not_ready_param.pop(param_name)
                    else:
                        break
            elif order > self.reduce_order:
                self._not_ready_param[pp.name] = (pp, comm_grad)
            else:
                raise Exception("dgc backward error!")

            return gg

        return reduce_grad

    def _dgc_op(self, param_var, clip_var, grad_var, u_var, v_var, k_var,
                encoded_var, gather_var, is_use_dgc):

        regular_type = self.regular_type
        regular_coeff = self.regular_coeff
        # The regularizer of the Parameters have higher priority
        if param_var.regularizer is not None:
            regular_type, regular_coeff = self._get_regularization_param(
                param_var.regularizer)

        paddle.fluid.framework._dygraph_tracer().trace_op(
            type="dgc_fuse",
            inputs={
                "U": u_var,
                "V": v_var,
                "Grad": clip_var,
                "Param": param_var,
                "current_step": self._global_step_var,
                "nranks": self._nranks_var,
            },
            outputs={
                "U_out": u_var,
                "V_out": v_var,
                "EncodeGrad": encoded_var,
                "k": k_var,
                "Grad_out": grad_var,
                "GatherBuff": gather_var,
            },
            attrs={
                "m": self._momentum,
                "sparsity": self._sparsity,
                "use_nesterov": self._use_nesterov,
                "rampup_begin_step": float(self._rampup_begin_step),
                "rampup_step": float(self._rampup_step),
                "regular_coeff": float(regular_coeff),
                "regular_type": int(regular_type),
                "is_use_dgc": is_use_dgc
            })

    def _clip_by_norm(self, x, max_norm, name=None):
        args = {'x': x, 'max_norm': max_norm, 'name': name}

        helper = LayerHelper("dgc_clip_by_norm_op", **args)

        if name is None:
            name = unique_name.generate_with_ignorable_key(".".join(
                [helper.name, 'tmp']))

        out = helper.create_variable(type=x.type,
                                     name=name,
                                     dtype=x.dtype,
                                     persistable=False)
        with paddle.no_grad():
            paddle.fluid.framework._dygraph_tracer().trace_op(
                type="dgc_clip_by_norm",
                inputs={
                    "X": x,
                    "current_step": self._global_step_var
                },
                attrs={
                    "max_norm": max_norm,
                    "rampup_begin_step": float(self._rampup_begin_step),
                },
                outputs={"Out": out})
        return out

    def _global_learning_rate(self):
        if isinstance(self._learning_rate, paddle.optimizer.lr.LRScheduler):
            return paddle.to_tensor([float(self._learning_rate())])

        return self._global_learning_rate_var

    def _create_regularization_of_grad(self, param, grad):
        """ Create and add backward regularization Operators
    
        Function helper of append_regularization_ops.
        """
        if isinstance(self._weight_decay, float):
            regularization = L2Decay(self._weight_decay)
        else:
            regularization = self._weight_decay

        # If no gradient or no regularization is specified,  then we don't need to do anything
        if grad is None or (
            (not hasattr(param, 'regularizer') or
             (hasattr(param, 'regularizer') and param.regularizer is None))
                and regularization is None):
            return grad
        regularization_term = None
        if hasattr(param, 'regularizer') and param.regularizer is not None:
            # Add variable for regularization term in grad block
            regularization_term = param.regularizer(param, grad, grad.block)
        elif regularization is not None:
            regularization_term = regularization(param, grad, grad.block)
        assert regularization_term is not None

        return _C_ops.sum([grad, regularization_term])

    def init_params(self):
        self.trainable_params = [p for p in self._parameters if p.trainable]
        num_trainable_params = len(self.trainable_params)

        for index, param in enumerate(self.trainable_params):
            self._reduce_var_ready[param.name] = 0
            self._reduce_var_index[
                param.name] = num_trainable_params - index - 1
            self._index_to_param[num_trainable_params - index - 1] = param.name

            self._reduce_grad_res[param.name] = paddle.zeros_like(
                param, dtype=param.dtype)

        self.trainable_params.reverse()

        assert len(self.trainable_params) == len(
            self._reduce_var_ready
        ), "there are at least two params with same name."

        assert len( self.trainable_params) > 0, \
            "This model does not have any parameters to train, and " \
            "does not need to use DataParallel"

        self.reduce_count = len(self.trainable_params)

        # step counter
        self._global_step_var = self._add_cpu_var(name="current_step", value=0)

        self._nranks_var = self._add_cpu_var(name="nranks",
                                             value=self._num_trainers)
        for param_var in self.trainable_params:
            # reuse velocity in dgc_op and dgc_momentum_op
            self._add_accumulator(self._u_velocity_acc_str, param_var)

            if not self._is_use_dgc(param_var, None):
                self._accumulators[self._v_velocity_acc_str][
                    param_var.name] = paddle.to_tensor([1], dtype=param.dtype)
                continue

            self._add_accumulator(self._v_velocity_acc_str, param_var)

    def _add_cpu_var(self, name, value=-1):
        helper = LayerHelper('global_step_counter')
        counter, is_new_var = helper.create_or_get_global_variable(
            name=name, dtype='float32', shape=[1], persistable=True)
        if is_new_var:
            helper.set_variable_initializer(counter,
                                            initializer=Constant(
                                                value=float(value),
                                                force_cpu=True))
            counter.stop_gradient = True
        return counter

    @imperative_base.no_grad
    def _clear_counters(self):
        """Reset all the grad reduce and call counters."""
        self.reduce_order = 0
        self._not_ready_param.clear()
        for k in self._reduce_var_ready.keys():
            self._reduce_var_ready[k] = 0

    def _is_use_dgc(self, param, grad):
        param_var, grad_var = param, grad
        var_numel = abs(reduce(lambda x, y: x * y, param_var.shape))
        if var_numel < 16384 or \
            param_var.type == core.VarDesc.VarType.SELECTED_ROWS  or \
            param_var.dtype != core.VarDesc.VarType.FP32 or \
            (grad_var is not None and grad_var.type == core.VarDesc.VarType.SELECTED_ROWS):
            return False
        return True

    def _get_regularization_param(self, regularization):
        regular_type = 0
        regular_coeff = 0.0

        if regularization is not None:
            if isinstance(regularization, float):
                regular_type = 2
                regular_coeff = regularization
                return regular_type, regular_coeff
            else:
                regular_coeff = regularization._regularization_coeff
                if isinstance(regularization, L1Decay):
                    regular_type = 1
                elif isinstance(regularization, L2Decay):
                    regular_type = 2
                else:
                    assert False, 'regularization must be None|L1Decay|L2Deacy'

        return regular_type, regular_coeff

    def _get_auxiliary_var(self, param):
        if param.name in self._dgc_vars.keys():
            return self._dgc_vars[param.name]
        k_var = self._add_cpu_var(name="k", value=0)
        encoded_var = paddle.to_tensor([0.0], dtype=param.dtype)
        gather_var = paddle.to_tensor([0.0], dtype=param.dtype)
        self._dgc_vars[param.name] = [k_var, encoded_var, gather_var]

        return k_var, encoded_var, gather_var

    def clear_grad(self, set_to_zero=True):
        core.clear_gradients(self.trainable_params, set_to_zero)

    @imperative_base.no_grad
    def step(self):
        """
           execute after loss.backward.
        """
        lr = self._global_learning_rate()

        for param in self.trainable_params:
            grad = self._reduce_grad_res[param.name]
            velocity_acc = self._accumulators[self._u_velocity_acc_str][
                param.name]

            assert velocity_acc is not None

            if not self._is_use_dgc(param, grad):
                if self._grad_clip is not None:
                    param, grad = self._grad_clip([(param, grad)])[0]
                else:
                    clip_attr = getattr(param, 'gradient_clip_attr', None)
                    assert clip_attr is None, "param does not support gradient_clip_attr in DGC now, you can globally clip in optimizer."
                #regularation
                grad = self._create_regularization_of_grad(param, grad)

                inputs = {
                    "Param": param,
                    "Grad": grad,
                    "Velocity": velocity_acc,
                    "LearningRate": lr,
                }
                outputs = {
                    "ParamOut": param,
                    "VelocityOut": velocity_acc,
                }
                attrs = {
                    "mu": self._momentum,
                    "use_nesterov": self._use_nesterov
                }
                _, _, _ = _C_ops.momentum(param, grad, velocity_acc, lr, (None),
                                          param, velocity_acc, (None), 'mu',
                                          self._momentum, 'use_nesterov',
                                          self._use_nesterov)
            else:
                opt_type = "dgc_momentum"
                inputs = {
                    "Param": param,
                    "Grad": grad,
                    "Velocity": velocity_acc,
                    "LearningRate": lr,
                    "current_step": self._global_step_var,
                    "nranks": self._nranks_var
                }
                outputs = {
                    "ParamOut": param,
                    "VelocityOut": velocity_acc,
                    'Grad_out': grad
                }
                attrs = {
                    "mu": self._momentum,
                    "use_nesterov": self._use_nesterov,
                    "rampup_begin_step": float(self._rampup_begin_step)
                }

                with paddle.no_grad():
                    paddle.fluid.framework._dygraph_tracer().trace_op(
                        type=opt_type,
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs)

        self._global_step_var = _C_ops.increment(self._global_step_var, 'step',
                                                 1)
        self._clear_counters()

    @framework.dygraph_only
    def state_dict(self):
        state_dict = {}
        for k, v in self._accumulators.items():
            state_dict[k] = v

        # global step if use lr decay
        if isinstance(self._learning_rate, LRScheduler):
            state_dict["LR_Scheduler"] = self._learning_rate.state_dict()
        state_dict["global_step"] = self._global_step_var.item()
        return state_dict

    @framework.dygraph_only
    def set_state_dict(self, state_dict):
        if isinstance(self._learning_rate, LRScheduler):
            self._learning_rate.set_dict(state_dict["LR_Scheduler"])

        if isinstance(self._learning_rate, LRScheduler):
            self._learning_rate.set_state_dict(state_dict["LR_Scheduler"])
            self._global_learning_rate_var[0] = float(self._learning_rate())

        # NOTE: exclude learning rate scheduler's state from
        # _accumulators_holder.
        state_dict = state_dict.copy()
        if "LR_Scheduler" in state_dict:
            state_dict.pop("LR_Scheduler")
        if "global_step" in state_dict:
            self._global_step_var[0] = state_dict["global_step"]
            state_dict.pop("global_step")

        for k, v in state_dict.items():
            self._accumulators[k] = v
