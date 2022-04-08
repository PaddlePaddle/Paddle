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

from paddle.fluid import framework, core, layers, unique_name
from paddle.fluid.framework import Variable
from paddle.fluid.clip import ClipGradByGlobalNorm
from paddle.fluid.initializer import Constant
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.optimizer import Optimizer
from paddle.distributed import get_rank, get_world_size
from paddle.fluid.executor import global_scope
from paddle.fluid.framework import name_scope
import numpy as np


class DistributedFusedLamb(Optimizer):
    def __init__(self,
                 learning_rate=0.001,
                 lamb_weight_decay=0.01,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-6,
                 parameters=None,
                 grad_clip=None,
                 exclude_from_weight_decay_fn=None,
                 clip_after_allreduce=True,
                 is_grad_scaled_by_nranks=True,
                 alignment=128,
                 use_master_param_norm=True,
                 name=None):
        assert not framework._non_static_mode(
        ), "DistributedFusedLamb does not support dygraph mode"
        super(DistributedFusedLamb, self).__init__(
            learning_rate=learning_rate, grad_clip=None, name=name)

        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._weight_decay = lamb_weight_decay if lamb_weight_decay is not None else 0.0
        if grad_clip is not None:
            assert isinstance(
                grad_clip, ClipGradByGlobalNorm
            ), "Only ClipGradByGlobalNorm is supported in DistributedFusedLamb"
            max_global_grad_norm = grad_clip.clip_norm
        else:
            max_global_grad_norm = -1.0
        self._max_global_grad_norm = max_global_grad_norm
        self._alignment = alignment if alignment is not None else -1
        self._clip_after_allreduce = clip_after_allreduce
        self._is_grad_scaled_by_nranks = is_grad_scaled_by_nranks
        self._exclude_from_weight_decay_fn = exclude_from_weight_decay_fn
        self._scale = None
        self._ring_id = 0
        self._use_master_param_norm = use_master_param_norm
        self.helper = LayerHelper('distributed_fused_lamb')
        self._supports_check_nan_inf = True  # very import flag for AMP

        main_block = self.helper.main_program.global_block()
        self._found_inf = main_block.create_var(
            name=unique_name.generate('found_inf'),
            shape=[1],
            dtype=core.VarDesc.VarType.BOOL)
        self._step = None

        self._param_to_master_param = {}

    def _set_step(self, step):
        self._step = step

    def _get_or_create_step(self):
        if self._step is None:
            self._step = self._create_persistable_var('step', dtype='int64')
        return self._step

    def _set_scale(self, scale):
        assert scale is not None
        if not isinstance(scale, Variable):
            scale = self._create_scale_from_constant(scale)
        self._scale = scale

    def _create_scale_from_constant(self, value):
        name = unique_name.generate('global_scale')
        return layers.create_global_var(
            name=name,
            shape=[1],
            dtype='float32',
            value=float(value),
            persistable=True)

    def _get_or_create_scale(self):
        if self._scale is None:
            self._scale = self._create_scale_from_constant(1.0)
        return self._scale

    def _create_persistable_var(self, name=None, shape=[-1], dtype='float32'):
        startup_block = self.helper.startup_program.global_block()
        if name is not None:
            name = unique_name.generate(name)
        startup_var = startup_block.create_var(
            name=name,
            shape=shape,
            dtype=dtype,
            persistable=True,
            stop_gradient=True)
        main_block = self.helper.main_program.global_block()
        main_var = main_block.create_var(
            name=startup_var.name,
            shape=startup_var.shape,
            dtype=startup_var.dtype,
            persistable=True,
            stop_gradient=True)
        return main_var

    def _get_parameter(self, name, scope=None):
        if scope is None:
            scope = global_scope()

        master_param = self._param_to_master_param.get(name)
        assert master_param is not None

        master_param_t = scope.find_var(master_param).get_tensor()
        assert master_param_t._dtype() == core.VarDesc.VarType.FP32

        param_t = scope.find_var(name).get_tensor()
        if param_t._dtype() == core.VarDesc.VarType.FP32:
            assert param_t._ptr() == master_param_t._ptr()
            return param_t, None
        else:
            assert param_t._dtype() == core.VarDesc.VarType.FP16
            assert param_t.shape() == master_param_t.shape()
            return param_t, master_param_t

    def apply_optimize(self, params_grads):
        self.apply_gradients(params_grads)

    def apply_gradients(self, params_grads):
        flattened = []
        for p, g in params_grads:
            flattened.extend([p, g])
        with flattened[0].block.program._optimized_guard(flattened), name_scope(
                "optimizer"):
            self._apply_gradients_impl(params_grads)

    def _apply_gradients_impl(self, params_grads):
        for p, g in params_grads:
            assert g.type == core.VarDesc.VarType.LOD_TENSOR, "Only support dense gradient"
            g.persistable = True  # the gradient must be persistable for fusion

        fp32_fused_param = self._create_persistable_var('fp32_fused_param')
        fp32_fused_grad = self._create_persistable_var('fp32_fused_grad')
        fp16_fused_param = self._create_persistable_var(
            'fp16_fused_param', dtype='float16')
        fp16_fused_grad = self._create_persistable_var(
            'fp16_fused_grad', dtype='float16')

        master_params = []
        for p, g in params_grads:
            master_p = self._create_persistable_var('master_weight')
            self._param_to_master_param[p.name] = master_p.name
            master_params.append(master_p)

        moment1 = self._create_persistable_var('moment1')
        moment1.is_distributed = True
        moment2 = self._create_persistable_var('moment2')
        moment2.is_distributed = True
        beta1pow = self._create_persistable_var('beta1pow')
        beta2pow = self._create_persistable_var('beta2pow')

        param_info = self._create_persistable_var('param_info', dtype='int32')
        param_info.is_distributed = True

        fused_offsets = self._create_persistable_var(
            'fused_offsets', dtype='int32')

        fp32_partial_fused_offsets = self._create_persistable_var(
            'fp32_partial_fused_offsets', dtype='int32')
        fp32_partial_fused_offsets.is_distributed = True

        fp16_partial_fused_offsets = self._create_persistable_var(
            'fp16_partial_fused_offsets', dtype='int32')
        fp16_partial_fused_offsets.is_distributed = True

        param_order = self._create_persistable_var('param_order', dtype='int32')
        param_order.is_distributed = True

        step = self._get_or_create_step()

        rank = get_rank()
        nranks = get_world_size()
        scale = self._get_or_create_scale()

        params = [p for p, _ in params_grads]
        grads = [g for _, g in params_grads]
        apply_weight_decay = [1] * len(params)
        if self._exclude_from_weight_decay_fn is not None:
            for i, p in enumerate(params):
                if self._exclude_from_weight_decay_fn(p):
                    apply_weight_decay[i] = 0

        startup_block = self.helper.startup_program.global_block()
        for g in grads:
            startup_block.create_var(
                name=g.name,
                type=g.type,
                dtype=g.dtype,
                persistable=g.persistable,
                shape=g.shape)

        startup_block.append_op(
            type='distributed_fused_lamb_init',
            inputs={
                'Param': params,
                'Grad': grads,
            },
            outputs={
                'FP32FusedParam': [fp32_fused_param],
                'FP32FusedGrad': [fp32_fused_grad],
                'FP16FusedParam': [fp16_fused_param],
                'FP16FusedGrad': [fp16_fused_grad],
                'Moment1': [moment1],
                'Moment2': [moment2],
                'Beta1Pow': [beta1pow],
                'Beta2Pow': [beta2pow],
                'GlobalScale': [scale],
                'ParamInfo': [param_info],
                'ParamOut': params,
                'MasterParamOut': master_params,
                'GradOut': grads,
                'FP32ShardFusedParamOffsets': [fp32_partial_fused_offsets],
                'FP16ShardFusedParamOffsets': [fp16_partial_fused_offsets],
                'FusedParamOffsets': [fused_offsets],
                'ParamOrder': [param_order],
                'Step': [step],
            },
            attrs={
                'alignment': self._alignment,
                'rank': rank,
                'nranks': nranks,
                'apply_weight_decay': apply_weight_decay,
                'moment1': 0.0,
                'moment2': 0.0,
                'beta1': self._beta1,
                'beta2': self._beta2,
            })

        main_block = self.helper.main_program.global_block()
        self._create_global_learning_rate()
        lr = None
        for p_g in params_grads:
            if lr is None:
                lr = self._create_param_lr(p_g)
            else:
                new_lr = self._create_param_lr(p_g)
                assert id(lr) == id(
                    new_lr
                ), "The learning rate for each parameter should be the same"
        assert lr is not None

        lamb_op = main_block.append_op(
            type='distributed_fused_lamb',
            inputs={
                'FP32FusedParam': [fp32_fused_param],
                'FP32FusedGrad': [fp32_fused_grad],
                'FP16FusedParam': [fp16_fused_param],
                'FP16FusedGrad': [fp16_fused_grad],
                'LearningRate': [lr],
                'Moment1': [moment1],
                'Moment2': [moment2],
                'Beta1Pow': [beta1pow],
                'Beta2Pow': [beta2pow],
                'GlobalScale': [scale],
                'ParamInfo': [param_info],
                'Param': params,
                'Grad': grads,
                'FusedParamOffsets': [fused_offsets],
                'FP32ShardFusedParamOffsets': [fp32_partial_fused_offsets],
                'FP16ShardFusedParamOffsets': [fp16_partial_fused_offsets],
                'ParamOrder': [param_order],
            },
            outputs={
                'FP32FusedParamOut': [fp32_fused_param],
                'FP16FusedParamOut': [fp16_fused_param],
                'Moment1Out': [moment1],
                'Moment2Out': [moment2],
                'Beta1PowOut': [beta1pow],
                'Beta2PowOut': [beta2pow],
                'ParamOut': params,
                'GradOut': grads,
                'FoundInf': [self._found_inf],
                'Step': [step],
            },
            attrs={
                'weight_decay': self._weight_decay,
                'beta1': self._beta1,
                'beta2': self._beta2,
                'epsilon': self._epsilon,
                'max_global_grad_norm': self._max_global_grad_norm,
                'clip_after_allreduce': self._clip_after_allreduce,
                'rank': rank,
                'ring_id': self._ring_id,
                'use_master_param_norm': self._use_master_param_norm,
                'is_grad_scaled_by_nranks': self._is_grad_scaled_by_nranks,
            })
        return [lamb_op]
