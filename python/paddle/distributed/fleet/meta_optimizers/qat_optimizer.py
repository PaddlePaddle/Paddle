# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import copy

import paddle
from paddle.static.quantization.quanter import (
    _quant_config_default,
    quant_aware,
)

from .meta_optimizer_base import MetaOptimizerBase


class QATOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = [
            "AMPOptimizer",
            "LarsOptimizer",
            "LambOptimizer",
            "GraphExecutionOptimizer",
            "RecomputeOptimizer",
            "GradientMergeOptimizer",
        ]
        self.meta_optimizers_black_list = []

    def _set_basic_info(
        self, loss, role_maker, user_defined_optimizer, user_defined_strategy
    ):
        super()._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy
        )

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

        if self.user_defined_strategy.qat:
            return True

        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.qat = False
        dist_strategy.qat_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        dist_strategy.qat = True
        dist_strategy.qat_configs = {
            'channel_wise_abs_max': True,
            'weight_bits': 8,
            'activation_bits': 8,
            'not_quant_pattern': [],
            'algo': "",
        }

    def _gen_qat_config(self):
        # Align the config to auto_parallel quantization pass
        config = self.user_defined_strategy.qat_configs
        qat_config = copy.deepcopy(_quant_config_default)
        qat_config['quantize_op_types'] = [
            'conv2d',
            'depthwise_conv2d',
            'mul',
            'matmul',
            'matmul_v2',
        ]
        qat_config['weight_quantize_type'] = (
            'channel_wise_abs_max'
            if config['channel_wise_abs_max']
            else 'abs_max'
        )
        qat_config['weight_bits'] = config['weight_bits']
        qat_config['activation_bits'] = config['activation_bits']
        qat_config['not_quant_pattern'] = list(config['not_quant_pattern'])
        return qat_config

    def _replace_program(self, main_program, refer_program):
        main_program._rebuild_from_desc(refer_program.desc)

    def minimize_impl(
        self, loss, startup_program=None, parameter_list=None, no_grad_set=None
    ):
        optimize_ops, params_grads = self.inner_opt.minimize(
            loss,
            startup_program,
            parameter_list,
            no_grad_set,
        )
        device = paddle.device.get_device()
        place = paddle.set_device(device)
        qat_config = self._gen_qat_config()
        qat_program = quant_aware(
            loss.block.program, place, config=qat_config, return_program=True
        )
        self._replace_program(loss.block.program, qat_program)
        return optimize_ops, params_grads

    def qat_init(self, place, scope=None, test_program=None):
        if test_program is not None:
            qat_config = self._gen_qat_config()
            qat_program = quant_aware(
                test_program,
                place,
                scope=scope,
                config=qat_config,
                for_test=True,
                return_program=True,
            )
            self._replace_program(test_program, qat_program)
