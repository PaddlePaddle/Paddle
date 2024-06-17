# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import test_op_translator

import paddle
from paddle.base import unique_name
from paddle.base.layer_helper import LayerHelper

paddle.pir_utils._switch_to_old_ir_()


class TestDistributedFusedLambInitOpTranslator(
    test_op_translator.TestOpTranslator
):
    def _create_persistable_var(self, name=None, shape=[-1], dtype='float32'):
        startup_block = self.helper.startup_program.global_block()
        if name is not None:
            name = unique_name.generate(name)
        startup_var = startup_block.create_var(
            name=name,
            shape=shape,
            dtype=dtype,
            persistable=True,
            stop_gradient=True,
        )
        main_block = self.helper.main_program.global_block()
        main_var = main_block.create_var(
            name=startup_var.name,
            shape=startup_var.shape,
            dtype=startup_var.dtype,
            persistable=True,
            stop_gradient=True,
        )
        return main_var

    def _create_scale_from_constant(self):
        name = unique_name.generate('global_scale')
        return paddle.static.create_global_var(
            name=name,
            shape=[1],
            dtype='float32',
            value=1.0,
            persistable=True,
        )

    def append_op(self):
        self.op_type = "distributed_fused_lamb_init"
        self.helper = LayerHelper('distributed_fused_lamb')
        rank = paddle.distributed.get_rank()
        nranks = paddle.distributed.get_world_size()
        local_rank = rank % nranks
        params = [paddle.ones(shape=(1, 1), dtype='float32')]
        grads = [paddle.ones(shape=(1, 1), dtype='float32')]
        apply_weight_decay = [1] * len(params)

        fp32_fused_param = self._create_persistable_var('fp32_fused_param')
        fp32_fused_grad = self._create_persistable_var('fp32_fused_grad')
        fp16_fused_param = self._create_persistable_var(
            'fp16_fused_param', dtype='float16'
        )
        fp16_fused_grad = self._create_persistable_var(
            'fp16_fused_grad', dtype='float16'
        )
        moment1 = self._create_persistable_var('moment1')
        moment1.is_distributed = True
        moment2 = self._create_persistable_var('moment2')
        moment2.is_distributed = True
        beta1pow = self._create_persistable_var('beta1pow')
        beta2pow = self._create_persistable_var('beta2pow')
        param_info = self._create_persistable_var('param_info', dtype='int32')
        param_info.is_distributed = True

        fused_offsets = self._create_persistable_var(
            'fused_offsets', dtype='int32'
        )

        fp32_partial_fused_offsets = self._create_persistable_var(
            'fp32_partial_fused_offsets', dtype='int32'
        )
        fp32_partial_fused_offsets.is_distributed = True

        fp16_partial_fused_offsets = self._create_persistable_var(
            'fp16_partial_fused_offsets', dtype='int32'
        )
        fp16_partial_fused_offsets.is_distributed = True

        param_order = self._create_persistable_var('param_order', dtype='int32')
        param_order.is_distributed = True

        scale = self._create_scale_from_constant()
        step = self._create_persistable_var('step', dtype='int64')

        master_params = []
        for p in params:
            master_p = self._create_persistable_var('master_weight')
            master_params.append(master_p)

        attrs = {
            'alignment': 128,
            'rank': local_rank,
            'nranks': nranks,
            'apply_weight_decay': apply_weight_decay,
            'moment1': 0.0,
            'moment2': 0.0,
            'beta1': 0.9,
            'beta2': 0.999,
        }
        helper = LayerHelper(self.op_type)
        helper.append_op(
            type=self.op_type,
            inputs={"Param": params, "Grad": grads},
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
            attrs=attrs,
        )

    def test_translator(self):
        self.check()


if __name__ == "__main__":
    unittest.main()
