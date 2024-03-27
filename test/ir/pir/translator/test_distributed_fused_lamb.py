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

import os
import unittest

import test_op_translator

import paddle
from paddle.base import core, unique_name
from paddle.base.layer_helper import LayerHelper


def init_communicator(block, rank, ranks, ring_id):
    eps = os.environ['PADDLE_TRAINER_ENDPOINTS']
    eps = [ep.strip() for ep in eps.split(",") if ep.strip()]
    cur_ep = eps[rank]
    other_eps = [eps[r] for r in ranks if r != rank]

    local_rank = ranks.index(rank)
    comm_var_name = unique_name.generate('comm_id')
    comm_id_var = block.create_var(
        name=comm_var_name, persistable=True, type=core.VarDesc.VarType.RAW
    )
    if core.is_compiled_with_cuda():
        block.append_op(
            type='c_gen_nccl_id',
            inputs={},
            outputs={'Out': comm_id_var},
            attrs={
                'rank': local_rank,
                'endpoint': cur_ep,
                'other_endpoints': other_eps,
                'ring_id': ring_id,
            },
        )
    elif core.is_compiled_with_xpu():
        block.append_op(
            type='c_gen_bkcl_id',
            inputs={},
            outputs={'Out': comm_id_var},
            attrs={
                'rank': local_rank,
                'endpoint': cur_ep,
                'other_endpoints': other_eps,
                'ring_id': ring_id,
            },
        )
    elif (
        paddle.distributed.ParallelEnv().device_type
        in paddle.device.get_all_custom_device_type()
    ):
        block.append_op(
            type='c_gen_xccl_id',
            inputs={},
            outputs={'Out': comm_id_var},
            attrs={
                'rank': local_rank,
                'endpoint': cur_ep,
                'other_endpoints': other_eps,
                'ring_id': ring_id,
            },
        )
    block.append_op(
        type='c_comm_init',
        inputs={'X': comm_id_var},
        outputs={},
        attrs={
            'nranks': len(ranks),
            'rank': local_rank,
            'ring_id': ring_id,
            'endpoints': ','.join(eps),
        },
    )
    tmp_var = block.create_var(name=unique_name.generate('tmp'))
    block.append_op(
        type='fill_constant', outputs={'Out': tmp_var}, attrs={'value': 1}
    )
    block.append_op(
        type='c_allreduce_sum',
        inputs={'X': tmp_var},
        outputs={'Out': tmp_var},
        attrs={'ring_id': ring_id, 'use_calc_stream': True},
    )
    block.append_op(
        type='c_sync_calc_stream',
        inputs={'X': tmp_var},
        outputs={'Out': tmp_var},
    )
    return ring_id


class TestDistributedFusedLambOpTranslator(test_op_translator.TestOpTranslator):
    def setUp(self):
        super().setUp()
        assert (
            not paddle.in_dynamic_mode()
        ), "DistributedFusedLamb does not support dygraph mode"
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._epsilon = 1e-6
        self._weight_decay = 0.01
        self._max_global_grad_norm = -1.0
        self._alignment = 128
        self._clip_after_allreduce = True
        self._is_grad_scaled_by_nranks = True
        self._scale = None
        self._use_master_param_norm = True
        self._gradient_accumulation_steps = 1
        self._use_master_acc_grad = True
        self._use_hierarchical_allreduce = False
        self.helper = LayerHelper("distributed_fused_lamb")

        main_block = self.helper.main_program.global_block()
        self._found_inf = main_block.create_var(
            name=unique_name.generate("found_inf"),
            shape=[1],
            dtype=core.VarDesc.VarType.BOOL,
        )
        self._step = None

        self._stop_update = main_block.create_var(
            name=unique_name.generate("stop_update"),
            shape=[1],
            dtype=core.VarDesc.VarType.BOOL,
        )

        self._param_to_master_param = {}

    def _create_persistable_var(self, name=None, shape=[-1], dtype="float32"):
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

    def _create_scale_from_constant(self, value):
        name = unique_name.generate('global_scale')
        return paddle.static.create_global_var(
            name=name,
            shape=[1],
            dtype='float32',
            value=float(value),
            persistable=True,
        )

    def append_op(self):
        self.op_type = "distributed_fused_lamb"
        params = [paddle.ones(shape=(1, 1), dtype="float32")]
        grads = [paddle.ones(shape=(1, 1), dtype="float32")]
        lr = paddle.to_tensor(0.001, dtype="float32")
        rank = paddle.distributed.get_rank()
        nranks = paddle.distributed.get_world_size()
        fp32_fused_param = self._create_persistable_var("fp32_fused_param")
        fp32_fused_grad = self._create_persistable_var("fp32_fused_grad")
        fp16_fused_param = self._create_persistable_var(
            "fp16_fused_param", dtype="float16"
        )
        fp16_fused_grad = self._create_persistable_var(
            "fp16_fused_grad", dtype="float16"
        )

        moment1 = self._create_persistable_var("moment1")
        moment1.is_distributed = True
        moment2 = self._create_persistable_var("moment2")
        moment2.is_distributed = True
        beta1pow = self._create_persistable_var("beta1pow")
        beta2pow = self._create_persistable_var("beta2pow")

        param_info = self._create_persistable_var("param_info", dtype="int32")
        param_info.is_distributed = True

        fused_offsets = self._create_persistable_var(
            "fused_offsets", dtype="int32"
        )

        fp32_partial_fused_offsets = self._create_persistable_var(
            "fp32_partial_fused_offsets", dtype="int32"
        )
        fp32_partial_fused_offsets.is_distributed = True

        fp16_partial_fused_offsets = self._create_persistable_var(
            "fp16_partial_fused_offsets", dtype="int32"
        )
        fp16_partial_fused_offsets.is_distributed = True

        param_order = self._create_persistable_var("param_order", dtype="int32")
        param_order.is_distributed = True

        fp32_acc_fused_grad = [
            self._create_persistable_var("fp32_acc_fused_grad")
        ]
        fp16_acc_fused_grad = [
            self._create_persistable_var("fp16_acc_fused_grad", dtype="float16")
        ]
        acc_step = [self._create_persistable_var("acc_step", dtype="int64")]

        scale = self._create_scale_from_constant(1.0)

        step = self._create_persistable_var('step', dtype='int64')

        ring_ids = []
        startup_block = self.helper.startup_program.global_block()
        ring_id = init_communicator(startup_block, rank, list(range(nranks)), 0)
        ring_ids.append(ring_id)

        attrs = {
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-6,
            "max_global_grad_norm": -1.0,
            "clip_after_allreduce": True,
            "rank": rank,
            "nranks": nranks,
            "ring_ids": ring_ids,
            "use_master_param_norm": True,
            "is_grad_scaled_by_nranks": True,
            "acc_steps": 1,
            "use_master_acc_grad": True,
            "use_hierarchical_allreduce": False,
        }

        helper = LayerHelper(self.op_type)
        helper.append_op(
            type=self.op_type,
            inputs={
                "FP32FusedParam": [fp32_fused_param],
                "FP32FusedGrad": [fp32_fused_grad],
                "FP16FusedParam": [fp16_fused_param],
                "FP16FusedGrad": [fp16_fused_grad],
                "LearningRate": [lr],
                "Moment1": [moment1],
                "Moment2": [moment2],
                "Beta1Pow": [beta1pow],
                "Beta2Pow": [beta2pow],
                "GlobalScale": [scale],
                "ParamInfo": [param_info],
                "Param": params,
                "Grad": grads,
                "FusedParamOffsets": [fused_offsets],
                "FP32ShardFusedParamOffsets": [fp32_partial_fused_offsets],
                "FP16ShardFusedParamOffsets": [fp16_partial_fused_offsets],
                "ParamOrder": [param_order],
            },
            outputs={
                "FP32FusedParamOut": [fp32_fused_param],
                "FP16FusedParamOut": [fp16_fused_param],
                "Moment1Out": [moment1],
                "Moment2Out": [moment2],
                "Beta1PowOut": [beta1pow],
                "Beta2PowOut": [beta2pow],
                "ParamOut": params,
                "GradOut": grads,
                "FoundInf": [self._found_inf],
                "FP32AccFusedGrad": fp32_acc_fused_grad,
                "FP16AccFusedGrad": fp16_acc_fused_grad,
                "AccStep": acc_step,
                "StopUpdate": self._stop_update,
                "Step": [step],
            },
            attrs=attrs,
        )

    def test_translator(self):
        self.check()


if __name__ == "__main__":
    unittest.main()
