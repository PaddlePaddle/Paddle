#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from types import MethodType

import numpy as np

import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.distributed import fleet
from paddle.fluid.dygraph import to_variable
from paddle.framework import core

from .base.topology import ParallelMode


def distributed_scaler(scaler):
    def unscale_method(self, optimizer):
        if not self._enable:
            return
        if getattr(optimizer, '_param_groups', None) and isinstance(
            optimizer._param_groups[0], dict
        ):
            param_grads = []
            param_grads_fp16 = []
            param_grads_fp32 = []
            for group in optimizer._param_groups:
                for param in group['params']:
                    if param._grad_ivar() is not None:
                        param_grads.append(param._grad_ivar())
                        if (
                            param._grad_ivar().dtype
                            == core.VarDesc.VarType.FP16
                        ):
                            param_grads_fp16.append(param._grad_ivar())
                        else:
                            param_grads_fp32.append(param._grad_ivar())
        else:
            strategy = fleet.fleet._user_defined_strategy
            sharding_stage_1_overlap = strategy.hybrid_configs[
                'sharding_configs'
            ].comm_overlap
            if sharding_stage_1_overlap:
                # If sharding stage 1 enable comm overlap and need do loss scale. Here we have to wait all comm tasks.
                # If no need do loss scale, the wait for all comm tasks will do in the optimizer step.
                assert hasattr(optimizer, "_comm_buffers")
                assert hasattr(optimizer, "_sharding_enable")
                if optimizer._sharding_enable:
                    # disable origin grad reduce in hybrid optimizer step
                    optimizer._sharding_enable = False
                for buffer in optimizer._comm_buffers:
                    buffer.scale_grads()
                # For sharding stage 1 under comm overlap, each rank only have to check finite for the response params.
                # For now, only sharding stage 1 contains this attr, this can be promoted to stage 2 and stage 3.
                assert hasattr(optimizer, "_local_parameter_list")
                parameters = optimizer._local_parameter_list
            else:
                parameters = optimizer._parameter_list
            param_grads_fp16 = [
                param._grad_ivar()
                for param in parameters
                if (param._grad_ivar() is not None)
                and (param._grad_ivar().dtype == core.VarDesc.VarType.FP16)
            ]
            param_grads_fp32 = [
                param._grad_ivar()
                for param in parameters
                if (param._grad_ivar() is not None)
                and (param._grad_ivar().dtype == core.VarDesc.VarType.FP32)
            ]
        temp_found_inf_fp16 = to_variable(np.array([0]).astype(np.bool_))
        temp_found_inf_fp32 = to_variable(np.array([0]).astype(np.bool_))
        self._found_inf = self._temp_found_inf_value_false
        if len(param_grads_fp16):
            _legacy_C_ops.check_finite_and_unscale(
                param_grads_fp16,
                self._scale,
                param_grads_fp16,
                temp_found_inf_fp16,
            )
            self._found_inf = _C_ops.bitwise_or(
                self._found_inf, temp_found_inf_fp16
            )
        if len(param_grads_fp32):
            _legacy_C_ops.check_finite_and_unscale(
                param_grads_fp32,
                self._scale,
                param_grads_fp32,
                temp_found_inf_fp32,
            )
            self._found_inf = _C_ops.bitwise_or(
                self._found_inf, temp_found_inf_fp32
            )

        self._found_inf = self._found_inf.cast("int32")

        # TODO(shenliang03) Since dp allreduce in the optimizer is
        # after the gradscaler, check_finite needs to synchronize global
        # information. In the future, we should use check_group to speed.
        paddle.distributed.all_reduce(
            self._found_inf, op=paddle.distributed.ReduceOp.MAX, group=None
        )
        self._found_inf = self._found_inf.cast("bool")

    # Only data_parallel doesn't need to modify scaler
    fleet_env = fleet.fleet
    if fleet_env._hcg.get_parallel_mode() is not ParallelMode.DATA_PARALLEL:
        scaler._unscale = MethodType(unscale_method, scaler)

    return scaler
