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

import paddle
import paddle.autograd as imperative_base
from paddle import _legacy_C_ops

from ...base.topology import ParallelMode

__all__ = []


class HybridParallelGradScaler:
    def __init__(self, scaler, hcg):
        self._scaler = scaler
        self._hcg = hcg
        self._use_dp_mode = (
            self._hcg.get_parallel_mode() == ParallelMode.DATA_PARALLEL
        )

    def scale(self, var):
        return self._scaler.scale(var)

    def minimize(self, optimizer, *args, **kwargs):
        if not self._enable:
            return optimizer.minimize(*args, **kwargs)

        #  unscale the grad
        self._unscale(optimizer)

        optimize_ops, params_grads = (None, None)

        if hasattr(optimizer, "_set_auxiliary_var"):
            optimizer._set_auxiliary_var('found_inf', self._found_inf)
            optimize_ops, params_grads = optimizer.minimize(*args, **kwargs)
            self._cache_founf_inf = optimizer._get_auxiliary_var('found_inf')
        else:
            if self._found_inf:
                self._cache_founf_inf = True
            else:
                optimize_ops, params_grads = optimizer.minimize(*args, **kwargs)
                self._cache_founf_inf = False

        if self._use_dynamic_loss_scaling:
            self._update()

        return optimize_ops, params_grads

    @imperative_base.no_grad()
    def _unscale(self, optimizer):
        if not self._enable:
            return
        param_grads = [
            param._grad_ivar()
            for param in optimizer._parameter_list
            if param._grad_ivar() is not None
        ]
        _legacy_C_ops.check_finite_and_unscale(
            param_grads, self._scale, param_grads, self._found_inf
        )
        # allreduce_max found_inf in check_group
        if not self._use_dp_mode:
            self._found_inf = paddle.cast(self._found_inf, dtype="int32")
            # TODO(shenliang03) Since the minimize call in the optimizer is
            # after the gradscaler, check_finite needs to synchronize global
            # information. In the future, we should use check_group
            paddle.distributed.all_reduce(
                self._found_inf, op=paddle.distributed.ReduceOp.MAX, group=None
            )
            self._found_inf = paddle.cast(self._found_inf, dtype="bool")

    def __getattr__(self, item):
        return getattr(self._scaler, item)
